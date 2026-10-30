"""Ядро: подготовка картинки схемы для Claude, проверка netlist, экспорт.

Можно использовать без интерфейса:
    python core.py prep схема.jpg [папка_проекта]
    python core.py check папка_проекта
    python core.py export папка_проекта
    python core.py zoom папка_проекта F5:G6     (крупно, для спорных мест)
"""
import csv
import json
import math
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

TILE_MAX = 1400      # макс. сторона фрагмента в px (больше Claude всё равно уменьшит)
MAX_UPSCALE = 4.0    # сильнее увеличивать бессмысленно
MARGIN = 32          # поле с подписями сетки вокруг фрагмента

TYPES = {
    "R": "резистор", "C": "конденсатор", "L": "дроссель/катушка", "D": "диод",
    "Z": "стабилитрон", "Q": "биполярный транзистор", "M": "MOSFET", "IGBT": "IGBT",
    "U": "микросхема/ОУ", "RV": "варистор", "K": "реле", "T": "трансформатор",
    "J": "разъём/вывод", "F": "предохранитель", "BR": "диодный мост",
    "OC": "оптопара/оптодрайвер",
    "SENSOR": "датчик КИП (давление, расход, уровень, температура…)", "RT": "терморезистор (Pt100, NTC)",
    "TC": "термопара", "XTAL": "кварцевый резонатор", "LED": "светодиод", "MCU": "микроконтроллер",
    "TERM": "клеммник", "ENC": "энкодер", "TACHO": "тахогенератор", "HALL": "датчик Холла / приближения",
    "X": "прочее",
}
# Порядок выводов для SPICE
PIN_ORDER = {"Q": ["C", "B", "E"], "M": ["D", "G", "S"], "IGBT": ["C", "G", "E"],
             "D": ["A", "K"], "Z": ["A", "K"]}
SPICE_PREFIX = {"R": "R", "C": "C", "L": "L", "D": "D", "Z": "D", "Q": "Q", "M": "M"}
GROUND_NAMES = {"GND", "0", "AGND", "DGND"}


# ---------- картинки ----------

def imread(path):
    data = np.fromfile(str(path), dtype=np.uint8)      # работает с кириллицей в пути
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Не удалось открыть картинку: {path}")
    return img


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".png", img)
    if not ok:
        raise ValueError(f"Не удалось сохранить {path}")
    buf.tofile(str(path))


def col_name(i):
    """0 -> A, 25 -> Z, 26 -> AA"""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(name):
    n = 0
    for ch in name:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def parse_cell(cell):
    """'E3' -> (col, row) с нуля, или None."""
    if not isinstance(cell, str):
        return None
    m = re.fullmatch(r"\s*([A-Za-z]+)(\d+)\s*", cell)
    if not m:
        return None
    return col_index(m.group(1).upper()), int(m.group(2)) - 1


def parse_range(rng):
    """'A1:D7' или ['A1','D7'] -> (c0, r0, c1, r1) включительно, или None."""
    if isinstance(rng, str):
        rng = rng.split(":")
    if not rng:
        return None
    a = parse_cell(rng[0])
    b = parse_cell(rng[-1])
    if not a or not b:
        return None
    return min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])


def estimate_skew(gray):
    """Угол наклона по преобладающим почти горизонтальным/вертикальным линиям."""
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=80,
                            minLineLength=max(gray.shape) // 15, maxLineGap=3)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(float):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        a = (a + 45) % 90 - 45
        if abs(a) < 10:
            angles.append(a)
    return float(np.median(angles)) if angles else 0.0


def rotate(img, angle):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def clean(img_bgr):
    """Убирает цветной фон/тени и растягивает контраст. Возвращает серое изображение."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = max(15, (max(gray.shape) // 40) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4)
    norm = cv2.divide(gray, bg, scale=255)
    lo = float(np.percentile(norm, 0.5))
    out = (norm.astype(np.float32) - lo) * 255.0 / max(245.0 - lo, 1.0)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_region(gray, grid, c0, r0, c1, r1, scale):
    """Фрагмент клеток [c0..c1)x[r0..r1) с сеткой и подписями на полях.
    gray может быть и цветным (BGR) — тогда цвета сохраняются."""
    cell, W, H = grid["cell"], grid["width"], grid["height"]
    x0, y0 = c0 * cell, r0 * cell
    x1, y1 = min(W, c1 * cell), min(H, r1 * cell)
    crop = gray[y0:y1, x0:x1]
    interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=interp)
    if scale > 1.2:
        blur = cv2.GaussianBlur(big, (0, 0), 1.0 * scale / 2)
        big = cv2.addWeighted(big, 1.6, blur, -0.6, 0)
    bgr = big if big.ndim == 3 else cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    h, w = big.shape[:2]

    overlay = bgr.copy()
    for c in range(c0 + 1, c1):
        x = round((c * cell - x0) * scale)
        cv2.line(overlay, (x, 0), (x, h - 1), (0, 0, 255), 1)
    for r in range(r0 + 1, r1):
        y = round((r * cell - y0) * scale)
        cv2.line(overlay, (0, y), (w - 1, y), (0, 0, 255), 1)
    bgr = cv2.addWeighted(overlay, 0.45, bgr, 0.55, 0)

    out = np.full((h + 2 * MARGIN, w + 2 * MARGIN, 3), 255, np.uint8)
    out[MARGIN:MARGIN + h, MARGIN:MARGIN + w] = bgr
    cv2.rectangle(out, (MARGIN - 1, MARGIN - 1), (MARGIN + w, MARGIN + h), (0, 0, 200), 1)
    font, fs, th, color = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2, (0, 0, 200)
    for c in range(c0, c1):
        a, b = max(c * cell, x0), min((c + 1) * cell, x1)
        if b <= a:
            continue
        cx = MARGIN + round(((a + b) / 2 - x0) * scale)
        t = col_name(c)
        tw = cv2.getTextSize(t, font, fs, th)[0][0]
        cv2.putText(out, t, (cx - tw // 2, MARGIN - 9), font, fs, color, th)
        cv2.putText(out, t, (cx - tw // 2, MARGIN + h + 23), font, fs, color, th)
    for r in range(r0, r1):
        a, b = max(r * cell, y0), min((r + 1) * cell, y1)
        if b <= a:
            continue
        cy = MARGIN + round(((a + b) / 2 - y0) * scale) + 7
        t = str(r + 1)
        tw = cv2.getTextSize(t, font, fs, th)[0][0]
        cv2.putText(out, t, (MARGIN // 2 - tw // 2, cy), font, fs, color, th)
        cv2.putText(out, t, (MARGIN + w + MARGIN // 2 - tw // 2, cy), font, fs, color, th)
    return out


def _starts(n, size, step):
    if n <= size:
        return [0]
    s = list(range(0, n - size + 1, step))
    if s[-1] != n - size:
        s.append(n - size)
    return s


def prepare(image_path, project_dir=None, cells_long=12, tile_cells=3, deskew=True,
            progress=None):
    """Готовит проект: чистая картинка, обзор с сеткой, фрагменты, manifest, задание."""
    image_path = Path(image_path)
    project = Path(project_dir) if project_dir else image_path.with_name(image_path.stem + "_project")
    project.mkdir(parents=True, exist_ok=True)
    tiles_dir = project / "tiles"
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir()

    original = project / ("original" + image_path.suffix.lower())
    if image_path.resolve() != original.resolve():
        shutil.copyfile(image_path, original)

    img = imread(original)
    angle = estimate_skew(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if deskew else 0.0
    if abs(angle) > 0.15:
        img = rotate(img, angle)
    else:
        angle = 0.0
    gray = clean(img)
    H, W = gray.shape
    imwrite(project / "source.png", img)
    imwrite(project / "clean.png", gray)

    cell = math.ceil(max(W, H) / max(1, cells_long))
    ncols, nrows = math.ceil(W / cell), math.ceil(H / cell)
    grid = {"cell": cell, "cols": ncols, "rows": nrows, "width": W, "height": H}

    ov_scale = min(MAX_UPSCALE, TILE_MAX / max(W, H))
    imwrite(project / "overview.png", render_region(gray, grid, 0, 0, ncols, nrows, ov_scale))

    tile_cells = max(1, tile_cells)
    step = max(1, tile_cells - 1)            # перекрытие в одну клетку
    tiles = []
    rows_s, cols_s = _starts(nrows, tile_cells, step), _starts(ncols, tile_cells, step)
    total, k = len(rows_s) * len(cols_s), 0
    for r0 in rows_s:
        for c0 in cols_s:
            c1, r1 = min(ncols, c0 + tile_cells), min(nrows, r0 + tile_cells)
            w_px = min(W, c1 * cell) - c0 * cell
            h_px = min(H, r1 * cell) - r0 * cell
            scale = min(MAX_UPSCALE, (TILE_MAX - 2 * MARGIN) / max(w_px, h_px))
            k += 1
            rng = f"{col_name(c0)}{r0 + 1}-{col_name(c1 - 1)}{r1}"
            name = f"t{k:02d}_{rng}.png"
            imwrite(tiles_dir / name, render_region(gray, grid, c0, r0, c1, r1, scale))
            tiles.append({"file": f"tiles/{name}", "cells": rng.replace("-", ":"),
                          "scale": round(scale, 2)})
            if progress:
                progress(k, total)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "image": image_path.name, "deskew_deg": round(angle, 2),
        "grid": grid, "tile_cells": tile_cells, "tiles": tiles,
    }
    (project / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (project / "CLAUDE_TASK.md").write_text(task_text(manifest), encoding="utf-8")
    if not (project / "notes.json").exists():
        (project / "notes.json").write_text("[]", encoding="utf-8")
    return project


def zoom(project, cells, sub=2):
    """Крупный фрагмент диапазона клеток ('F6' или 'F5:G6') с мелкой сеткой: каждая
    клетка делится на sub×sub частей. Сохраняет в zoom/ и возвращает путь."""
    project = Path(project)
    grid = load_json(project / "manifest.json")["grid"]
    rng = parse_range(cells)
    if not rng:
        raise ValueError(f"Не понял диапазон клеток: {cells}")
    c0, r0, c1, r1 = rng
    gray = imread(project / "clean.png")[:, :, 0]
    fine = dict(grid, cell=grid["cell"] / sub)
    scale = min(8.0, (TILE_MAX - 2 * MARGIN) / (max(c1 - c0 + 1, r1 - r0 + 1) * grid["cell"]))
    img = render_region(gray, grid, c0, r0, c1 + 1, r1 + 1, scale)
    # мелкая сетка пунктиром поверх, подписи остаются от основной сетки
    h, w = img.shape[:2]
    for i in range(1, (c1 - c0 + 1) * sub):
        if i % sub:
            x = MARGIN + round(i * fine["cell"] * scale)
            img[MARGIN:h - MARGIN:6, x] = (255, 120, 120)
    for i in range(1, (r1 - r0 + 1) * sub):
        if i % sub:
            y = MARGIN + round(i * fine["cell"] * scale)
            img[y, MARGIN:w - MARGIN:6] = (255, 120, 120)
    out = project / "zoom"
    out.mkdir(exist_ok=True)
    path = out / f"zoom_{cells.replace(':', '-')}.png"
    imwrite(path, img)
    return path


def task_text(manifest):
    g = manifest["grid"]
    last = f"{col_name(g['cols'] - 1)}{g['rows']}"
    return f"""# Задание: перевести схему в netlist

Сетка: столбцы A..{col_name(g['cols'] - 1)}, строки 1..{g['rows']} (клетка = {g['cell']} px оригинала, весь лист A1:{last}).

1. Прочитай `overview.png` — общий вид, найди функциональные блоки.
2. Прочитай все фрагменты из `tiles/` (список в `manifest.json`). Соседние перекрываются на одну клетку.
3. Прочитай `notes.json` — вопросы и поправки пользователя по клеткам. Они важнее твоих догадок.
4. Запиши `netlist.json` (формат ниже). Программа подхватит файл сама и покажет ошибки.
5. Где не уверен — `"confidence": "low"` и вопрос в `questions`, а не догадка молча.
6. Спорное место (точка соединения, мелкий номинал) — увеличь его:
   `python "{Path(__file__).resolve()}" zoom "<папка проекта>" F5:G6` → файл в `zoom/`.
7. Проверка без интерфейса: `python "{Path(__file__).resolve()}" check "<папка проекта>"`.

## Формат netlist.json

```json
{{
  "title": "Название схемы",
  "components": [
    {{"ref": "R1", "type": "R", "value": "270k", "cell": "E1",
      "pins": {{"1": "VBUS", "2": "N_DIV1"}}}},
    {{"ref": "Q1", "type": "IGBT", "value": "FS25R12W1T4", "cell": "R6",
      "pins": {{"C": "VBUS", "G": "G1", "E": "U"}}, "confidence": "low",
      "note": "общий модуль Q1-Q6"}}
  ],
  "nets": {{"VBUS": {{"label": "+600V"}}, "G1": {{"port": true, "label": "к CPU"}}}},
  "blocks": [{{"name": "Выпрямитель", "cells": "A3:D7", "refs": ["D1", "D2"],
              "description": "Трёхфазный мост"}}],
  "questions": [{{"cell": "G3", "text": "Номинал конденсатора со знаком '?'"}}]
}}
```

Типы: {", ".join(f"{k} ({v})" for k, v in TYPES.items())}.
Выводы: Q → C/B/E, M → D/G/S, IGBT → C/G/E, D/Z → A/K, R/C/L → 1/2, ОУ → номера выводов.
Электролит — выводы `+` и `-`. МК (MCU) — имена выводов как на схеме (VDD, VSS, PA0, ADC1, SDA, NRST, XTAL1…).
Датчик 4–20 мА (SENSOR) — `+`/`-`, в номинале диапазон и сигнал: «PT 0-16 bar 4-20mA».
Энкодер (ENC) — A, B, Z, +U, 0V и «1024 имп» в номинале; тахогенератор (TACHO) — «20V/1000rpm»;
терморезистор (RT) — «Pt100» / «NTC 10k B3950»; термопара (TC) — «тип K»; кварц — «8MHz CL=20p».
Мощность резистора, если указана, — в номинале: «47 5W». Напряжения питания — в метках цепей: «+24 В», «+3,3 В».
Земля: `GND`. Цепь с одним выводом помечай `"port": true`, если это внешний вывод или метка.

Если в `analysis/` есть разбор без ИИ (`analysis/tiles/*.png` — цепи разными цветами,
`netlist_auto.json` — черновик), используй его как подсказку: одинаковый цвет = одна цепь. Но сверяй с картинкой —
анализатор ошибается на сканах и надписях.
"""


# ---------- netlist ----------

def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def net_index(nl):
    idx = {}
    for comp in nl.get("components", []):
        for pin, net in (comp.get("pins") or {}).items():
            if net:
                idx.setdefault(str(net), []).append((comp.get("ref", "?"), str(pin)))
    return idx


def check(nl, manifest=None):
    """Список замечаний: dict(level, ref|net, cell, text)."""
    issues = []
    grid = (manifest or {}).get("grid")

    def add(level, text, ref="", cell=""):
        issues.append({"level": level, "ref": ref, "cell": cell, "text": text})

    seen = {}
    for comp in nl.get("components", []):
        ref, cell = comp.get("ref", ""), comp.get("cell", "")
        if not ref:
            add("error", "Компонент без обозначения", cell=cell)
            continue
        if ref in seen:
            add("error", f"Повтор обозначения {ref}", ref, cell)
        seen[ref] = comp
        t = comp.get("type", "")
        if t not in TYPES:
            add("warn", f"Неизвестный тип '{t}'", ref, cell)
        pc = parse_cell(cell)
        if not pc:
            add("warn", "Не указана клетка (cell)", ref)
        elif grid and not (0 <= pc[0] < grid["cols"] and 0 <= pc[1] < grid["rows"]):
            add("error", f"Клетка {cell} вне листа", ref, cell)
        pins = comp.get("pins") or {}
        if not pins:
            add("error", "Нет выводов", ref, cell)
        elif len(pins) > 1 and len(set(pins.values())) == 1:
            add("warn", "Все выводы на одной цепи (закорочен?)", ref, cell)
        if t in PIN_ORDER:
            missing = [p for p in PIN_ORDER[t] if p not in pins]
            if missing:
                add("warn", f"Нет выводов {', '.join(missing)}", ref, cell)
        if t in ("R", "C", "L", "Z", "RV") and not comp.get("value"):
            add("warn", "Нет номинала", ref, cell)
        if comp.get("confidence") == "low":
            add("info", "Низкая уверенность" + (f": {comp['note']}" if comp.get("note") else ""),
                ref, cell)

    nets = nl.get("nets", {})
    for net, conns in sorted(net_index(nl).items()):
        meta = nets.get(net, {})
        if len(conns) == 1 and not meta.get("port") and net.upper() not in GROUND_NAMES:
            ref = conns[0][0]
            add("warn", f"Цепь '{net}' подключена только к {ref}.{conns[0][1]} (обрыв?)",
                ref, seen.get(ref, {}).get("cell", ""))
    for q in nl.get("questions", []):
        add("question", q.get("text", ""), cell=q.get("cell", ""))
    order = {"error": 0, "warn": 1, "question": 2, "info": 3}
    issues.sort(key=lambda i: order.get(i["level"], 9))
    return issues


def _spice_node(n):
    return "0" if n.upper() in GROUND_NAMES else re.sub(r"[^A-Za-z0-9_]", "_", n.replace("+", "P").replace("-", "N"))


def export_spice(nl):
    lines = [f"* {nl.get('title', 'schematic')}", "* сгенерировано Schematic Helper, проверь вручную", ""]
    for comp in nl.get("components", []):
        t, ref, pins = comp.get("type", "X"), comp.get("ref", "?"), comp.get("pins") or {}
        order = PIN_ORDER.get(t) or list(pins)
        nodes = " ".join(_spice_node(pins.get(p, "NC_" + ref + "_" + p)) for p in order)
        # «560uF/400V» -> 560uF, «0,47uF» -> 0.47uF; не-латиницу в модель не пускаем
        value = str(comp.get("value", "")).split("/")[0].strip().replace(",", ".")
        if not value or not value.isascii() or "?" in value:
            value = "UNKNOWN_" + t
        prefix = SPICE_PREFIX.get(t, "X")
        name = re.sub(r"[^A-Za-z0-9_]", "_", ref if ref.upper().startswith(prefix) else prefix + "_" + ref)
        comment = f"   ; {comp['note']}" if comp.get("note") else ""
        if prefix == "X":
            lines.append(f"{name} {nodes} {re.sub(r'[^A-Za-z0-9_]', '_', value)}   ; выводы: {' '.join(order)}{comment}")
        else:
            lines.append(f"{name} {nodes} {value}{comment}")
    lines += ["", ".end", ""]
    return "\n".join(lines)


def export_all(project):
    project = Path(project)
    nl = load_json(project / "netlist.json")
    if nl is None:
        raise FileNotFoundError("netlist.json ещё нет")
    out = project / "export"
    out.mkdir(exist_ok=True)
    (out / "schematic.cir").write_text(export_spice(nl), encoding="utf-8")

    groups = {}
    for comp in nl.get("components", []):
        key = (comp.get("type", ""), str(comp.get("value", "")))
        groups.setdefault(key, []).append(comp.get("ref", "?"))
    with open(out / "bom.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["Тип", "Описание", "Номинал", "Кол-во", "Обозначения"])
        for (t, v), refs in sorted(groups.items()):
            w.writerow([t, TYPES.get(t, ""), v, len(refs), ", ".join(refs)])

    idx = net_index(nl)
    md = [f"# {nl.get('title', 'Схема')}", ""]
    for b in nl.get("blocks", []):
        md += [f"## {b.get('name', '')} ({b.get('cells', '')})", "", b.get("description", ""), "",
               "Компоненты: " + ", ".join(b.get("refs", [])), ""]
    md += ["## Цепи", ""]
    for net, conns in sorted(idx.items()):
        label = nl.get("nets", {}).get(net, {}).get("label", "")
        md.append(f"- **{net}**{' (' + label + ')' if label else ''}: "
                  + ", ".join(f"{r}.{p}" for r, p in conns))
    issues = check(nl, load_json(project / "manifest.json"))
    if issues:
        md += ["", "## Замечания", ""]
        md += [f"- [{i['level']}] {i['ref'] or i['cell']}: {i['text']}" for i in issues]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "prep":
        p = prepare(arg, sys.argv[3] if len(sys.argv) > 3 else None,
                    progress=lambda k, n: print(f"\r{k}/{n}", end=""))
        print(f"\nГотово: {p}")
    elif cmd == "check":
        p = Path(arg)
        for i in check(load_json(p / "netlist.json", {}), load_json(p / "manifest.json")):
            print(f"[{i['level']}] {i['ref'] or '-'} {i['cell']}: {i['text']}")
    elif cmd == "export":
        print(export_all(arg))
    elif cmd == "zoom":
        print(zoom(arg, sys.argv[3]))
