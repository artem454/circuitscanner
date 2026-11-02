"""Эталоны условных обозначений из библиотек САПР (Altium Designer .SchLib, sPlan 8 .lib).

    python libsyms.py altium папка_SchLib [--show файл.png]   — извлечь, показать сводку и добавить в обучение
    python libsyms.py splan папка_lib [--show файл.png]       — то же для библиотеки sPlan (ГОСТ)

Как читается: .SchLib — составной файл OLE, внутри записи вида «|RECORD=13|Location.X=…|Corner.X=…|».
Берутся линии (13), ломаные (6), многоугольники (7), прямоугольники (14), дуги (12, 11) и эллипсы (8)
первой части компонента; надписи, параметры и выводы пропускаются. Координаты — в единицах 10 mil
(целая часть + *_Frac / 100000), ось Y вверх. Тип — по папке и имени библиотеки (TYPES ниже).
Дальше — как у эталонов из PDF (pdfsyms.add_to_training): символ размножается и проходит через
анализатор; источник примеров — «altium».
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# имя файла библиотеки (без расширения) → тип; чего нет — пропускается (разъёмы, модули, корпуса)
TYPES = {"resistor": "R", "cercapacitor": "C", "polarcapacitor": "C", "inductor": "L", "fuse": "F",
         "diode": "D", "zennerdiode": "Z", "led": "LED", "npn": "Q", "pnp": "Q", "mosfet_n": "M",
         "mosfet_p": "M", "crystal_2p": "XTAL", "crystal_4p_1_3": "XTAL", "crystal_4p_1_4": "XTAL",
         "diode_bridge": "BR"}
# bat и varistor в этой базе рисуются не тем, что ждёт классификатор (таблица, косая без корпуса) — не берём


def _records(data):
    for m in re.finditer(rb"\|RECORD=(\d+)\|([^\x00]*)", data):
        kv = {}
        for part in m.group(2).split(b"|"):
            if b"=" in part:
                k, v = part.split(b"=", 1)
                kv[k.decode("latin1")] = v.decode("latin1")
        yield int(m.group(1)), kv


def _num(kv, key):
    """Координата Altium: целая часть + дробная (_Frac, в стотысячных)."""
    try:
        return float(kv.get(key, 0)) + float(kv.get(key + "_Frac", 0)) / 100000
    except ValueError:
        return 0.0


def parse(path):
    """[(имя компонента, [примитивы])]; примитив — ("line", [(x, y), …]) / ("poly", …) / ("arc", cx, cy, rx, ry, a0, a1)."""
    data = Path(path).read_bytes()
    comps, cur = [], None
    for rec, kv in _records(data):
        if rec == 1:
            cur = (kv.get("LibReference", Path(path).stem), [])
            comps.append(cur)
            continue
        if cur is None or kv.get("OwnerPartId", "1") not in ("1", "-1") or kv.get("OwnerPartDisplayMode", "0") != "0":
            continue
        prims = cur[1]
        if rec == 13:
            prims.append(("line", [(_num(kv, "Location.X"), _num(kv, "Location.Y")),
                                   (_num(kv, "Corner.X"), _num(kv, "Corner.Y"))]))
        elif rec in (6, 7):
            n = int(kv.get("LocationCount", 0) or 0)
            pts = [(_num(kv, f"X{i}"), _num(kv, f"Y{i}")) for i in range(1, n + 1)]
            if len(pts) >= 2:
                prims.append(("poly" if rec == 7 else "line", pts))
        elif rec == 14:
            x0, y0, x1, y1 = (_num(kv, "Location.X"), _num(kv, "Location.Y"), _num(kv, "Corner.X"), _num(kv, "Corner.Y"))
            prims.append(("line", [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]))
        elif rec in (8, 11, 12):
            r = _num(kv, "Radius")
            r2 = _num(kv, "SecondaryRadius") if rec in (8, 11) else r
            a0, a1 = (0.0, 360.0) if rec == 8 else (float(kv.get("StartAngle", 0)), float(kv.get("EndAngle", 360)))
            prims.append(("arc", _num(kv, "Location.X"), _num(kv, "Location.Y"), r, r2 or r, a0, a1))
    return [c for c in comps if c[1]]


TWO_LEAD = {"R", "C", "L", "F", "D", "Z", "LED", "XTAL"}


def with_leads(prims):
    """Выводы в .SchLib хранятся в двоичном виде и не читаются; двухвыводному символу дорисовываем
    провода слева и справа по средней линии — иначе анализатор не отличит, например, пластины
    конденсатора от обрывков провода."""
    xs, ys = [], []
    for p in prims:
        if p[0] == "arc":
            xs += [p[1] - p[3], p[1] + p[3]]
            ys += [p[2] - p[4], p[2] + p[4]]
        else:
            xs += [x for x, _ in p[1]]
            ys += [y for _, y in p[1]]
    x0, x1, yc = min(xs), max(xs), (min(ys) + max(ys)) / 2
    lead = 0.6 * max(x1 - x0, max(ys) - min(ys))
    return prims + [("line", [(x0 - lead, yc), (x0, yc)]), ("line", [(x1, yc), (x1 + lead, yc)])]


def _extent(prims):
    xs, ys = [], []
    for p in prims:
        if p[0] == "arc":
            xs += [p[1] - p[3], p[1] + p[3]]
            ys += [p[2] - p[4], p[2] + p[4]]
        else:
            xs += [x for x, _ in p[1]]
            ys += [y for _, y in p[1]]
    return min(xs), min(ys), max(xs), max(ys)


def render(prims, size=160, pad=6, body_prims=None):
    """Примитивы → серая картинка (чёрное на белом), символ вписан в size×size. С body_prims —
    ещё и рамка тела символа (без дорисованных выводов) в пикселях картинки."""
    xs, ys = [], []
    for p in prims:
        if p[0] == "arc":
            _, cx, cy, rx, ry, _, _ = p
            xs += [cx - rx, cx + rx]
            ys += [cy - ry, cy + ry]
        else:
            xs += [x for x, _ in p[1]]
            ys += [y for _, y in p[1]]
    x0, y0 = min(xs), min(ys)
    span = max(max(xs) - x0, max(ys) - y0, 1e-6)
    k = (size - 2 * pad) / span
    W = int((max(xs) - x0) * k) + 2 * pad + 1
    H = int((max(ys) - y0) * k) + 2 * pad + 1
    img = np.full((H, W), 255, np.uint8)
    tr = lambda x, y: (int(round((x - x0) * k + pad)), int(round(H - 1 - ((y - y0) * k + pad))))   # Y вверх
    t = max(1, size // 60)
    for p in prims:
        if p[0] == "arc":
            _, cx, cy, rx, ry, a0, a1 = p
            if a1 <= a0:
                a1 += 360
            c = tr(cx, cy)
            # в картинке ось Y вниз — углы зеркалятся
            cv2.ellipse(img, c, (max(1, int(rx * k)), max(1, int(ry * k))), 0, -a1, -a0, 0, t, cv2.LINE_AA)
        else:
            pts = np.array([tr(x, y) for x, y in p[1]], np.int32)
            if p[0] == "poly":
                cv2.fillPoly(img, [pts], 0, cv2.LINE_AA)
            cv2.polylines(img, [pts], False, 0, t, cv2.LINE_AA)
    if body_prims is None:
        return img
    bx0, by0, bx1, by1 = _extent(body_prims)
    (a0, b1), (a1, b0) = tr(bx0, by0), tr(bx1, by1)
    return img, [a0, b0, a1, b1]


# --- sPlan 8 (.lib): двоичный формат. Каждая фигура — заголовок в 24 байта: тип, 00, флаг, 04,
# 11 байт, цвет заливки (3 байта), 6 нулей; дальше точки float32 (x, y). 04 — ломаная и 05 — залитый
# многоугольник: число отрезков n, затем n+1 точек; 01 — окружность/дуга, тоже как ломаная (n, n+1
# точек); 02 — прямоугольник, 5 точек. Надписи (обозначение, номинал) пропускаются. Компонент
# заканчивается признаком 01 01 01 01 и своим названием (UTF-16 с длиной), фигуры до него — его.
_SPLAN_HDR = re.compile(rb"([\x01\x02\x04\x05])\x00[\x00\x01]\x04[\x00-\xff]{11}[\x00-\xff]{3}\x00{6}", re.S)
_SPLAN_NAME = re.compile(rb"\x01\x01\x01\x01([\x01-\x50])\x00\x00\x00", re.S)


def _pts(data, pos, n):
    import struct
    if pos + 8 * n > len(data):
        return None
    v = struct.unpack_from(f"<{2 * n}f", data, pos)
    if not all(abs(t) < 20000 for t in v):
        return None
    return [(v[2 * i], -v[2 * i + 1]) for i in range(n)]          # в sPlan ось Y вниз


def parse_splan(path):
    """[(название компонента, [примитивы])] из библиотеки sPlan."""
    import struct
    data = Path(path).read_bytes()
    names = []
    for m in _SPLAN_NAME.finditer(data):
        n = m.group(1)[0]
        s = data[m.end():m.end() + 2 * n]
        try:
            names.append((m.start(), s.decode("utf-16-le")))
        except UnicodeDecodeError:
            pass
    comps, prev = [], 0
    for end, name in names:
        prims = []
        for h in _SPLAN_HDR.finditer(data, prev, end):
            t, p = h.group(1)[0], h.end()
            if t == 2:
                pts = _pts(data, p, 5)
            else:
                cnt = struct.unpack_from("<I", data, p)[0]
                pts = _pts(data, p + 4, cnt + 1) if 1 <= cnt <= 200 else None
            if pts:
                prims.append(("poly" if t == 5 else "line", pts))
        if prims:
            comps.append((name, prims))
        prev = end
    return comps


def splan_items(root):
    """Компоненты из папки библиотек sPlan → [{"img", "body", "type", "name"}]; тип — по названию
    компонента (pdfsyms.name_type), неопознанные пропускаются."""
    import pdfsyms
    items, per_type = [], {}
    for f in sorted(Path(root).rglob("*.lib")):
        for name, prims in parse_splan(f):
            typ = pdfsyms.name_type(name)
            if not typ or typ in ("PORT", "GND") or "RGB" in name:
                continue                                 # RGB-светодиод в библиотеке — составной, читается обрывками
            per_type[typ] = per_type.get(typ, 0) + 1
            if per_type[typ] > 6:
                continue                                 # 17 резисторов разной мощности перекосили бы обучение
            img, body = render(prims, body_prims=prims)
            items.append({"img": img, "body": body, "type": typ, "name": f"{f.stem}/{name}"})
    return items


def altium_items(root):
    """Все компоненты известных типов из папки с .SchLib → [{"img", "type", "name"}]."""
    items = []
    for f in sorted(Path(root).rglob("*")):
        if f.suffix.lower() != ".schlib":
            continue
        typ = TYPES.get(f.stem.lower())
        if not typ:
            continue
        for name, prims in parse(f):
            full = with_leads(prims) if typ in TWO_LEAD else prims
            img, body = render(full, body_prims=prims)
            items.append({"img": img, "body": body, "type": typ, "name": f"{f.stem}/{name}"})
    return items


if __name__ == "__main__":
    import pdfsyms
    if sys.argv[1:2] not in (["altium"], ["splan"]) or len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    items = altium_items(sys.argv[2]) if src == "altium" else splan_items(sys.argv[2])
    from collections import Counter
    print("символов:", len(items), dict(Counter(i["type"] for i in items)))
    if "--show" in sys.argv:
        pdfsyms.contact_sheet(items, sys.argv[sys.argv.index("--show") + 1])
        print("лист символов:", sys.argv[sys.argv.index("--show") + 1])
    else:
        print("добавлено в обучение:", pdfsyms.add_to_training(items, per=24, src=src))
