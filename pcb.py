"""Печатная плата по netlist.json: посадочные места → расстановка → трассировка → Gerber.

    python pcb.py папка_проекта [ширина_мм высота_мм]

Результат в папке проекта/pcb/:
  board.png, board.svg — вид платы (нижний слой синий, верхний красный, переходные — зелёные);
  *.gbr — Gerber RS-274X (медь сверху/снизу, маска, шелкография, контур), drill.drl — сверловка Excellon;
  pcb.json — расстановка и дорожки; report.md — что разведено, что нет и что проверить.

Как устроено:
  • посадочные места выводные (THT) по типу и номиналу: резисторы (по мощности), конденсаторы и электролиты
    (по ёмкости и напряжению), диоды, TO-92/TO-220/TO-247 (корпус из справочника, если указан), DIP по числу
    выводов, мосты, реле (контакты «K1.1» садятся в корпус реле K1), клеммники, предохранители, кварцы;
  • детали вне платы (двигатели, лампы, кнопки, датчики, контакты пускателей) и выводы-метки схемы
    становятся клеммниками;
  • расстановка: клеммники к краю, остальное по одной — туда, где короче связи с уже поставленными;
  • трассировка — волновой поиск (A*) по сетке 0,635 мм на двух слоях с переходными отверстиями;
    силовые цепи шире (1–2 мм), у высоковольтных зазор 2,5 мм; неразведённое — в отчёт.
"""
import heapq
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

import core
import diagnose as dg
import parts

MIL = 2.54
G = 0.635                                   # шаг сетки трассировки, мм
VIA_D, VIA_DRILL = 1.2, 0.6
OFFBOARD = {"MOTOR", "LAMP", "BTN", "QF", "SENSOR", "ENC", "TACHO", "HALL", "TC", "SW"}
PANEL_WORDS = re.compile(r"пускател|контактор|LC1|КМИ|ПМЛ|ПМА|КМ\d|тепл|LRD|РТИ|РТЛ|автомат|380|~\s*2[23]0|230\s*V\s*AC|AC\s*230", re.I)


def is_panel(comp):
    """Аппарат щита (на плату не ставится): пускатели и контакторы, автоматы, тепловые реле,
    двигатели, кнопки, лампы, датчики. Маленькое реле с катушкой до 48 В — на плату."""
    t = comp.get("type")
    if t in OFFBOARD:
        return True
    if t == "K":
        txt = f"{comp.get('ref', '')} {comp.get('value', '')} {comp.get('note', '')}"
        if re.search(r"SRD|HF\d|RT\d|JQC|TRU|G5L|G2R|герк|reed", txt, re.I):
            return False
        m = re.search(r"(\d+(?:[.,]\d+)?)\s*[VВ]", txt)
        if m and float(m.group(1).replace(",", ".")) <= 48 and not PANEL_WORDS.search(txt):
            return False
        return bool(PANEL_WORDS.search(txt)) or str(comp.get("ref", "")).upper().startswith(("KM", "КМ"))
    return False


def panel_share(nl):
    """Доля аппаратов щита среди деталей: почти одна релейно-контакторная автоматика — плата не нужна."""
    comps = [c for c in nl.get("components", []) if c.get("pins")]
    if not comps:
        return 0.0
    relays = {c["ref"] for c in comps if c.get("type") == "K" and is_panel(c)}
    panel = sum(1 for c in comps if is_panel(c) or (c.get("type") == "SW" and c["ref"].split(".")[0] in relays)
                or c.get("type") in ("F", "QF"))
    return panel / len(comps)


# ---------- посадочные места ----------

def _row(names, pitch, drill=0.9, pad=1.8, square_first=True):
    n = len(names)
    return [{"name": nm, "x": (i - (n - 1) / 2) * pitch, "y": 0.0, "drill": drill, "d": pad,
             "shape": "square" if i == 0 and square_first else "round"} for i, nm in enumerate(names)]


def _dip(names_by_num, n):
    pads = []
    rows = n // 2
    for k in range(1, n + 1):
        if k <= rows:
            x, y = (k - 1 - (rows - 1) / 2) * MIL, 3.81
        else:
            x, y = (n - k - (rows - 1) / 2) * MIL, -3.81
        pads.append({"name": names_by_num.get(k, str(k)), "x": x, "y": y, "drill": 0.8, "d": 1.6,
                     "shape": "square" if k == 1 else "round"})
    return pads


def _cap_size(val, vmax):
    """Диаметр электролита (мм) по CV — по типовым рядам производителей."""
    cv = (val or 1e-5) * 1e6 * (vmax or 25)
    for lim, d in ((300, 5), (1500, 6.3), (4000, 8), (12000, 10), (40000, 13), (120000, 16), (300000, 22), (1e12, 30)):
        if cv <= lim:
            return d


def footprint(comp):
    """Посадочное место детали: выводы (мм, относительно центра) и размер корпуса."""
    t = comp.get("type", "X")
    v = str(comp.get("value", ""))
    pins = list((comp.get("pins") or {}).keys())
    val, vmax = dg.parse_value(v)
    name, d = parts.lookup(v)
    pkg = (d or {}).get("pkg", "")
    note = ""
    if t == "R" and len(pins) == 3:                                   # потенциометр / подстроечник
        pads, w, h, note = _row(pins, 5.08), 10, 10, "подстроечный резистор"
    elif t in ("R", "RT", "L"):
        p = re.search(r"(\d+(?:[.,]\d+)?)\s*(W|Вт)", v + " " + str(comp.get("note", "")), re.I)
        pw = float(p.group(1).replace(",", ".")) if p else 0.25
        pitch = 10.16 if pw <= 0.5 else 15.24 if pw <= 2 else 25.4
        pads, w, h = _row(pins[:2] or ["1", "2"], pitch, drill=1.0 if pw > 1 else 0.8), pitch - 2.5, 2.5 + pw
        note = f"выводной, {pw:g} Вт"
    elif t == "C" and "+" in pins:
        dia = _cap_size(val, vmax)
        pitch = 2.5 if dia <= 6.3 else 3.5 if dia <= 8 else 5.0 if dia <= 13 else 7.5 if dia <= 18 else 10
        pads, w, h = _row(["+", "-"], pitch, drill=1.0 if dia > 10 else 0.8), dia, dia
        note = f"электролит ⌀{dia} мм"
    elif t == "C":
        big = (vmax or 0) >= 250 or (val or 0) >= 0.47e-6
        pitch = 15.0 if (val or 0) >= 1e-6 else 10.0 if big else 5.08
        pads, w, h = _row(pins[:2] or ["1", "2"], pitch, square_first=False), pitch + 3, 5 if big else 3
        note = "плёночный" if big else "керамический"
    elif t in ("D", "Z"):
        heavy = re.search(r"540|560|580|BYV|MUR8|30C|SR5|1N54", v, re.I)
        pitch = 12.7 if heavy else 10.16
        pads, w, h = _row(["K", "A"] if "K" in pins else pins, pitch, drill=1.2 if heavy else 0.9), pitch - 4, 3.5
        note = "DO-201" if heavy else "DO-41"
    elif t == "LED":
        pads, w, h, note = _row(["K", "A"] if "K" in pins else pins, MIL), 5, 5, "светодиод 5 мм"
    elif t in ("Q", "M", "IGBT", "VS"):
        order = {"Q": ["E", "B", "C"], "M": ["G", "D", "S"], "IGBT": ["G", "C", "E"]}.get(t)
        if t == "VS":
            order = ["A1", "A2", "G"] if "A1" in pins else ["K", "A", "G"]
        order = [p for p in (order or pins) if p in pins] + [p for p in pins if p not in (order or [])]
        if "247" in pkg or t == "IGBT":
            pads, w, h, note = _row(order, 5.45, drill=1.4, pad=2.6), 16, 5, "TO-247"
        elif "220" in pkg or t in ("M", "VS") or (d and d.get("imax", 0) > 1.5):
            pads, w, h, note = _row(order, MIL, drill=1.1, pad=2.0), 10.5, 4.6, "TO-220"
        else:
            pads, w, h, note = _row(order, MIL, drill=0.8, pad=1.5), 5, 4, "TO-92"
        note += " — проверьте цоколёвку по даташиту"
    elif t == "XTAL":
        pads, w, h, note = _row(pins[:2] or ["1", "2"], 4.88), 11, 4.5, "HC-49"
    elif t == "RV":
        pads, w, h, note = _row(pins[:2] or ["1", "2"], 7.5), 14, 6, "варистор ⌀14"
    elif t == "F":
        pads, w, h, note = _row(pins[:2] or ["1", "2"], 22.6, drill=1.3, pad=3.0), 26, 8, "держатель 5×20"
    elif t == "BR":
        order = [p for p in ("+", "AC1", "AC2", "-") if p in pins] or pins
        big = re.search(r"KBPC|GBPC|GBJ|KBU|25|35|50", v, re.I)
        pads, w, h = _row(order, 7.5 if big else 5.0, drill=1.3 if big else 1.0, pad=2.6 if big else 2.0), \
            (30 if big else 20), (8 if big else 4)
        note = "мост KBU/GBJ" if big else "мост KBP"
    elif t in ("U", "MCU", "OC"):
        nums = [int(p) for p in pins if str(p).isdigit()]
        n = max(nums) if nums and len(nums) >= len(pins) * 0.8 else len(pins)
        n = max(4, n + (n % 2))
        if t == "OC" and n <= 6:
            n = 6 if any(str(p) in ("5", "6") for p in pins) else 4
        by_num = {int(p): p for p in pins if str(p).isdigit()}
        named = [p for p in pins if not str(p).isdigit()]
        for k in range(1, n + 1):                        # именованные выводы — по порядку (цоколёвку проверить!)
            if k not in by_num and named:
                by_num[k] = named.pop(0)
        pads = _dip(by_num, n)
        w, h = n / 2 * MIL, 6.4
        note = f"DIP-{n}" + (" — выводы по именам расставлены условно, сверьте с даташитом" if
                             any(not str(p).isdigit() for p in pins) else "")
    elif t == "K":
        pads = [{"name": "A1", "x": -6.0, "y": 3.0, "drill": 1.0, "d": 2.0, "shape": "square"},
                {"name": "A2", "x": -6.0, "y": -3.0, "drill": 1.0, "d": 2.0, "shape": "round"}]
        w, h, note = 19, 15.5, "реле (типа SRD/HF)"
    elif t == "T":
        half = (len(pins) + 1) // 2
        pads = [{"name": p, "x": (i - (half - 1) / 2) * 5.0, "y": 10.0 if i < half else -10.0, "drill": 1.2,
                 "d": 2.4, "shape": "round"} for i, p in enumerate(pins)]
        for i, p in enumerate(pads[half:]):
            p["x"] = (i - (len(pins) - half - 1) / 2) * 5.0
        w, h, note = max(5 * half + 6, 25), 26, "трансформатор (габарит условный)"
    else:                                                # клеммник: J, TERM, X и детали вне платы
        pads, w, h = _row(pins or ["1"], 5.08, drill=1.3, pad=2.6), max(len(pins), 1) * 5.08 + 1, 8
        note = "клеммник 5,08"
    return {"pads": pads, "w": w, "h": h, "note": note}


# ---------- подготовка компонентов ----------

def build_parts(nl):
    """Детали платы: реле с контактами, клеммники для деталей вне платы и выводов схемы."""
    comps = {c["ref"]: dict(c) for c in nl.get("components", []) if c.get("ref") and c.get("pins")}
    out = {}
    relays = {r for r, c in comps.items() if c.get("type") == "K" and not is_panel(c)}
    panel_relays = {r for r, c in comps.items() if c.get("type") == "K" and is_panel(c)}
    absorbed = set()
    for r in relays:
        c = comps[r]
        fp = footprint(c)
        pins = dict(c["pins"])
        k = 0
        for r2, c2 in comps.items():                     # контакты K1.1, K1.2 → в корпус реле K1
            if r2.startswith(r + ".") and c2.get("type") == "SW" and len(c2["pins"]) == 2:
                for pn, net in c2["pins"].items():
                    nm = f"{r2.split('.', 1)[1]}_{pn}"
                    fp["pads"].append({"name": nm, "x": 1.0 + 5.0 * (k // 2), "y": 3.0 if k % 2 == 0 else -3.0,
                                       "drill": 1.3, "d": 2.4, "shape": "round"})
                    pins[nm] = net
                    k += 1
                absorbed.add(r2)
        fp["w"] = max(fp["w"], 8 + 5 * ((k + 1) // 2) + 4)
        out[r] = {"ref": r, "type": "K", "value": c.get("value", ""), "pins": pins, "fp": fp}
    for r, c in comps.items():
        if r in out or r in absorbed:
            continue
        t = c.get("type")
        if is_panel(c) or r in panel_relays or (t == "SW" and r not in absorbed):
            pins = {f"{r}:{pn}": net for pn, net in c["pins"].items()}
            fp = footprint({"type": "TERM", "pins": pins})
            fp["note"] = f"клеммник к {r} ({c.get('value', '') or t}) — аппарат щита / деталь вне платы"
            out["X_" + r] = {"ref": "X_" + r, "type": "TERM", "value": f"к {r}", "pins": pins, "fp": fp, "edge": True}
        else:
            out[r] = {"ref": r, "type": t, "value": c.get("value", ""), "pins": c["pins"], "fp": footprint(c),
                      "edge": t in ("J", "TERM")}       # «X» (неопознанное) — обычная деталь, не разъём
    # выводы-метки схемы, которые не выходят на разъём, — общий клеммник XP
    on_term = {n for p in out.values() if p.get("edge") for n in p["pins"].values()}
    used = {n for p in out.values() for n in p["pins"].values()}
    ports = [n for n, m in (nl.get("nets") or {}).items() if m.get("port") and n in used and n not in on_term]
    if ports:
        pins = {f"P{i + 1}": n for i, n in enumerate(ports)}
        fp = footprint({"type": "TERM", "pins": pins})
        fp["note"] = "внешние выводы схемы: " + ", ".join(ports)
        out["XP"] = {"ref": "XP", "type": "TERM", "value": "выводы", "pins": pins, "fp": fp, "edge": True}
    return out


def pad_positions(p, x, y, rot):
    out = []
    for pd in p["fp"]["pads"]:
        px, py = pd["x"], pd["y"]
        if rot == 90:
            px, py = -py, px
        out.append((pd, x + px, y + py))
    return out


def body_box(p, x, y, rot, margin=1.27):
    w, h = p["fp"]["w"], p["fp"]["h"]
    xs = [pd["x"] for pd in p["fp"]["pads"]] + [-w / 2, w / 2]
    ys = [pd["y"] for pd in p["fp"]["pads"]] + [-h / 2, h / 2]
    x0, x1, y0, y1 = min(xs) - 1.5, max(xs) + 1.5, min(ys) - 1.5, max(ys) + 1.5
    if rot == 90:
        x0, x1, y0, y1 = -y1, -y0, x0, x1
    return x + x0 - margin, y + y0 - margin, x + x1 + margin, y + y1 + margin


# ---------- расстановка ----------

def place(P, W=None, H=None):
    """Жадная расстановка по длине связей. Возвращает (ширина, высота, {ref: (x, y, rot)})."""
    area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in (body_box(p, 0, 0, 0) for p in P.values()))
    if not W:
        W = math.ceil(math.sqrt(area * 1.35 * 1.4) / MIL) * MIL
        H = math.ceil(area * 1.35 / W / MIL) * MIL
        # разъёмы стоят по периметру — периметр должен вместить их длину
        edge_len = sum(body_box(p, 0, 0, 90)[3] - body_box(p, 0, 0, 90)[1] for p in P.values() if p.get("edge"))
        while 2 * (W + H) * 0.8 < edge_len:
            W, H = W * 1.1, H * 1.1
    for attempt in range(25):
        pos = _try_place(P, W, H)
        if pos:
            return W, H, pos
        W, H = W * 1.1, H * 1.1                       # не поместилось — плата больше (до ×10 по площади)
    raise RuntimeError(f"не удалось расставить {len(P)} деталей даже на плате {W:.0f}×{H:.0f} мм")


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _try_place(P, W, H):
    pos, boxes = {}, []
    nets_pins = {}
    edge = [r for r, p in P.items() if p.get("edge")]
    # клеммники — по периметру: левый, правый, верхний, нижний край; не уместившиеся — внутрь, как детали
    fill = {"L": 2.0, "R": 2.0, "T": 2.0, "B": 2.0}
    for r in sorted(edge, key=lambda r: -len(P[r]["pins"])):
        placed = False
        for side in sorted(fill, key=lambda k: fill[k] / (H if k in "LR" else W)):
            rot = 90 if side in "LR" else 0
            b0 = body_box(P[r], 0, 0, rot)
            length = (b0[3] - b0[1]) if side in "LR" else (b0[2] - b0[0])
            limit = H if side in "LR" else W
            if fill[side] + length > limit - 2.0:
                continue
            if side == "L":
                x, y = -b0[0] + 0.5, fill[side] - b0[1]
            elif side == "R":
                x, y = W - b0[2] - 0.5, fill[side] - b0[1]
            elif side == "T":
                x, y = fill[side] - b0[0], -b0[1] + 0.5
            else:
                x, y = fill[side] - b0[0], H - b0[3] - 0.5
            bb = body_box(P[r], round(x / G) * G, round(y / G) * G, rot)
            if any(_overlap(bb, o) for o in boxes):
                fill[side] += length
                continue
            fill[side] += length
            placed = True
            break
        if not placed:
            continue
        pos[r] = (round(x / G) * G, round(y / G) * G, rot)
        boxes.append(body_box(P[r], *pos[r]))
        for pd, px, py in pad_positions(P[r], *pos[r]):
            net = P[r]["pins"].get(pd["name"])
            if net:
                nets_pins.setdefault(net, []).append((px, py))
    rest = [r for r in P if r not in pos]
    degree = {r: sum(len([q for q in P if net in P[q]["pins"].values()]) for net in P[r]["pins"].values()) for r in rest}
    rest.sort(key=lambda r: (-degree[r], r))
    step = MIL
    for r in rest:
        best = None
        for rot in (0, 90):
            b0 = body_box(P[r], 0, 0, rot)
            xs = np.arange(-b0[0] + 0.5, W - b0[2] - 0.5, step)
            ys = np.arange(-b0[1] + 0.5, H - b0[3] - 0.5, step)
            if not len(xs) or not len(ys):
                continue
            X, Y = np.meshgrid(xs, ys)
            X, Y = X.ravel(), Y.ravel()
            ok = np.ones(len(X), bool)
            if boxes:
                B = np.array(boxes)
                ok = ~(((X[:, None] + b0[0]) < B[None, :, 2]) & ((X[:, None] + b0[2]) > B[None, :, 0]) &
                       ((Y[:, None] + b0[1]) < B[None, :, 3]) & ((Y[:, None] + b0[3]) > B[None, :, 1])).any(1)
            if not ok.any():
                continue
            X, Y = X[ok], Y[ok]
            cost = 0.02 * (np.abs(X - W / 2) + np.abs(Y - H / 2))
            for pd, px, py in pad_positions(P[r], 0, 0, rot):
                pts = nets_pins.get(P[r]["pins"].get(pd["name"]))
                if pts:
                    Q = np.array(pts)
                    cost = cost + (np.abs(X[:, None] + px - Q[None, :, 0]) + np.abs(Y[:, None] + py - Q[None, :, 1])).min(1)
            k = int(np.argmin(cost))
            if best is None or cost[k] < best[0]:
                best = (float(cost[k]), float(X[k]), float(Y[k]), rot)
        if best is None:
            return None
        _, x, y, rot = best
        pos[r] = (round(x / G) * G, round(y / G) * G, rot)
        boxes.append(body_box(P[r], *pos[r]))
        for pd, px, py in pad_positions(P[r], *pos[r]):
            net = P[r]["pins"].get(pd["name"])
            if net:
                nets_pins.setdefault(net, []).append((px, py))
    return pos


# ---------- схема соединений щита (когда плата не нужна) ----------

def wiring_list(nl):
    """Перечень проводов щита: каждая цепь — провод с номером, по порядку подключения аппаратов."""
    comps = [c for c in nl.get("components", []) if c.get("pins")]
    nets = {}
    for c in comps:
        for pn, net in c["pins"].items():
            nets.setdefault(net, []).append(f"{c['ref']}:{pn}")
    meta = nl.get("nets") or {}
    rows = []
    for net, ends in sorted(nets.items(), key=lambda kv: kv[0]):
        if len(ends) < 2 and not meta.get(net, {}).get("port"):
            continue
        lbl = meta.get(net, {}).get("label") or ""
        v = dg.label_voltage(lbl or net) or 0
        power = v >= 300 or re.match(r"^(L\d|[UVW]\d?|A|B|C|M[UVW])$", net)
        rows.append({"wire": net, "label": lbl, "ends": ends, "hops": [f"{a} → {b}" for a, b in zip(ends, ends[1:])],
                     "section": "2,5 мм²" if power else "1,0 мм²", "external": bool(meta.get(net, {}).get("port"))})
    return rows


def wiring_markdown(nl, rows):
    md = [f"# Схема соединений щита: {nl.get('title', '')}", "",
          "Схема — релейно-контакторная автоматика: аппараты (пускатели, автоматы, тепловые реле, кнопки) ставятся "
          "на DIN-рейку и соединяются проводами, печатная плата не нужна.", "",
          "Маркировка провода = имя цепи. Сечение: силовые цепи (фазы, двигатель) 2,5 мм², управление 1,0 мм² "
          "(уточните по току двигателя и номиналу автомата).", "",
          "| Провод | Метка | Сечение | Соединения (по порядку) |", "|---|---|---|---|"]
    for r in rows:
        ext = " (внешний ввод)" if r["external"] else ""
        md.append(f"| {r['wire']}{ext} | {r['label']} | {r['section']} | {'; '.join(r['hops']) or r['ends'][0]} |")
    return "\n".join(md) + "\n"


# ---------- трассировка ----------

def net_classes(nl):
    """(ширина дорожки, зазор, высоковольтная) по цепи: силовые шире, высоковольтные с зазором 2,5 мм."""
    circ = dg.Circuit(nl)
    power_types = {"BR", "IGBT", "M", "VS", "F", "QF", "T", "MOTOR"}
    cls = {}
    for net, conns in circ.nets.items():
        v = abs(circ.volt.get(net, 0) or 0)
        lbl = str((nl.get("nets") or {}).get(net, {}).get("label", ""))
        if not v:
            v = dg.label_voltage(lbl or net) or 0
        power = net.upper() in core.GROUND_NAMES or v > 0 or any(
            circ.comps[r].get("type") in power_types for r, _ in conns) or any(
            circ.comps[r].get("type") == "C" and (dg.parse_value(circ.comps[r].get("value"))[0] or 0) >= 470e-6
            for r, _ in conns)
        hv = v >= 100 or bool(re.search(r"~|сеть|380|230|220|300", lbl + net))
        cls[net] = (2.0 if hv and power else 1.2 if power else 0.5, 2.5 if hv else 0.4, hv)
    return cls


class Router:
    def __init__(self, W, H):
        self.nx, self.ny = int(math.ceil(W / G)) + 1, int(math.ceil(H / G)) + 1
        self.own = np.full((2, self.ny, self.nx), -1, np.int32)     # чья медь (−1 — свободно)
        self.halo = np.full((2, self.ny, self.nx), -1, np.int32)    # зона зазора: номер цепи или −2 (две цепи)

    def cell(self, x, y):
        return int(round(y / G)), int(round(x / G))

    @staticmethod
    def _disk(r):
        rr = int(math.ceil(r / G))
        return [(dy, dx) for dy in range(-rr, rr + 1) for dx in range(-rr, rr + 1) if (dy * dy + dx * dx) * G * G <= r * r + 1e-9]

    def mark(self, cells, nid, radius, clear):
        core_d, halo_d = self._disk(radius), self._disk(radius + clear)
        for L, y, x in cells:
            for l in ((0, 1) if L is None else (L,)):
                for dy, dx in core_d:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < self.ny and 0 <= xx < self.nx:
                        self.own[l, yy, xx] = nid
                for dy, dx in halo_d:
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < self.ny and 0 <= xx < self.nx:
                        h = self.halo[l, yy, xx]
                        self.halo[l, yy, xx] = nid if h in (-1, nid) else -2

    def free(self, l, y, x, nid, rad):
        """Центр дорожки цепи nid в клетке: в радиусе полуширины+зазора нет чужой меди, клетка не в чужом зазоре."""
        y0, y1, x0, x1 = y - rad, y + rad + 1, x - rad, x + rad + 1
        if y0 < 0 or x0 < 0 or y1 > self.ny or x1 > self.nx:
            return False
        o = self.own[l, y0:y1, x0:x1]
        return not ((o >= 0) & (o != nid)).any() and self.halo[l, y, x] in (-1, nid)

    def route(self, nid, sources, targets, rad, box, limit=250000):
        """A* от клеток дерева цепи до площадки; 8 направлений, переход между слоями (дорого)."""
        y0, y1, x0, x1 = box
        tset = set(targets)
        ty = np.array([t[1] for t in targets])
        tx = np.array([t[2] for t in targets])

        def h(y, x):
            dy, dx = np.abs(ty - y), np.abs(tx - x)
            return float((np.maximum(dy, dx) + 0.414 * np.minimum(dy, dx)).min())

        openq, came, gbest, cache = [], {}, {}, {}
        for s in sources:
            gbest[s] = 0.0
            heapq.heappush(openq, (h(s[1], s[2]), 0.0, s))
        steps = [(0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]

        def ok(l, y, x):
            k = (l, y, x)
            if k not in cache:
                cache[k] = y0 <= y <= y1 and x0 <= x <= x1 and (k in tset or self.free(l, y, x, nid, rad))
            return cache[k]

        n = 0
        while openq:
            f, gc, cur = heapq.heappop(openq)
            if gc > gbest.get(cur, 1e18):
                continue
            if cur in tset:
                path = [cur]
                while path[-1] in came:
                    path.append(came[path[-1]])
                return path[::-1]
            n += 1
            if n > limit:
                return None
            l, y, x = cur
            for dy, dx, c in steps:
                if not ok(l, y + dy, x + dx):
                    continue
                if dy and dx and not (ok(l, y + dy, x) and ok(l, y, x + dx)):
                    continue
                nc = gc + c * (1.0 if l == 0 else 1.15)          # нижний слой предпочтительнее
                nxt = (l, y + dy, x + dx)
                if nc < gbest.get(nxt, 1e18):
                    gbest[nxt], came[nxt] = nc, cur
                    heapq.heappush(openq, (nc + h(y + dy, x + dx), nc, nxt))
            o = 1 - l
            vr = max(rad, int(math.ceil((VIA_D / 2 + 0.4) / G)) - 1)
            if ok(o, y, x) and self.free(o, y, x, nid, vr) and self.free(l, y, x, nid, vr):
                nc = gc + 8.0
                nxt = (o, y, x)
                if nc < gbest.get(nxt, 1e18):
                    gbest[nxt], came[nxt] = nc, cur
                    heapq.heappush(openq, (nc + h(y, x), nc, nxt))
        return None


def route_board(P, pos, W, H, classes, attempts=3, progress=None):
    """Развести все цепи. Возвращает (площадки, дорожки, переходные, неразведённые связи)."""
    pads = []
    for r, p in P.items():
        for pd, x, y in pad_positions(p, *pos[r]):
            pads.append({"ref": r, "pad": pd, "x": x, "y": y, "net": p["pins"].get(pd["name"])})
    nets = {}
    for i, pd in enumerate(pads):
        if pd["net"]:
            nets.setdefault(pd["net"], []).append(i)
    names = sorted(nets)
    nid = {n: i for i, n in enumerate(names)}
    dflt = (0.5, 0.4, False)

    def hpwl(n):
        xs = [pads[i]["x"] for i in nets[n]]
        ys = [pads[i]["y"] for i in nets[n]]
        return max(xs) - min(xs) + max(ys) - min(ys)

    order = sorted(names, key=lambda n: (-classes.get(n, dflt)[0], hpwl(n)))
    best = None
    for attempt in range(attempts):
        R = Router(W, H)
        for pd in pads:                                      # площадки — на обоих слоях
            clear = classes.get(pd["net"], dflt)[1] if pd["net"] else 0.4
            R.mark([(None, *R.cell(pd["x"], pd["y"]))], nid.get(pd["net"], -3), pd["pad"]["d"] / 2, clear)
        tracks, vias, failed = [], [], []
        for k, n in enumerate(order):
            if progress and k % 10 == 0:
                progress(f"трассировка: попытка {attempt + 1}, цепь {k + 1}/{len(order)}")
            if len(nets[n]) < 2:
                continue
            width, clear, _ = classes.get(n, dflt)
            rad = max(0, int(math.ceil((width / 2 + clear) / G)) - 1)
            idx = sorted(nets[n], key=lambda i: (pads[i]["x"], pads[i]["y"]))
            tree = {(l, *R.cell(pads[idx[0]]["x"], pads[idx[0]]["y"])) for l in (0, 1)}
            left = idx[1:]
            while left:
                left.sort(key=lambda i: min(abs(pads[i]["x"] - c[2] * G) + abs(pads[i]["y"] - c[1] * G) for c in tree))
                i = left.pop(0)
                cy, cx = R.cell(pads[i]["x"], pads[i]["y"])
                ys = [c[1] for c in tree] + [cy]
                xs = [c[2] for c in tree] + [cx]
                m = 40
                box = (max(0, min(ys) - m), min(R.ny - 1, max(ys) + m), max(0, min(xs) - m), min(R.nx - 1, max(xs) + m))
                path = R.route(nid[n], list(tree), [(0, cy, cx), (1, cy, cx)], rad, box)
                if path is None:
                    failed.append((n, pads[i]["ref"], pads[i]["pad"]["name"]))
                    continue
                R.mark(path, nid[n], width / 2, clear)
                seg = [path[0]]
                for a, b in zip(path, path[1:]):
                    if a[0] != b[0]:
                        tracks.append({"net": n, "layer": a[0], "w": width, "pts": [(c[2] * G, c[1] * G) for c in seg]})
                        vias.append({"net": n, "x": a[2] * G, "y": a[1] * G})
                        seg = [b]
                    else:
                        seg.append(b)
                tracks.append({"net": n, "layer": seg[0][0], "w": width, "pts": [(c[2] * G, c[1] * G) for c in seg]})
                tree |= set(path)
        if best is None or len(failed) < len(best[2]):
            best = (tracks, vias, failed)
        if not failed:
            break
        bad = {f[0] for f in failed}                          # не разведённые — первыми в следующей попытке
        order = [n for n in order if n in bad] + [n for n in order if n not in bad]
    tracks, vias, failed = best
    for t in tracks:                                          # убрать промежуточные точки на прямых участках
        pts = t["pts"]
        keep = [pts[0]]
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            if abs((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])) > 1e-6:
                keep.append(b)
        if len(pts) > 1:
            keep.append(pts[-1])
        t["pts"] = keep
    return pads, tracks, vias, failed


# ---------- шрифт для шелкографии (штрихи на сетке 4×6) ----------

_F = {
    "0": "0,0 4,0 4,6 0,6 0,0 4,6", "1": "1,1 2,0 2,6;1,6 3,6", "2": "0,1 1,0 3,0 4,1 4,2 0,6 4,6",
    "3": "0,0 4,0 2,2 4,3 4,5 3,6 0,6", "4": "3,6 3,0 0,4 4,4", "5": "4,0 0,0 0,2 3,2 4,3 4,5 3,6 0,6",
    "6": "4,0 1,0 0,2 0,6 4,6 4,3 0,3", "7": "0,0 4,0 1,6", "8": "0,0 4,0 4,6 0,6 0,0;0,3 4,3",
    "9": "4,3 0,3 0,0 4,0 4,6 0,6", "A": "0,6 2,0 4,6;1,3 3,3", "B": "0,0 3,0 4,1 3,3 4,5 3,6 0,6 0,0;0,3 3,3",
    "C": "4,0 0,0 0,6 4,6", "D": "0,0 3,0 4,2 4,4 3,6 0,6 0,0", "E": "4,0 0,0 0,6 4,6;0,3 3,3",
    "F": "4,0 0,0 0,6;0,3 3,3", "G": "4,0 0,0 0,6 4,6 4,3 2,3", "H": "0,0 0,6;4,0 4,6;0,3 4,3",
    "I": "2,0 2,6;1,0 3,0;1,6 3,6", "J": "4,0 4,6 0,6 0,4", "K": "0,0 0,6;4,0 0,3 4,6", "L": "0,0 0,6 4,6",
    "M": "0,6 0,0 2,3 4,0 4,6", "N": "0,6 0,0 4,6 4,0", "O": "0,0 4,0 4,6 0,6 0,0", "P": "0,6 0,0 4,0 4,3 0,3",
    "Q": "0,0 4,0 4,6 0,6 0,0;2,4 4,6", "R": "0,6 0,0 4,0 4,3 0,3 4,6", "S": "4,0 0,0 0,3 4,3 4,6 0,6",
    "T": "0,0 4,0;2,0 2,6", "U": "0,0 0,6 4,6 4,0", "V": "0,0 2,6 4,0", "W": "0,0 1,6 2,3 3,6 4,0",
    "X": "0,0 4,6;4,0 0,6", "Y": "0,0 2,3 4,0;2,3 2,6", "Z": "0,0 4,0 0,6 4,6", "-": "0,3 4,3",
    "+": "0,3 4,3;2,1 2,5", ".": "2,5 2,6", "_": "0,6 4,6", "?": "0,1 1,0 3,0 4,1 2,3 2,4;2,6 2,6",
    "/": "0,6 4,0", ":": "2,1 2,2;2,4 2,5",
}
_LAT = str.maketrans("АВЕКМНОРСТХаеорсухКнДЛПУ", "ABEKMHOPCTXAEOPCYXKHDLPY")


def text_strokes(s, x, y, h=1.5):
    """Текст штрихами: список полилиний (мм). Кириллица — похожими латинскими буквами."""
    k, out = h / 6, []
    for i, ch in enumerate(str(s).upper().translate(_LAT)):
        g = _F.get(ch)
        if not g:
            continue
        for part in g.split(";"):
            pts = [tuple(map(float, p.split(","))) for p in part.split()]
            out.append([(x + (i * 5.5 + px) * k, y + py * k) for px, py in pts])
    return out


# ---------- вывод ----------

def silk(P, pos):
    lines = []
    for r, p in P.items():
        x0, y0, x1, y1 = body_box(p, *pos[r], margin=0)
        lines.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
        lines += text_strokes(r, x0, y0 - 2.2)
    return lines


def render_png(path, W, H, P, pos, pads, tracks, vias, failed, scale=8):
    img = np.full((int(H * scale) + 1, int(W * scale) + 1, 3), 245, np.uint8)
    S = lambda x, y: (int(round(x * scale)), int(round(y * scale)))
    cv2.rectangle(img, S(0, 0), S(W, H), (40, 90, 40), 2)
    for t in sorted(tracks, key=lambda t: -t["layer"]):
        col = (200, 120, 60) if t["layer"] == 0 else (60, 60, 210)        # нижний — синий, верхний — красный
        for a, b in zip(t["pts"], t["pts"][1:]):
            cv2.line(img, S(*a), S(*b), col, max(1, int(t["w"] * scale)), cv2.LINE_AA)
    for pl in silk(P, pos):
        for a, b in zip(pl, pl[1:]):
            cv2.line(img, S(*a), S(*b), (90, 90, 90), 1, cv2.LINE_AA)
    for pd in pads:
        r = int(pd["pad"]["d"] / 2 * scale)
        c = S(pd["x"], pd["y"])
        if pd["pad"]["shape"] == "square":
            cv2.rectangle(img, (c[0] - r, c[1] - r), (c[0] + r, c[1] + r), (30, 160, 200), -1)
        else:
            cv2.circle(img, c, r, (30, 160, 200), -1, cv2.LINE_AA)
        cv2.circle(img, c, max(1, int(pd["pad"]["drill"] / 2 * scale)), (255, 255, 255), -1)
    for v in vias:
        cv2.circle(img, S(v["x"], v["y"]), int(VIA_D / 2 * scale), (60, 170, 60), -1, cv2.LINE_AA)
    by = {(pd["ref"], pd["pad"]["name"]): pd for pd in pads}
    for n, ref, pn in failed:                                        # неразведённые — пунктир к ближайшей площадке цепи
        a = by[(ref, pn)]
        others = [q for q in pads if q["net"] == n and q is not a]
        if others:
            b = min(others, key=lambda q: abs(q["x"] - a["x"]) + abs(q["y"] - a["y"]))
            cv2.line(img, S(a["x"], a["y"]), S(b["x"], b["y"]), (0, 0, 255), 1, cv2.LINE_8)
    core.imwrite(path, img)


def _g_num(v):
    return str(int(round(v * 1e6)))


def gerber(path, W, H, items, name):
    """Gerber RS-274X: items = [("flash", d, shape, x, y) | ("line", w, pts)]; Y вверх (как в CAD)."""
    aps, out = {}, []

    def ap(kind, d):
        key = (kind, round(d, 3))
        if key not in aps:
            aps[key] = 10 + len(aps)
        return aps[key]

    body = []
    for it in items:
        if it[0] == "flash":
            _, d, shape, x, y = it
            body.append(f"D{ap('R' if shape == 'square' else 'C', d)}*")
            body.append(f"X{_g_num(x)}Y{_g_num(H - y)}D03*")
        else:
            _, w, pts = it
            body.append(f"D{ap('C', w)}*")
            body.append(f"X{_g_num(pts[0][0])}Y{_g_num(H - pts[0][1])}D02*")
            for x, y in pts[1:]:
                body.append(f"X{_g_num(x)}Y{_g_num(H - y)}D01*")
    out += ["G04 Schematic Scanner PCB: " + name + "*", "%FSLAX46Y46*%", "%MOMM*%", "%LPD*%"]
    for (kind, d), n in sorted(aps.items(), key=lambda kv: kv[1]):
        out.append(f"%ADD{n}{'R' if kind == 'R' else 'C'},{d:.3f}{'X' + format(d, '.3f') if kind == 'R' else ''}*%")
    out += ["G01*"] + body + ["M02*"]
    Path(path).write_text("\n".join(out) + "\n", encoding="ascii", errors="replace")


def drill_file(path, H, pads, vias):
    tools = {}
    holes = [(pd["pad"]["drill"], pd["x"], pd["y"]) for pd in pads] + [(VIA_DRILL, v["x"], v["y"]) for v in vias]
    for d, _, _ in holes:
        tools.setdefault(round(d, 2), len(tools) + 1)
    out = ["M48", "METRIC,TZ"] + [f"T{n}C{d:.2f}" for d, n in tools.items()] + ["%"]
    for d, n in tools.items():
        out.append(f"T{n}")
        out += [f"X{x:.3f}Y{H - y:.3f}" for dd, x, y in holes if round(dd, 2) == d]
    out.append("M30")
    Path(path).write_text("\n".join(out) + "\n", encoding="ascii")


def make(project, W=None, H=None, progress=None):
    """Плата (или схема соединений щита) по netlist.json проекта. Возвращает сводку."""
    project = Path(project)
    nl = core.load_json(project / "netlist.json")
    if not nl:
        raise FileNotFoundError("нет netlist.json — сначала распознайте схему")
    comps = nl.get("components", [])
    low = sum(1 for c in comps if c.get("confidence") == "low")
    if nl.get("generated") == "autonet" and low > 0.3 * max(1, len(comps)):
        raise ValueError(f"netlist — черновик без ИИ: {low} из {len(comps)} деталей под сомнением. Плата повторит "
                         "все ошибки распознавания. Сначала исправьте сомнительные детали (вкладка «Компоненты» → "
                         "«Исправить деталь») или отдайте схему Claude, потом стройте плату.")
    out = project / "pcb"
    out.mkdir(exist_ok=True)
    share = panel_share(nl)
    if share >= 0.6:                                             # почти один щит — провода, а не плата
        rows = wiring_list(nl)
        (out / "wiring.md").write_text(wiring_markdown(nl, rows), encoding="utf-8")
        return {"mode": "wiring", "wires": len(rows), "panel_share": share, "dir": str(out)}
    if progress:
        progress("посадочные места и расстановка")
    P = build_parts(nl)
    W, H, pos = place(P, W, H)
    classes = net_classes(nl)
    pads, tracks, vias, failed = route_board(P, pos, W, H, classes, progress=progress)
    if progress:
        progress("Gerber, сверловка, картинка")
    render_png(out / "board.png", W, H, P, pos, pads, tracks, vias, failed)
    pad_items = [("flash", pd["pad"]["d"], pd["pad"]["shape"], pd["x"], pd["y"]) for pd in pads]
    via_items = [("flash", VIA_D, "round", v["x"], v["y"]) for v in vias]
    for layer, nm in ((1, "F_Cu"), (0, "B_Cu")):
        gerber(out / f"{nm}.gbr", W, H, pad_items + via_items +
               [("line", t["w"], t["pts"]) for t in tracks if t["layer"] == layer and len(t["pts"]) > 1], nm)
    mask = [("flash", pd["pad"]["d"] + 0.1, pd["pad"]["shape"], pd["x"], pd["y"]) for pd in pads]
    gerber(out / "F_Mask.gbr", W, H, mask, "F_Mask")
    gerber(out / "B_Mask.gbr", W, H, mask, "B_Mask")
    gerber(out / "F_SilkS.gbr", W, H, [("line", 0.15, pl) for pl in silk(P, pos)], "F_SilkS")
    gerber(out / "Edge_Cuts.gbr", W, H, [("line", 0.1, [(0, 0), (W, 0), (W, H), (0, H), (0, 0)])], "Edge_Cuts")
    drill_file(out / "drill.drl", H, pads, vias)
    data = {"board": [W, H], "placement": {r: {"x": x, "y": y, "rot": rot, "fp": P[r]["fp"]["note"]}
                                           for r, (x, y, rot) in pos.items()},
            "tracks": tracks, "vias": vias, "unrouted": [list(f) for f in failed]}
    (out / "pcb.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    nets_total = len({pd["net"] for pd in pads if pd["net"]})
    md = [f"# Печатная плата: {nl.get('title', '')}", "",
          f"Размер {W:.1f} × {H:.1f} мм, деталей {len(P)}, цепей {nets_total}, дорожек {len(tracks)}, "
          f"переходных {len(vias)}, не разведено связей: {len(failed)}.", "",
          "Нижний слой — основной (выводной монтаж), верхний — перемычки. Силовые цепи 1,2–2 мм, высоковольтные "
          "с зазором 2,5 мм. Файлы для производства: *.gbr (RS-274X) и drill.drl (Excellon).", ""]
    if failed:
        md += ["## Не разведено (провести перемычкой или поправить расстановку)", ""]
        md += [f"- цепь **{n}**: к {r}.{pn}" for n, r, pn in failed]
        md.append("")
    md += ["## Проверить перед заказом", "",
           "- цоколёвку транзисторов, тиристоров и микросхем с именованными выводами (корпус выбран по типу);",
           "- размеры крупных деталей (трансформаторы, реле, радиаторы) — габариты условные;",
           "- зазоры высоковольтных цепей и ширину силовых дорожек под ваш ток.", "",
           "## Посадочные места", "", "| Деталь | Номинал | Корпус |", "|---|---|---|"]
    md += [f"| {r} | {P[r]['value']} | {P[r]['fp']['note']} |" for r in P]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"mode": "pcb", "board": [W, H], "parts": len(P), "tracks": len(tracks), "vias": len(vias),
            "unrouted": len(failed), "dir": str(out)}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    size = [float(v) for v in sys.argv[2:4]] if len(sys.argv) >= 4 else [None, None]
    print(make(sys.argv[1], *size, progress=lambda m: print("…", m, flush=True)))
