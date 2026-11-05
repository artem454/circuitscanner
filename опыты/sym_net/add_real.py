# Примеры символов с 15 проверенных сложных листов. python add_real.py
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import make_sym as M
from make_sym import *

ROOT_R = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
HARD = ("01_", "04_", "08_")
PART_OF = {"T": "L", "BR": "D"}                         # обмотка и диод внутри большой рамки — не обломок
EXTRA = {"RP": "R", "DIAC": "VS", "Z": "D"}


def truth_cls(res, boxes, sy):
    cx, cy = (sy["box"][0] + sy["box"][2]) / 2, (sy["box"][1] + sy["box"][3]) / 2
    best = None
    for b, cls, whole in boxes:
        if b[0] - 3 <= cx <= b[2] + 3 and b[1] - 3 <= cy <= b[3] + 3:
            area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            if best is None or area < best[1]:
                best = (cls, area, whole)
    if best is None:
        return "OTHER"
    sa = (sy["box"][2] - sy["box"][0]) * (sy["box"][3] - sy["box"][1])
    return best[0] if best[2] or sa >= 0.15 * best[1] else "OTHER"


def one(path):
    p = Path(path)
    nl = core.load_json(p / "netlist.json")
    g = core.imread(p / "clean.png")[:, :, 0]
    res = parse(g)
    src = "test" if p.name.startswith(HARD) else "real"
    REAL = dict(learn.TYPE_MAP)
    REAL.update(EXTRA)
    boxes = []
    for c in nl["components"]:
        t = c.get("type", "")
        whole = t in PART_OF
        cls = PART_OF.get(t) or REAL.get(t)
        for b in (c.get("section_boxes") or {"": c.get("box")}).values():
            if b and cls:
                boxes.append((b, cls, whole))
    for s in nl.get("power_symbols", []):
        boxes.append((s["box"], "GND" if s.get("type") == "GND" else "PORT", False))
    pins = pins_count(res)
    C, T, Y = [], [], []
    for sy in res["symbols"]:
        cls = truth_cls(res, boxes, sy)
        cls = MERGE.get(cls, cls)
        if cls in IDX:
            C.append(crop(g, sy["box"]))
            T.append(table(res, sy, pins.get(sy["id"], 0)))
            Y.append(IDX[cls])
    save(f"h_{src}_{p.name[:2]}", C, T, Y, src)
    return p.name[:20], src, len(Y), sum(y == 0 for y in Y)


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        R = list(pool.map(one, [str(p) for p in sorted(ROOT_R.iterdir())]))
    for r in R:
        print(r)
    print("новых настоящих", sum(r[2] for r in R if r[1] == "real"), "из них мусора", sum(r[3] for r in R if r[1] == "real"))
