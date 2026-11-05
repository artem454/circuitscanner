# Почему корпус микросхемы не стал рамкой: проверка каждого большого контура. python rects.py
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main")
import cv2
import numpy as np
import analyzer as A
import core

OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")


def job(p):
    p = Path(p)
    nl = core.load_json(p / "netlist.json")
    g0 = core.imread(p / "clean.png")[:, :, 0]
    P = A.load_params()
    H0, W0 = g0.shape
    s = int(max(1, min(4, round(P["target"] / max(H0, W0)))))
    g = cv2.resize(g0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else g0
    ink = A.binarize(g)
    sw = A.stroke_width(ink)
    ink = A.drop_big_text(ink, sw, P)
    words, tm = A.find_text(ink, sw, P)
    work = cv2.subtract(ink, tm)
    H, W = work.shape
    ics = [(c["ref"], c["type"], [v * s for v in c["box"]]) for c in nl["components"] if c["type"] in ("U", "LOGIC", "OPAMP", "OC", "BR", "T", "X") and (c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) * s * s > 900]
    cnts, hier = cv2.findContours(work, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    L = int(max(P["min_len"] * sw, 12))
    for ref, typ, b in ics:
        best = None
        for i, c in enumerate(cnts):
            if hier[0][i][3] < 0:
                continue
            x, y, w, h = cv2.boundingRect(c)
            ov = max(0, min(x + w, b[2]) - max(x, b[0])) * max(0, min(y + h, b[3]) - max(y, b[1]))
            area = (b[2] - b[0]) * (b[3] - b[1])
            if ov > 0.5 * area and (best is None or w * h < best[2] * best[3]):
                best = (x, y, w, h, c)
        if best is None:
            out.append((p.name[:12], ref, typ, "нет контура"))
            continue
        x, y, w, h, c = best
        big = max(w, h) / max(W, H)
        rectness = cv2.contourArea(c) / (w * h)
        inner = work[y + 2 * sw:y + h - 2 * sw, x + 2 * sw:x + w - 2 * sw]
        fill = inner.mean() / 255 if inner.size else 0
        marks = A.only_small_marks(inner, sw, P) if inner.size else True
        cont = A.edge_continuations(work, [x - sw, y - sw, x + w + sw, y + h + sw], [[x - sw, y - sw, x + w + sw, y + h + sw]], sw, L)
        fails = []
        if big > 0.3:
            fails.append(f"размер {big:.2f}")
        if rectness < 0.85:
            fails.append(f"прямоуг {rectness:.2f}")
        if fill > P["rect_fill"] and not marks:
            fails.append(f"внутри {fill:.3f}")
        if cont >= P["rect_cont"]:
            fails.append(f"продолж {cont}")
        out.append((p.name[:12], ref, typ, fails or "принят"))
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        for r in pool.map(job, [str(p) for p in sorted(OUT.iterdir()) if not p.name.startswith(("01_", "04_", "08_"))]):
            for x in r:
                print(*x)
