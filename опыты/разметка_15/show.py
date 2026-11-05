# Лист на ручную сверку: python show.py <№> → грид и черновик с номерами. Части: show_<№>_a.png, _b.png
import json
import sys
from pathlib import Path
sys.path.insert(0, "code")
import cv2
import numpy as np
import core

OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
d = next(p for p in sorted(OUT.iterdir()) if p.name.startswith(f"{int(sys.argv[1]):02d}_"))
g = core.imread(d / "source.png")
nl = core.load_json(d / "netlist.json")
H, W = g.shape[:2]
k = 2 if W <= 1150 else 1
v = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
for x in range(0, W, 50):
    cv2.line(v, (x * k, 0), (x * k, H * k), (255, 210, 150) if x % 100 else (230, 150, 60), 1)
    if x % 100 == 0:
        cv2.putText(v, str(x), (x * k + 2, 12), 0, 0.42, (200, 90, 0), 1, cv2.LINE_AA)
for y in range(0, H, 50):
    cv2.line(v, (0, y * k), (W * k, y * k), (255, 210, 150) if y % 100 else (230, 150, 60), 1)
    if y % 100 == 0:
        cv2.putText(v, str(y), (2, y * k + 12), 0, 0.42, (200, 90, 0), 1, cv2.LINE_AA)
clean = v.copy()
for i, c in enumerate(nl["components"]):
    x0, y0, x1, y1 = (int(a * k) for a in c["box"])
    cv2.rectangle(v, (x0, y0), (x1, y1), (0, 0, 230), 1)
    cv2.putText(v, f"{i}:{c['type']}:{c['ref']}", (x0, max(10, y0 - 3)), 0, 0.4, (0, 0, 230), 1, cv2.LINE_AA)
    for p in c.get("pin_xy", {}).values():
        cv2.circle(v, (int(p[0] * k), int(p[1] * k)), 3, (0, 170, 0), -1)
core.imwrite(f"show_{sys.argv[1]}_clean.png", clean)
core.imwrite(f"show_{sys.argv[1]}_draft.png", v)
print(d.name, W, H, len(nl["components"]))
print(json.dumps([[i, c["type"], c["ref"], c.get("value", ""), [round(v) for v in c["box"]], {n: [round(a) for a in xy] + [c["pins"][n]] for n, xy in c.get("pin_xy", {}).items()}] for i, c in enumerate(nl["components"])], ensure_ascii=False)[:7000])
