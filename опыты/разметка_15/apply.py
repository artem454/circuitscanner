# Лист по картинке → netlist.json: python apply.py <№>
# Файл NN.json: {"comps":[[ref,type,value,box,{вывод:[x,y,цепь]}],...], "power":[[type,box,net]]}
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "../code")
import cv2
import numpy as np
import core
import diagnose
import physics

n = int(sys.argv[1])
OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
d = next(p for p in sorted(OUT.iterdir()) if p.name.startswith(f"{n:02d}_"))
src = json.load(open(f"{n:02d}.json", encoding="utf-8"))
if not (d / "netlist_scanner.json").exists():
    shutil.copyfile(d / "netlist.json", d / "netlist_scanner.json")
draft = core.load_json(d / "netlist_scanner.json")
man = core.load_json(d / "manifest.json")
cell = man["grid"]["cell"]


def cn(b):
    return f"{core.col_name(int(((b[0] + b[2]) / 2) // cell))}{int(((b[1] + b[3]) / 2) // cell) + 1}"


K = src.get("scale", 1)
if K != 1:                                             # координаты сняты с уменьшенной картинки
    sc = lambda v: [round(a * K, 1) for a in v]
    for c in src["comps"]:
        c[3] = sc(c[3])
        c[4] = {p: sc(v[:2]) + [v[2]] for p, v in c[4].items()}
        if len(c) > 5:
            c[5] = {k: sc(b) for k, b in c[5].items()}
    src["power"] = [[t, sc(b), n] for t, b, n in src.get("power", [])]
comps = []
for ref, typ, val, box, pins, *sections in src["comps"]:
    comps.append({"ref": ref, "type": typ, "value": val, "pins": {p: v[2] for p, v in pins.items()}, "box": box,
                  "pin_xy": {p: v[:2] for p, v in pins.items()}, "cell": cn(box)})
    if sections:
        comps[-1]["section_boxes"] = sections[0]
power = [{"type": t, "box": b, "net": net, "cell": cn(b)} for t, b, net in src.get("power", [])]
nets = defaultdict(list)
for c in comps:
    for p, net in c["pins"].items():
        nets[net].append(f"{c['ref']}.{p}")
meta = {}
for c in comps:
    if c["type"] == "J":
        for net in c["pins"].values():
            meta.setdefault(net, {})["port"] = True
nl = {"title": d.name, "generated": "claude", "components": comps, "nets": meta, "power_symbols": power, "questions": []}
(d / "netlist.json").write_text(json.dumps(nl, ensure_ascii=False, indent=1), encoding="utf-8")
print(d.name, "деталей", len(comps), "цепей", len(nets), "знаков", len(power))
print("одиночные цепи:", {k: v for k, v in nets.items() if len(v) < 2})
g = core.imread(d / "clean.png")[:, :, 0]
ink = cv2.dilate((g < 170).astype(np.uint8), np.ones((5, 5), np.uint8))
print("выводы мимо чернил:", [f"{c['ref']}.{p}" for c in comps for p, (x, y) in c["pin_xy"].items()
                              if not ink[min(int(y), g.shape[0] - 1), min(int(x), g.shape[1] - 1)]])
refs = [c["ref"] for c in comps]
print("повторы:", [r for r in set(refs) if refs.count(r) > 1])
issues = core.check(nl, man)
print("ошибки проверки:", [str(i)[:110] for i in issues if i.get("level") == "error"][:12])
dp = [(xy, c["pins"].get(p)) for c in draft["components"] for p, xy in (c.get("pin_xy") or {}).items()]


def near(xy):
    b = min(dp, key=lambda q: abs(q[0][0] - xy[0]) + abs(q[0][1] - xy[1]))
    return b[1] if abs(b[0][0] - xy[0]) + abs(b[0][1] - xy[1]) <= 12 else None


mapped = [(f"{c['ref']}.{p}", c["pins"][p], near(xy)) for c in comps for p, xy in c["pin_xy"].items()]
merged = defaultdict(set)
split = defaultdict(set)
for nm, net, dn in mapped:
    if dn:
        merged[dn].add(net)
        split[net].add(dn)
print("анализатор объединяет разные цепи листа:", {k: sorted(v) for k, v in merged.items() if len(v) > 1})
print("одну цепь листа анализатор рвёт:", {k: len(v) for k, v in split.items() if len(v) > 1})
print("выводов сопоставлено", sum(1 for x in mapped if x[2]), "из", len(mapped))
try:
    r = physics.Physics(nl, {}).run()
    diagnose.Circuit(nl)
    print("физика ок, находок", len(r["findings"]))
except Exception as e:
    print("ФИЗИКА", type(e).__name__, str(e)[:120])
v = cv2.cvtColor(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
v = (255 - (255 - v.astype(np.float32)) * 0.5).astype(np.uint8)
names = sorted(nets)
pal = [tuple(int(a) for a in cv2.cvtColor(np.uint8([[[i * 37 % 180, 230, 200]]]), cv2.COLOR_HSV2BGR)[0][0]) for i in range(len(names))]
for c in comps:
    x0, y0, x1, y1 = (int(a * 2) for a in c["box"])
    cv2.rectangle(v, (x0, y0), (x1, y1), (0, 140, 0), 1)
    cv2.putText(v, f"{c['ref']} {c['value']}", (x0, max(10, y0 - 3)), 0, 0.4, (0, 110, 0), 1, cv2.LINE_AA)
    for p, (x, y) in c["pin_xy"].items():
        col = pal[names.index(c["pins"][p])]
        cv2.circle(v, (int(x * 2), int(y * 2)), 4, col, -1)
        cv2.putText(v, c["pins"][p], (int(x * 2) + 5, int(y * 2) + 4), 0, 0.33, col, 1, cv2.LINE_AA)
core.imwrite(f"check_{n:02d}.png", v)
