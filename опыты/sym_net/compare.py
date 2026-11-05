# Детали и связи на учебных листах для заданной папки кода: python compare.py <папка кода> <метка>
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
CODE = sys.argv[1]
sys.path.insert(0, CODE)
import numpy as np
import analyzer
import autonet
import core
import learn
import ocr as ocrmod

R = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
TEST = ("01_", "04_", "08_")


def job(p):
    p = Path(p)
    g = core.imread(p / "clean.png")[:, :, 0]
    nl = core.load_json(p / "netlist.json")
    res = analyzer.analyze_gray(g, analyzer.load_params(), False)
    res["classified"] = analyzer.classify(res)
    autonet.read_texts(res, g)
    autonet.read_inside(res, g)
    pred = autonet.build(res, cell=None)
    units = [(b, c) for c in pred["components"] for b in (c.get("section_boxes") or {"": c["box"]}).values()]
    used, found, typ = set(), 0, 0
    n = 0
    for c in nl["components"]:
        if c.get("type") in ("GND", "PORT"):
            continue
        for b in (c.get("section_boxes") or {"": c["box"]}).values():
            n += 1
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            hit = next((i for i, (u, _) in enumerate(units) if i not in used and u[0] - 6 <= cx <= u[2] + 6 and u[1] - 6 <= cy <= u[3] + 6), None)
            if hit is not None:
                used.add(hit)
                found += 1
                t = units[hit][1]["type"]
                typ += t == c["type"] or (t, c["type"]) in autonet.COMPAT
    extra = sum(1 for i, (u, c) in enumerate(units) if i not in used and c.get("type") not in ("GND", "PORT"))
    f1 = learn.score(res, learn.truth_from_netlist(nl))["f1"]
    return p.name[:16], n, found, extra, typ, round(f1, 3)


if __name__ == "__main__":
    S = [str(p) for p in sorted(R.iterdir()) if not p.name.startswith(TEST)]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(job, S))
    Path(f"cmp_{sys.argv[2]}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(sys.argv[2], "деталей", sum(r[1] for r in rows), "найдено", sum(r[2] for r in rows), "ложных", sum(r[3] for r in rows), "тип", sum(r[4] for r in rows), "F1", round(float(np.mean([r[5] for r in rows])), 3))
