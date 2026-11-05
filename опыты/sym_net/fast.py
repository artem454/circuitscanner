# Быстрый замер соединений на учебных листах: python fast.py <метка> [test]. Без чтения подписей.
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main/опыты")
import enhance_eval as e
from enhance_eval import *

OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
TEST = ("01_", "04_", "08_")


def job(path):
    p = Path(path)
    g = core.imread(p / "clean.png")[:, :, 0]
    res = analyzer.analyze_gray(g, analyzer.load_params(json.loads(__import__("os").environ.get("EXTRA", "{}"))), False)
    res["classified"] = analyzer.classify(res)
    s = learn.score(res, learn.truth_from_netlist(core.load_json(p / "netlist.json")))
    return p.name[:20], round(s["f1"], 3), round(s["precision"], 3), round(s["recall"], 3), max([len(n["pins"]) for n in res["nets"]] or [0])


if __name__ == "__main__":
    test = len(sys.argv) > 2
    tests = [str(p) for p, t in e.sheets() if t]
    if test:
        S = tests
    else:
        S = [str(p) for p, t in e.sheets() if not t] + [str(p) for p in sorted(OUT.iterdir()) if not p.name.startswith(TEST)]
        S = [p for p in dict.fromkeys(S) if p not in tests]
    with ProcessPoolExecutor(8) as pool:
        R = list(pool.map(job, S))
    old = {}
    f = Path(f"fast_{'t' if test else 'u'}_base.json")
    if f.exists():
        old = {r[0]: r for r in json.loads(f.read_text(encoding="utf-8"))}
    else:
        f.write_text(json.dumps(R, ensure_ascii=False), encoding="utf-8")
    ch = [(r[0], old[r[0]][1], r[1], old[r[0]][4], r[4]) for r in R if r[0] in old and abs(old[r[0]][1] - r[1]) > 0.004]
    print(sys.argv[1], "листов", len(R), "F1", round(np.mean([r[1] for r in R]), 3), "точность", round(np.mean([r[2] for r in R]), 3), "полнота", round(np.mean([r[3] for r in R]), 3))
    print("изменились (F1 было стало, крупнейшая цепь):", ch)
