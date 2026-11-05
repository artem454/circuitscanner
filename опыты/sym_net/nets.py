# Соединения на проверенных листах: точность и полнота цепей, рисунок самой большой цепи.
#   python nets.py          — таблица по учебным листам
#   python nets.py draw NN  — рисунок цепей сложного листа NN
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main/опыты")
import enhance_eval as e
from enhance_eval import *

OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
HARD_TEST = ("01_", "04_", "08_")


def parse(g):
    res = analyzer.analyze_gray(g, analyzer.load_params(), False)
    res["classified"] = analyzer.classify(res)
    autonet.read_texts(res, g)
    autonet.read_inside(res, g)
    return res


def one_sheet(path):
    p = Path(path)
    g = core.imread(p / "clean.png")[:, :, 0]
    nl = core.load_json(p / "netlist.json")
    res = parse(g)
    s = learn.score(res, learn.truth_from_netlist(nl))
    big = sorted((len(n["pins"]) for n in res["nets"]), reverse=True)[:3]
    return p.name[:22], g.shape[1], g.shape[0], res["scale"], res["stroke"], round(s["f1"], 3), round(s["precision"], 3), round(s["recall"], 3), len(res["nets"]), big, len(res["symbols"]), len(res["segments"]), len(res.get("dots", []))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = sys.argv[2]
        d = next(p for p in sorted(OUT.iterdir()) if p.name.startswith(n + "_"))
        g = core.imread(d / "clean.png")[:, :, 0]
        res = parse(g)
        k = 2 if max(g.shape) < 1200 else 1
        v = cv2.cvtColor(cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR)
        v = (255 - (255 - v.astype(np.float32)) * 0.35).astype(np.uint8)
        nets = sorted(res["nets"], key=lambda t: -len(t["pins"]))
        cols = [(0, 0, 230), (0, 150, 0), (230, 0, 0), (0, 160, 200), (200, 0, 180), (120, 120, 0)]
        for i, net in enumerate(nets):
            c = cols[i] if i < len(cols) else (150, 150, 150)
            for si in net["segments"]:
                x0, y0, x1, y1 = (int(a * k) for a in res["segments"][si])
                cv2.line(v, (x0, y0), (x1, y1), c, 2 if i < len(cols) else 1)
            for q in net["pins"]:
                cv2.circle(v, (int(q["x"] * k), int(q["y"] * k)), 4, c, 1)
        for s in res["symbols"]:
            x0, y0, x1, y1 = (int(a * k) for a in s["box"])
            cv2.rectangle(v, (x0, y0), (x1, y1), (0, 0, 0) if not s.get("junk") else (200, 200, 200), 1)
        core.imwrite(f"nets_{n}.png", v)
        print(d.name, "цепей", len(nets), [len(t["pins"]) for t in nets[:6]])
    else:
        S = [str(p) for p, t in e.sheets() if not t] + [str(p) for p in sorted(OUT.iterdir()) if not p.name.startswith(HARD_TEST)]
        S = list(dict.fromkeys(S))
        with ProcessPoolExecutor(6) as pool:
            R = list(pool.map(one_sheet, S))
        R.sort(key=lambda r: r[5])
        print("лист | ширина высота | масштаб толщина | F1 точность полнота | цепей | крупнейшие | символов отрезков точек")
        for r in R:
            print(*r)
        print("средняя F1", round(np.mean([r[5] for r in R]), 3), "точность", round(np.mean([r[6] for r in R]), 3), "полнота", round(np.mean([r[7] for r in R]), 3))
