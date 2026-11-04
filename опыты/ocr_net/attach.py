# Привязка подписей: разбор один раз (кеш), дальше только autonet.build — проверка за секунды.
#   python attach.py cache   — посчитать кеш разборов
#   python attach.py train|test [show] — верные и неверные подписи
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main/опыты")
import enhance_eval as e
from enhance_eval import *

C = Path("C:/Users/Artem/scanner_ocr/cache")


def make(a):
    p, t = a
    p = Path(p)
    g = core.imread(p / "clean.png")[:, :, 0]
    res = analyzer.analyze_gray(g, analyzer.load_params(), False)
    res["classified"] = analyzer.classify(res)
    autonet.read_texts(res, g)
    autonet.read_inside(res, g)
    (C / (p.name[:12] + ".pkl")).write_bytes(pickle.dumps((str(p), t, res)))


def rate(pred, tr, show):
    r = dict(v_ok=0, v_bad=0, v_miss=0, r_ok=0, r_bad=0, r_miss=0)
    bad = []
    for c in tr["components"]:
        if not c.get("box") or c.get("type") in ("GND", "PORT"):
            continue
        b = c["box"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        u = next((u for u in pred["components"] if u["box"][0] - 6 <= cx <= u["box"][2] + 6 and u["box"][1] - 6 <= cy <= u["box"][3] + 6), None)
        if u is None:
            continue
        v, w = norm_value(u.get("value")), norm_value(c.get("value"))
        if c.get("value"):
            r["v_ok" if v == w else "v_bad" if u.get("value") else "v_miss"] += 1
            if v != w:
                bad.append(("н", c["ref"], c["value"], u["ref"], u.get("value")))
        rr, rw = str(u["ref"]).split(".")[0], str(c["ref"]).split(".")[0]
        if "?" in rr:
            r["r_miss"] += 1
        else:
            r["r_ok" if rr == rw else "r_bad"] += 1
            if rr != rw:
                bad.append(("о", c["ref"], c.get("value"), u["ref"], u.get("value")))
    return r, bad


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        C.mkdir(exist_ok=True)
        with ProcessPoolExecutor(6) as pool:
            list(pool.map(make, [(str(p), t) for p, t in e.sheets()]))
        print("готово")
    else:
        want = sys.argv[1] == "test"
        S = dict(v_ok=0, v_bad=0, v_miss=0, r_ok=0, r_bad=0, r_miss=0)
        for f in sorted(C.glob("*.pkl")):
            path, t, res = pickle.loads(f.read_bytes())
            if t != want:
                continue
            pred = autonet.build(res, cell=None)
            r, bad = rate(pred, core.load_json(Path(path) / "netlist.json"), False)
            for k in S:
                S[k] += r[k]
            if len(sys.argv) > 2:
                print(f.stem[:8], bad[:18])
        print(sys.argv[1], S)
