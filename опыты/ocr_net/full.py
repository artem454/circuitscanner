# Полный разбор 20 листов с ответом: верные и неверные подписи. python full.py <метка>
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main/опыты")
import enhance_eval as e
from enhance_eval import *


def nv(v):
    return norm_value(v)


autonet.LABEL_SURE = float(__import__("os").environ.get("SURE", autonet.LABEL_SURE))


def job(a):
    path, test = a
    p = Path(path)
    r = one((path, test, "как есть", ""))
    g = core.imread(p / "clean.png")[:, :, 0]
    res = analyzer.analyze_gray(g, analyzer.load_params(), False)
    res["classified"] = analyzer.classify(res)
    autonet.read_texts(res, g)
    autonet.read_inside(res, g)
    pred = autonet.build(res, cell=None)
    tr = core.load_json(p / "netlist.json")
    r.update(v_ok=0, v_bad=0, v_bad_flag=0, v_ok_flag=0, r_ok=0, r_bad=0, r_bad_flag=0, r_ok_flag=0)
    for u in pred["components"]:
        cx, cy = (u["box"][0] + u["box"][2]) / 2, (u["box"][1] + u["box"][3]) / 2
        c = next((c for c in tr["components"] if c.get("box") and c["box"][0] - 6 <= cx <= c["box"][2] + 6 and c["box"][1] - 6 <= cy <= c["box"][3] + 6), None)
        d = u.get("doubtful") or []
        if u.get("value"):
            ok = bool(c) and nv(u["value"]) == nv(c.get("value"))
            r["v_ok" if ok else "v_bad"] += 1
            r["v_ok_flag" if ok else "v_bad_flag"] += "value" in d
        if "?" not in u["ref"]:
            ok = bool(c) and str(u["ref"]).split(".")[0] == str(c.get("ref")).split(".")[0]
            r["r_ok" if ok else "r_bad"] += 1
            r["r_ok_flag" if ok else "r_bad_flag"] += "ref" in d
    return r


def show(R, tag):
    S = {f: sum(r[f] for r in R) for f in R[0] if isinstance(R[0][f], (int, float)) and f not in ("test",)}
    print(f"{tag}: тип {S['type']}, ложных {S['extra']}, F1 {np.mean([r['f1'] for r in R]):.3f}; НОМИНАЛЫ верных {S['v_ok']} неверных {S['v_bad']} (помечено {S['v_bad_flag']} неверных, {S['v_ok_flag']} верных); "
          f"ОБОЗНАЧЕНИЯ верных {S['r_ok']} неверных {S['r_bad']} (помечено {S['r_bad_flag']}, {S['r_ok_flag']})")


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        R = list(pool.map(job, [(str(p), t) for p, t in e.sheets()]))
    Path(f"C:/Users/Artem/scanner_ocr/full_{sys.argv[1]}.json").write_text(json.dumps(R, ensure_ascii=False), encoding="utf-8")
    show(R, sys.argv[1] + " все")
    show([r for r in R if r["test"]], sys.argv[1] + " тест")
