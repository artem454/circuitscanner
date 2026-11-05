# Итог сканера на всех тестовых листах: python testset.py <метка>
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main/опыты")
import enhance_eval as e
from enhance_eval import *

HARD = ("01_", "04_", "08_")
F = ("n", "found", "extra", "type", "ref", "ref_n", "val", "val_n")


def show(R, tag):
    S = {f: sum(r[f] for r in R) for f in F}
    print(f"{tag} ({len(R)} лист.): найдено {S['found']}/{S['n']}, ложных {S['extra']}, тип {S['type']}/{S['found']}, обозначение {S['ref']}/{S['ref_n']}, номинал {S['val']}/{S['val_n']}, F1 {np.mean([r['f1'] for r in R]):.3f}")


if __name__ == "__main__":
    S = [p for p, t in e.sheets() if t]
    with ProcessPoolExecutor(6) as pool:
        R = list(pool.map(one, [(str(p), True, "как есть", "") for p in S]))
    for r, p in zip(R, S):
        r["hard"] = p.name.startswith(HARD)
        r["name"] = p.name[:24]
    Path(f"C:/Users/Artem/scanner_sym/test_{sys.argv[1]}.json").write_text(json.dumps(R, ensure_ascii=False), encoding="utf-8")
    show(R, sys.argv[1] + " все")
    show([r for r in R if r["hard"]], sys.argv[1] + " сложные")
    show([r for r in R if not r["hard"]], sys.argv[1] + " прежние")
