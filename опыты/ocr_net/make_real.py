# Настоящие надписи с известным текстом: разбор находит надпись, проверенный netlist даёт её текст.
#   python make_real.py   — пишет real/*.png и real/labels.json. Файлы проектов не меняет.
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path

SCANNER = "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main"
sys.path.insert(0, SCANNER)
import cv2
import analyzer
import autonet
import core
import learn

ROOT = Path(__file__).parent
OUT = ROOT / "real"


def canon(s):
    s = str(s or "").upper().replace("µ", "U").replace("Μ", "U").replace(" ", "").replace(",", ".")
    return "".join({"О": "O", "Q": "O", "I": "1", "L": "1", "С": "C", "Р": "P", "К": "K", "М": "M", "В": "B", "Н": "H", "Т": "T", "А": "A", "Х": "X", "Е": "E"}.get(c, c) for c in s)


def sim(a, b):
    return SequenceMatcher(None, canon(a), canon(b)).ratio()


def one(a):
    path, test = a
    p = Path(path)
    nl = core.load_json(p / "netlist.json")
    g = core.imread(p / "clean.png")[:, :, 0]
    res = analyzer.analyze_gray(g, analyzer.load_params(), False)
    res["classified"] = analyzer.classify(res)
    autonet.read_texts(res, g)
    labels = []
    for c in nl.get("components", []):
        b = c.get("box")
        if not b:
            continue
        ref, val = c.get("ref"), str(c.get("value") or "")
        names = [x for x in (ref, val, val.split("/")[0], val.split(" ")[0]) if x]
        if ref and val:
            names.append(f"{ref} {val.split('/')[0].split(' ')[0]}")
        for s in nl.get("power_symbols", []):
            pass
        labels.append((b, names))
    out, used = [], set()
    cands = []
    for ti, t in enumerate(res["texts"]):
        tb = t["box"]
        tc = ((tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
        for b, names in labels:
            size = max(b[2] - b[0], b[3] - b[1], 20)
            if analyzer.box_dist(b, *tc) > 3 * size:
                continue
            for n in names:
                r = sim(t.get("text", ""), n)
                if r >= 0.6:
                    cands.append((r, len(n), ti, n))
    taken = set()
    for r, _, ti, n in sorted(cands, reverse=True):
        if ti in used or (ti, n) in taken:
            continue
        used.add(ti)
        t = res["texts"][ti]
        x0, y0, x1, y1 = (int(v) for v in t["box"])
        crop = g[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4]
        if t.get("vertical"):
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        if crop.size == 0:
            continue
        name = f"{p.name[:10]}_{ti}.png"
        out.append({"file": name, "text": n, "old": t.get("text", ""), "conf": t.get("conf", 0), "sheet": p.name[:10], "test": test, "r": round(r, 2)})
        core.imwrite(OUT / name, crop)
    return out


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tests = [str(p) for p in learn.test_sheets()]
    seen = [str(p) for p in learn.registered()]
    jobs = [(p, False) for p in seen if p not in tests] + [(p, True) for p in tests]
    jobs = [j for j in jobs if (Path(j[0]) / "netlist.json").exists() and (Path(j[0]) / "clean.png").exists()]
    with ProcessPoolExecutor(6) as pool:
        rows = [x for r in pool.map(one, jobs) for x in r]
    (OUT / "labels.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    te = [r for r in rows if r["test"]]
    tr = [r for r in rows if not r["test"]]
    for name, R in (("обучение", tr), ("тест", te)):
        ok = sum(r["old"].replace(" ", "") == r["text"].replace(" ", "") for r in R)
        print(name, "листов", len({r["sheet"] for r in R}), "надписей", len(R), "точно прочитано", ok, f"{ok / max(1, len(R)):.3f}")
    print([(r["text"], r["old"]) for r in te[:30]])
