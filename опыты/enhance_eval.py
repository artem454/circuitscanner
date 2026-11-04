# Сравнение: без улучшения, UltraSharpV2, Text2HD на проверенных листах (готовых и ухудшенных вдвое с JPEG).
#   python опыты/enhance_eval.py prep  — посчитать улучшенные копии (видеокарта, кеш в WORK)
#   python опыты/enhance_eval.py eval  — разбор и таблица. Файлы проектов и пул поколений не меняются.
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import cv2
import numpy as np
import analyzer
import autonet
import core
import enhance
import learn
import ocr as ocrmod

WORK = Path(r"C:\Users\Artem\chaiNNer\work\enh")         # chaiNNer не видит файлы из AppData
VARIANTS = ("", "ultrasharp", "text2hd")
QUALITIES = ("как есть", "хуже")
DEGRADE_SCALE, DEGRADE_JPEG = 0.5, 40


def sheets():
    tests = [str(p) for p in learn.test_sheets()]
    seen = [str(p) for p in learn.registered()]
    out = []
    for p in seen + [t for t in tests if t not in seen]:
        p = Path(p)
        nl = core.load_json(p / "netlist.json")
        if nl and learn.truth_from_netlist(nl) and (p / "source.png").exists():
            out.append((p, str(p) in tests))
    return out


def stem(p, q):
    return p.name[:12] + ("_d" if q == "хуже" else "")


def base_image(p, q):
    """Цветная картинка варианта качества."""
    img = core.imread(p / "source.png")
    if q == "хуже":
        img = cv2.resize(img, None, fx=DEGRADE_SCALE, fy=DEGRADE_SCALE, interpolation=cv2.INTER_AREA)
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, DEGRADE_JPEG])[1], cv2.IMREAD_COLOR)
    return img


def prep():
    WORK.mkdir(exist_ok=True)
    times, names = {}, []
    for p, _ in sheets():
        for q in QUALITIES:
            core.imwrite(WORK / (stem(p, q) + ".png"), base_image(p, q))
            names.append(stem(p, q))
    for key in ("text2hd", "ultrasharp"):
        secs = enhance.run_model(key, [(WORK / (s + ".png"), WORK / f"{s}__{key}.png") for s in names])
        for s, t in zip(names, secs):
            times[f"{s}__{key}"] = t
            core.imwrite(WORK / f"{s}__{key}.png", core.clean(core.imread(WORK / f"{s}__{key}.png")))
        print(key, "секунд", round(sum(secs)), flush=True)
    (WORK / "times.json").write_text(json.dumps(times), encoding="utf-8")


def scaled_truth(nl, k):
    def sb(b):
        return [v * k for v in b]
    out = dict(nl, components=[])
    for c in nl["components"]:
        c = dict(c, box=sb(c["box"]))
        c["pin_xy"] = {n: sb(v) for n, v in c["pin_xy"].items()} if c.get("pin_xy") else c.get("pin_xy")
        if c.get("section_boxes"):
            c["section_boxes"] = {n: sb(b) for n, b in c["section_boxes"].items()}
        out["components"].append(c)
    out["power_symbols"] = [dict(s, box=sb(s["box"])) for s in nl.get("power_symbols", [])]
    return out


RU_UNITS = (("мкф", "μ"), ("мкгн", "μH"), ("нф", "n"), ("пф", "p"), ("ком", "k"), ("мом", "M"), ("ом", ""))


def norm_value(v):
    """Ответ и разбор к одному виду: русские единицы и разные символы «мю» — не ошибка чтения."""
    v = str(v or "").split("/")[0].split(" ")[0].replace("µ", "μ")
    low = v.lower()
    for ru, en in RU_UNITS:
        if low.endswith(ru):
            v = v[:len(v) - len(ru)] + en
            break
    return ocrmod.fix_value(v) or v


def match(pred, nl, tol):
    """Сверка найденного с ответом: центр рамки ответа попадает в рамку разбора (одна к одной)."""
    units = []
    for c in pred["components"]:
        for b in (c.get("section_boxes") or {"": c["box"]}).values():
            units.append((b, c))
    used, r = set(), dict(n=0, found=0, type=0, val_n=0, val=0, ref_n=0, ref=0)
    for c in nl["components"]:
        if c.get("type") in ("GND", "PORT"):
            continue
        for b in (c.get("section_boxes") or {"": c["box"]}).values():
            r["n"] += 1
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            hit = next((i for i, (u, _) in enumerate(units)
                        if i not in used and u[0] - tol <= cx <= u[2] + tol and u[1] - tol <= cy <= u[3] + tol), None)
            r["val_n"] += bool(c.get("value"))
            r["ref_n"] += bool(c.get("ref"))
            if hit is None:
                continue
            used.add(hit)
            p = units[hit][1]
            r["found"] += 1
            r["type"] += p["type"] == c["type"] or (p["type"], c["type"]) in autonet.COMPAT
            if c.get("value"):
                r["val"] += norm_value(p.get("value")) == norm_value(c["value"])
            if c.get("ref"):
                r["ref"] += str(p.get("ref", "")).split(".")[0] == str(c["ref"]).split(".")[0]
    power = [s["box"] for s in nl.get("power_symbols", [])]
    extra = 0
    for i, (u, c) in enumerate(units):
        if i in used or c.get("type") in ("GND", "PORT"):
            continue
        cx, cy = (u[0] + u[2]) / 2, (u[1] + u[3]) / 2
        extra += not any(b[0] - tol <= cx <= b[2] + tol and b[1] - tol <= cy <= b[3] + tol for b in power)
    r["extra"] = extra
    return r


def one(a):
    path, test, q, key = a
    p = Path(path)
    s = stem(p, q)
    k = DEGRADE_SCALE if q == "хуже" else 1.0
    gray0 = core.imread(p / "clean.png")[:, :, 0] if q == "как есть" else core.clean(core.imread(WORK / (s + ".png")))
    better = core.imread(WORK / f"{s}__{key}.png")[:, :, 0] if key else None
    truth = core.load_json(p / "netlist.json")
    truth = scaled_truth(truth, k) if k != 1 else truth
    t0 = time.time()
    res = analyzer.analyze_gray(gray0, analyzer.load_params(), False, None, enhanced=better)
    res["classified"] = analyzer.classify(res)
    autonet.read_texts(res, gray0)
    autonet.read_inside(res, gray0)
    pred = autonet.build(res, cell=None)
    secs = time.time() - t0
    r = match(pred, truth, 6 * k)
    r.update(f1=learn.score(res, learn.truth_from_netlist(truth))["f1"], secs=secs, sheet=p.name[:8], test=test, q=q, key=key)
    return r


def table(rows, title):
    print(f"\n### {title}")
    print("| Вариант | Листов | Найдено | Ложные | Тип верен | Обозначение | Номинал | Цепи F1 | Секунд: улучшение + разбор |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in VARIANTS:
        R = [r for r in rows if r["key"] == key]
        S = {f: sum(r[f] for r in R) for f in ("n", "found", "extra", "type", "ref", "ref_n", "val", "val_n", "enh", "secs")}
        print(f"| {enhance.TITLES[key]} | {len(R)} | {S['found']}/{S['n']} | {S['extra']} | {S['type']}/{S['found']} | "
              f"{S['ref']}/{S['ref_n']} | {S['val']}/{S['val_n']} | {np.mean([r['f1'] for r in R]):.3f} | "
              f"{S['enh']:.0f} + {S['secs']:.0f} |")


def evaluate():
    times = json.loads((WORK / "times.json").read_text(encoding="utf-8"))
    jobs = [(str(p), t, q, key) for p, t in sheets() for q in QUALITIES for key in VARIANTS]
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(one, jobs))
    for r, (path, _, q, key) in zip(rows, jobs):
        r["enh"] = times.get(f"{stem(Path(path), q)}__{key}", 0) if key else 0
    (WORK / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=0), encoding="utf-8")
    for q in QUALITIES:
        table([r for r in rows if r["q"] == q], f"Качество: {q} — все листы")
        table([r for r in rows if r["q"] == q and r["test"]], f"Качество: {q} — только тестовые")


if __name__ == "__main__":
    prep() if sys.argv[1] == "prep" else evaluate()
