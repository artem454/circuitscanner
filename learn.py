"""Самообучение анализатора.

    python learn.py eval   [N]          — оценить текущие параметры на N синтетических схемах
    python learn.py tune   [N] [шагов]  — подобрать параметры (сохраняются в learned/params.json)
    python learn.py symbols [N]         — обучить классификатор символов на синтетике
    python learn.py add папка_проекта   — добавить проверенный проект (netlist.json) в обучение
    python learn.py auto [N] [шагов]    — дообучить всё: проекты, символы, параметры

Оценка: для каждой пары выводов сравнивается «в одной цепи / в разных» по ответу
и по анализатору (F1), плюс доля найденных деталей. Подбор — случайный поиск
вокруг лучшего найденного; каждый шаг пишется в learned/history.jsonl, так что
обучение можно продолжать с того места, где остановились.
"""
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import analyzer
import core
import synth

LEARNED = analyzer.LEARNED
SEED0 = 5000                     # фиксированный набор схем для сравнения параметров

# пределы для подбора: (мин, макс, целое?)
SPACE = {
    "text_max": (7, 18, False), "min_len": (2.5, 11, False), "tol": (1.2, 3.5, False),
    "dot_d": (1.4, 2.6, False), "dot_min": (1.8, 3.2, False), "dot_max": (5, 12, False),
    "rect_fill": (0.01, 0.1, False), "rect_cont": (1, 4, True), "plate_thick": (1.4, 2.6, False),
    "stroke_len": (1.5, 4.0, False), "sym_area": (2, 8, False), "sym_grow": (2, 5, False),
    "pin_dist": (1.2, 3.5, False),
}


def prep(img):
    """Та же очистка, что у настоящих фото (core.clean), но без выпрямления:
    поворот синтетики в ответе уже учтён."""
    return core.clean(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


# ---------- оценка ----------

def score(res, truth):
    """Сравнить результат анализатора с ответом. Возвращает словарь метрик."""
    an_pins = [(p["x"], p["y"], n["id"], p["symbol"]) for n in res["nets"] for p in n["pins"]]
    A = np.array([(x, y) for x, y, _, _ in an_pins]).reshape(-1, 2)
    R = max(8.0, 4 * res["stroke"])
    taken = set()
    pred = []
    # неподключённые выводы (цепь из одного вывода) находить не нужно
    size = {}
    for p in truth["pins"]:
        size[p["net"]] = size.get(p["net"], 0) + 1
    truth = dict(truth, pins=[p for p in truth["pins"] if size[p["net"]] > 1])
    for tp in truth["pins"]:
        best = None
        if len(A):
            # вывод анализатора может оказаться где угодно на ножке: от тела до конца
            ax, ay, bx, by = tp["x"], tp["y"], tp["ox"], tp["oy"]
            L2 = (bx - ax) ** 2 + (by - ay) ** 2 or 1.0
            u = np.clip(((A[:, 0] - ax) * (bx - ax) + (A[:, 1] - ay) * (by - ay)) / L2, 0, 1)
            d = np.hypot(A[:, 0] - (ax + u * (bx - ax)), A[:, 1] - (ay + u * (by - ay)))
            for k in np.argsort(d)[:4]:
                if d[k] <= R and k not in taken:
                    best = int(k)
                    break
        if best is not None:
            taken.add(best)
            pred.append(an_pins[best][2])
        else:
            pred.append(None)
    tp_ = fp = fn = 0
    tn = [p["net"] for p in truth["pins"]]
    n = len(tn)
    for i in range(n):
        for j in range(i + 1, n):
            same_t = tn[i] == tn[j]
            same_p = pred[i] is not None and pred[i] == pred[j]
            if same_t and same_p:
                tp_ += 1
            elif same_p:
                fp += 1
            elif same_t:
                fn += 1
    prec = tp_ / (tp_ + fp) if tp_ + fp else 1.0
    rec = tp_ / (tp_ + fn) if tp_ + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    found = 0
    for c in truth["comps"]:
        cx, cy = (c["box"][0] + c["box"][2]) / 2, (c["box"][1] + c["box"][3]) / 2
        if any(s["box"][0] - R <= cx <= s["box"][2] + R and s["box"][1] - R <= cy <= s["box"][3] + R
               for s in res["symbols"]):
            found += 1
    comp_rec = found / max(1, len(truth["comps"]))
    pin_rec = sum(p is not None for p in pred) / max(1, n)
    return {"f1": f1, "precision": prec, "recall": rec, "comps": comp_rec, "pins": pin_rec,
            "score": 0.6 * f1 + 0.2 * comp_rec + 0.2 * pin_rec}


def _eval_one(args):
    seed, params = args
    img, truth = synth.generate(seed)
    res = analyzer.analyze_gray(prep(img), params)
    return score(res, truth)


def evaluate(params, n=24, pool=None):
    jobs = [(SEED0 + i, params) for i in range(n)]
    rows = list(pool.map(_eval_one, jobs)) if pool else [_eval_one(j) for j in jobs]
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    real = [p for p in registered() if (Path(p) / "netlist.json").exists()]
    if real:
        rr = list(pool.map(_real_one, [(p, params) for p in real])) if pool else             [_real_one((p, params)) for p in real]
        m["real"] = float(np.mean(rr))
        m["score"] = 0.7 * m["score"] + 0.3 * m["real"]
    return m


# ---------- реальные проверенные проекты ----------

def registered():
    return (core.load_json(LEARNED / "projects.json") or [])


def register(project):
    LEARNED.mkdir(exist_ok=True)
    lst = registered()
    p = str(Path(project).resolve())
    if p not in lst:
        lst.append(p)
        (LEARNED / "projects.json").write_text(json.dumps(lst, ensure_ascii=False, indent=1), encoding="utf-8")


def score_real(res, nl, cell_size):
    """Слабая проверка по проверенному netlist.json (там только клетки, не координаты):
    доля пар клеток, соединённых в netlist, которые анализатор тоже соединил
    (с допуском в одну соседнюю клетку)."""
    def cell_xy(c):
        pc = core.parse_cell(c or "")
        return pc

    need = set()
    for net, conns in core.net_index(nl).items():
        if net.upper() in core.GROUND_NAMES:
            continue                                   # земля часто нарисована значками, а не проводом
        cells = {cell_xy(c.get("cell")) for c in nl.get("components", [])
                 if any(r == c.get("ref") for r, _ in conns)} - {None}
        cells = sorted(cells)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                need.add((cells[i], cells[j]))
    if not need:
        return 1.0
    an = []
    for n in res["nets"]:
        an.append({(int(p["x"] // cell_size), int(p["y"] // cell_size)) for p in n["pins"]})

    def near(a, cells):
        return any(abs(a[0] - c[0]) <= 1 and abs(a[1] - c[1]) <= 1 for c in cells)

    ok = sum(1 for a, b in need if any(near(a, cs) and near(b, cs) for cs in an))
    return ok / len(need)


_BASE_DOTS = {}


def _real_one(args):
    """Слабая оценка на проверенном проекте + страховка от переобучения: если точек
    соединения стало намного меньше, чем при исходных параметрах, — штраф."""
    project, params = args
    project = Path(project)
    gray = core.imread(project / "clean.png")[:, :, 0]
    res = analyzer.analyze_gray(gray, params)
    man = core.load_json(project / "manifest.json")
    s = score_real(res, core.load_json(project / "netlist.json"), man["grid"]["cell"])
    if str(project) not in _BASE_DOTS:
        _BASE_DOTS[str(project)] = len(analyzer.analyze_gray(gray, dict(analyzer.DEFAULTS))["dots"])
    base = _BASE_DOTS[str(project)]
    if base >= 5 and len(res["dots"]) < 0.6 * base:
        s *= len(res["dots"]) / base
    return s


# ---------- подбор параметров ----------

def perturb(p, rng, strength):
    q = dict(p)
    keys = rng.sample(list(SPACE), k=rng.randint(1, 3))
    for k in keys:
        lo, hi, is_int = SPACE[k]
        if is_int:
            q[k] = int(min(hi, max(lo, q[k] + rng.choice([-1, 1]))))
        else:
            v = q[k] * math.exp(rng.gauss(0, strength))
            q[k] = round(float(min(hi, max(lo, v))), 3)
    return q


def save_params(params, metrics, note):
    LEARNED.mkdir(exist_ok=True)
    (LEARNED / "params.json").write_text(json.dumps(
        {"params": params, "metrics": metrics, "note": note,
         "updated": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")


def log(row):
    LEARNED.mkdir(exist_ok=True)
    with open(LEARNED / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tune(n=24, steps=60, seed=None):
    rng = random.Random(seed)
    best = analyzer.load_params()
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as pool:
        best_m = evaluate(best, n, pool)
        print(f"старт: score {best_m['score']:.4f}  F1 {best_m['f1']:.3f}  детали {best_m['comps']:.3f}  "
              f"выводы {best_m['pins']:.3f}")
        log({"step": 0, "metrics": best_m, "params": best, "accepted": True})
        for step in range(1, steps + 1):
            strength = 0.25 * (1 - step / (steps + 1)) + 0.05
            cand = perturb(best, rng, strength)
            m = evaluate(cand, n, pool)
            ok = m["score"] > best_m["score"] + 1e-4
            log({"step": step, "metrics": m, "params": cand, "accepted": ok})
            if ok:
                best, best_m = cand, m
                save_params(best, best_m, f"подобрано на {n} синтетических схемах")
                changed = {k: v for k, v in cand.items() if analyzer.DEFAULTS.get(k) != v}
                print(f"шаг {step}: лучше → score {m['score']:.4f}  F1 {m['f1']:.3f}  {changed}")
            else:
                print(f"шаг {step}: {m['score']:.4f}", end="\r")
    print(f"\nитог: score {best_m['score']:.4f}  F1 {best_m['f1']:.3f}")
    return best, best_m


# ---------- классификатор символов ----------

def _symbols_one(args):
    seed, params = args
    img, truth = synth.generate(seed)
    res = analyzer.analyze_gray(prep(img), params)
    X, y = [], []
    for sy, f in zip(res["symbols"], res["_features"]):
        b = sy["box"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        lab = "OTHER"
        best = 1e9
        for c in truth["comps"]:
            cb = c["box"]
            if cb[0] - 4 <= cx <= cb[2] + 4 and cb[1] - 4 <= cy <= cb[3] + 4:
                d = abs(cx - (cb[0] + cb[2]) / 2) + abs(cy - (cb[1] + cb[3]) / 2)
                if d < best:
                    best, lab = d, c["type"]
        X.append(f)
        y.append(lab)
    return X, y


def save_examples(X, y, src):
    """Добавить примеры к learned/symbols.npz (src: synth|real)."""
    LEARNED.mkdir(exist_ok=True)
    path = LEARNED / "symbols.npz"
    X = np.array(X, np.float32)
    y = np.array(y)
    s = np.array([src] * len(y))
    if path.exists():
        d = np.load(path, allow_pickle=False)
        keep = d["src"] != src if src == "synth" else np.ones(len(d["y"]), bool)
        X = np.concatenate([d["X"][keep], X])
        y = np.concatenate([d["y"][keep], y])
        s = np.concatenate([d["src"][keep], s])
    np.savez_compressed(path, X=X.astype(np.float16), y=y, src=s)    # половинная точность: файл вдвое меньше
    return len(y)


def train_symbols(n=80):
    params = analyzer.load_params()
    workers = max(1, (os.cpu_count() or 2) - 1)
    X, y = [], []
    with ProcessPoolExecutor(workers) as pool:
        for Xi, yi in pool.map(_symbols_one, [(9000 + i, params) for i in range(n)]):
            X += Xi
            y += yi
    total = save_examples(X, y, "synth")
    # проверка на отложенных схемах
    model = analyzer.SymbolModel()
    ok = cnt = 0
    for seed in range(20000, 20010):
        Xi, yi = _symbols_one((seed, params))
        for f, lab in zip(Xi, yi):
            cnt += 1
            ok += model.predict(np.asarray(f))[0] == lab
    stats = {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"примеров: {total} (синтетика {len(y)}: {stats}); точность на новых схемах: {ok / max(1, cnt):.3f}")


TYPE_MAP = {"R": "R", "RV": "R", "C": "C", "L": "L", "D": "D", "Z": "D", "Q": "Q", "M": "Q",
            "IGBT": "Q", "U": "U", "OC": "U", "BR": "U", "J": "PORT", "K": "OTHER", "T": "L", "X": "OTHER",
            "SENSOR": "SENSOR", "RT": "RT", "TC": "TC", "XTAL": "XTAL", "LED": "LED", "MCU": "MCU", "TERM": "TERM",
            "ENC": "ENC", "TACHO": "TACHO", "HALL": "HALL"}


def add_project(project):
    """Проверенный проект: символ ↔ компонент из netlist.json по клетке (один к одному)."""
    project = Path(project)
    nl = core.load_json(project / "netlist.json")
    if not nl:
        raise SystemExit("в проекте нет netlist.json")
    register(project)
    res = analyzer.analyze(project, ocr=False)
    by_cell_c, by_cell_s = {}, {}
    for c in nl.get("components", []):
        by_cell_c.setdefault(c.get("cell", "").upper(), []).append(c)
    for sy, f in zip(res["symbols"], res["_features"]):
        by_cell_s.setdefault(sy["cell"].upper(), []).append(f)
    X, y = [], []
    for cell, comps in by_cell_c.items():
        syms = by_cell_s.get(cell, [])
        types = {TYPE_MAP.get(c.get("type", "X"), "OTHER") for c in comps}
        if len(syms) == 1 and len(comps) == 1 or (len(types) == 1 and len(syms) == len(comps)):
            for f in syms:
                X.append(f)
                y.append(types.pop() if len(types) == 1 else "OTHER")
                types = types or {y[-1]}
    if not X:
        print("не нашлось клеток, где символ и компонент сопоставляются однозначно")
        return
    total = save_examples(X, y, "real")
    print(f"добавлено {len(X)} примеров из {project.name}; всего {total}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        workers = max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(workers) as pool:
            m = evaluate(analyzer.load_params(), n, pool)
        print(json.dumps({k: round(v, 4) for k, v in m.items()}, ensure_ascii=False))
    elif cmd == "tune":
        tune(int(sys.argv[2]) if len(sys.argv) > 2 else 24, int(sys.argv[3]) if len(sys.argv) > 3 else 60)
    elif cmd == "symbols":
        train_symbols(int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    elif cmd == "add":
        add_project(sys.argv[2])
    elif cmd == "auto":
        # всё сразу: проекты → классификатор → короткий подбор параметров
        for p in registered():
            if (Path(p) / "netlist.json").exists():
                add_project(p)
        train_symbols(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
        tune(24, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        print(__doc__)
