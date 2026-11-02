"""Самообучение анализатора.

    python learn.py eval   [N]          — оценить текущие параметры на N синтетических схемах
    python learn.py tune   [N] [шагов]  — подобрать параметры (сохраняются в learned/params.json)
    python learn.py symbols [N]         — обучить классификатор символов на синтетике
    python learn.py add папка_проекта   — добавить проверенный проект (netlist.json) в обучение
    python learn.py auto [N] [шагов]    — дообучить всё: проекты, символы, параметры
    python learn.py curriculum [N] [шагов] — по ступеням: от чистых простых схем до фото сложных
    python learn.py guard проект …      — реальные схемы без ответа: страховка от переобучения
    python learn.py ocr [N]             — дообучить распознавание надписей на N учебных схемах
    python learn.py holdout проект …    — отложить проверенный лист в тест: в обучении и подборе не участвует

Оценка: для каждой пары выводов сравнивается «в одной цепи / в разных» по ответу
и по анализатору (F1); детали и выводы сопоставляются один к одному, так что лишние
находки снижают точность (precision), а пропуски — полноту (recall); в score идут F1.
Подбор — случайный поиск вокруг лучшего найденного на учебном наборе (SEED0); шаг
принимается, только если не ухудшилась и отдельная проверочная синтетика (SEED_VAL),
которую подбор не оптимизирует. Отложенные листы (learned/test_sheets.json) видит
только `learn.py eval`. Каждый шаг пишется в learned/history.jsonl, так что обучение
можно продолжать с того места, где остановились.
"""
import json
import math
import os
import random
import re
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
SEED0 = 5000                     # учебный набор схем для сравнения параметров при подборе
SEED_VAL = 60000                 # проверочный набор: подбор его не оптимизирует, только сверяется (9000+ — примеры символов)
VAL_TOL = 0.002                  # насколько может просесть проверка, чтобы шаг всё же приняли

# пределы для подбора: (мин, макс, целое?)
SPACE = {
    "text_max": (9, 18, False), "min_len": (2.5, 11, False), "tol": (1.2, 3.5, False),
    "dot_d": (1.4, 2.6, False), "dot_min": (1.8, 3.2, False), "dot_max": (5, 12, False),
    "rect_fill": (0.01, 0.1, False), "rect_cont": (1, 4, True), "plate_thick": (1.4, 2.6, False),
    "stroke_len": (1.5, 4.0, False), "sym_area": (2, 8, False), "sym_grow": (2, 5, False),
    "pin_dist": (1.2, 3.5, False),
}


def prep(img, truth=None):
    """Тот же путь, что у настоящей картинки: выравнивание листа на фото (core.rectify) и
    очистка (core.clean). Если лист выровнен, ответ пересчитывается теми же преобразованиями."""
    bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rect, Hw = core.rectify(bgr) if img.ndim == 3 else (bgr, None)
    if truth is not None and Hw is not None:
        tp = lambda x, y: cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), Hw)[0][0]
        truth = dict(truth)
        pins = []
        for q in truth["pins"]:
            a, b = tp(q["x"], q["y"]), tp(q["ox"], q["oy"])
            pins.append(dict(q, x=float(a[0]), y=float(a[1]), ox=float(b[0]), oy=float(b[1])))
        comps = []
        for c in truth["comps"]:
            b = c["box"]
            pts = np.array([tp(x, y) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3]))])
            comps.append(dict(c, box=[float(pts[:, 0].min()), float(pts[:, 1].min()),
                                      float(pts[:, 0].max()), float(pts[:, 1].max())]))
        truth["pins"], truth["comps"] = pins, comps
    gray = core.clean(rect)
    return (gray, truth) if truth is not None else gray


# ---------- оценка ----------

def score(res, truth):
    """Сравнить результат анализатора с ответом. Возвращает словарь метрик.
    truth["parts_only"] — в ответе только детали (проверенный netlist: земля и питание там — свойства
    цепей), поэтому символы земли/питания анализатора и их выводы не считаются ни верными, ни лишними."""
    if truth.get("parts_only"):
        skip = {s["id"] for s in res["symbols"] if s["kind"] in ("GND", "PORT")}
        res = dict(res, symbols=[s for s in res["symbols"] if s["id"] not in skip],
                   nets=[dict(n, pins=[p for p in n["pins"] if p["symbol"] not in skip]) for n in res["nets"]])
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
    # выводы анализатора в цепях из 2+ выводов, не совпавшие ни с одним выводом ответа, — лишние
    per_net = {}
    for _, _, nid, _ in an_pins:
        per_net[nid] = per_net.get(nid, 0) + 1
    multi = {i for i, (_, _, nid, _) in enumerate(an_pins) if per_net[nid] > 1}
    pin_prec = len(taken & multi) / len(multi) if multi else 1.0
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
    found = _match_comps(truth["comps"], res["symbols"], R)
    comp_rec = found / max(1, len(truth["comps"]))
    comp_prec = found / len(res["symbols"]) if res["symbols"] else 1.0
    pin_rec = sum(p is not None for p in pred) / max(1, n)
    return {"f1": f1, "precision": prec, "recall": rec,
            "comps": comp_rec, "comps_prec": comp_prec, "comps_f1": _f1(comp_prec, comp_rec),
            "pins": pin_rec, "pins_prec": pin_prec, "pins_f1": _f1(pin_prec, pin_rec),
            "extra_symbols": len(res["symbols"]) - found,
            "score": 0.6 * f1 + 0.2 * _f1(comp_prec, comp_rec) + 0.2 * _f1(pin_prec, pin_rec)}


def _f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def _match_comps(comps, symbols, R):
    """Сколько деталей ответа нашлось, если каждый символ анализатора засчитывается не больше
    одного раза: пара годится, когда центр одной рамки лежит в другой (с допуском R); пары
    разбираются от ближайших центров. Лишние и раздробленные символы остаются без пары."""
    def center(b):
        return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

    def inside(pt, b):
        return b[0] - R <= pt[0] <= b[2] + R and b[1] - R <= pt[1] <= b[3] + R

    pairs = []
    for i, c in enumerate(comps):
        ct = center(c["box"])
        for j, s in enumerate(symbols):
            cs = center(s["box"])
            if inside(ct, s["box"]) or inside(cs, c["box"]):
                pairs.append((math.hypot(ct[0] - cs[0], ct[1] - cs[1]), i, j))
    used_c, used_s = set(), set()
    for _, i, j in sorted(pairs):
        if i not in used_c and j not in used_s:
            used_c.add(i)
            used_s.add(j)
    return len(used_c)


LEVELS_ALL = (1, 2, 3, 4, 5)


def _eval_one(args):
    seed, params, level = (args + (None,))[:3]
    img, truth = synth.generate(seed, level=level)
    gray, truth = prep(img, truth)
    m = score(analyzer.analyze_gray(gray, params), truth)
    m["level"] = truth.get("level", 0)
    return m


def evaluate(params, n=24, pool=None, levels=LEVELS_ALL, seed0=SEED0, real=True, test=False):
    """Средние метрики на n схемах; уровни сложности по кругу. В m["by_level"] — score по уровням.
    real — подмешать учебные проверенные листы; test — отдельно посчитать отложенные листы
    (m["test_real"], в score не входит)."""
    jobs = [(seed0 + i, params, levels[i % len(levels)]) for i in range(n)]
    rows = list(pool.map(_eval_one, jobs)) if pool else [_eval_one(j) for j in jobs]
    keys = [k for k in rows[0] if k != "level"]
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    m["by_level"] = {lv: round(float(np.mean([r["score"] for r in rows if r["level"] == lv])), 3)
                     for lv in sorted({r["level"] for r in rows})}
    guards = core.load_json(LEARNED / "guard.json") or []
    if guards:                                   # страховка на реальных схемах без ответа
        gf = list(pool.map(_guard_one, [(g, params) for g in guards])) if pool else \
            [_guard_one((g, params)) for g in guards]
        m["guard"] = float(np.mean(gf))
        m["score"] *= m["guard"]
    run = lambda ps: list(pool.map(_real_one, [(p, params) for p in ps])) if pool else \
        [_real_one((p, params)) for p in ps]
    train = [p for p in registered() if (Path(p) / "netlist.json").exists()] if real else []
    if train:
        m["real"] = float(np.mean(run(train)))
        m["score"] = 0.7 * m["score"] + 0.3 * m["real"]
    held = [p for p in test_sheets() if (Path(p) / "netlist.json").exists()] if test else []
    if held:
        m["test_real"] = float(np.mean(run(held)))
    return m


# ---------- реальные проверенные проекты ----------

def registered():
    return (core.load_json(LEARNED / "projects.json") or [])


def test_sheets():
    """Отложенные проверенные листы: только для итоговой оценки, никогда для обучения."""
    return core.load_json(LEARNED / "test_sheets.json") or []


def holdout(project):
    """Перевести проверенный лист в тест целиком: убрать из учебных проектов и удалить
    уже взятые с него примеры символов, чтобы он больше нигде не участвовал в обучении."""
    p = str(Path(project).resolve())
    LEARNED.mkdir(exist_ok=True)
    tests = test_sheets()
    if p not in tests:
        tests.append(p)
        (LEARNED / "test_sheets.json").write_text(json.dumps(tests, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    lst = [q for q in registered() if q != p]
    (LEARNED / "projects.json").write_text(json.dumps(lst, ensure_ascii=False, indent=1), encoding="utf-8")
    import generations
    name = Path(p).name
    pf = generations._pool_file(p)
    pool = generations._load(pf)
    gi = np.flatnonzero(pool["status"] == "gen") if pool else []
    own = {"X": {pool["feat"][i].astype(np.float16).tobytes() for i in gi} if pool else set(),
           "crops": {pool["crop"][i].tobytes() for i in gi} if pool else set()}
    removed = 0
    for path, field in ((LEARNED / "symbols.npz", "X"), (LEARNED / "symdata.npz", "crops")):
        if path.exists():
            d = dict(np.load(path, allow_pickle=False))
            src = d["src"].astype(str)
            vals = d[field].astype(np.float16) if field == "X" else d[field]
            # примеры с листа, ваши правки на нём и метки поколения, выданные его символам
            drop = (src == "real:" + name) | np.char.startswith(src, f"user:{name}:")
            drop |= np.array([s == "gen" and vals[i].tobytes() in own[field] for i, s in enumerate(src)], bool)
            removed += int(drop.sum())
            np.savez_compressed(path, **{k: v[~drop] for k, v in d.items()})
    if pf.exists():                              # пул листа — в сторону: поколения его больше не видят
        (LEARNED / "pool_test").mkdir(exist_ok=True)
        pf.replace(LEARNED / "pool_test" / pf.name)
    print(f"{name}: отложен в тест, убрано примеров символов: {removed}; "
          f"учебных листов {len(lst)}, тестовых {len(tests)}")


def register(project):
    p = str(Path(project).resolve())
    if p in test_sheets():
        return                                   # тестовый лист в обучение не попадает
    LEARNED.mkdir(exist_ok=True)
    lst = registered()
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
_BASE_GUARD = {}


def _guard_one(args):
    """Реальная схема без ответа: при новых параметрах не должно резко стать меньше надписей
    и точек соединения или резко больше «цепей» из одного вывода (распавшийся текст), чем при
    исходных. Возвращает множитель 0…1."""
    project, params = args
    gray = core.imread(Path(project) / "clean.png")[:, :, 0]

    def stats(res):
        single = sum(1 for n in res["nets"] if len(n["pins"]) + len(n["labels"]) < 2)
        return len(res["texts"]), len(res["dots"]), single, len(res["segments"]), len(res["symbols"])

    if project not in _BASE_GUARD:
        _BASE_GUARD[project] = stats(analyzer.analyze_gray(gray, dict(analyzer.DEFAULTS)))
    t0, d0, s0, g0, y0 = _BASE_GUARD[project]
    t1, d1, s1, g1, y1 = stats(analyzer.analyze_gray(gray, params))
    f = 1.0
    if g1 > 2 * g0 + 30:                        # провода «размножились» — рвутся символы и текст
        f *= (2 * g0 + 30) / g1
    if y1 > 2 * y0 + 15:
        f *= (2 * y0 + 15) / y1
    if t0 >= 10 and t1 < 0.75 * t0:
        f *= t1 / (0.75 * t0)
    if d0 >= 5 and d1 < 0.6 * d0:
        f *= d1 / (0.6 * d0)
    if s1 > 1.6 * s0 + 10:
        f *= (1.6 * s0 + 10) / s1
    return f


def truth_from_netlist(nl):
    """Проверенный netlist с рамками и точками выводов (box, pin_xy у каждой детали) → ответ в
    формате синтетики для score(). None, если хоть у одной детали координат нет."""
    comps = [c for c in (nl or {}).get("components", []) if c.get("type") not in ("GND", "PORT")]
    if not comps or not all(c.get("box") and c.get("pin_xy") for c in comps):
        return None
    ids = {}
    pins = []
    for c in comps:
        for pn, net in (c.get("pins") or {}).items():
            xy = c["pin_xy"].get(pn)
            if xy is None or not net:
                continue
            pins.append({"comp": c["ref"], "pin": pn, "x": xy[0], "y": xy[1], "ox": xy[0], "oy": xy[1],
                         "net": ids.setdefault(net, len(ids))})
    out = []
    for c in comps:                                    # микросхема из секций — каждая секция своим символом
        for s, b in (c.get("section_boxes") or {"": c["box"]}).items():
            out.append({"type": c.get("type"), "ref": f"{c.get('ref')}.{s}" if s else c.get("ref"), "box": b})
    return {"comps": out, "pins": pins, "parts_only": True}


def _real_one(args):
    """Слабая оценка на проверенном проекте + страховка от переобучения: если точек
    соединения стало намного меньше, чем при исходных параметрах, — штраф."""
    project, params = args
    project = Path(project)
    gray = core.imread(project / "clean.png")[:, :, 0]
    res = analyzer.analyze_gray(gray, params)
    man = core.load_json(project / "manifest.json")
    nl = core.load_json(project / "netlist.json")
    truth = truth_from_netlist(nl)
    # лист с точками выводов — та же строгая оценка, что на синтетике; иначе — слабая, по клеткам
    s = score(res, truth)["score"] if truth else score_real(res, nl, man["grid"]["cell"])
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


def tune(n=24, steps=60, seed=None, levels=LEVELS_ALL):
    rng = random.Random(seed)
    best = analyzer.load_params()
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as pool:
        best_m = evaluate(best, n, pool, levels)
        best_v = evaluate(best, n, pool, levels, seed0=SEED_VAL, real=False)["score"]
        best_m["val"] = best_v
        print(f"старт: score {best_m['score']:.4f}  проверка {best_v:.4f}  F1 {best_m['f1']:.3f}  "
              f"детали P/R {best_m['comps_prec']:.3f}/{best_m['comps']:.3f}  "
              f"выводы P/R {best_m['pins_prec']:.3f}/{best_m['pins']:.3f}")
        log({"step": 0, "metrics": best_m, "params": best, "accepted": True})
        for step in range(1, steps + 1):
            strength = 0.25 * (1 - step / (steps + 1)) + 0.05
            cand = perturb(best, rng, strength)
            m = evaluate(cand, n, pool, levels)
            ok = m["score"] > best_m["score"] + 1e-4
            if ok:                               # улучшение на учебном наборе сверяем с проверочным
                m["val"] = evaluate(cand, n, pool, levels, seed0=SEED_VAL, real=False)["score"]
                ok = m["val"] >= best_v - VAL_TOL
            log({"step": step, "metrics": m, "params": cand, "accepted": ok})
            if ok:
                best, best_m, best_v = cand, m, max(best_v, m["val"])
                save_params(best, best_m, f"подобрано на {n} синтетических схемах, сверено с проверочными")
                changed = {k: v for k, v in cand.items() if analyzer.DEFAULTS.get(k) != v}
                print(f"шаг {step}: лучше → score {m['score']:.4f}  проверка {m['val']:.4f}  "
                      f"F1 {m['f1']:.3f}  {changed}")
            elif "val" in m:
                print(f"шаг {step}: {m['score']:.4f}, но проверка {m['val']:.4f} < {best_v:.4f} — отклонено")
            else:
                print(f"шаг {step}: {m['score']:.4f}", end="\r")
    print(f"\nитог: score {best_m['score']:.4f}  проверка {best_m['val']:.4f}  F1 {best_m['f1']:.3f}  "
          f"по уровням {best_m['by_level']}")
    return best, best_m


def curriculum(n=30, steps=25):
    """Обучение от простого к сложному: сначала чистые маленькие схемы, потом добавляются
    сканы, плотные листы, схемы управления и фото телефоном. Параметры переходят со ступени
    на ступень; на каждой ступени выше оценивается всё пройденное, чтобы не забывать простое."""
    stages = [(1, 2), (1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5)]
    for k, lv in enumerate(stages, 1):
        print(f"=== ступень {k}: уровни {lv}")
        tune(n, steps, seed=k, levels=lv)


# ---------- классификатор символов ----------

def _symbols_one(args):
    seed, params = args
    img, truth = synth.generate(seed)
    gray, truth = prep(img, truth)
    res = analyzer.analyze_gray(gray, params)
    X, y, Cr = [], [], []
    for sy, f, cr in zip(res["symbols"], res["_features"], res["_crops"]):
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
        Cr.append(cr)
    return X, y, Cr


def save_examples(X, y, src):
    """Добавить примеры к learned/symbols.npz (src: synth|real)."""
    LEARNED.mkdir(exist_ok=True)
    path = LEARNED / "symbols.npz"
    X = np.array(X, np.float32)
    y = np.array(y)
    s = np.array([src] * len(y))
    if path.exists():
        d = np.load(path, allow_pickle=False)
        keep = d["src"] != src                  # пример с того же листа заменяется, а не копится
        X = np.concatenate([d["X"][keep], X])
        y = np.concatenate([d["y"][keep], y])
        s = np.concatenate([d["src"][keep], s])
    np.savez_compressed(path, X=X.astype(np.float16), y=y, src=s)    # половинная точность: файл вдвое меньше
    return len(y)


def _ocr_one(args):
    """Учебная схема: для каждой подписи (обозначение/номинал) найти надпись анализатора
    рядом с деталью и вырезать буквы, если их число совпало с ответом."""
    import ocr as ocrmod
    seed, params = args
    img, truth = synth.generate(seed)
    gray, truth = prep(img, truth)
    res = analyzer.analyze_gray(gray, params)
    eng = _ocr_one.engine = getattr(_ocr_one, "engine", None) or ocrmod.OCR()
    out = []
    for c in truth["comps"]:
        for label in (c.get("ref"), c.get("value")):
            if not label:
                continue
            b = c["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            size = max(b[2] - b[0], b[3] - b[1])
            best = None
            for t in res["texts"]:
                tb = t["box"]
                d = analyzer.box_dist(b, (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2)
                if d < 1.5 * size and (best is None or d < best[0]):
                    best = (d, t)
            if best:
                out += [(f.tolist(), ch) for f, ch in
                        ocrmod.learn_word(eng, gray, best[1]["box"], best[1]["vertical"], label)]
    return out


def train_ocr(n=300):
    """Самообучение OCR на учебных схемах (буквы в настоящих условиях: мелкий кегль, скан, фото)
    и на проверенных проектах (обозначения из netlist.json)."""
    import ocr as ocrmod
    params = analyzer.load_params()
    workers = max(1, (os.cpu_count() or 2) - 1)
    X, y = [], []
    with ProcessPoolExecutor(workers) as pool:
        for pairs in pool.map(_ocr_one, [(40000 + i, params) for i in range(n)], chunksize=4):
            for f, ch in pairs:
                X.append(f)
                y.append(ch)
    eng = ocrmod.OCR()
    for proj in registered():                         # реальные: обозначение у детали в своей клетке
        proj = Path(proj)
        nl = core.load_json(proj / "netlist.json")
        an = core.load_json(proj / "analysis" / "analysis.json")
        man = core.load_json(proj / "manifest.json")
        if not (nl and an and man):
            continue
        gray = core.imread(proj / "clean.png")[:, :, 0]
        cell = man["grid"]["cell"]
        refs = {c["ref"]: core.parse_cell(c.get("cell", "")) for c in nl["components"]}
        vals = [(str(c.get("value", "")).split("/")[0].split(" ")[0], core.parse_cell(c.get("cell", "")))
                for c in nl["components"] if c.get("value") and c.get("type") in ("R", "C", "L")]
        for t in an["texts"]:
            tb = t["box"]
            cc = (int((tb[0] + tb[2]) / 2 // cell), int((tb[1] + tb[3]) / 2 // cell))
            for ref, pc in refs.items():
                if pc and abs(pc[0] - cc[0]) <= 1 and abs(pc[1] - cc[1]) <= 1 and \
                        ocrmod.fix_ref(eng.word(gray, tb, t["vertical"])[0] or "") == ref:
                    for f, ch in ocrmod.learn_word(eng, gray, tb, t["vertical"], ref):
                        X.append(f.tolist())
                        y.append(ch)
            # номинал: в той же клетке, число букв совпадает, цифры согласуются с прочитанным
            raw = eng.word(gray, tb, t["vertical"])[0]
            for v, pc in vals:
                want = [ch for ch in v.replace("u", "μ") if ch not in " .,-"]
                got = [ch for ch in raw if ch not in " .,-"]
                if pc and abs(pc[0] - cc[0]) <= 1 and abs(pc[1] - cc[1]) <= 1 and len(want) == len(got)                         and want and want[0] == got[0]:
                    for f, ch in ocrmod.learn_word(eng, gray, tb, t["vertical"], "".join(want)):
                        X.append(f.tolist())
                        y.append(ch)
                    break
    total = ocrmod.save_extra(X, y)
    print(f"вырезано букв из схем: {len(y)}, всего в дообучении: {total}")


def train_symbols(n=80):
    params = analyzer.load_params()
    workers = max(1, (os.cpu_count() or 2) - 1)
    import symclass
    X, y, Cr = [], [], []
    with ProcessPoolExecutor(workers) as pool:
        for Xi, yi, ci in pool.map(_symbols_one, [(9000 + i, params) for i in range(n)]):
            X += Xi
            y += yi
            Cr += ci
    total = save_examples(X, y, "synth")
    symclass.save([c for c, _ in Cr], [d for _, d in Cr], y, "synth")
    # проверка на отложенных схемах
    model = analyzer.SymbolModel()
    ok = cnt = 0
    for seed in range(20000, 20010):
        Xi, yi, _ = _symbols_one((seed, params))
        for f, lab in zip(Xi, yi):
            cnt += 1
            ok += model.predict(np.asarray(f))[0] == lab
    stats = {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"примеров: {total} (синтетика {len(y)}: {stats}); точность на новых схемах: {ok / max(1, cnt):.3f}")


TYPE_MAP = {"R": "R", "RV": "R", "C": "C", "L": "L", "D": "D", "Z": "D", "Q": "Q", "M": "Q",
            "IGBT": "Q", "VS": "VS", "U": "U", "OC": "OC", "RP": "R", "CV": "C", "VC": "D", "JFET": "Q", "DIAC": "VS",
            "OPAMP": "OPAMP", "COMP": "OPAMP", "LOGIC": "LOGIC", "REG": "U", "MEM": "U", "BR": "U", "J": "PORT", "K": "COIL", "T": "L", "X": "OTHER",
            "SENSOR": "SENSOR", "RT": "RT", "TC": "TC", "XTAL": "XTAL", "LED": "LED", "MCU": "MCU", "TERM": "TERM",
            "ENC": "ENC", "TACHO": "TACHO", "HALL": "HALL",
            "SW": "SW_NO", "BTN": "BTN", "LAMP": "LAMP", "MOTOR": "MOTOR", "QF": "QF", "F": "FU"}


# корпуса-прямоугольники (R, U, разъёмы) классификатор не трогает — для разметки «символов» они не в счёт
RECT_CLASSES = {"R", "U", "PORT", "MCU", "TERM", "OTHER"}


def _cell_xy(cell):
    m = re.match(r"([A-Z]+)(\d+)", str(cell).upper())
    if not m:
        return None
    col = 0
    for ch in m.group(1):
        col = col * 26 + ord(ch) - 64
    return col - 1, int(m.group(2)) - 1


def _examples_by_box(parts, res, power_symbols=()):
    """Лист с рамками деталей (netlist после проверки): символ анализатора получает тип детали, рамку
    которой он задевает — ровно одной; символ далеко от всех деталей — «не деталь» (OTHER). Символ
    у корпуса микросхемы — скорее обрывок ножек/надписей, его не размечаем. power_symbols — знаки
    земли и питания (в netlist это цепи, а не детали): {"type": "GND"|"PORT", "box"} → классы GND/PORT."""
    boxes = []
    for c in parts:
        for b in (c.get("section_boxes") or {"": c["box"]}).values():
            boxes.append((b, TYPE_MAP.get(c.get("type", "X"), "OTHER")))
    for s in power_symbols:
        boxes.append((s["box"], "GND" if s.get("type") == "GND" else "PORT"))
    sw = res.get("stroke", 2)
    X, y, Cr = [], [], []
    for sy, f, cr in zip(res["symbols"], res["_features"], res["_crops"]):
        if sy["kind"] in ("R", "U", "GND", "PORT"):
            continue
        x0, y0, x1, y1 = sy["box"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hit, near = [], False
        for b, cls in boxes:
            pad = 0.25 * max(b[2] - b[0], b[3] - b[1])
            if b[0] - pad <= cx <= b[2] + pad and b[1] - pad <= cy <= b[3] + pad:
                hit.append(cls)
            if analyzer.box_dist(b, cx, cy) <= 6 * sw + pad:
                near = True
        if len(hit) == 1 and hit[0] not in ("U", "MCU", "TERM", "OTHER"):
            X.append(f); y.append(hit[0]); Cr.append(cr)
        elif not hit and not near:
            X.append(f); y.append("OTHER"); Cr.append(cr)
    return X, y, Cr


def real_examples(project, res=None):
    """Примеры символов с проверенного листа. Клетки netlist.json приблизительны, поэтому:
    «не деталь» (OTHER) — символ, рядом с которым (его клетки и соседние) по ответу нет ни одной
    символьной детали; тип — когда в клетке ровно одна символьная деталь и ровно один символ."""
    project = Path(project)
    nl = core.load_json(project / "netlist.json")
    cell = core.load_json(project / "manifest.json")["grid"]["cell"]
    res = res or analyzer.analyze(project, ocr=False)
    parts = [c for c in nl.get("components", []) if c.get("type") not in ("GND", "PORT")]
    if parts and all(c.get("box") for c in parts):
        return _examples_by_box(parts, res, nl.get("power_symbols", []))
    comps = {}
    for c in nl.get("components", []):
        cls = TYPE_MAP.get(c.get("type", "X"), "OTHER")
        xy = _cell_xy(c.get("cell", ""))
        if cls not in RECT_CLASSES and xy:
            comps.setdefault(xy, []).append(cls)
    cand = []
    for sy, f, cr in zip(res["symbols"], res["_features"], res["_crops"]):
        if sy["kind"] in ("R", "U") or sy.get("kind") in ("GND", "PORT"):
            continue
        x0, y0, x1, y1 = sy["box"]
        span = {(int(x // cell), int(y // cell)) for x in (x0, x1) for y in (y0, y1)}
        cand.append((span, (f, cr)))
    X, y = [], []
    per_cell = {}
    for span, f in cand:
        near = {(cx + dx, cy + dy) for cx, cy in span for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        if not any(comps.get(c) for c in near):
            X.append(f)
            y.append("OTHER")
        elif len(span) == 1:
            per_cell.setdefault(next(iter(span)), []).append(f)
    for xy, fs in per_cell.items():
        if len(fs) == 1 and len(comps.get(xy, [])) == 1:
            X.append(fs[0])
            y.append(comps[xy][0])
    return [f for f, _ in X], y, [c for _, c in X]


def add_project(project):
    """Проверенный проект → примеры для классификатора символов (заменяют прежние с этого листа)."""
    project = Path(project)
    if str(project.resolve()) in test_sheets():
        print(f"{project.name}: тестовый лист, в обучение не добавляется")
        return
    nl = core.load_json(project / "netlist.json")
    if not nl:
        raise SystemExit("в проекте нет netlist.json")
    if not nl.get("verified") and str(project.resolve()) not in registered():
        # прежние учебные листы остаются; новый — только после кнопки «Лист проверен»
        print(f"{project.name}: лист не отмечен как проверенный (кнопка «Лист проверен»), в обучение не добавляется")
        return
    register(project)
    X, y, Cr = real_examples(project)
    if not X:
        print("не нашлось символов, которые можно однозначно разметить")
        return
    total = save_examples(X, y, "real:" + project.name)
    import symclass
    symclass.save([c for c, _ in Cr], [d for _, d in Cr], y, "real:" + project.name)
    stats = {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"{project.name}: {len(X)} примеров {stats}; всего {total}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "eval":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
        workers = max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(workers) as pool:
            P = analyzer.load_params()
            m = evaluate(P, n, pool, test=True)
            v = evaluate(P, n, pool, seed0=SEED_VAL, real=False)
            m["val"] = v["score"]
            m["val_by_level"] = v["by_level"]
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}, ensure_ascii=False))
    elif cmd == "tune":
        tune(int(sys.argv[2]) if len(sys.argv) > 2 else 24, int(sys.argv[3]) if len(sys.argv) > 3 else 60)
    elif cmd == "symbols":
        train_symbols(int(sys.argv[2]) if len(sys.argv) > 2 else 80)
    elif cmd == "add":
        add_project(sys.argv[2])
    elif cmd == "holdout":
        for p in sys.argv[2:]:
            holdout(p)
    elif cmd == "guard":
        # python learn.py guard папка_проекта … — добавить реальные схемы (без ответа) в страховку
        lst = core.load_json(LEARNED / "guard.json") or []
        for p in sys.argv[2:]:
            p = str(Path(p).resolve())
            if p not in lst:
                lst.append(p)
        LEARNED.mkdir(exist_ok=True)
        (LEARNED / "guard.json").write_text(json.dumps(lst, ensure_ascii=False, indent=1), encoding="utf-8")
        print("страховочных схем:", len(lst))
    elif cmd == "ocr":
        train_ocr(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif cmd == "curriculum":
        curriculum(int(sys.argv[2]) if len(sys.argv) > 2 else 30, int(sys.argv[3]) if len(sys.argv) > 3 else 25)
    elif cmd == "auto":
        # всё сразу: проекты → классификатор → короткий подбор параметров
        for p in registered():
            if (Path(p) / "netlist.json").exists():
                add_project(p)
        train_symbols(int(sys.argv[2]) if len(sys.argv) > 2 else 60)
        tune(24, int(sys.argv[3]) if len(sys.argv) > 3 else 20)
    else:
        print(__doc__)
