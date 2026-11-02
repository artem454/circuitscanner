"""Самообучение по поколениям: каждая схема — эталоны, ни один символ не забывается.

Пул (learned/pool/<проект>.npz): вырез каждого символа каждой отсканированной схемы, его признаки,
метка и статус: unknown — ещё не определён; gen — определён поколением (обе модели уверенно согласны);
user — указан вами вручную (не меняется).

Поколение (run_generation):
  1. обе модели (kNN по картинке и модель формы) смотрят на весь пул;
  2. символ, по которому они уверенно сходятся, получает метку поколения; прежняя метка поколения
     снимается, если модели разошлись (ошибки не закрепляются);
  3. метки поколения становятся примерами обеих моделей (источник «gen», заменяется целиком),
     модель формы переобучается — следующее поколение видит больше и пробует то, что не смогло это;
  4. итог — в learned/generations.json: сколько символов, сколько определено (%), сколько новых и снятых.

    python generations.py             — провести поколение
    python generations.py stats       — таблица поколений
"""
import json
import time
from pathlib import Path

import numpy as np

LEARNED = Path(__file__).resolve().parent / "learned"
POOL = LEARNED / "pool"
GENS = LEARNED / "generations.json"
SKIP_KINDS = {"R", "U"}                                  # корпуса-прямоугольники определяются правилами
KNN_CONF, SHAPE_P = 0.85, 0.85                           # уверенность обеих моделей для метки поколения
GATE_PREC, GATE_MIN = 0.6, 5                             # метки поколения — только классам с точностью ≥60 % на ≥5 ответах проверенных листов
AUTO_EVERY = 10                                          # новое поколение — после стольких новых схем


def _pool_file(project):
    import hashlib
    p = str(Path(project).resolve())
    return POOL / (Path(p).name[:40] + "_" + hashlib.md5(p.encode("utf-8")).hexdigest()[:8] + ".npz")


def _load(path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _save(path, d):
    POOL.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **d)


def add_from_analysis(project, res):
    """Символы разобранной схемы — в пул (новые; уже известные по месту не дублируются)."""
    tests = json.loads((LEARNED / "test_sheets.json").read_text(encoding="utf-8")) \
        if (LEARNED / "test_sheets.json").exists() else []
    if str(Path(project).resolve()) in tests:
        return                                           # тестовый лист в обучение не попадает
    path = _pool_file(project)
    old = _load(path)
    keys = set(old["key"].tolist()) if old else set()
    rows = []
    for sy, f, (cr, dims) in zip(res.get("symbols", []), res.get("_features", []), res.get("_crops", [])):
        if sy["kind"] in SKIP_KINDS:
            continue
        key = "%d_%d_%d_%d" % tuple(int(round(v / 4)) for v in sy["box"])  # место символа (с допуском 4 px)
        if key in keys:
            continue
        keys.add(key)
        rows.append((key, cr, dims, np.asarray(f, np.float16), [float(v) for v in sy["box"]]))
    if not rows:
        return 0
    new = {"key": np.array([r[0] for r in rows]), "crop": np.array([r[1] for r in rows], np.uint8),
           "dims": np.array([r[2] for r in rows], np.float32), "feat": np.array([r[3] for r in rows], np.float16),
           "box": np.array([r[4] for r in rows], np.float32), "label": np.array([""] * len(rows), dtype="<U16"),
           "status": np.array(["unknown"] * len(rows), dtype="<U8"), "gen": np.zeros(len(rows), np.int16),
           "project": np.array([str(Path(project).resolve())] * len(rows))}
    if old:
        new = {k: np.concatenate([old[k], new[k]]) for k in new}
    _save(path, new)
    _bump_pending(project)
    return len(rows)


def mark_user(project, box, label):
    """Ваша правка: символ в этом месте — проверенный, с этой меткой (поколения его не трогают)."""
    path = _pool_file(project)
    d = _load(path)
    if not d or not len(d["box"]):
        return False
    b = np.array(box, np.float32)
    dist = np.abs(d["box"] - b[None]).sum(1)
    i = int(np.argmin(dist))
    if dist[i] > 0.6 * (abs(b[2] - b[0]) + abs(b[3] - b[1])) + 20:
        return False
    d["label"] = d["label"].astype("<U16")
    d["label"][i], d["status"][i] = label, "user"
    _save(path, d)
    return True


def _state():
    return json.loads(GENS.read_text(encoding="utf-8")) if GENS.exists() else {"pending": [], "history": []}


def _bump_pending(project):
    st = _state()
    p = str(Path(project).resolve())
    if p not in st["pending"]:
        st["pending"].append(p)
        GENS.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def due():
    """Пора ли новое поколение (накопилось AUTO_EVERY новых схем)."""
    return len(_state()["pending"]) >= AUTO_EVERY


def validate(knn, shape, agree_only=False):
    """Точность каждого класса на проверенных листах (по типам деталей в клетках символа).
    Класс, который там часто ошибается, не получает меток поколения — ошибки не закрепляются."""
    import core
    import autonet
    projs = core.load_json(LEARNED / "projects.json") or []
    hit, cnt = {}, {}
    for p in projs:
        d = _load(_pool_file(p))
        nl = core.load_json(Path(p) / "netlist.json")
        man = core.load_json(Path(p) / "manifest.json")
        if not d or not nl or not man:
            continue
        cell = man["grid"]["cell"]
        types = {}
        for c in nl.get("components", []):
            types.setdefault(str(c.get("cell", "")).upper(), set()).add(c.get("type"))
        for i in range(len(d["key"])):
            lab, c1 = knn.predict(d["feat"][i].astype(np.float32))
            if lab in ("OTHER", "GND", "PORT") or (c1 < KNN_CONF and not agree_only):   # земля/питание — цепь, не деталь
                continue
            if shape is not None:
                lab2, p2 = shape.predict(d["crop"][i], d["dims"][i], trusted_only=False)
                if lab2 != lab or (p2 < SHAPE_P and not agree_only):
                    continue                          # agree_only — как в netlist: согласны — уверенно
            x0, y0, x1, y1 = d["box"][i]
            cells = {chr(65 + int(x // cell)) + str(int(y // cell) + 1) for x in (x0, x1) for y in (y0, y1)}
            truth = set().union(*(types.get(c, set()) for c in cells))
            t = autonet.KIND2TYPE.get(lab, lab)
            ok = t in truth or (t == "SW" and bool(truth & {"BTN", "SW"})) or (t == "D" and bool(truth & {"Z", "VS"}))
            cnt[lab] = cnt.get(lab, 0) + 1
            hit[lab] = hit.get(lab, 0) + ok
    return {k: (hit[k] / cnt[k], cnt[k]) for k in cnt}


def _overall(prec):
    """Общая точность уверенных ответов на проверенных листах (по всем классам)."""
    n = sum(c for _, c in prec.values())
    return round(sum(p * c for p, c in prec.values()) / n, 3) if n else None


def _hint_holdout(path):
    """Каждая пятая схема пула — проверочная для подсказок по обозначениям: её подсказки никогда
    не становятся метками, по ним меряется, стали ли модели точнее после поколения."""
    import hashlib
    return int(hashlib.md5(path.name.encode("utf-8")).hexdigest(), 16) % 5 == 0


def _hints(d):
    return d["hint"] if "hint" in d else np.array([""] * len(d["key"]))


def hint_check(knn, shape, files, trusted):
    """По подсказкам из обозначений (классы, прошедшие проверку на синтетике):
    prec — точность уверенного согласия обеих моделей по каждому классу на всём пуле
    (для допуска класса к меткам без подсказки); acc — доля проверочных подсказок, которые
    модели называют так же (первый ответ kNN и модели формы), — общая точность «до/после»."""
    cnt, hit = {}, {}
    acc_n = acc_ok = 0
    for path in files:
        d = _load(path)
        hints = _hints(d)
        hold = _hint_holdout(path)
        for i in np.flatnonzero(np.isin(hints, list(trusted))):
            h = str(hints[i])
            lab1, c1 = knn.predict(d["feat"][i].astype(np.float32))
            lab2, p2 = shape.predict(d["crop"][i], d["dims"][i], trusted_only=False) if shape else (lab1, 1.0)
            if hold:
                acc_n += 2
                acc_ok += (lab1 == h) + (lab2 == h)
            if lab1 == lab2 and c1 >= KNN_CONF and p2 >= SHAPE_P and lab1 not in ("OTHER", "GND", "PORT"):
                cnt[lab1] = cnt.get(lab1, 0) + 1
                hit[lab1] = hit.get(lab1, 0) + (lab1 == h)
    return {k: (hit[k] / cnt[k], cnt[k]) for k in cnt}, (round(acc_ok / acc_n, 3) if acc_n else None), acc_n // 2


def run_generation(progress=None, retrain=True):
    import analyzer
    import learn
    import symclass
    say = progress or (lambda m: None)
    files = sorted(POOL.glob("*.npz")) if POOL.exists() else []
    if not files:
        return None
    st = _state()
    g = len(st["history"]) + 1
    # отпечаток данных: схемы в пуле, ваши правки, все примеры — без новых данных повтор отклонённого бессмыслен
    import reflabels
    finger = [len(files), sum(int((_load(f)["status"] == "user").sum()) for f in files),
              int(np.load(LEARNED / "symbols.npz")["y"].shape[0]) if (LEARNED / "symbols.npz").exists() else 0,
              sum(int((_hints(_load(f)) != "").sum()) for f in files)]
    if st["history"] and st["history"][-1].get("rejected") and st["history"][-1].get("data") == finger:
        say("новых схем и правок нет — прошлое поколение отклонено, повтор не нужен")
        return None
    knn = analyzer.SymbolModel()
    shape = symclass.Model() if symclass.MODEL.exists() else None
    trusted = reflabels.trusted_classes()                # классы, чьим подсказкам по обозначениям можно верить
    say(f"поколение {g}: проверка классов на проверенных листах и по обозначениям на схемах")
    prec = validate(knn, shape)
    hprec, hacc_before, hacc_n = hint_check(knn, shape, files, trusted)
    # класс допущен к меткам без подсказки, если доказал точность на проверенных листах или по обозначениям;
    # но если проверенные листы говорят против (достаточно ответов, точность низкая) — нет
    allowed = {k for k, (p, n) in prec.items() if n >= GATE_MIN and p >= GATE_PREC}
    allowed |= {k for k, (p, n) in hprec.items() if n >= reflabels.HINT_MIN and p >= reflabels.HINT_PREC}
    allowed -= {k for k, (p, n) in prec.items() if n >= GATE_MIN and p < GATE_PREC}
    blocked = (set(prec) | set(hprec)) - allowed
    acc_before = _overall(validate(knn, shape, agree_only=True))
    ver_n = sum(n for _, n in validate(knn, shape, agree_only=True).values())
    by_hint = 0
    import shutil
    backup = {}                                          # для отката, если поколение ухудшит точность
    for f in (LEARNED / "symbols.npz", LEARNED / "symdata.npz", LEARNED / "symbol_mlp.npz"):
        if f.exists():
            backup[f] = f.with_suffix(".bak")
            shutil.copyfile(f, backup[f])
    pool_backup = {}
    total = user = gen = new = revoked = 0
    ex_f, ex_c, ex_d, ex_y = [], [], [], []
    for fi, path in enumerate(files):
        say(f"поколение {g}: схема {fi + 1}/{len(files)}")
        d = _load(path)
        d["label"] = d["label"].astype("<U16")
        hints = _hints(d)
        hold = _hint_holdout(path)
        pool_backup[path] = (d["status"].copy(), d["label"].copy(), d["gen"].copy())
        for i in range(len(d["key"])):
            total += 1
            if d["status"][i] == "user":
                user += 1
                continue
            h = str(hints[i]) if str(hints[i]) in trusted else ""
            if h and hold:                               # проверочная подсказка — не метка
                if d["status"][i] == "gen":
                    d["status"][i], d["label"][i] = "unknown", ""
                continue
            lab1, c1 = knn.predict(d["feat"][i].astype(np.float32))
            lab2, p2 = shape.predict(d["crop"][i], d["dims"][i], trusted_only=False) if shape is not None else (lab1, 1.0)
            if h and (lab1 == h or lab2 == h):           # обозначение на схеме + согласие хотя бы одной модели
                ok, lab1 = True, h
                by_hint += 1
            else:
                ok = lab1 in allowed and c1 >= KNN_CONF and not h   # уверенные модели против обозначения — не метка
                if ok and shape is not None:
                    lab2, p2 = shape.predict(d["crop"][i], d["dims"][i])
                    ok = lab2 == lab1 and p2 >= SHAPE_P
            if ok:
                if d["status"][i] != "gen" or d["label"][i] != lab1:
                    new += d["status"][i] != "gen"
                    d["label"][i], d["gen"][i] = lab1, g
                d["status"][i] = "gen"
                gen += 1
                ex_f.append(d["feat"][i].astype(np.float32))
                ex_c.append(d["crop"][i])
                ex_d.append(d["dims"][i])
                ex_y.append(lab1)
            elif d["status"][i] == "gen":                # модели разошлись — метку снять, не закреплять ошибку
                d["status"][i], d["label"][i] = "unknown", ""
                revoked += 1
        _save(path, d)
    say(f"поколение {g}: обучение на {len(ex_y)} метках поколения")
    learn.save_examples(ex_f, ex_y, "gen")
    symclass.save(ex_c, ex_d, ex_y, "gen")
    if retrain:
        net, meta = symclass.train()
        symclass.save_model(net, meta)
    analyzer._MODEL.clear()
    say(f"поколение {g}: проверка точности после обучения")
    knn2, shape2 = analyzer.SymbolModel(), symclass.Model() if symclass.MODEL.exists() else None
    acc_after = _overall(validate(knn2, shape2, agree_only=True))
    _, hacc_after, _ = hint_check(knn2, shape2, files, trusted)
    # откат, если упала точность по проверочным обозначениям (их сотни) или заметно — на проверенных
    # листах; допуск по проверенным листам тем шире, чем меньше на них ответов
    tol = max(0.005, 1.0 / max(1, ver_n) ** 0.5)
    rejected = (hacc_before is not None and hacc_after is not None and hacc_after < hacc_before - 0.005) or \
        (acc_before is not None and acc_after is not None and acc_after < acc_before - tol)
    if rejected:                                         # точность упала — поколение откатить целиком
        for f, b in backup.items():
            shutil.copyfile(b, f)
        for path, (stt, lab, gg) in pool_backup.items():
            d = _load(path)
            d["status"], d["label"], d["gen"] = stt, lab, gg
            _save(path, d)
        analyzer._MODEL.clear()
    for b in backup.values():
        b.unlink(missing_ok=True)
    row = {"gen": g, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "schematics": len(files), "symbols": total,
           "user": user, "by_generation": gen, "unknown": total - user - gen, "new": new, "revoked": revoked,
           "resolved_pct": round(100 * (user + gen) / max(1, total), 1), "blocked": sorted(blocked),
           "precision_before": acc_before, "precision_after": acc_after, "rejected": bool(rejected),
           "by_hint": by_hint, "hint_check_n": hacc_n, "hint_acc_before": hacc_before, "hint_acc_after": hacc_after,
           "hint_classes": sorted(trusted), "data": finger,
           "class_precision": {k: round(p, 2) for k, (p, n) in sorted(prec.items())},
           "hint_class_precision": {k: [round(p, 2), n] for k, (p, n) in sorted(hprec.items())}}
    st["history"].append(row)
    st["pending"] = []
    GENS.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return row


def stats_text():
    h = _state()["history"]
    if not h:
        return "Поколений ещё не было."
    lines = ["Пок. | схем | символов | определено | ваши | новых | снято | без меток (точность не доказана)"]
    for r in h:
        acc = (f" | точность {r['precision_before']}→{r['precision_after']}" if r.get("precision_after") is not None else "")
        lines.append(f"{r['gen']:4} | {r['schematics']:4} | {r['symbols']:8} | {r['resolved_pct']:9.1f}% | "
                     f"{r['user']:4} | {r['new']:5} | {r['revoked']:5} | {', '.join(r.get('blocked', [])) or '—'}"
                     + acc + (" | ОТКЛОНЕНО: точность упала, откат" if r.get("rejected") else ""))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["stats"]:
        print(stats_text())
    else:
        print(run_generation(progress=print))
