"""Проверенные листы учат, где модели ошибаются (поиск трудных примеров).

Для каждого символа проверенного листа модели дают тип. Сверка с проверенным netlist по клеткам:
  • в клетках символа и вокруг (3×3) нет ни одной детали этого типа — модель ошиблась: символ становится
    примером «не деталь» (OTHER) — так уходят ложные «дроссели», «двигатели» из обрывков надписей и дуг;
  • тип совпал с деталью в своей клетке — подтверждённый пример этого типа.
Источник примеров «mine:<лист>», при повторе заменяется.

    python mining.py            — добыть примеры со всех проверенных листов и переобучить модель формы
    python mining.py check      — честная проверка: учиться на половине листов, мерить на другой
"""
import json
import sys
from pathlib import Path

import numpy as np

import analyzer
import autonet
import core
import learn
import symclass

LEARNED = Path(__file__).resolve().parent / "learned"


def _truth(project):
    nl = core.load_json(Path(project) / "netlist.json")
    man = core.load_json(Path(project) / "manifest.json")
    if not nl or not man:
        return None, None
    by = {}
    for c in nl.get("components", []):
        xy = learn._cell_xy(c.get("cell", ""))
        if xy:
            by.setdefault(xy, set()).add(c.get("type"))
    return by, man["grid"]["cell"]


def _match(t, truth):
    return t in truth or (t == "SW" and bool(truth & {"BTN", "SW"})) or (t == "D" and bool(truth & {"Z", "VS"})) \
        or (t == "U" and bool(truth & {"MCU", "OC", "OPAMP", "LOGIC", "REG"}))


def analyze_sheet(project):
    """(признаки, вырезы, размеры, предсказание kNN, тип в своей клетке?, тип рядом?) для каждого символа."""
    by, cell = _truth(project)
    if by is None:
        return []
    res = analyzer.analyze(project, ocr=False)
    rows = []
    for sy, f, (cr, dims) in zip(res["symbols"], res["_features"], res["_crops"]):
        if sy["kind"] in ("R", "U", "?", "GND", "PORT"):
            continue
        t = autonet.KIND2TYPE.get(sy["kind"], sy["kind"])
        x0, y0, x1, y1 = sy["box"]
        own = {(int(x // cell), int(y // cell)) for x in (x0, x1) for y in (y0, y1)}
        near = {(cx + dx, cy + dy) for cx, cy in own for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
        t_own = set().union(*(by.get(c, set()) for c in own))
        t_near = set().union(*(by.get(c, set()) for c in near))
        rows.append({"f": f, "crop": cr, "dims": dims, "kind": sy["kind"], "uncertain": bool(sy.get("uncertain")),
                     "ok_own": _match(t, t_own), "ok_near": _match(t, t_near)})
    return rows


def mine(rows):
    """Трудные примеры: уверенная ошибка → OTHER; совпадение в своей клетке → подтверждённый тип."""
    X, C, D, Y = [], [], [], []
    for r in rows:
        if not r["ok_near"]:
            X.append(r["f"]), C.append(r["crop"]), D.append(r["dims"]), Y.append("OTHER")
        elif r["ok_own"] and not r["uncertain"]:
            X.append(r["f"]), C.append(r["crop"]), D.append(r["dims"]), Y.append(r["kind"])
    return X, C, D, Y


def save(project_name, X, C, D, Y):
    src = "mine:" + project_name
    learn.save_examples(X, Y, src)
    symclass.save(C, D, Y, src)


def precision(rows):
    sure = [r for r in rows if not r["uncertain"]]
    ok = sum(r["ok_own"] or r["ok_near"] and False for r in sure)
    return len(sure), sum(r["ok_own"] for r in sure)


def _clear_mine():
    d = np.load(learn.LEARNED / "symbols.npz")
    keep = ~np.char.startswith(d["src"].astype(str), "mine:")
    np.savez_compressed(learn.LEARNED / "symbols.npz", X=d["X"][keep], y=d["y"][keep], src=d["src"][keep])
    d = dict(np.load(symclass.DATA))
    keep = ~np.char.startswith(d["src"].astype(str), "mine:")
    np.savez_compressed(symclass.DATA, **{k: v[keep] for k, v in d.items()})


def run(projects):
    for p in projects:
        rows = analyze_sheet(p)
        save(Path(p).name, *mine(rows))
    net, meta = symclass.train()
    symclass.save_model(net, meta)
    analyzer._MODEL.clear()


def check(projects):
    """Честно: учимся на половине листов, меряем на другой (и наоборот)."""
    half = [projects[0::2], projects[1::2]]
    total = {"before": [0, 0], "after": [0, 0]}
    for k in (0, 1):
        train, test = half[k], half[1 - k]
        _clear_mine()
        net, meta = symclass.train()
        symclass.save_model(net, meta)
        analyzer._MODEL.clear()
        for p in test:
            n, ok = precision(analyze_sheet(p))
            total["before"][0] += n
            total["before"][1] += ok
        run(train)
        for p in test:
            n, ok = precision(analyze_sheet(p))
            total["after"][0] += n
            total["after"][1] += ok
    for k, (n, ok) in total.items():
        print(f"{'до' if k == 'before' else 'после'}: уверенных {n}, верных {ok}, точность {ok / max(1, n):.3f}")
    return total


if __name__ == "__main__":
    projs = json.load(open(LEARNED / "projects.json", encoding="utf-8"))
    if sys.argv[1:2] == ["check"]:
        check(projs)
    else:
        run(projs)
        print("готово")
