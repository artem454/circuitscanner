"""Метки символов по обозначениям на самой схеме: R1, C3, VD2, VT1, DA1, VS1… — буква и есть тип.

Сопоставление обозначений символам — только по геометрии, без мнения классификатора (иначе
проверка «модель согласна с обозначением» была бы по кругу): кандидаты — пары «надпись-обозначение
↔ символ» рядом друг с другом; пары разбираются глобально от самых близких, каждая надпись и каждый
символ — не больше одного раза; неоднозначные (вторая по близости пара почти так же хороша)
пропускаются — лучше без метки, чем с чужой.

Подсказка (hint) пишется в пул поколений (learned/pool) рядом с символом; поколение берёт её
меткой, только если с ней согласна хотя бы одна модель и класс прошёл проверку на синтетике.

    python reflabels.py check [N]   — точность подсказок на N синтетических схемах с ответом
    python reflabels.py pool        — проставить подсказки всем схемам пула
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import analyzer
import core

LEARNED = analyzer.LEARNED
CHECK = LEARNED / "reflabels_check.json"
# буквы обозначения → класс классификатора; неоднозначные буквы (D — диод или микросхема, K — катушка
# или контакт, T — транзистор или трансформатор, HL, SA, U, X…) не используются
PREFIX2CLASS = {"R": "R", "RP": "R", "C": "C", "L": "L", "VD": "D", "VT": "Q", "Q": "Q", "DA": "OPAMP",
                "VS": "VS", "ZQ": "XTAL", "FU": "FU", "QF": "QF", "SB": "BTN", "RK": "RT"}
NOT_PARTS = {"GND", "PORT"}
MAX_D = 2.5              # дальше двух с половиной размеров символа надпись к нему не относится
MARGIN = 0.3             # вторая по близости пара должна быть хуже хотя бы на столько (в размерах символа)
MIN_CONF = 0.4           # уверенность OCR в надписи
HINT_PREC, HINT_MIN = 0.8, 15   # класс подсказок годен, если на синтетике верно ≥80 % при ≥15 подсказках


def ref_class(ref):
    import autonet
    import re
    if re.fullmatch(r"DD\d+\.\d{1,2}", ref or ""):
        return "LOGIC"                                  # секция цифровой микросхемы — логический элемент
    return PREFIX2CLASS.get(autonet.prefix(ref or ""))


def match(symbols, texts, stroke=2.0):
    """[(индекс символа, обозначение, класс)] — однозначные пары по геометрии."""
    def size(b):
        return max(b[2] - b[0], b[3] - b[1], 6 * stroke)

    pairs = []
    for ti, t in enumerate(texts):
        cls = ref_class(t.get("ref"))
        if not cls or t.get("conf", 1.0) < MIN_CONF:
            continue
        cx, cy = analyzer.box_center(t["box"])
        for si, s in enumerate(symbols):
            if s["kind"] in NOT_PARTS:
                continue
            c = analyzer.box_dist(s["box"], cx, cy) / size(s["box"])
            if c <= MAX_D:
                pairs.append((c, ti, si))
    best_t, best_s = {}, {}                     # две лучшие стоимости для каждой надписи и символа
    for c, ti, si in pairs:
        best_t.setdefault(ti, []).append(c)
        best_s.setdefault(si, []).append(c)
    second = lambda lst, c: sorted(lst)[1] if len(lst) > 1 else c + 99
    out, used_t, used_s = [], set(), set()
    for c, ti, si in sorted(pairs):
        if ti in used_t or si in used_s:
            continue
        used_t.add(ti)
        used_s.add(si)
        if second(best_t[ti], c) - c < MARGIN or second(best_s[si], c) - c < MARGIN:
            continue                            # спорная пара: соседний символ или надпись почти так же близко
        out.append((si, texts[ti]["ref"], ref_class(texts[ti]["ref"])))
    return out


# ---------- проверка на синтетике ----------

def _check_one(args):
    import autonet
    import learn
    import ocr as ocrmod
    import synth
    seed, params = args
    eng = _check_one.engine = getattr(_check_one, "engine", None) or ocrmod.OCR()
    img, truth = synth.generate(seed, level=1 + seed % 5)
    gray, truth = learn.prep(img, truth)
    res = analyzer.analyze_gray(gray, params)
    autonet.read_texts(res, gray, eng)
    rows = []
    for si, ref, cls in match(res["symbols"], res["texts"], res.get("stroke", 2)):
        b = res["symbols"][si]["box"]
        cx, cy = analyzer.box_center(b)
        hit = [c for c in truth["comps"] if c["box"][0] - 4 <= cx <= c["box"][2] + 4 and c["box"][1] - 4 <= cy <= c["box"][3] + 4]
        want = learn.TYPE_MAP.get(hit[0]["type"], hit[0]["type"]) if len(hit) == 1 else "OTHER"
        rows.append((cls, want, agrees(res["_features"][si], *res["_crops"][si], cls)))
    return rows


_MODELS = {}


def agrees(feat, crop, dims, cls):
    """Согласна ли с подсказкой хотя бы одна модель (kNN по картинке или модель формы) — их первый ответ."""
    import symclass
    if not _MODELS:
        _MODELS["knn"] = analyzer.SymbolModel()
        _MODELS["shape"] = symclass.Model() if symclass.MODEL.exists() else None
    if _MODELS["knn"].ready() and _MODELS["knn"].predict(np.asarray(feat, np.float32))[0] == cls:
        return True
    return _MODELS["shape"] is not None and _MODELS["shape"].predict(crop, dims, trusted_only=False)[0] == cls


def check(n=150):
    """Точность подсказок по классам на синтетике (там ответ известен). Пишет learned/reflabels_check.json."""
    import os
    params = analyzer.load_params()
    cnt, hit, conf, acnt, ahit = {}, {}, {}, {}, {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for rows in pool.map(_check_one, [(70000 + i, params) for i in range(n)], chunksize=2):
            for cls, want, agree in rows:
                cnt[cls] = cnt.get(cls, 0) + 1
                hit[cls] = hit.get(cls, 0) + (cls == want)
                if agree:                                 # то, что реально станет меткой
                    acnt[cls] = acnt.get(cls, 0) + 1
                    ahit[cls] = ahit.get(cls, 0) + (cls == want)
                if cls != want:
                    conf.setdefault(cls, {}).setdefault(want, 0)
                    conf[cls][want] += 1
    # n/prec — подсказки с согласием модели (по ним решается допуск класса); raw — все подсказки
    stats = {k: {"n": acnt.get(k, 0), "prec": round(ahit.get(k, 0) / acnt[k], 3) if acnt.get(k) else 0.0,
                 "raw_n": cnt[k], "raw_prec": round(hit[k] / cnt[k], 3), "ошибки": conf.get(k, {})}
             for k in sorted(cnt)}
    CHECK.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    return stats


def trusted_classes():
    """Классы, подсказкам которых можно верить (по последней проверке на синтетике)."""
    st = core.load_json(CHECK) or {}
    return {k for k, v in st.items() if v["n"] >= HINT_MIN and v["prec"] >= HINT_PREC}


# ---------- пул ----------

def _pool_one(path):
    import autonet
    import generations
    import ocr as ocrmod
    eng = _pool_one.engine = getattr(_pool_one, "engine", None) or ocrmod.OCR()
    d = generations._load(path)
    if not d or not len(d["key"]):
        return path, 0, 0
    project = Path(str(d["project"][0]))
    res = core.load_json(project / "analysis" / "analysis.json")
    if not res or not (project / "clean.png").exists():
        return path, 0, 0
    gray = core.imread(project / "clean.png")[:, :, 0]
    autonet.read_texts(res, gray, eng)
    key_of = {"%d_%d_%d_%d" % tuple(int(round(v / 4)) for v in s["box"]): si for si, s in enumerate(res["symbols"])}
    hint_of = {si: (ref, cls) for si, ref, cls in match(res["symbols"], res["texts"], res.get("stroke", 2))}
    hint = np.array([""] * len(d["key"]), dtype="<U8")
    href = np.array([""] * len(d["key"]), dtype="<U12")
    for i, k in enumerate(d["key"]):
        si = key_of.get(str(k))
        if si in hint_of:
            href[i], hint[i] = hint_of[si]
    d["hint"], d["hint_ref"] = hint, href
    generations._save(path, d)
    return path, len(d["key"]), int((hint != "").sum())


def label_pool(progress=print):
    import os
    import generations
    files = sorted(generations.POOL.glob("*.npz")) if generations.POOL.exists() else []
    total = hinted = 0
    per = {}
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for k, (path, n, h) in enumerate(pool.map(_pool_one, files, chunksize=4), 1):
            total += n
            hinted += h
            if k % 100 == 0:
                progress(f"{k}/{len(files)} схем: символов {total}, с подсказкой {hinted}")
    for path in files:
        d = generations._load(path)
        for c in d.get("hint", []):
            if c:
                per[str(c)] = per.get(str(c), 0) + 1
    progress(f"итог: схем {len(files)}, символов {total}, с подсказкой {hinted}: {dict(sorted(per.items()))}")
    return per


if __name__ == "__main__":
    if sys.argv[1:2] == ["check"]:
        st = check(int(sys.argv[2]) if len(sys.argv) > 2 else 150)
        for k, v in st.items():
            print(f"{k:6} подсказок {v['raw_n']:5} (верно {v['raw_prec']:.3f}); с согласием модели "
                  f"{v['n']:5} (верно {v['prec']:.3f})  ошибки {v['ошибки']}")
        print("годные классы:", sorted(trusted_classes()))
    elif sys.argv[1:2] == ["pool"]:
        label_pool()
    else:
        print(__doc__)
