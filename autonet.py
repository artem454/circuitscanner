"""Самостоятельная сборка netlist.json — без ИИ.

Берёт разбор анализатора (провода, точки, символы, надписи, цепи) и:
  1. читает надписи своим OCR (ocr.py) или Tesseract, если он установлен;
  2. сопоставляет символам обозначения (R12, VT3…) и номиналы (2,2k, 100n) —
     по расстоянию и по совместимости буквы обозначения с типом символа;
  3. уточняет тип детали: класс символа от классификатора + буква обозначения;
  4. именует цепи по меткам (+5V, GND…) и знакам земли; одинаковые метки — одна цепь;
  5. называет выводы (1/2 у двухвыводных, по геометрии у остальных);
  6. пишет netlist.json в формате программы — с пометками уверенности и вопросами.

    python autonet.py папка_проекта         — собрать netlist.json (если он уже есть — в netlist_autonet.json)
"""
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

import analyzer
import core
import ocr as ocrmod

# класс символа анализатора → тип netlist
KIND2TYPE = {"R": "R", "C": "C", "L": "L", "D": "D", "Z": "Z", "Q": "Q", "M": "M", "U": "U", "J": "J",
             "OPAMP": "U", "MCU": "MCU", "SENSOR": "SENSOR", "RT": "RT", "TC": "TC", "XTAL": "XTAL", "LED": "LED",
             "TERM": "TERM", "ENC": "ENC", "TACHO": "TACHO", "HALL": "HALL", "SW_NO": "SW", "SW_NC": "SW",
             "BTN": "BTN", "COIL": "K", "LAMP": "LAMP", "THERMAL": "SW", "FU": "F", "QF": "QF", "MOTOR": "MOTOR"}
# буквы обозначения → тип
PREFIX2TYPE = {"R": "R", "RP": "R", "C": "C", "L": "L", "VD": "D", "D": "D", "VT": "Q", "Q": "Q", "T": "Q",
               "DA": "U", "DD": "U", "U": "U", "IC": "U", "K": "K", "KM": "K", "KK": "SW", "SB": "BTN", "SA": "SW",
               "HL": "LAMP", "FU": "F", "F": "F", "QF": "QF", "XT": "TERM", "XS": "J", "XP": "J", "X": "J", "J": "J",
               "BQ": "ENC", "BH": "HALL", "KT": "K", "BR": "TACHO", "BP": "SENSOR", "BK": "TC", "RK": "RT", "ZQ": "XTAL", "M": "MOTOR",
               "VS": "VS", "TR": "T", "Z": "Z"}
# совместимость типов (класс символа и буква обозначения могут разойтись законно)
COMPAT = {("Q", "M"), ("M", "Q"), ("D", "Z"), ("Z", "D"), ("D", "LED"), ("LED", "D"), ("LAMP", "LED"),
          ("U", "MCU"), ("MCU", "U"), ("SW", "BTN"), ("BTN", "SW"), ("R", "RT"), ("RT", "R"), ("J", "TERM"),
          ("TERM", "J"), ("Q", "IGBT"), ("K", "SW")}
NOT_PARTS = {"GND", "PORT"}
GENERIC = {"U", "X", None}         # «прямоугольник» и неопознанное — тип берём из буквы обозначения


def bare_number(v):
    return v is not None and v.replace(".", "").isdigit()


def unit_fits(v, typ):
    """Подходит ли номинал по единицам к типу: пико/нано/микро — ёмкость (у дросселя — мкГн),
    кило/мега/Ом — сопротивление. Голое число годится всем."""
    if not v or typ is None:
        return True
    cap = bool(re.search(r"[pnμ]|F$", v))
    res = bool(re.search(r"[kMΩ]", v))
    if typ in ("R", "RT"):
        return not cap
    if typ == "C":
        return not res
    if typ == "L":
        return not res and "F" not in v
    return True


def prefix(ref):
    m = re.match(r"[A-Z]+", ref or "")
    return m.group(0) if m else ""


def read_texts(res, gray, engine=None):
    """Текст каждой надписи: от Tesseract (если был) или своим OCR. Координаты res — оригинал."""
    engine = engine or ocrmod.OCR()
    for t in res["texts"]:
        if not t.get("text"):
            w, c = engine.word(gray, t["box"], t.get("vertical", False))
            t["text"], t["conf"] = w, round(c, 2)
        t["ref"] = ocrmod.fix_ref(t["text"])
        t["value"] = ocrmod.fix_value(t["text"]) if not t["ref"] else None
        t["net"] = ocrmod.fix_net(t["text"])
    return res


def read_inside(res, gray, engine=None):
    """Номинал внутри корпуса (резистор-прямоугольник с «270» внутри): читаем внутренность
    рамки, отрезав её края. Для символов-прямоугольников без найденного номинала рядом."""
    engine = engine or ocrmod.OCR()
    sw = max(1, int(round(res.get("stroke", 1.5))))
    for s in res["symbols"]:
        if s["kind"] not in ("R", "U", "?"):
            continue
        x0, y0, x1, y1 = s["box"]
        w, h = x1 - x0, y1 - y0
        if min(w, h) < 6 or max(w, h) > 12 * min(w, h) or max(w, h) > 300:
            continue
        m = 2 * sw + 1
        inner = gray[int(y0 + m):int(y1 - m), int(x0 + m):int(x1 - m)]
        if inner.size == 0 or inner.min() > 128:
            continue
        txt, conf = engine.word(gray, (x0 + m, y0 + m, x1 - m, y1 - m), vertical=h > 1.6 * w)
        v = ocrmod.fix_value(txt)
        if v and conf > 0.3:
            s["inside_value"] = v
    return res


def _center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def build(res, title="Автоматический разбор (без ИИ)", cell=None):
    """res — результат analyzer.analyze_gray/analyze с прочитанными надписями. Возвращает netlist."""
    syms = {s["id"]: s for s in res["symbols"] if s["kind"] not in NOT_PARTS}
    texts = res["texts"]
    stroke = res.get("stroke", 2)

    def size(b):
        return max(b[2] - b[0], b[3] - b[1], 6 * stroke)

    # --- обозначения: жадно по расстоянию с учётом совместимости ---
    pairs = []
    for t in texts:
        if not t["ref"]:
            continue
        tc = _center(t["box"])
        pt = PREFIX2TYPE.get(prefix(t["ref"]))
        for sid, s in syms.items():
            d = analyzer.box_dist(s["box"], *tc)
            if d > 2.5 * size(s["box"]):
                continue
            st = KIND2TYPE.get(s["kind"])
            cost = d / size(s["box"])
            if st and pt and st != pt and (st, pt) not in COMPAT and st not in GENERIC:
                cost += 1.5                                  # тип не сходится — можно, но хуже
            pairs.append((cost, t["id"], sid))
    used_t, ref_of = set(), {}
    for cost, tid, sid in sorted(pairs):
        if tid in used_t or sid in ref_of:
            continue
        used_t.add(tid)
        ref_of[sid] = next(t for t in texts if t["id"] == tid)["ref"]
    # --- номиналы ---
    vpairs = []
    for t in texts:
        if not t["value"] or t["id"] in used_t:
            continue
        tc = _center(t["box"])
        for sid, s in syms.items():
            st = KIND2TYPE.get(s["kind"])
            pt = PREFIX2TYPE.get(prefix(ref_of.get(sid, "")))
            if st in GENERIC or not st:
                st = pt
            if st not in ("R", "C", "L", "RT", "Z", "F", None):
                continue
            if not unit_fits(t["value"], st):
                continue                                     # «10n» у резистора, «2.7k» у конденсатора
            d = analyzer.box_dist(s["box"], *tc)
            if bare_number(t["value"]) and d > 0.35 * size(s["box"]):
                continue                                     # «7», «10» в стороне — скорее номер вывода
            if d <= 3 * size(s["box"]):
                vpairs.append((d / size(s["box"]), t["id"], sid))
    val_of = {sid: s["inside_value"] for sid, s in syms.items() if s.get("inside_value")}
    for cost, tid, sid in sorted(vpairs):
        if tid in used_t or sid in val_of:
            continue
        used_t.add(tid)
        val_of[sid] = next(t for t in texts if t["id"] == tid)["value"]

    # --- цепи: имя по метке, земля по знаку, одинаковые метки — одна цепь ---
    gnd_syms = {s["id"] for s in res["symbols"] if s["kind"] == "GND"}
    tmap = {t["id"]: t for t in texts}
    names, parent = {}, {}

    def find(a):
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    for n in res["nets"]:
        nm = None
        if any(p["symbol"] in gnd_syms for p in n["pins"]):
            nm = "GND"
        for lid in n["labels"]:
            if tmap.get(lid, {}).get("net"):
                nm = tmap[lid]["net"]
        if nm in ("0V",):
            nm = "GND"
        names[n["id"]] = nm
    by_name = {}
    for nid, nm in names.items():
        if nm:
            if nm in by_name:
                parent[find(nid)] = find(by_name[nm])
            else:
                by_name[nm] = nid
    final = {}
    k = 0
    for n in res["nets"]:
        root = find(n["id"])
        if root not in final:
            nm = names.get(root) or next((names[x["id"]] for x in res["nets"] if find(x["id"]) == root and names[x["id"]]), None)
            if not nm:
                k += 1
                nm = f"N{k}"
            final[root] = nm

    # --- компоненты и выводы ---
    pins_of = {}
    for n in res["nets"]:
        for p in n["pins"]:
            pins_of.setdefault(p["symbol"], []).append((p["x"], p["y"], final[find(n["id"])]))
    comps, counter, questions = [], {}, []
    for sid, s in syms.items():
        pins = pins_of.get(sid, [])
        if not pins:
            continue
        st = KIND2TYPE.get(s["kind"])
        ref = ref_of.get(sid)
        pt = PREFIX2TYPE.get(prefix(ref)) if ref else None
        typ = st or pt or "X"
        low = False
        if pt and (st in GENERIC or not st):
            typ = pt
        elif st and pt and st != pt and (st, pt) not in COMPAT:
            low = True
        if not ref:
            letter = {"R": "R", "C": "C", "L": "L", "D": "VD", "Z": "VD", "Q": "VT", "M": "VT", "U": "DA",
                      "MCU": "DD", "K": "K", "SW": "SA", "BTN": "SB", "LAMP": "HL", "F": "FU", "QF": "QF",
                      "J": "X", "TERM": "XT"}.get(typ, "X")
            counter[letter] = counter.get(letter, 0) + 1
            ref = f"{letter}?{counter[letter]}"
            low = True
        # выводы: по порядку слева направо / сверху вниз
        b = s["box"]
        horiz = (b[2] - b[0]) >= (b[3] - b[1])
        pins.sort(key=lambda p: (p[0], p[1]) if horiz else (p[1], p[0]))
        if len(pins) == 2 and typ in ("D", "Z", "LED"):
            pin_names = ["1", "2"]                           # анод/катод по картинке не определить без формы
            low = True
        else:
            pin_names = [str(i + 1) for i in range(len(pins))]
        cx, cy = _center(b)
        c = {"ref": ref, "type": typ, "value": val_of.get(sid, ""), "pins": {pn: p[2] for pn, p in zip(pin_names, pins)},
             "box": [round(v, 1) for v in b]}
        if cell:
            c["cell"] = f"{core.col_name(int(cx // cell))}{int(cy // cell) + 1}"
        if low or typ == "X":
            c["confidence"] = "low"
        comps.append(c)
    # одинаковые обозначения (OCR ошибся или схема) — второму добавить суффикс
    seen = {}
    for c in comps:
        if c["ref"] in seen:
            seen[c["ref"]] += 1
            c["ref"] = f"{c['ref']}_{seen[c['ref']]}"
            c["confidence"] = "low"
        else:
            seen[c["ref"]] = 1
    nets_meta = {}
    for n in res["nets"]:
        nm = final[find(n["id"])]
        if n["dangling"] or n["labels"]:
            nets_meta.setdefault(nm, {})["port"] = True
    unk = [c for c in comps if c["type"] == "X"]
    if unk:
        questions.append({"cell": unk[0].get("cell", ""), "text": f"Не опознано символов: {len(unk)} (тип X)."})
    noref = [c for c in comps if "?" in c["ref"]]
    if noref:
        questions.append({"cell": noref[0].get("cell", ""),
                          "text": f"Без прочитанного обозначения: {len(noref)} деталей (имена вида R?3)."})
    return {"title": title, "generated": "autonet", "components": comps, "nets": nets_meta, "questions": questions}


def run(project, progress=None, overwrite=False):
    """Собрать netlist. Если проверенный netlist.json уже есть и overwrite=False —
    результат пишется в netlist_autonet.json, чтобы его не затереть."""
    project = Path(project)
    manifest = core.load_json(project / "manifest.json")
    gray = core.imread(project / "clean.png")[:, :, 0]
    res = analyzer.analyze(project, ocr=True, progress=progress)
    read_texts(res, gray)
    read_inside(res, gray)
    nl = build(res, cell=manifest["grid"]["cell"])
    target = project / "netlist.json"
    if target.exists() and not overwrite:
        target = project / "netlist_autonet.json"
    elif target.exists():
        shutil.copyfile(target, project / "netlist_prev.json")
    target.write_text(json.dumps(nl, ensure_ascii=False, indent=1), encoding="utf-8")
    nl["_file"] = target.name
    return nl


# ---------- оценка на схеме с ответом ----------

def score(nl, truth):
    """Тип, обозначение, номинал — по совпадению рамок деталей; цепи — learn.score."""
    T = [c for c in truth["comps"] if c["type"] not in NOT_PARTS]
    used = set()
    typ_ok = ref_ok = ref_n = val_ok = val_n = found = 0
    for c in T:
        cx, cy = _center(c["box"])
        best = None
        for i, p in enumerate(nl["components"]):
            if i in used:
                continue
            b = p["box"]
            if b[0] - 6 <= cx <= b[2] + 6 and b[1] - 6 <= cy <= b[3] + 6:
                best = i
                break
        if best is None:
            continue
        used.add(best)
        found += 1
        p = nl["components"][best]
        want = KIND2TYPE.get(c["type"], c["type"]) if c["type"] not in ("SW_NO", "SW_NC", "THERMAL") else "SW"
        want = {"OPAMP": "U", "COIL": "K", "FU": "F"}.get(c["type"], want)
        typ_ok += p["type"] == want or (p["type"], want) in COMPAT
        if c.get("ref"):
            ref_n += 1
            ref_ok += p["ref"] == c["ref"]
        if c.get("value"):
            val_n += 1
            val_ok += p["value"] == ocrmod.fix_value(c["value"])
    n = max(1, len(T))
    return {"found": found / n, "type": typ_ok / found if found else None,
            "ref": ref_ok / ref_n if ref_n else None, "value": val_ok / val_n if val_n else None}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    nl = run(sys.argv[1], progress=lambda m: print("…", m))
    n_ref = sum(1 for c in nl["components"] if "?" not in c["ref"])
    print(f"компонентов {len(nl['components'])}, с прочитанным обозначением {n_ref}, "
          f"с номиналом {sum(1 for c in nl['components'] if c['value'])}, цепей {len({n for c in nl['components'] for n in c['pins'].values()})}")
