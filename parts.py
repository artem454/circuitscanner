"""Справочник деталей: встроенная база (parts_db.json) + ваши детали и импорт из даташитов
(learned/parts_user.json). Ваши записи важнее встроенных.

    python parts.py find IRFP460             — найти деталь (по номиналу из схемы)
    python parts.py list [MOSFET|D|REG|…]    — список
    python parts.py add NAME kind=MOSFET vmax=500 imax=20 rds=0.27 qg=210n
    python parts.py import даташит.pdf [NAME] — вытащить параметры из PDF (проверьте результат!)
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = HERE / "learned" / "parts_user.json"

_DB = None


def norm(s):
    return re.sub(r"[\s_\-]", "", str(s or "")).upper().replace("Ё", "Е")


def db():
    global _DB
    if _DB is None:
        base = json.loads((HERE / "parts_db.json").read_text(encoding="utf-8"))
        base.pop("_about", None)
        user = json.loads(USER.read_text(encoding="utf-8")) if USER.exists() else {}
        _DB = {**base, **user}
        for k in user:
            _DB[k]["_user"] = True
    return _DB


def reload():
    global _DB
    _DB = None
    return db()


def lookup(value):
    """Деталь по номиналу из схемы ('IRFP350', 'L7812CV', 'SMBJ5.0A 5V'). Самое длинное совпадение имени.
    Возвращает (имя, параметры) или (None, None). Для семейств (TVS, стабилитроны) подставляет напряжения."""
    v = norm(value)
    if not v:
        return None, None
    best = None
    for name, d in db().items():
        n = norm(name)
        if n and n in v and (best is None or len(n) > len(norm(best))):
            best = name
    if not best:
        return None, None
    d = dict(db()[best])
    if d.get("family"):
        d = _family(best, d, str(value))
    return best, d


def _family(name, d, value):
    m = re.search(re.escape(name).replace(r"\.", r"\.?") + r"\s*-?\s*(\d+(?:[.,]\d+)?)", value, re.I)
    num = float(m.group(1).replace(",", ".")) if m else None
    if d["kind"] == "TVS" and num:
        if d.get("by_vbr"):
            vbr = num
            vwm = round(num / 1.18, 2)
        else:
            vwm = num
            vbr = round(num * 1.11 + 0.9, 2)                 # SMBJ5.0A: 6,4 В; SMBJ24A: ≈27 В
        d.update(vwm=vwm, vbr=vbr, vc=round(vbr * 1.45, 2), vmax=vwm)   # ограничение при Iимп ≈ 1,45·Uпроб
    if d["kind"] == "Z":
        m1 = re.search(r"(\d+)[VВ](\d+)", value)                 # «5V1» = 5,1 В
        m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*[VВ](?!\d)", value)
        m3 = re.search(r"(?:BZX\d\d|BZV\d\d)C?(\d+(?:[.,]\d+)?)$", value.strip(), re.I)
        if m1:
            d["vz"] = float(f"{m1.group(1)}.{m1.group(2)}")
        elif m2:
            d["vz"] = float(m2.group(1).replace(",", "."))
        elif m3:
            d["vz"] = float(m3.group(1).replace(",", "."))
    return d


def save_user(name, params):
    USER.parent.mkdir(exist_ok=True)
    user = json.loads(USER.read_text(encoding="utf-8")) if USER.exists() else {}
    user[name] = params
    USER.write_text(json.dumps(user, ensure_ascii=False, indent=1), encoding="utf-8")
    reload()


# ---------- импорт из даташита ----------

SI = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "m": 1e-3, "": 1.0, "k": 1e3, "M": 1e6}

# (параметр, шаблоны строки, единица) — первая подходящая строка, первое число с нужной единицей
PATTERNS = [
    ("vmax", [r"V\s*\(?BR\)?\s*DSS", r"\bV\s*DSS\b", r"Drain[- ]to[- ]Source Voltage", r"\bV\s*CES\b",
              r"Collector[- ]Emitter Voltage", r"\bV\s*RRM\b", r"Repetitive Peak Reverse Voltage",
              r"\bV\s*CEO\b"], "V"),
    ("imax", [r"Continuous Drain Current", r"\bI\s*D\b.*T\s*C\s*=\s*25", r"Continuous Collector Current",
              r"\bI\s*C\b.*T\s*C\s*=\s*25", r"Average (Rectified|Forward) Current", r"\bI\s*F\s*\(?AV\)?"], "A"),
    ("rds", [r"R\s*DS\s*\(?on\)?", r"Static Drain[- ]to[- ]Source On[- ]Resistance"], "Ω"),
    ("qg", [r"\bQ\s*g\b(?!s|d)", r"Total Gate Charge"], "C"),
    ("ciss", [r"\bC\s*iss\b", r"Input Capacitance"], "F"),
    ("vce", [r"V\s*CE\s*\(?sat\)?", r"Collector[- ]Emitter Saturation Voltage"], "V"),
    ("vf", [r"\bV\s*F\b", r"Forward Voltage"], "V"),
    ("trr", [r"\bt\s*rr\b", r"Reverse Recovery Time"], "s"),
    ("rth_jc", [r"R\s*(θ|th|Θ)\s*\(?J-?C\)?", r"Junction[- ]to[- ]Case"], "K/W"),
    ("rth_ja", [r"R\s*(θ|th|Θ)\s*\(?J-?A\)?", r"Junction[- ]to[- ]Ambient"], "K/W"),
    ("vin_max", [r"Input Voltage", r"\bV\s*IN\b.*max"], "V"),
    ("vref", [r"Reference Voltage", r"\bV\s*REF\b", r"Feedback Voltage", r"\bV\s*FB\b"], "V"),
]
UNIT_RE = {
    "V": r"(-?\d+(?:[.,]\d+)?)\s*(k|m)?\s*V\b",
    "A": r"(-?\d+(?:[.,]\d+)?)\s*(m)?\s*A\b",
    "Ω": r"(\d+(?:[.,]\d+)?)\s*(m|k)?\s*(?:Ω|Ohm|ohm|W\b)",
    "C": r"(\d+(?:[.,]\d+)?)\s*(n|µ|μ|u|p)\s*C\b",
    "F": r"(\d+(?:[.,]\d+)?)\s*(p|n|µ|μ|u)\s*F\b",
    "s": r"(\d+(?:[.,]\d+)?)\s*(n|µ|μ|u|m)\s*s\b",
    "K/W": r"(\d+(?:[.,]\d+)?)\s*(?:°C/W|℃/W|K/W|C/W)",
}
# если в строке нет единицы, а таблица даёт её в отдельном столбце — берём число из строки,
# умножая на типичный для этого параметра множитель
BARE_UNIT = {"rds": 1.0, "qg": 1e-9, "ciss": 1e-12, "trr": 1e-9}


def kind_from_text(text):
    t = text[:4000].upper()
    if "IGBT" in t or "INSULATED GATE" in t:
        return "IGBT"
    if "MOSFET" in t or "FIELD EFFECT" in t or "HEXFET" in t:
        return "MOSFET"
    if "REGULATOR" in t or "LDO" in t:
        return "REG"
    if "RECTIFIER" in t or "DIODE" in t or "SCHOTTKY" in t:
        return "D"
    if "TRANSISTOR" in t:
        return "BJT"
    return "X"


def parse_text(text):
    """Параметры из текста даташита. Возвращает (параметры, откуда взято)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    found, where = {}, {}
    for key, pats, unit in PATTERNS:
        for pat in pats:
            for line in lines:
                if not re.search(pat, line, re.I):
                    continue
                m = re.search(UNIT_RE[unit], line)
                val = None
                if m:
                    pre = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
                    val = float(m.group(1).replace(",", ".")) * SI.get(pre or "", 1.0)
                elif key in BARE_UNIT:
                    nums = re.findall(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w.])", re.sub(pat, " ", line, flags=re.I))
                    if nums:
                        val = float(nums[0].replace(",", ".")) * BARE_UNIT[key]
                if val is not None and val != 0:
                    found[key] = abs(val) if key != "vref" else val
                    where[key] = line[:120]
                    break
            if key in found:
                break
    return found, where


def import_pdf(path, name=None):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("нужен pypdf: python -m pip install pypdf")
    reader = PdfReader(str(path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:8])
    params, where = parse_text(text)
    params["kind"] = kind_from_text(text)
    if not name:
        m = re.search(r"\b([A-Z]{1,6}\d[A-Z0-9\-]{2,})\b", text[:2000])
        name = m.group(1) if m else Path(path).stem
    params.update(source=Path(path).name, verified=False)
    save_user(name, params)
    return name, params, where


def parse_add(args):
    out = {}
    for a in args:
        k, _, v = a.partition("=")
        m = re.fullmatch(r"(-?\d+(?:[.,]\d+)?)\s*([pnuµmkM]?)", v.strip())
        out[k] = float(m.group(1).replace(",", ".")) * SI.get(m.group(2), 1.0) if m else v
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        print(__doc__)
        sys.exit(1)
    if a[0] == "find":
        n, d = lookup(" ".join(a[1:]))
        print(n, json.dumps(d, ensure_ascii=False) if d else "— нет в базе")
    elif a[0] == "list":
        for n, d in sorted(db().items()):
            if len(a) < 2 or d.get("kind") == a[1].upper():
                print(f"{n:16} {d.get('kind', ''):8} " + ", ".join(f"{k}={v}" for k, v in d.items()
                                                              if k not in ("kind", "pins", "note", "_user")))
        print(f"всего: {len(db())}")
    elif a[0] == "add":
        save_user(a[1], parse_add(a[2:]))
        print("добавлено:", a[1], db()[a[1]])
    elif a[0] == "import":
        n, p, w = import_pdf(a[1], a[2] if len(a) > 2 else None)
        print(f"{n}: " + json.dumps(p, ensure_ascii=False))
        for k, line in w.items():
            print(f"  {k:7} ← {line}")
        print("Проверьте значения по даташиту: таблицы в PDF распознаются не идеально.")
