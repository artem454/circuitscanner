"""Моделирование неисправностей и поиск поломки по измерениям.

Для каждой детали схемы перебираются типичные поломки (обрыв, КЗ, уход номинала,
пробой ключа, обрыв датчика…). Для каждой физический решатель (physics.Physics)
заново считает напряжения узлов. Получается «словарь неисправностей»:
  • какие симптомы даёт каждая поломка — им обучается диагностика (diagnose.py);
  • по вашим измерениям находится поломка, которая им соответствует лучше всего
    (сравнение с допуском, вероятность ∝ априорная частота · exp(−½·Σ((Uмод−Uизм)/σ)²));
  • подсказывается, где мерить дальше: точка, где у главных кандидатов самые разные напряжения.

    python faults.py папка_проекта                        — словарь неисправностей (faults.json)
    python faults.py папка_проекта ADC0N=0 +24V=24 AI1=0.02 — поиск по измерениям
"""
import copy
import json
import os
import math
import sys
from pathlib import Path

import core
import diagnose as dg
import physics

# (вид, априорная частота, описание)
FAULTS = {
    "R": [("open", 0.6, "обрыв"), ("short", 0.05, "замыкание"), ("drift", 0.3, "сопротивление выросло в 3 раза")],
    "RT": [("open", 0.6, "обрыв"), ("short", 0.3, "замыкание")],
    "C": [("short", 0.5, "пробой (КЗ)"), ("leak", 0.3, "утечка ≈ 1 кОм")],
    "D": [("short", 0.8, "пробой"), ("open", 0.4, "обрыв")],
    "Z": [("short", 0.6, "пробой"), ("open", 0.3, "обрыв")],
    "LED": [("open", 0.5, "обрыв"), ("short", 0.2, "замыкание")],
    "L": [("open", 0.3, "обрыв")], "F": [("open", 0.8, "сгорел")],
    "M": [("short_ds", 1.0, "пробой сток-исток"), ("short_gs", 0.4, "пробой затвор-исток")],
    "IGBT": [("short_ce", 1.0, "пробой коллектор-эмиттер"), ("short_ge", 0.4, "пробой затвор-эмиттер")],
    "Q": [("short_ce", 0.6, "пробой коллектор-эмиттер"), ("short_be", 0.4, "пробой база-эмиттер")],
    "SENSOR": [("open", 1.0, "обрыв петли (0 мА)"), ("stuck", 0.4, "зашкал (22 мА)")],
}
SHORT_R, LEAK_R = 0.05, 1e3


def _apply(nl, ref, kind):
    """Копия netlist с поломкой."""
    nl = copy.deepcopy(nl)
    comps = nl["components"]
    c = next(x for x in comps if x.get("ref") == ref)
    p = c.get("pins") or {}

    def as_r(a, b, r):
        return {"ref": f"{ref}~", "type": "R", "value": f"{r:g}", "cell": c.get("cell", ""), "pins": {"1": a, "2": b}}

    if kind == "open":
        if c.get("type") == "SENSOR":
            c["value"] = "отключён"
        else:
            comps.remove(c)
    elif kind == "short":
        nets = list(p.values())
        if c.get("type") in ("D", "Z", "LED"):
            nets = [p.get("A"), p.get("K")]
        comps.remove(c)
        comps.append(as_r(nets[0], nets[1], SHORT_R))
    elif kind == "drift":
        v, _ = dg.parse_value(c.get("value"))
        c["value"] = f"{v * 3:g}"
    elif kind == "leak":
        nets = [p.get("+"), p.get("-")] if "+" in p else list(p.values())
        comps.append(as_r(nets[0], nets[1], LEAK_R))
    elif kind == "stuck":
        c["value"] = "4-20mA зашкал"
        c["_current"] = 0.022
    else:                                        # short_ds / short_ce / short_gs …
        a, b = {"short_ds": ("D", "S"), "short_gs": ("G", "S"), "short_ce": ("C", "E"),
                "short_ge": ("G", "E"), "short_be": ("B", "E")}[kind]
        if a in p and b in p:
            comps.append(as_r(p[a], p[b], SHORT_R))
    return nl


def _powers(ph):
    out = {}
    for ref, a, b, g in getattr(ph, "_elems", []):
        if a in ph.U and b in ph.U:
            out[ref] = (ph.U[a] - ph.U[b]) ** 2 * g
    return out


def _symptoms(base, ph, circ, nominal_p):
    """Симптомы, которые даёт поломка, — по тому, как изменились напряжения и мощности."""
    s = set()
    adc = circ.mcu_pins("adc")
    vcc = circ.mcu_pins("vcc")
    def reach(types):
        """Цепи датчиков обратной связи и цепи за одним резистором от них (делитель, подтяжка)."""
        nets = {n for r, c in circ.comps.items() if c.get("type") in types for n in circ.pins(r).values()
                if n.upper() not in core.GROUND_NAMES}
        more = {x for n in nets for r in circ.neighbors(n) if circ.comps[r].get("type") == "R"
                for x in circ.pins(r).values() if x.upper() not in core.GROUND_NAMES and x not in vcc}
        return nets | more

    fb_nets = reach(("TACHO",))
    enc = reach(("ENC", "HALL"))
    ports = {n for n, m in circ.meta.items() if m.get("port")}
    for n, u0 in base.items():
        if n.upper() in core.GROUND_NAMES:
            continue
        u1 = ph.U.get(n)
        if u1 is None:                          # узел «повис» — ни к чему не подключён
            if n in adc:
                s |= {"wrong_reading", "noisy"}
            if n in vcc:
                s |= {"mcu_hang", "unstable"}
            if n in enc:
                s.add("position_error")
            elif n in fb_nets:
                s.add("overspeed")
            if n in ports:
                s.add("no_output")
            continue
        d = abs(u1 - u0)
        big = d > max(0.1 * abs(u0), 0.2)
        if not big:
            continue
        if n in adc:
            s.add("wrong_reading")
            if abs(u1) < 0.1 or u1 > 3.6:
                s.add("no_signal")
        if n in vcc:
            s |= {"mcu_hang", "unstable"}
        if n in fb_nets:
            s.add("overspeed" if abs(u1) < abs(u0) else "speed_hunt")
        if n in enc:
            s.add("position_error")
        if n in ports and n not in adc:
            s.add("wrong_reading" if circ.sensor_nets() & {n} else "no_output")
    p1 = _powers(ph)
    for ref, p in p1.items():
        base_p = nominal_p.get(ref, 0)
        rated = 0.25
        if ref in circ.comps:
            rated, _ = physics.power_rating(circ.comps[ref])
        if p > 50:
            s |= {"fuse", "smoke"}
        elif p > rated and p > 2 * base_p:
            s |= {"overheat", "smoke"}
    return s


def simulate(project=None, nl=None, settings=None):
    """Словарь неисправностей: [{ref, kind, desc, prior, U, symptoms}]."""
    if project is not None:
        project = Path(project)
        nl = nl or core.load_json(project / "netlist.json")
        settings = settings if settings is not None else (core.load_json(project / "physics.json") or {})
    base_ph = physics.Physics(nl, settings)
    base_ph.solve_dc()
    base = dict(base_ph.U)
    nominal_p = _powers(base_ph)
    circ = base_ph.c
    out = [{"ref": "", "kind": "ok", "desc": "исправно", "prior": 2.0, "U": base, "symptoms": []}]
    for c in nl.get("components", []):
        for kind, prior, desc in FAULTS.get(c.get("type"), []):
            if kind == "drift" and not dg.parse_value(c.get("value"))[0]:
                continue
            try:
                ph = physics.Physics(_apply(nl, c["ref"], kind), settings)
                ph.solve_dc()
            except Exception:
                continue
            syms = _symptoms(base, ph, circ, nominal_p)
            out.append({"ref": c["ref"], "kind": kind, "desc": desc, "prior": prior,
                        "U": {n: round(v, 4) for n, v in ph.U.items()}, "symptoms": sorted(syms)})
    if project is not None:
        (project / "faults.json").write_text(json.dumps(
            [{k: v for k, v in f.items() if k != "U"} for f in out], ensure_ascii=False, indent=1), encoding="utf-8")
        learn_stats(str(project.resolve()), out, circ)
    return out


# ---------- обучение диагностики на моделировании ----------

STATS = Path(__file__).resolve().parent / "learned" / "sim_stats.json"


def learn_stats(key, faults, circ):
    """Счётчики «тип|роль → симптом» по всем смоделированным схемам. Проект пересчитывается
    целиком, поэтому повторное моделирование не задваивает статистику."""
    stats = core.load_json(STATS, {}) or {}
    cnt = {}
    for f in faults:
        if not f["ref"] or f["ref"] not in circ.comps:
            continue
        t = circ.comps[f["ref"]].get("type", "X")
        for role in circ.roles.get(f["ref"], ["signal"]):
            k = f"{t}|{role}"
            cnt[k + "|*"] = cnt.get(k + "|*", 0) + 1
            for s in f["symptoms"]:
                cnt[f"{k}|{s}"] = cnt.get(f"{k}|{s}", 0) + 1
    stats[key] = cnt
    STATS.parent.mkdir(exist_ok=True)
    tmp = STATS.with_suffix(f".{os.getpid()}.tmp")     # читатель никогда не увидит пустой файл
    tmp.write_text(json.dumps(stats, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATS)


def stats_factor(typ, roles, symptoms):
    """Насколько поломки деталей такого типа и роли (по всем схемам) дают эти симптомы:
    множитель 0,7…1,8 относительно средней детали."""
    stats = core.load_json(STATS, {}) or {}
    if not stats or not symptoms:
        return 1.0, 0
    tot = {}
    for cnt in stats.values():
        for k, v in cnt.items():
            tot[k] = tot.get(k, 0) + v
    all_n = sum(v for k, v in tot.items() if k.endswith("|*")) or 1
    all_hit = sum(v for k, v in tot.items() if k.split("|")[-1] in symptoms) or 0
    base = (all_hit + 0.5) / (all_n + 1)
    best, n_best = 0.0, 0
    for role in roles:
        k = f"{typ}|{role}"
        n = tot.get(k + "|*", 0)
        if not n:
            continue
        hit = sum(tot.get(f"{k}|{s}", 0) for s in symptoms)
        best = max(best, (hit + 0.5) / (n + 1))
        n_best = max(n_best, n)
    if not n_best:
        return 1.0, 0
    return max(0.7, min(1.8, best / base)), n_best


def by_measurements(faults, meas, repairs_factor=None):
    """Вероятности поломок по измерениям {цепь: В}."""
    rows = []
    for f in faults:
        chi, used, detail = 0.0, 0, []
        for n, um in meas.items():
            us = f["U"].get(n)
            if us is None:
                continue
            sig = max(0.05 * abs(um), 0.1)                     # допуск мультиметра и номиналов
            chi += ((us - um) / sig) ** 2
            used += 1
            detail.append((n, us, um))
        prior = f["prior"] * (repairs_factor(f) if repairs_factor else 1.0)
        rows.append(dict(f, chi=chi, used=used, detail=detail, w=prior * math.exp(-0.5 * min(chi, 700))))
    tot = sum(r["w"] for r in rows) or 1.0
    for r in rows:
        r["p"] = r["w"] / tot
    rows.sort(key=lambda r: -r["p"])
    return rows


def next_test_points(ranked, measured, fixed, k=8, n=3):
    """Где мерить дальше: цепь с наибольшим разбросом напряжений у главных кандидатов."""
    top = [f for f in ranked[:k] if f["p"] > 0.01]
    if len(top) < 2:
        return []
    nets = set().union(*(f["U"].keys() for f in top)) - set(measured) - set(fixed)
    scored = []
    for net in nets:
        vals = [(f["U"].get(net), f["p"]) for f in top if f["U"].get(net) is not None]
        if len(vals) < 2:
            continue
        wsum = sum(p for _, p in vals) or 1e-9
        mean = sum(v * p for v, p in vals) / wsum
        var = sum(p * (v - mean) ** 2 for v, p in vals) / wsum
        spread = max(v for v, _ in vals) - min(v for v, _ in vals)
        if spread > 0.2:
            scored.append((var, net, vals))
    scored.sort(key=lambda x: -x[0])
    return [(net, {f"{f['ref']} {f['desc']}".strip(): round(f["U"].get(net), 2) for f in top
                   if f["U"].get(net) is not None}) for _, net, _ in scored[:n]]


def symptom_index(faults):
    """ref → множество симптомов, которые даёт хотя бы одна его поломка (для diagnose.py)."""
    idx = {}
    for f in faults:
        if f["ref"]:
            idx.setdefault(f["ref"], set()).update(f["symptoms"])
    return idx


def report(ranked, tests, meas):
    lines = ["Поиск поломки по измерениям:", ""]
    for r in ranked[:8]:
        who = f"{r['ref']} — {r['desc']}" if r["ref"] else "схема исправна"
        mism = ", ".join(f"{n}: модель {us:.2f} / изм. {um:.2f}" for n, us, um in r["detail"][:4])
        lines.append(f"{r['p'] * 100:5.1f}%  {who}   ({mism})")
    if tests:
        lines += ["", "Чтобы различить кандидатов, измерьте:"]
        for net, vals in tests:
            lines.append(f"  {net}: " + "; ".join(f"{k} → {v} В" for k, v in list(vals.items())[:5]))
    elif len([r for r in ranked if r["p"] > 0.05]) > 1:
        lines += ["", "Напряжениями кандидатов не различить — проверьте их омметром на выпаянных деталях: "
                  + ", ".join(r["ref"] for r in ranked if r["p"] > 0.05 and r["ref"])]
    floating = [r["ref"] for r in ranked[:8] if r["ref"] and not r["used"] and meas]
    if floating:
        lines.append("Узел после поломки «висит» (показание случайное): " + ", ".join(floating))
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    proj = Path(sys.argv[1])
    fl = simulate(proj)
    meas = {}
    for a in sys.argv[2:]:
        k, _, v = a.partition("=")
        meas[k] = float(v.replace(",", "."))
    if not meas:
        n_sym = sum(1 for f in fl if f["symptoms"])
        print(f"поломок смоделировано: {len(fl) - 1}, из них дают заметные симптомы: {n_sym}")
        for f in fl[1:]:
            if f["symptoms"]:
                print(f"  {f['ref']:8} {f['desc']:28} → {', '.join(f['symptoms'])}")
    else:
        ranked = by_measurements(fl, meas)
        base = physics.Physics(core.load_json(proj / "netlist.json"), core.load_json(proj / "physics.json") or {})
        base.solve_dc()
        print(report(ranked, next_test_points(ranked, meas, base.fixed), meas))
