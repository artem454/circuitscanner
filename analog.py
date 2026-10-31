"""Аналоговые и логические узлы: ОУ, компараторы, триггеры Шмитта, дешифраторы и шифраторы.

Что распознаётся по netlist и справочнику (parts_db.json):
  • каскад на ОУ: повторитель, неинвертирующий усилитель K = 1 + Rос/Rз;
  • триггер Шмитта на ОУ/компараторе — резистор с выхода на неинвертирующий вход (положительная ОС):
      инвертирующий: Uпор = (Uоп·Gоп + Uвых·Gос)/(Gоп + Gос) при Uвых = Uвых.макс и Uвых.мин;
      неинвертирующий: Uвкл = Uоп + (Uоп − Uвых.мин)·Rвх/Rос, Uвыкл = Uоп − (Uвых.макс − Uоп)·Rвх/Rос;
  • компаратор без гистерезиса (на медленном или зашумлённом сигнале выход «дребезжит»);
  • логический триггер Шмитта (74HC14, CD40106…): пороги по даташиту при данном питании,
    RC-генератор (резистор с выхода на вход, конденсатор со входа на общий): f ≈ 1/(k·R·C);
  • дешифратор/шифратор: уровни разрешающих входов (на «вечно запрещённом» все выходы неактивны),
    какая комбинация адреса включает какой выход.
"""
import math

import core
import diagnose as dg
import parts

# цоколёвка, если в справочнике нет: (выход, вход −, вход +) по секциям
DUAL8 = {"A": (1, 2, 3), "B": (7, 6, 5)}
QUAD14 = {"A": (1, 2, 3), "B": (7, 6, 5), "C": (8, 9, 10), "D": (14, 13, 12)}
SINGLE = {"A": (6, 2, 3)}
LM311 = {"A": (7, 3, 2)}                                 # у LM311: 2 — вход +, 3 — вход −
HEX_INV = [(1, 2), (3, 4), (5, 6), (9, 8), (11, 10), (13, 12)]       # 74HC14/CD40106: вход → выход


def fmt(x, unit=""):
    from physics import fmt as f
    return f(x, unit)


def _sections(comp, d):
    pins = comp.get("pins") or {}
    named = {str(k).upper(): v for k, v in pins.items()}
    if {"OUT", "IN-", "IN+"} <= set(named):
        return {"A": (named["OUT"], named["IN-"], named["IN+"])}
    num = {int(k): v for k, v in pins.items() if str(k).isdigit()}
    if not num:
        return {}
    v = str(comp.get("value", "")).upper()
    if d and d.get("pins"):
        table = {s: tuple(p) for s, p in d["pins"].items()}
    elif "311" in v or "СА3" in v:
        table = LM311
    elif max(num) >= 14 or "324" in v or "339" in v:
        table = QUAD14
    elif max(num) <= 8 and len(num) >= 7:
        table = DUAL8
    else:
        table = SINGLE
    return {s: (num.get(o), num.get(i), num.get(p)) for s, (o, i, p) in table.items()
            if num.get(o) and num.get(i) and num.get(p)}


class Analog:
    def __init__(self, phys):
        self.p = phys
        self.c = phys.c

    def U(self, n):
        return self.p.U.get(n) if n is not None else None

    def rails(self, ref):
        vs = [self.U(n) for n in self.c.pins(ref).values() if self.U(n) is not None]
        hi = max([v for v in vs if v > 0.5], default=None)
        lo = min([v for v in vs if v < -0.5], default=0.0)
        return hi, lo

    def res_between(self, a, b):
        for r in self.c.neighbors(a):
            comp = self.c.comps[r]
            if comp.get("type") == "R" and b in self.c.pins(r).values() and a in self.c.pins(r).values():
                R, _ = dg.parse_value(comp.get("value"))
                if R:
                    return r, R
        return None, None

    def feedback_path(self, a, b, depth=3):
        """Есть ли путь между цепями a и b через резисторы/конденсаторы (не через общий провод и питание)."""
        seen, frontier = {a}, [a]
        for _ in range(depth):
            nxt = []
            for n in frontier:
                for r in self.c.neighbors(n):
                    if self.c.comps[r].get("type") not in ("R", "C"):
                        continue
                    for m in self.c.pins(r).values():
                        if m == b:
                            return True
                        if m not in seen and m.upper() not in core.GROUND_NAMES and self.U(m) is None:
                            seen.add(m)
                            nxt.append(m)
            frontier = nxt
        return False

    def resistors_at(self, n, exclude=()):
        out = []
        for r in self.c.neighbors(n):
            comp = self.c.comps[r]
            if comp.get("type") != "R" or r in exclude:
                continue
            R, _ = dg.parse_value(comp.get("value"))
            other = [m for m in self.c.pins(r).values() if m != n]
            if R and other:
                out.append((r, R, other[0]))
        return out

    # ---------- ОУ и компараторы ----------

    def amplifiers(self):
        for ref, comp in self.c.comps.items():
            name, d = parts.lookup(comp.get("value", ""))
            kind = (d or {}).get("kind")
            if kind not in ("OPAMP", "COMP"):
                continue
            hi, lo = self.rails(ref)
            for sec, (out, nin, nip) in _sections(comp, d).items():
                tag = f"{ref}.{sec}" if len(_sections(comp, d)) > 1 else ref
                pull = [(r, R, m) for r, R, m in self.resistors_at(out) if self.U(m) and self.U(m) > 0.5
                        and m not in (nin, nip)]
                voh = self.U(pull[0][2]) if pull and (d or {}).get("open_collector") else hi
                vol = lo if lo is not None else 0.0
                rf_pos, Rf_pos = self.res_between(out, nip)
                rf_neg, Rf_neg = self.res_between(out, nin)
                if rf_pos and voh:
                    self.schmitt(tag, ref, rf_pos, Rf_pos, out, nin, nip, voh, vol)
                elif kind == "COMP" or (not rf_neg and out != nin and not self.feedback_path(out, nin)):
                    sig = [n for n in (nin, nip) if self.U(n) is None]
                    self.p.add("средний", [ref], f"{tag}: компаратор без гистерезиса",
                               "нет резистора с выхода на неинвертирующий вход (положительной ОС)",
                               "на медленно меняющемся или зашумлённом сигнале выход многократно переключается "
                               "около порога («дребезг»): ложные срабатывания, сбои счёта; обычно добавляют "
                               "гистерезис 1…5 % через резистор ОС в сотни кОм" + (f"; сигнальные входы: {', '.join(sig)}" if sig else ""))
                elif out == nin:
                    self.p.add("информация", [ref], f"{tag}: повторитель напряжения", "K = 1, выход = вход +",
                               "буфер: высокое входное и низкое выходное сопротивление")
                elif not rf_neg:
                    self.p.add("информация", [ref], f"{tag}: усилитель с цепью ОС из нескольких элементов",
                               "обратная связь с выхода на вход − через R/C-цепочку (подстроечник, интегратор, фильтр)",
                               "коэффициент задаётся всей цепью ОС; при подстройке смотрите выход осциллографом — "
                               "без насыщения у шин питания")
                elif rf_neg:
                    rg = [(r, R, m) for r, R, m in self.resistors_at(nin, exclude={rf_neg})
                          if m.upper() in core.GROUND_NAMES or (self.U(m) is not None and abs(self.U(m)) < 1e-6)]
                    if rg:
                        K = 1 + Rf_neg / rg[0][1]
                        self.p.add("информация", [ref, rf_neg, rg[0][0]], f"{tag}: неинвертирующий усилитель",
                                   f"K = 1 + Rос/Rз = 1 + {fmt(Rf_neg, 'Ом')}/{fmt(rg[0][1], 'Ом')} = {K:.2f}",
                                   f"выход = {K:.2f}·U+; насыщение у шин питания — входной сигнал не больше "
                                   f"{(hi or 5) / K:.2f} В" if hi else f"выход = {K:.2f}·U+")

    def schmitt(self, tag, ref, rf, Rf, out, nin, nip, voh, vol):
        others = self.resistors_at(nip, exclude={rf})
        known = [(r, R, m) for r, R, m in others if self.U(m) is not None]
        unknown = [(r, R, m) for r, R, m in others if self.U(m) is None]
        uref_in = self.U(nin)
        if known and uref_in is None:                     # инвертирующий: сигнал на «−»
            G = sum(1 / R for _, R, _ in known) + 1 / Rf
            base = sum(self.U(m) / R for _, R, m in known)
            th_hi, th_lo = (base + voh / Rf) / G, (base + vol / Rf) / G
            refs = [ref, rf] + [r for r, _, _ in known]
            self.p.add("информация", refs, f"{tag}: триггер Шмитта (инвертирующий)",
                       f"Uпор = (ΣU·G + Uвых/Rос)/(ΣG + 1/Rос): верхний {th_hi:.2f} В (при Uвых = {voh:.1f} В), "
                       f"нижний {th_lo:.2f} В (при Uвых = {vol:.1f} В), гистерезис {th_hi - th_lo:.2f} В",
                       f"выход переключается в 0, когда вход «−» поднимется выше {th_hi:.2f} В, и обратно — "
                       f"ниже {th_lo:.2f} В; если сигнал не выходит за эти пороги — выход не меняется")
        elif unknown and uref_in is not None:             # неинвертирующий: сигнал через Rвх на «+»
            r_in, Rin, _ = unknown[0]
            on = uref_in + (uref_in - vol) * Rin / Rf
            off = uref_in - (voh - uref_in) * Rin / Rf
            self.p.add("информация", [ref, rf, r_in], f"{tag}: триггер Шмитта (неинвертирующий)",
                       f"Uвкл = Uоп + (Uоп − Uвых.мин)·Rвх/Rос = {on:.2f} В; Uвыкл = Uоп − (Uвых.макс − Uоп)·Rвх/Rос = "
                       f"{off:.2f} В (Uоп = {uref_in:.2f} В, Rвх = {fmt(Rin, 'Ом')}, Rос = {fmt(Rf, 'Ом')})",
                       f"гистерезис {on - off:.2f} В; выход включится при входе выше {on:.2f} В и выключится ниже {off:.2f} В")
        else:
            self.p.add("информация", [ref, rf], f"{tag}: положительная ОС (гистерезис)",
                       f"Rос = {fmt(Rf, 'Ом')} с выхода на вход +", "пороги зависят от источника опорного напряжения — "
                       "задайте напряжения цепей метками, чтобы посчитать их")

    # ---------- логика ----------

    def logic(self):
        for ref, comp in self.c.comps.items():
            name, d = parts.lookup(comp.get("value", ""))
            if not d:
                continue
            if d["kind"] == "SCHMITT":
                self.schmitt_logic(ref, comp, name, d)
            elif d["kind"] in ("DECODER", "ENCODER"):
                self.decoder(ref, comp, name, d)

    def schmitt_logic(self, ref, comp, name, d):
        num = {int(k): v for k, v in self.c.pins(ref).items() if str(k).isdigit()}
        vcc = self.U(num.get(14)) or self.U(self.c.pins(ref).get("VCC")) or self.U(self.c.pins(ref).get("VDD"))
        if vcc:
            if d.get("vt_pos_abs"):
                tp, tn = d["vt_pos_abs"], d["vt_neg_abs"]
            else:
                tp, tn = d["vt_pos"] * vcc, d["vt_neg"] * vcc
            self.p.add("информация", [ref], f"{ref} {name}: пороги триггера Шмитта при {vcc:.1f} В",
                       f"VT+ ≈ {tp:.2f} В, VT− ≈ {tn:.2f} В, гистерезис ≈ {tp - tn:.2f} В ({d.get('src', '')})",
                       "вход между порогами не меняет выход — так фронты медленных и зашумлённых сигналов "
                       "становятся чистыми; разброс порогов между экземплярами до ±30 %")
        if "инверт" not in d.get("func", ""):
            return
        for a, y in HEX_INV:                               # RC-генератор на инверторе Шмитта
            na, ny = num.get(a), num.get(y)
            if not na or not ny:
                continue
            r, R = self.res_between(na, ny)
            caps = [(cc, dg.parse_value(self.c.comps[cc].get("value"))[0]) for cc in self.c.neighbors(na)
                    if self.c.comps[cc].get("type") == "C" and any(
                        m.upper() in core.GROUND_NAMES for m in self.c.pins(cc).values())]
            if r and caps and caps[0][1]:
                k = d.get("osc_k", 1.0)
                f = 1 / (k * R * caps[0][1])
                self.p.add("информация", [ref, r, caps[0][0]], f"{ref}: генератор на триггере Шмитта (вход {a}, выход {y})",
                           f"f ≈ 1/({k}·R·C) = 1/({k}·{fmt(R, 'Ом')}·{fmt(caps[0][1], 'Ф')}) = {fmt(f, 'Гц')}",
                           "частота зависит от питания и порогов конкретного экземпляра (±30 %); "
                           "нет генерации — проверьте питание, C на КЗ, R на обрыв")

    def decoder(self, ref, comp, name, d):
        pins = self.c.pins(ref)
        pmap = d.get("pins") or {}
        by_name = {str(k).upper(): v for k, v in pins.items()}
        for k, n in pins.items():                          # номера выводов → имена по справочнику
            for nm, num in pmap.items():
                if str(num) == str(k):
                    by_name[nm.upper()] = n
        bad = []
        for en, want in d.get("enable", {}).items():
            n = by_name.get(en.upper())
            if n is None:
                continue
            v = self.U(n)
            if n.upper() in core.GROUND_NAMES:
                v = 0.0
            if v is None:
                continue
            level = 1 if v > 1.5 else 0
            if level != want:
                bad.append(f"{en} = {level} (нужно {want})")
        what = "дешифратор" if d["kind"] == "DECODER" else "шифратор"
        act = "низкий (0)" if d.get("outputs_active") == 0 else "высокий (1)"
        if bad:
            self.p.add("высокий", [ref], f"{ref} {name}: {what} заблокирован",
                       "разрешающие входы: " + ", ".join(bad),
                       f"при таком подключении ни один выход не станет активным — проверьте схему или распознавание")
        else:
            inputs = d.get("inputs") or []
            rule = ""
            if d["kind"] == "DECODER" and inputs:
                rule = (f"; выход Yn активен ({act}), когда двоичный код на входах {''.join(reversed(inputs))} = n"
                        if len(inputs) <= 3 else f"; активен выход с номером кода на входах {', '.join(inputs)}")
            en = ", ".join(f"{k}={v}" for k, v in d.get("enable", {}).items())
            self.p.add("информация", [ref], f"{ref} {name}: {d.get('func', what)}",
                       (f"условие работы: {en}" if en else "без входов разрешения") + rule,
                       "неисправность: один выход «молчит» — внутренний обрыв или КЗ этого выхода на нагрузку; "
                       "все молчат — питание, разрешение, адрес не меняется (проверьте источник кода)")

    def run(self):
        self.amplifiers()
        self.logic()
