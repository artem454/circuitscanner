"""Физические расчёты по схеме: законы Кирхгофа, Ома, Джоуля–Ленца и правила
электротехники. Каждый вывод сопровождается формулой и подставленными числами.

  1. Режим по постоянному току — метод узловых потенциалов (MNA):
     G·U = I, где G — матрица проводимостей. Резисторы — проводимости 1/R,
     дроссели и предохранители — почти КЗ, конденсаторы — разрыв, диоды и
     стабилитроны — кусочно-линейная модель (Uпр ≈ 0,7 В, пробой при Uст),
     транзисторы и микросхемы — разрыв (их режим без сигналов не определён).
     Известные напряжения — из меток цепей («+600V», «+12 В») и земли.
  2. По найденным U и I: мощность P = U²/R, запас по мощности (правило:
     резистор нагружают не более чем на 50–60 %), предельное рабочее напряжение
     резистора (≈250 В для 0,25 Вт), ток и мощность стабилитронов.
  3. Расчёты узлов: делитель, предзаряд (E = ½CU², Iпик = U/R, τ = RC),
     выравнивание последовательных конденсаторов (Iут ≈ 0,01·C·U), выпрямитель
     и пульсации (ΔU = I/(2fC)), линейный стабилизатор (P = (Uвх−Uвых)·I,
     Tj = Ta + P·Rθ), бутстреп (C ≥ 15·Qg/ΔU), ток затвора, усилитель шунта
     (K = −Rос/Rвх), фильтр (fср = 1/(2πRC)), генератор XR2206 (f = 1/(RC)),
     счётчик-делитель, нагрев ключа (P = I²·Rси·1,8, Tj = Ta + P·Rθ), ресурс
     электролита (L = L₀·2^((Tmax−T)/10)).

Условия работы (ток нагрузки, частота ШИМ, температура) задаются в
physics.json в папке проекта; без них расчёты, которым они нужны, пропускаются.

    python physics.py папка_проекта
"""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

import core
import diagnose as dg
import parts

# Параметры деталей — из справочника parts.py (parts_db.json + ваши/импортированные из даташитов)

# Термопары: коэффициент Зеебека около 0…500 °C, мкВ/°C
SEEBECK = {"K": 41, "J": 52, "T": 43, "E": 61, "N": 27, "S": 6, "R": 6, "B": 1, "L": 66, "ХА": 41, "ХК": 66}
PT_A, PT_B = 3.9083e-3, -5.775e-7          # Каллендар–Ван Дюзен, IEC 60751 (T ≥ 0 °C)

DEFAULT_SETTINGS = {
    "ambient_C": 40,            # температура внутри корпуса
    "load_current_A": None,     # ток нагрузки силовой части (действующий)
    "switch_freq_Hz": None,     # частота переключения ключей
    "logic_current_A": 0.1,     # ток потребления от линейного стабилизатора
    "mains_freq_Hz": 50,
    "heatsink_K_per_W": None,   # тепловое сопротивление радиатора ключей
    "cap_life_h": 2000, "cap_tmax_C": 85,
    "max_freq_Hz": 1e6,         # до какой частоты смотреть ВЧ-поведение
    # КИП
    "loop_supply_V": None,      # питание токовой петли, если не подписано на схеме (обычно 24 В)
    "tx_min_V": 12.0,           # минимальное напряжение на клеммах 2-проводного датчика
    "wire_ohm": 20.0,           # сопротивление линии туда-обратно, Ом
    "barrier_ohm": 0.0,         # искрозащитный барьер в петле, Ом (обычно 250–350)
    "adc_vref_V": None,         # опорное напряжение АЦП, если отличается от питания МК
    "i2c_bus_F": 100e-12,       # ёмкость шины I²C
    "i2c_speed_Hz": 100e3,
    # обратная связь по скорости/положению
    "motor_rpm_max": 3000,
    "encoder_ppr": None,        # импульсов на оборот (если не указано в номинале энкодера)
    "cable_m": 10,              # длина кабеля датчика
    "cable_pF_per_m": 100,
    "tacho_ripple_per_rev": 20, # пульсаций тахогенератора за оборот (число коллекторных пластин)
    "loop_bw_Hz": None,         # полоса контура регулирования скорости
}

VF = 0.7


def part(comp):
    """(имя, параметры) детали по номиналу из справочника."""
    return parts.lookup(comp.get("value", ""))


def fmt(x, unit=""):
    """Число в инженерной записи: 0.00022 → '220 мк'."""
    if x is None:
        return "?"
    a = abs(x)
    for k, p in ((1e9, "Г"), (1e6, "М"), (1e3, "к"), (1, ""), (1e-3, "м"), (1e-6, "мк"), (1e-9, "н"), (1e-12, "п")):
        if a >= k * 0.9999 or k == 1e-12:
            s = f"{x / k:.3g}".replace(".", ",")
            return f"{s} {p}{unit}".strip()
    return f"{x:g} {unit}"


def power_rating(comp):
    """Мощность резистора по номиналу или примечанию ('1W', '5 Вт'); иначе 0,25 Вт."""
    t = (str(comp.get("value", "")) + " " + str(comp.get("note", ""))).replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:W|Вт)\b", t, re.I)
    return (float(m.group(1)), True) if m else (0.25, False)


class Physics:
    def __init__(self, nl, settings=None):
        self.c = dg.Circuit(nl)
        self.S = dict(DEFAULT_SETTINGS, **(settings or {}))
        self.findings = []
        self.stress = {}           # ref → коэффициент нагрузки (1 = на пределе)
        self.stress_sym = {}       # ref → симптомы, к которым ведёт эта перегрузка
        self.U = {}
        self.fixed = {}

    def add(self, level, refs, title, formula, conclusion):
        self.findings.append({"level": level, "refs": list(refs), "title": title,
                              "formula": formula, "conclusion": conclusion})

    def mark(self, ref, k, syms=("overheat", "smoke", "after_time", "fuse")):
        self.stress[ref] = max(self.stress.get(ref, 0), k)
        self.stress_sym.setdefault(ref, set()).update(syms)

    # ---------- метод узловых потенциалов ----------

    def solve_dc(self):
        c = self.c
        fixed = {}
        for n in c.nets:
            if n.upper() in core.GROUND_NAMES:
                fixed[n] = 0.0
        for n, val in c.volt.items():
            if n not in fixed and not n.endswith("_IN"):         # сетевые входы — переменка
                lbl = str(c.meta.get(n, {}).get("label", "")) + n
                fixed[n] = -abs(val) if re.search(r"[-−]\s*\d", lbl) else val
        # тахогенератор — источник напряжения U = Kт·n (на половине максимальных оборотов)
        for ref, comp in c.comps.items():
            if comp.get("type") != "TACHO":
                continue
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*[VВ]\s*/\s*(\d+)", str(comp.get("value", "")))
            p = list(c.pins(ref).values())
            gnd = [n for n in p if n.upper() in core.GROUND_NAMES]
            if m and gnd and len(p) == 2:
                ke = float(m.group(1).replace(",", ".")) / float(m.group(2))
                top = [n for n in p if n not in gnd][0]
                fixed.setdefault(top, ke * self.S["motor_rpm_max"] * 0.5)
        free = [n for n in c.nets if n not in fixed]
        idx = {n: i for i, n in enumerate(free)}
        N = len(free)
        if N == 0:
            self.U, self.fixed = dict(fixed), fixed
            return
        elems, diodes, isrc = [], [], []
        for ref, comp in c.comps.items():
            t, p = comp.get("type"), c.pins(ref)
            val, _ = dg.parse_value(comp.get("value"))
            vs_ = str(comp.get("value", "")) + " " + str(comp.get("note", ""))
            if t == "RT" and len(p) == 2:                      # терморезистор — сопротивление при 25 °C
                if re.search(r"NTC", vs_, re.I):
                    val = val or 10e3
                else:
                    r0 = 1000.0 if "1000" in vs_ else 100.0
                    val = r0 * (1 + PT_A * 25 + PT_B * 625)
                t = "R"
            if t == "SENSOR" and "+" in p and "-" in p and re.search(r"4.?20", vs_):
                isrc.append((ref, p["+"], p["-"], comp.get("_current", 0.020)))   # худший случай — 20 мА
                continue
            if t == "R" and val and len(p) == 2:
                a, b = list(p.values())
                elems.append((ref, a, b, 1.0 / max(val, 1e-6)))
            elif t in ("L", "F") and len(p) == 2:
                a, b = list(p.values())
                elems.append((ref, a, b, 1e3))
            elif t in ("D", "Z") and "A" in p and "K" in p:
                vz = None
                if t == "Z":
                    _, pd_ = parts.lookup(comp.get("value", ""))
                    if pd_ and pd_.get("vbr"):
                        vz = pd_["vbr"]                       # TVS: начинает проводить на пробое, не на Uраб
                    elif pd_ and pd_.get("vz"):
                        vz = pd_["vz"]
                    else:
                        m = re.search(r"(\d+(?:[.,]\d+)?)\s*[VВ]", str(comp.get("value", "")))
                        vz = float(m.group(1).replace(",", ".")) if m else None
                diodes.append([ref, p["A"], p["K"], vz, 0])     # 0 выкл, 1 прямо, -1 пробой
            elif t == "BR":
                ac = [n for k, n in p.items() if k.upper().startswith("AC")]
                for i, a in enumerate(ac):
                    if "P" in p:
                        diodes.append([f"{ref}.{i}+", a, p["P"], None, 0])
                    if "N" in p:
                        diodes.append([f"{ref}.{i}-", p["N"], a, None, 0])
        connected = set(fixed)
        changed = True
        links = [(a, b) for _, a, b, _ in elems] + [(d[1], d[2]) for d in diodes] + [(a, b) for _, a, b, _ in isrc]
        while changed:                                           # какие узлы привязаны к источникам
            changed = False
            for a, b in links:
                if (a in connected) != (b in connected):
                    connected |= {a, b}
                    changed = True
        g_on, g_off = 100.0, 1e-9
        for _ in range(30):
            G = np.zeros((N, N))
            I = np.zeros(N)

            def stamp(a, b, g, src=0.0):
                # ток из a в b: g·(Ua − Ub − src)
                for n1, n2, sgn in ((a, b, 1), (b, a, -1)):
                    if n1 in idx:
                        i = idx[n1]
                        G[i, i] += g
                        if n2 in idx:
                            G[i, idx[n2]] -= g
                        else:
                            I[i] += g * fixed.get(n2, 0.0)
                        I[i] += sgn * g * src

            for i in range(N):
                G[i, i] += 1e-12
            for _, a, b, g in elems:
                stamp(a, b, g)
            for _, a, b, cur in isrc:                          # источник тока из a в b
                if a in idx:
                    I[idx[a]] -= cur
                if b in idx:
                    I[idx[b]] += cur
            for d in diodes:
                _, a, k, vz, st = d
                if st == 1:
                    stamp(a, k, g_on, VF)
                elif st == -1:
                    stamp(k, a, g_on, vz)
                else:
                    stamp(a, k, g_off)
            Uf = np.linalg.solve(G, I)
            U = dict(fixed)
            U.update({n: float(Uf[idx[n]]) for n in free})
            flip = False
            for d in diodes:
                _, a, k, vz, st = d
                va, vk = U.get(a, 0), U.get(k, 0)
                new = st
                if st == 0 and va - vk > VF + 1e-3:
                    new = 1
                elif st == 0 and vz and vk - va > vz + 1e-3:
                    new = -1
                elif st == 1 and va - vk < VF - 1e-3:
                    new = 0
                elif st == -1 and vk - va < vz - 1e-3:
                    new = 0
                if new != st:
                    d[4] = new
                    flip = True
            if not flip:
                break
        self.U = {n: v for n, v in U.items() if n in connected}
        self.fixed = fixed
        self._elems, self._diodes = elems, diodes

    # ---------- проверки по найденному режиму ----------

    def check_resistors(self):
        for ref, a, b, g in self._elems:
            comp = self.c.comps[ref]
            if comp.get("type") != "R" or a not in self.U or b not in self.U:
                continue
            u = abs(self.U[a] - self.U[b])
            if u < 1e-3:
                continue
            r = 1 / g
            p = u * u / r
            prated, known = power_rating(comp)
            k = p / prated
            self.mark(ref, k)
            f = f"P = U²/R = ({fmt(u, 'В')})² / {fmt(r, 'Ом')} = {fmt(p, 'Вт')}"
            if k > 1:
                self.add("высокий", [ref], f"{ref}: перегрузка по мощности", f,
                         f"больше допустимых {fmt(prated, 'Вт')}{'' if known else ' (мощность не указана, принято 0,25 Вт)'} — резистор сгорит")
            elif k > 0.6:
                self.add("средний", [ref], f"{ref}: мало запаса по мощности", f,
                         f"{k * 100:.0f} % от {fmt(prated, 'Вт')}; по правилу нагружают не более 50–60 %, иначе перегрев и уход номинала")
            if u > 250 and prated <= 0.5:
                self.mark(ref, u / 250)
                self.add("средний", [ref], f"{ref}: превышено рабочее напряжение резистора",
                         f"U = {fmt(u, 'В')} > ≈250 В (предел для 0,25–0,5 Вт)",
                         "плёнку пробивает дугой по спирали; высокое напряжение делят цепочкой резисторов")

    def check_diodes(self):
        for ref, a, k, vz, st in self._diodes:
            if "." in ref or a not in self.U or k not in self.U:
                continue
            comp = self.c.comps[ref]
            vr = self.U[k] - self.U[a]
            name, d = part(comp)
            vmax = (d or {}).get("vmax") or dg.part_vmax(comp)
            if st == -1 and vz:
                i = sum(g * (self.U.get(x, 0) - self.U.get(y, 0)) for r2, x, y, g in self._elems
                        if k in (x, y) and False)  # ток считаем через баланс узла ниже
                i = self._node_current(k, exclude=ref)
                p = abs(i) * vz
                pz = 1.3 if "1N47" in str(comp.get("value", "")) else 0.5
                self.mark(ref, p / pz)
                self.add("средний" if p > 0.6 * pz else "информация", [ref], f"{ref}: стабилитрон в пробое",
                         f"Iст ≈ {fmt(abs(i), 'А')}, P = Iст·Uст = {fmt(p, 'Вт')}",
                         "перегрузка — стабилитрон перегреется" if p > pz else "в пределах нормы")
            elif vmax and vr > 0.8 * vmax:
                self.mark(ref, vr / vmax)
                self.add("высокий" if vr > vmax else "средний", [ref], f"{ref}: обратное напряжение близко к пределу",
                         f"Uобр = {fmt(vr, 'В')}, Uобр.макс = {fmt(vmax, 'В')}", "нужен диод на большее напряжение")

    def _node_current(self, node, exclude=None):
        """Сумма токов, втекающих в узел через резисторы (для тока стабилитрона)."""
        s = 0.0
        for ref, a, b, g in self._elems:
            if ref == exclude:
                continue
            if a == node and b in self.U:
                s += g * (self.U[b] - self.U[node])
            elif b == node and a in self.U:
                s += g * (self.U[a] - self.U[node])
        return s

    def report_dividers(self):
        """Порты с вычисленным (а не заданным) напряжением — выходы делителей, сигналы на CPU."""
        for n, meta in self.c.meta.items():
            if meta.get("port") and n in self.U and n not in self.fixed and abs(self.U[n]) > 1e-3 \
                    and not any(abs(self.U[n] - v) < 0.01 for v in self.fixed.values()):   # просто подтяжка к питанию
                src = max((v for v in self.fixed.values()), default=0)
                self.add("информация", [], f"Напряжение в цепи {n} ({meta.get('label', '')})",
                         f"по узловым потенциалам: U({n}) = {fmt(self.U[n], 'В')}",
                         f"при {fmt(src, 'В')} на шине; проверьте, что вход, куда идёт сигнал, это допускает")
        for n in self.U:                                   # внутренние узлы делителей с портом через повторитель
            pass

    # ---------- расчёты узлов ----------

    def bus_caps(self):
        c = self.c
        groups = {}
        for ref, comp in c.comps.items():
            if "bus_cap" in c.roles[ref]:
                groups.setdefault(comp.get("value"), []).append(ref)
        out = []
        for val, refs in groups.items():
            C, vrat = dg.parse_value(val)
            if not C:
                continue
            ser = c._series_partners(refs[0])
            n_ser = 1 + len(ser)
            n_par = max(1, len(refs) // n_ser)
            Ctot = C * n_par / n_ser
            vbus = max(v for r in refs for v in [c.span_v(r) or 0])
            hi = max((c.net_v(nn) or 0) for r in refs for nn in c.pins(r).values())
            vbus = max(vbus, hi)
            out.append((refs, C, vrat, n_ser, n_par, Ctot, vbus))
            if n_ser > 1 and vrat and vbus:
                ileak = 0.01 * C * 1e6 * vrat * 1e-6          # Iут ≤ 0,01·C[мкФ]·U[В] мкА
                di = 0.5 * ileak
                rmax = (n_ser * vrat - vbus) / ((n_ser - 1) * di)
                has_bal = any(c.comps[r].get("type") == "R" for n in c.pins(refs[0]).values()
                              for r in c.neighbors(n) if n not in ("GND",) and r not in refs and
                              set(c.pins(r).values()) <= set(c.pins(refs[0]).values()) | set(c.pins(ser[0]).values()))
                f = (f"Iут ≤ 0,01·C·U = 0,01·{fmt(C * 1e6)}мкФ·{vrat:g}В = {fmt(ileak, 'А')}; "
                     f"разброс ΔI ≈ 50 % = {fmt(di, 'А')}; Rвыр ≤ (n·Uном − Uшины)/((n−1)·ΔI) = "
                     f"({n_ser}·{vrat:g} − {vbus:.0f})/({n_ser - 1}·{fmt(di, 'А')}) = {fmt(rmax, 'Ом')}")
                if not has_bal:
                    for r in refs:
                        self.mark(r, 1.3)
                    pr = vbus / n_ser * (vbus / n_ser) / max(rmax, 1)
                    self.add("высокий", refs, "Нет выравнивающих резисторов на последовательных конденсаторах", f,
                             f"поставить по резистору ≤ {fmt(rmax, 'Ом')} параллельно каждому ряду; "
                             f"мощность каждого U²/R = {fmt(pr, 'Вт')} — брать с запасом ×2")
            if vrat and vbus:
                k = vbus / (n_ser * vrat)
                for r in refs:
                    self.mark(r, k / 0.8)
                if k > 0.8:
                    self.add("средний" if k <= 1 else "высокий", refs, "Конденсаторы шины работают выше 80 % номинала",
                             f"Uшины/(n·Uном) = {vbus:.0f}/({n_ser}·{vrat:g}) = {k * 100:.0f} %",
                             "правило: электролиты нагружают не более 80 % по напряжению")
        return out

    def precharge(self, caps):
        c = self.c
        for ref, comp in c.comps.items():
            if comp.get("type") != "R" or "precharge" not in c.roles[ref]:
                continue
            R, _ = dg.parse_value(comp.get("value"))
            if not R or not caps:
                continue
            refs, C, vrat, n_ser, n_par, Ctot, vbus = max(caps, key=lambda x: x[5] * x[6] ** 2)
            E = 0.5 * Ctot * vbus ** 2
            ipk = vbus / R
            tau = R * Ctot
            prated, known = power_rating(comp)
            f = (f"E = ½·C·U² = ½·{fmt(Ctot, 'Ф')}·({vbus:.0f} В)² = {fmt(E, 'Дж')}; "
                 f"Iпик = U/R = {vbus:.0f}/{R:g} = {fmt(ipk, 'А')}; τ = R·C = {fmt(tau, 'с')}, заряд ≈ 5τ = {fmt(5 * tau, 'с')}")
            self.mark(ref, E / (prated * 5) if prated else 1)      # ≈ 5 с допустимой перегрузки
            self.add("средний" if E > 10 else "информация", [ref], f"{ref}: предзаряд звена", f,
                     f"при каждом включении резистор поглощает {fmt(E, 'Дж')} (столько же, сколько запасает конденсатор). "
                     + ("Нужен импульсный/проволочный резистор, ≥5 Вт; обычный " + fmt(prated, "Вт") + " сгорит или уйдёт в обрыв."
                        if E > 10 * prated else "Допустимо.")
                     + " Реле должно замыкаться не раньше чем через 5τ.")

    def transformer_supply(self):
        c = self.c
        for ref, comp in c.comps.items():
            if comp.get("type") != "T":
                continue
            m = re.search(r"(\d+(?:[.,]\d+)?)\s*[VВ]", str(comp.get("value", "")))
            if not m:
                continue
            u2 = float(m.group(1).replace(",", "."))
            sec = [n for k, n in c.pins(ref).items() if k.upper().startswith("S")]
            br = [r for n in sec for r in c.neighbors(n) if c.comps[r].get("type") == "BR"]
            if not br:
                continue
            vdc = u2 * math.sqrt(2) - 2 * VF
            pnet = c.pins(br[0]).get("P")
            self.fixed.setdefault(pnet, vdc)
            caps = [r for r in c.neighbors(pnet) if c.comps[r].get("type") == "C"]
            Csum = sum((dg.parse_value(c.comps[r].get("value"))[0] or 0) for r in caps)
            I = self.S["logic_current_A"]
            fr = 2 * self.S["mains_freq_Hz"]
            rip = I / (fr * Csum) if Csum else None
            self.add("информация", [ref] + br, f"Выпрямитель {ref}→{br[0]}",
                     f"Uхх ≈ U₂·√2 − 2·Uпр = {u2:g}·1,414 − 1,4 = {vdc:.1f} В"
                     + (f"; пульсации ΔU = I/(2f·C) = {I:g}/({fr:g}·{fmt(Csum, 'Ф')}) = {rip:.1f} В" if rip else ""),
                     "сеть +10 % даёт ещё +10 % к этому напряжению")
            for r in caps:
                _, vr = dg.parse_value(c.comps[r].get("value"))
                if vr and vdc * 1.1 > 0.8 * vr:
                    self.mark(r, vdc * 1.1 / vr / 0.8)
                    self.add("средний", [r], f"{r}: мал запас по напряжению",
                             f"Uмакс ≈ 1,1·{vdc:.1f} = {vdc * 1.1:.1f} В при номинале {vr:g} В", "нужен конденсатор на большее напряжение")
            for reg in [r for r in c.neighbors(pnet) if part(c.comps[r])[1] and part(c.comps[r])[1]["kind"] == "REG"]:
                name, d = part(c.comps[reg])
                vin_min = max(0.0, vdc - (rip or 0))       # ΔU ≥ Uвх: ёмкость не держит напряжение вовсе
                p = (vdc - d["vout"]) * I
                tj = self.S["ambient_C"] + p * d["rth_ja"]
                starved = vin_min < d["vout"] + d["dropout"]
                supply_syms = ("unstable", "mcu_hang", "no_power", "noise", "weak", "overheat")
                self.mark(reg, max(tj / 125, 1.5 if starved else 0), supply_syms)
                if starved:
                    for r in caps:                      # мала ёмкость фильтра — её и подозревать
                        self.mark(r, min(3.0, (rip or 0) / (0.2 * vdc)), supply_syms)
                lvl = "высокий" if tj > 125 or starved else (
                    "средний" if tj > 100 or vdc * 1.1 > d["vin_max"] * 0.9 else "информация")
                self.add(lvl, [reg] + caps, f"{reg} {name}: режим стабилизатора",
                         f"P = (Uвх − Uвых)·I = ({vdc:.1f} − {d['vout']})·{I:g} = {fmt(p, 'Вт')}; "
                         f"Tj = Ta + P·Rθja = {self.S['ambient_C']} + {p:.2f}·{d['rth_ja']} = {tj:.0f} °C; "
                         f"Uвх.мин = {vin_min:.1f} В при нужных ≥ {d['vout'] + d['dropout']:.1f} В",
                         ("перегрев — нужен радиатор или меньшее Uвх; " if tj > 100 else "")
                         + ("провалы пульсаций ниже Uвых+2 В — стабилизатор выпадает из режима, логика сбоит; увеличьте ёмкость; "
                            if vin_min < d["vout"] + d["dropout"] else "")
                         + f"ток нагрузки принят {I:g} А (logic_current_A в physics.json)")

    def gate_drive(self):
        c = self.c
        f_sw = self.S["switch_freq_Hz"]
        for ref, comp in c.comps.items():
            if "power_switch" not in c.roles[ref]:
                continue
            name, d = part(comp)
            if not d or "qg" not in d:
                continue
            g = c.pins(ref).get("G")
            rg = [r for r in c.neighbors(g) if c.comps[r].get("type") == "R" and r != ref]
            drv = [r for r in rg for n in c.pins(r).values() for x in c.neighbors(n)
                   if part(c.comps[x])[1] and part(c.comps[x])[1]["kind"] == "DRV"]
            if not rg:
                continue
            R, _ = dg.parse_value(c.comps[rg[0]].get("value"))
            vdrv = 12.0
            dname, dd = (part(c.comps[[x for n in c.pins(rg[0]).values() for x in c.neighbors(n)
                                       if part(c.comps[x])[1] and part(c.comps[x])[1]["kind"] == "DRV"][0]])
                         if drv else (None, None))
            ig = vdrv / (R + 10)
            lim = min(dd["io_src"], ig) if dd else ig
            t = d["qg"] / lim                     # полный заряд затвора
            t_sw = t / 3                          # сам фронт напряжения — плато Миллера, ≈ треть Qg
            self.add("информация", [ref, rg[0]], f"{ref}: управление затвором",
                     f"Iзатв = Uдр/(Rз + Rдр) = {vdrv:g}/({R:g}+10) = {fmt(ig, 'А')}"
                     + (f", ограничено драйвером {dname} ({fmt(dd['io_src'], 'А')})" if dd and dd["io_src"] < ig else "")
                     + f"; заряд затвора Qg/I = {fmt(d['qg'], 'Кл')}/{fmt(lim, 'А')} = {fmt(t, 'с')}, "
                     f"фронт (плато Миллера ≈ Qg/3) ≈ {fmt(t_sw, 'с')}",
                     ("фронт медленный — потери на переключение растут; " if t_sw > 0.5e-6 else "")
                     + (f"при f = {fmt(f_sw, 'Гц')} фронты занимают {t_sw * f_sw * 2 * 100:.1f} % периода" if f_sw else
                        "задайте switch_freq_Hz, чтобы оценить потери"))
            # бутстреп
            vb = [n for n in c.nets if re.search(r"VB|BOOT", n.upper()) and
                  c.pins(ref).get("S") in [nn for r in c.neighbors(n) for nn in c.pins(r).values()]]
            for n in vb:
                cb = [r for r in c.neighbors(n) if c.comps[r].get("type") == "C"]
                Cb = sum((dg.parse_value(c.comps[r].get("value"))[0] or 0) for r in cb)
                need = 15 * d["qg"] / 1.0
                if Cb:
                    self.add("информация" if Cb >= need else "средний", cb, f"{ref}: конденсатор бутстрепа",
                             f"C ≥ 15·Qg/ΔU = 15·{fmt(d['qg'], 'Кл')}/1 В = {fmt(need, 'Ф')}; стоит {fmt(Cb, 'Ф')}",
                             "достаточно" if Cb >= need else "мало — верхний ключ недооткрывается и греется")
                for dref in [r for r in c.neighbors(n) if c.comps[r].get("type") == "D"]:
                    dn, ddat = part(c.comps[dref])
                    if ddat and ddat.get("trr", 0) > 1e-6:
                        self.mark(dref, 1.2)
                        self.add("средний", [dref], f"{dref}: медленный диод бутстрепа",
                                 f"trr = {fmt(ddat['trr'], 'с')}" + (f", период 1/f = {fmt(1 / f_sw, 'с')}" if f_sw else ""),
                                 "пока диод восстанавливается, конденсатор бутстрепа разряжается обратно; нужен UF4007/FR107 (trr < 0,1 мкс)")
            # нагрев
            I = self.S["load_current_A"]
            if I:
                ta = self.S["ambient_C"]
                if d["kind"] == "MOSFET":
                    p = I * I * d["rds"] * 1.8 * 0.5          # полумост: каждый ключ ~половину времени
                    form = f"P = I²·Rси(горяч.)·D = {I:g}²·{d['rds']:g}·1,8·0,5 = {fmt(p, 'Вт')}"
                else:
                    p = d["vce"] * I * 0.5 * 0.9
                    form = f"P = Uкэ.нас·I·D = {d['vce']}·{I:g}·0,5·0,9 = {fmt(p, 'Вт')}"
                vbus = max(c.volt.values(), default=0)
                if f_sw and vbus:
                    psw = 0.5 * vbus * I * 2 * t_sw * f_sw
                    p += psw
                    form += (f"; Pпер = ½·U·I·(tвкл+tвыкл)·f = ½·{vbus:.0f}·{I:g}·{fmt(2 * t_sw, 'с')}·"
                             f"{fmt(f_sw, 'Гц')} = {fmt(psw, 'Вт')}")
                hs = self.S["heatsink_K_per_W"]
                rjc = d.get("rth_jc", 1.0)
                if hs:
                    rth = rjc + 0.5 + hs                              # кристалл–корпус + прокладка + радиатор
                    tj = ta + p * rth
                    self.mark(ref, tj / 150)
                    self.add("высокий" if tj > 150 else "средний" if tj > 110 else "информация", [ref],
                             f"{ref}: нагрев ключа",
                             form + f"; Tj = Ta + P·(Rθjc + Rθcs + Rθsa) = {ta} + {p:.1f}·({rjc}+0,5+{hs}) = {tj:.0f} °C",
                             "перегрев — нужен радиатор лучше или ключ с меньшими потерями" if tj > 110 else "в норме")
                else:
                    need = (110 - ta) / p - rjc - 0.5 if p else None   # держать Tj ≤ 110 °C
                    self.mark(ref, 1.0 if need is not None and need < 1 else 0.6)
                    lvl = "высокий" if need is not None and need <= 0 else (
                        "средний" if need is not None and need < 1 else "информация")
                    self.add(lvl, [ref], f"{ref}: потери и нужный радиатор",
                             form + f"; для Tj ≤ 110 °C: Rθsa ≤ (110 − Ta)/P − Rθjc − Rθcs = "
                                    f"(110 − {ta})/{p:.1f} − {rjc} − 0,5 = {need:.2f} К/Вт",
                             ("никакой радиатор не удержит 110 °C: снизьте частоту или ток, ускорьте переключение "
                              "(меньше Rз, мощнее драйвер); " if need is not None and need <= 0 else
                              "радиатор должен быть очень хорошим (с обдувом); " if need is not None and need < 1 else "")
                             + "задайте heatsink_K_per_W в physics.json, чтобы посчитать температуру кристалла. "
                               "Потери переключения оценены по заряду затвора — точнее по Eon/Eoff из даташита")
                if I > d["imax"] * 0.7:
                    self.mark(ref, I / d["imax"] / 0.7)
                    self.add("средний", [ref], f"{ref}: ток близок к паспортному",
                             f"I = {I:g} А при Iмакс = {d['imax']} А (при 25 °C; при 100 °C ≈ 0,6·Iмакс)",
                             "правило: ключ выбирают с запасом по току ×1,5–2")

    def opamp_stages(self):
        c = self.c
        for ref, comp in c.comps.items():
            name, d = part(comp)
            if not d or d["kind"] != "OPAMP":
                continue
            p = {int(k): n for k, n in c.pins(ref).items() if str(k).isdigit()}
            for sec, (o, inn, inp) in d["pins"].items():
                if o not in p or inn not in p:
                    continue
                out, nin = p[o], p[inn]
                rf = [r for r in c.neighbors(nin) if c.comps[r].get("type") == "R" and out in c.pins(r).values()]
                if not rf:
                    continue
                Rf, _ = dg.parse_value(c.comps[rf[0]].get("value"))
                rin = [r for r in c.neighbors(nin) if c.comps[r].get("type") == "R" and r not in rf]
                if not rin or not Rf:
                    continue
                Ri, _ = dg.parse_value(c.comps[rin[0]].get("value"))
                K = -Rf / Ri
                cf = [r for r in c.neighbors(nin) if c.comps[r].get("type") == "C" and out in c.pins(r).values()]
                form = f"K = −Rос/Rвх = −{fmt(Rf, 'Ом')}/{fmt(Ri, 'Ом')} = {K:.1f}"
                if cf:
                    Cf, _ = dg.parse_value(c.comps[cf[0]].get("value"))
                    if Cf:
                        form += f"; fср = 1/(2π·Rос·C) = {fmt(1 / (2 * math.pi * Rf * Cf), 'Гц')}"
                src = [n for n in c.pins(rin[0]).values() if n != nin]
                sh = [r for n in src for r in c.neighbors(n) if "shunt" in c.roles.get(r, [])]
                concl = "инвертирующий усилитель"
                if sh:
                    Rs, _ = dg.parse_value(c.comps[sh[0]].get("value"))
                    kt = K * Rs
                    vmax = 13.5
                    imax = vmax / abs(kt)
                    form += f"; с шунтом {sh[0]} {fmt(Rs, 'Ом')}: Kт = K·Rш = {fmt(kt, 'В/А')}, Iмакс = 13,5 В/|Kт| = {fmt(imax, 'А')}"
                    concl = f"датчик тока: выход {fmt(kt, 'В/А')}; выше {fmt(imax, 'А')} усилитель насыщается и защита по току «слепнет»"
                    I = self.S["load_current_A"]
                    if I and I * 1.414 * 2 > imax:
                        self.add("средний", [ref, sh[0]], f"{ref}.{sec}: диапазон датчика тока мал",
                                 form, f"пиковый ток пуска ≈ 2·√2·I = {2.83 * I:.1f} А больше {imax:.1f} А")
                        continue
                self.add("информация", [ref] + rf + rin, f"{ref}.{sec}: каскад на ОУ", form, concl)

    def oscillators(self):
        c = self.c
        for ref, comp in c.comps.items():
            v = str(comp.get("value", "")).upper()
            if "XR2206" not in v:
                continue
            p = c.pins(ref)
            cap = [r for r in c.neighbors(p.get("5", "")) if c.comps[r].get("type") == "C" and p.get("6") in c.pins(r).values()]
            rs = [r for r in c.neighbors(p.get("7", "")) if c.comps[r].get("type") == "R"]
            if not cap or not rs:
                continue
            C, _ = dg.parse_value(c.comps[cap[0]].get("value"))
            div = 1
            clk = p.get("11")
            for r in c.neighbors(clk or ""):
                if "561ИЕ8" in str(c.comps[r].get("value", "")).upper() or "4017" in str(c.comps[r].get("value", "")):
                    rst = c.pins(r).get("15")
                    outs = {"3": 0, "2": 1, "4": 2, "7": 3, "10": 4, "1": 5, "5": 6, "6": 7, "9": 8, "11": 9}
                    hits = [outs[k] for k, n in c.pins(r).items() if n == rst and k in outs]
                    div = min(hits) if hits else 10
            lines = []
            for r in rs:
                R, _ = dg.parse_value(c.comps[r].get("value"))
                if R and C:
                    f = 1 / (R * C)
                    lines.append(f"{r}: f = 1/(R·C) = 1/({fmt(R, 'Ом')}·{fmt(C, 'Ф')}) = {f:.1f} Гц → /{div} = {f / div:.1f} Гц")
            self.add("информация", [ref, cap[0]] + rs, f"{ref}: частота генератора",
                     "; ".join(lines),
                     "последовательно с ними подстроечный резистор снижает частоту; выход после делителя-счётчика"
                     if div > 1 else "")

    # ---------- переменный ток и ВЧ (до max_freq_Hz) ----------

    def cap_model(self, ref, w):
        """Реальный конденсатор: Z = ESR + jωESL + 1/(jωC)."""
        comp = self.c.comps[ref]
        C, _ = dg.parse_value(comp.get("value"))
        if not C:
            return None
        if "+" in self.c.pins(ref):                         # электролит
            esr, esl = 0.15 / (2 * math.pi * 100 * C), 15e-9  # tgδ ≈ 0,15 на 100 Гц
        else:
            esr, esl = 0.01, 2e-9
        return esr + 1j * w * esl + 1 / (1j * w * C)

    def ac_transfer(self, src, f):
        """Комплексные потенциалы узлов при 1 В переменного на цепи src; остальные
        источники питания — короткое замыкание по переменному току."""
        c = self.c
        w = 2 * math.pi * f
        acgnd = {n for n in self.fixed if n != src}
        free = [n for n in c.nets if n not in acgnd and n != src]
        idx = {n: i for i, n in enumerate(free)}
        N = len(free)
        if not N:
            return {}
        Y = np.zeros((N, N), complex)
        I = np.zeros(N, complex)

        def stamp(a, b, y):
            for n1, n2 in ((a, b), (b, a)):
                if n1 in idx:
                    i = idx[n1]
                    Y[i, i] += y
                    if n2 in idx:
                        Y[i, idx[n2]] -= y
                    elif n2 == src:
                        I[i] += y

        for i in range(N):
            Y[i, i] += 1e-12
        for ref, comp in c.comps.items():
            t, p = comp.get("type"), c.pins(ref)
            if t not in ("R", "C", "L", "F") or len(p) != 2:
                continue
            a, b = list(p.values())
            v, _ = dg.parse_value(comp.get("value"))
            if t == "R" and v:
                stamp(a, b, 1 / v)
            elif t == "C":
                z = self.cap_model(ref, w)
                if z:
                    stamp(a, b, 1 / z)
            elif t == "L":
                stamp(a, b, 1 / (1j * w * v + 0.01) if v else 1e3)
            elif t == "F":
                stamp(a, b, 1e3)
        try:
            Uf = np.linalg.solve(Y, I)
        except np.linalg.LinAlgError:
            return {}
        U = {n: Uf[idx[n]] for n in free}
        U[src] = 1.0
        return U

    def hf_checks(self):
        c = self.c
        fmax = float(self.S.get("max_freq_Hz") or 1e6)
        f_sw = self.S["switch_freq_Hz"]
        # 1. собственный резонанс конденсаторов шины: выше него конденсатор — индуктивность
        done = set()
        for ref, comp in c.comps.items():
            if "bus_cap" not in c.roles[ref] or "+" not in c.pins(ref):
                continue
            net = c.pins(ref)["+"]
            midpoint = any(c.comps[r].get("type") == "C" and c.pins(r).get("-") == net for r in c.neighbors(net))
            if net in done or net not in self.fixed or midpoint:  # средняя точка — не шина
                continue
            done.add(net)
            C, _ = dg.parse_value(comp.get("value"))
            if not C:
                continue
            srf = 1 / (2 * math.pi * math.sqrt(15e-9 * C))
            film = [r for r in c.neighbors(net) if c.comps[r].get("type") == "C" and "+" not in c.pins(r)
                    and any(n.upper() in core.GROUND_NAMES for n in c.pins(r).values())]
            fl = ""
            if film:
                Cf, _ = dg.parse_value(c.comps[film[0]].get("value"))
                if Cf:
                    fl = (f"; плёночный {film[0]} {fmt(Cf, 'Ф')}: fрез = 1/(2π√(2 нГн·C)) = "
                          f"{fmt(1 / (2 * math.pi * math.sqrt(2e-9 * Cf)), 'Гц')}")
            self.add("средний" if not film else "информация", [ref] + film, f"Шина {net}: конденсаторы на ВЧ",
                     f"fрез = 1/(2π√(ESL·C)) = 1/(2π√(15 нГн·{fmt(C, 'Ф')})) = {fmt(srf, 'Гц')}{fl}",
                     "выше fрез электролит работает как индуктивность. "
                     + ("Плёночного конденсатора рядом нет — ВЧ-ток переключения идёт через выводы, на ключах выбросы."
                        if not film else "Фронты шунтирует плёночный конденсатор — правильно, если он стоит у самих ключей."))
        # 2. LC-резонансы
        seen = set()
        for ref, comp in c.comps.items():
            if comp.get("type") != "L":
                continue
            L, _ = dg.parse_value(comp.get("value"))
            if not L:
                continue
            for n in c.pins(ref).values():
                for r in c.neighbors(n):
                    if c.comps[r].get("type") == "C" and (ref, r) not in seen:
                        seen.add((ref, r))
                        C, _ = dg.parse_value(c.comps[r].get("value"))
                        if C:
                            f0 = 1 / (2 * math.pi * math.sqrt(L * C))
                            self.add("информация", [ref, r], f"LC-контур {ref}–{r}",
                                     f"f₀ = 1/(2π√(LC)) = {fmt(f0, 'Гц')}; ρ = √(L/C) = {fmt(math.sqrt(L / C), 'Ом')}",
                                     "резонансный подъём на f₀; для фильтра сопротивление нагрузки должно быть ≈ ρ"
                                     + ("" if f0 <= fmax else " (выше рассматриваемого диапазона)"))
        # 3. полоса ОУ: GBW и скорость нарастания
        for f_ in self.findings:
            if not f_["refs"] or "K = −" not in f_["formula"]:
                continue
            name, d = part(c.comps.get(f_["refs"][0], {}))
            if not d or "gbw" not in d:
                continue
            m = re.search(r"= (-?\d+(?:\.\d+)?)\s*(?:;|$)", f_["formula"].split(";")[0] + ";")
            if not m:
                continue
            K = abs(float(m.group(1)))
            bw = d["gbw"] / (1 + K)
            fpb = d["sr"] / (2 * math.pi * 10)
            f_["formula"] += (f"; полоса = GBW/(1+|K|) = {fmt(d['gbw'], 'Гц')}/{1 + K:.0f} = {fmt(bw, 'Гц')}; "
                              f"SR/(2π·10 В) = {fmt(fpb, 'Гц')}")
            if f_sw and bw < 10 * f_sw:
                self.add("средний", f_["refs"][:1], f"{f_['refs'][0]}: узкая полоса для сигнала с ключей",
                         f"полоса {fmt(bw, 'Гц')} < 10·fШИМ = {fmt(10 * f_sw, 'Гц')}",
                         "фронты тока сглаживаются, защита по току срабатывает с задержкой")
        # 4. быстродействие драйверов, оптопар, логики против частоты
        for ref, comp in c.comps.items():
            name, d = part(comp)
            if not d or not f_sw:
                continue
            if d.get("tpd"):
                share = d["tpd"] * f_sw * 100
                if share > 2:
                    self.add("средний" if share > 5 else "информация", [ref], f"{ref} {name}: задержка на частоте ШИМ",
                             f"tзд·f = {fmt(d['tpd'], 'с')}·{fmt(f_sw, 'Гц')} = {share:.1f} % периода",
                             "разброс задержек между верхним и нижним каналом съедает мёртвое время — риск сквозного тока")
            if d.get("fmax") and f_sw > 0.5 * d["fmax"]:
                self.add("средний", [ref], f"{ref}: работа близко к предельной частоте",
                         f"f = {fmt(f_sw, 'Гц')}, fмакс ≈ {fmt(d['fmax'], 'Гц')}", "фронты затягиваются, возможны сбои")
        # 5. звон в цепи затвора: Lконт и Ciss
        for ref, comp in c.comps.items():
            name, d = part(comp)
            if not d or not d.get("ciss") or "power_switch" not in c.roles[ref]:
                continue
            Lg = 30e-9
            f_r = 1 / (2 * math.pi * math.sqrt(Lg * d["ciss"]))
            rcrit = 2 * math.sqrt(Lg / d["ciss"])
            g = c.pins(ref).get("G")
            if any(c.comps[r].get("type") == "X" for r in c.neighbors(g or "")):
                continue                                          # затвор уходит в неразобранный блок
            rg = [r for r in c.neighbors(g or "") if c.comps[r].get("type") == "R" and r != ref]
            Rg = dg.parse_value(c.comps[rg[0]].get("value"))[0] if rg else None
            ok = Rg is not None and Rg >= rcrit
            self.add("информация" if ok else "средний", [ref] + rg[:1], f"{ref}: контур затвора",
                     f"f = 1/(2π√(Lконт·Ciss)) = 1/(2π√(30 нГн·{fmt(d['ciss'], 'Ф')})) = {fmt(f_r, 'Гц')}; "
                     f"для апериодичности Rз ≥ 2√(L/Ciss) = {fmt(rcrit, 'Ом')}" + (f", стоит {fmt(Rg, 'Ом')}" if Rg else ""),
                     "звон погашен" if ok else "без достаточного Rз затвор звенит, ключ может ложно открыться")
        # 6. скин-эффект
        lt = [r for r, comp in c.comps.items() if comp.get("type") in ("L", "T")]
        if lt:
            fs = sorted({f for f in (f_sw, fmax) if f})
            self.add("информация", lt, "Скин-эффект в обмотках (медь)",
                     "δ = 66/√f мм; " + "; ".join(f"при {fmt(f, 'Гц')}: δ = {66 / math.sqrt(f):.3f} мм" for f in fs),
                     "провод толще 2δ на этой частоте работает только поверхностью и греется; на ВЧ — литцендрат или фольга")
        # 7. АЧХ: помеха с силовых шин в сигнальные выходы
        ports = [n for n, m in c.meta.items() if m.get("port") and n not in self.fixed and n != "PE"
                 and not n.endswith("_IN")
                 and not re.search(r"фаза|выход|корпус|сеть|земл|питание", str(m.get("label", "")).lower())]
        srcs = sorted((n for n, v in self.fixed.items() if abs(v) >= 50), key=lambda n: -abs(self.fixed[n]))
        freqs = [f for f in (1e3, 1e4, 1e5, 1e6) if f <= fmax]
        for s_ in srcs[:2]:
            for pnet in ports:
                hs = []
                for f in freqs:
                    U = self.ac_transfer(s_, f)
                    if pnet in U:
                        hs.append((f, abs(U[pnet])))
                if hs and max(h for _, h in hs) > 1e-3:
                    worst = max(hs, key=lambda x: x[1])
                    self.add("средний" if worst[1] > 0.01 else "информация", [], f"Помеха с {s_} в {pnet}",
                             "|H(f)| = " + ", ".join(f"{fmt(f, 'Гц')}: {20 * math.log10(max(h, 1e-12)):.0f} дБ" for f, h in hs),
                             f"1 В помехи на {s_} даёт {fmt(worst[1], 'В')} в {pnet} на {fmt(worst[0], 'Гц')}"
                             + ("; поставьте RC-фильтр перед входом АЦП" if worst[1] > 0.01 else ""))

    # ---------- КИП: токовая петля, термодатчики, микроконтроллер ----------

    def _val(self, ref):
        return dg.parse_value(self.c.comps[ref].get("value"))[0]

    def _mcu(self):
        for ref, comp in self.c.comps.items():
            if comp.get("type") == "MCU":
                v = str(comp.get("value", "")).upper()
                name, d = parts.lookup(comp.get("value", ""))
                if d and d.get("kind") == "MCU":
                    return ref, name, d
                return ref, comp.get("value", "МК"), dict(vcc=3.3, i_pin=8e-3, i_abs=20e-3, adc_bits=12,
                                                         adc_rmax=10e3, c_adc=10e-12)
        return None, None, None

    def loop_4_20(self):
        """Токовая петля 4–20 мА: нагрузка, запас по напряжению, разрешение АЦП, NAMUR NE43."""
        c = self.c
        mref, mname, md = self._mcu()
        vref = self.S["adc_vref_V"] or (md or {}).get("vcc", 3.3)
        for ref, comp in c.comps.items():
            if "burden" not in c.roles[ref]:
                continue
            Rb = self._val(ref)
            if not Rb:
                continue
            top = [n for n in c.pins(ref).values() if n.upper() not in core.GROUND_NAMES]
            if not top:
                continue
            node = top[0]
            sens = [r for r in c.comps if c.comps[r].get("type") == "SENSOR" and node in c.pins(r).values()]
            vs = self.S["loop_supply_V"]
            for s in sens:
                for n in c.pins(s).values():
                    if n != node and c.net_v(n):
                        vs = c.net_v(n)
            vs = vs or 24.0
            rw, rbar = self.S["wire_ohm"], self.S["barrier_ohm"]
            vtx = vs - 0.022 * (Rb + rw + rbar)          # при 22 мА (верх с запасом NAMUR)
            rmax = (vs - self.S["tx_min_V"]) / 0.022 - rw - rbar
            u20, u4 = 0.020 * Rb, 0.004 * Rb
            lsb = vref / 2 ** (md or {}).get("adc_bits", 12)
            res = lsb / Rb
            refs = [ref] + sens
            f = (f"U = I·Rн: 4 мА → {u4:.2f} В, 20 мА → {u20:.2f} В; NAMUR NE43: < 3,6 мА ({0.0036 * Rb:.2f} В) и "
                 f"> 21 мА ({0.021 * Rb:.2f} В) — авария датчика; "
                 f"Uдатч = Uпит − I·(Rн+Rлин+Rбар) = {vs:g} − 0,022·({Rb:g}+{rw:g}+{rbar:g}) = {vtx:.1f} В "
                 f"(нужно ≥ {self.S['tx_min_V']:g} В); Rмакс петли = (Uпит − Uмин)/22 мА = {rmax:.0f} Ом")
            concl = []
            lvl = "информация"
            if vtx < self.S["tx_min_V"]:
                lvl = "высокий"
                concl.append("датчику не хватает напряжения у верхней границы — показания «упрутся» ниже 20 мА")
            if 0.021 * Rb > vref:
                lvl = "высокий" if lvl == "информация" else lvl
                concl.append(f"21 мА дают {0.021 * Rb:.2f} В > Uоп АЦП {vref:g} В — верх шкалы и аварию не различить; "
                             f"нужен Rн ≤ {vref / 0.021:.0f} Ом или делитель")
            if mref:
                f += f"; разрешение: Uоп/2ⁿ/Rн = {vref:g}/{2 ** md.get('adc_bits', 12)}/{Rb:g} = {res * 1e6:.1f} мкА на шаг АЦП"
                concl.append(f"{0.016 / res:.0f} шагов АЦП на 16 мА ({100 * res / 0.016:.3f} % шкалы на шаг)")
            pb = (vs / Rb) ** 2 * Rb                                # линия закорочена: весь Uпит на Rн
            prated, _ = power_rating(c.comps[ref])
            prot = [r for r in c.neighbors(node) if "loop_protection" in c.roles.get(r, [])]
            f += f"; при КЗ линии: I = Uпит/Rн = {vs / Rb * 1000:.0f} мА, P = U²/R = {fmt(pb, 'Вт')}"
            if pb > prated and not any(c.comps[r].get("type") == "F" for r in prot):
                self.mark(ref, pb / prated)
                lvl = "высокий"
                concl.append(f"при замыкании кабеля Rн сгорит ({fmt(pb, 'Вт')} > {fmt(prated, 'Вт')}) — нужен "
                             "самовосстанавливающийся предохранитель (PTC) или ограничитель тока в петле")
            if not any(c.comps[r].get("type") in ("Z", "D") for r in c.neighbors(node)):
                concl.append("нет TVS/стабилитрона на входе: помеха с полевого кабеля попадёт прямо на АЦП")
                lvl = "средний" if lvl == "информация" else lvl
            self.add(lvl, refs, f"Токовая петля 4–20 мА на {ref} ({fmt(Rb, 'Ом')})", f,
                     "; ".join(concl) + ". Пересчёт: X = Xмин + (I − 4)/16·(Xмакс − Xмин); "
                     "у расходомера по перепаду давления Q = Qмакс·√((I − 4)/16)")

    def temp_sensors(self):
        c = self.c
        mref, mname, md = self._mcu()
        vref = self.S["adc_vref_V"] or (md or {}).get("vcc", 3.3)
        bits = (md or {}).get("adc_bits", 12)
        for ref, comp in c.comps.items():
            t = comp.get("type")
            val_s = str(comp.get("value", "")) + " " + str(comp.get("note", ""))
            if t == "RT" or re.search(r"PT ?100|PT ?1000|NTC|ТСП|ТСМ|50М|100П", val_s, re.I):
                pins = list(c.pins(ref).values())
                ser = [(r, n) for n in pins for r in c.neighbors(n) if r != ref and c.comps[r].get("type") == "R"]
                src = None
                for r, n in ser:
                    for nn in c.pins(r).values():
                        if nn != n and c.net_v(nn):
                            src = (r, c.net_v(nn))
                if re.search(r"NTC", val_s, re.I):
                    R25 = dg.parse_value(comp.get("value"))[0] or 10e3
                    m = re.search(r"B\s*=?\s*(\d{4})", val_s)
                    B = float(m.group(1)) if m else 3950.0
                    Rt = lambda T: R25 * math.exp(B * (1 / (T + 273.15) - 1 / 298.15))
                    f = (f"R(T) = R25·exp(B·(1/T − 1/298,15 К)), B = {B:.0f}: "
                         f"R(0 °C) = {fmt(Rt(0), 'Ом')}, R(25) = {fmt(Rt(25), 'Ом')}, R(85) = {fmt(Rt(85), 'Ом')}")
                    if src:
                        Rs = self._val(src[0])
                        U = lambda T: src[1] * Rt(T) / (Rt(T) + Rs)
                        sens = (U(26) - U(24)) / 2
                        f += (f"; делитель с {src[0]} {fmt(Rs, 'Ом')} от {src[1]:g} В: U(0) = {U(0):.2f} В, U(25) = {U(25):.2f} В, "
                              f"U(85) = {U(85):.2f} В; чувствительность при 25 °C {sens * 1000:.1f} мВ/°C = "
                              f"{abs(sens) / (vref / 2 ** bits):.1f} шага АЦП/°C")
                    self.add("информация", [ref] + ([src[0]] if src else []), f"{ref}: термистор NTC", f,
                             "характеристика нелинейная: пересчёт по формуле или таблице; лучшая чувствительность "
                             "около температуры, где R(T) ≈ Rпослед")
                else:
                    R0 = 1000.0 if re.search(r"1000", val_s) else 100.0
                    Rt = lambda T: R0 * (1 + PT_A * T + PT_B * T * T)
                    f = (f"R(T) = R0·(1 + A·T + B·T²), A = 3,9083·10⁻³, B = −5,775·10⁻⁷: R(0) = {R0:g} Ом, "
                         f"R(100) = {Rt(100):.2f} Ом, R(200) = {Rt(200):.2f} Ом; ≈ {R0 * 0.00385:.3f} Ом/°C")
                    concl = []
                    lvl = "информация"
                    if src:
                        Rs = self._val(src[0])
                        I = src[1] / (Rs + R0)
                        p = I * I * R0
                        dT = p * 1000 * 0.4                      # ≈ 0,4 °C/мВт (плёночный, неподвижный воздух)
                        f += (f"; ток I = U/(Rдоп + R0) = {src[1]:g}/({fmt(Rs, 'Ом')} + {R0:g}) = {fmt(I, 'А')}, "
                              f"самонагрев P = I²R = {fmt(p, 'Вт')} → ΔT ≈ 0,4 °C/мВт·P = {dT:.2f} °C; "
                              f"сигнал {I * R0 * 0.00385 * 1000:.2f} мВ/°C")
                        if I > (1.2e-3 if R0 == 100 else 0.5e-3):
                            lvl = "средний"
                            self.mark(ref, I / 1e-3)
                            concl.append("ток великоват — самонагрев завышает показания; для Pt100 берут ≤ 1 мА, Pt1000 ≤ 0,3 мА")
                    concl.append("при 2-проводном подключении каждый 1 Ом проводов = +2,6 °C (Pt100); "
                                 "нужна 3- или 4-проводная схема")
                    self.add(lvl, [ref] + ([src[0]] if src else []), f"{ref}: термосопротивление Pt{R0:g}", f, "; ".join(concl))
            if t == "TC" or re.search(r"термопар|\bTC\b|тип [KJTENSRB]\b|ТХА|ТХК", val_s, re.I):
                m = re.search(r"(?:тип|type)\s*([KJTENSRBL])\b|\b(ХА|ХК)\b|\(([KJTENSRBL])\)", val_s, re.I)
                tp = (m.group(1) or m.group(2) or m.group(3)).upper() if m else "K"
                s_uv = SEEBECK.get(tp, 41)
                tmax = 1000 if tp in "KN" else 700 if tp == "J" else 350 if tp == "T" else 1300
                umax = s_uv * 1e-6 * tmax
                gain = vref * 0.9 / umax
                self.add("информация", [ref], f"{ref}: термопара типа {tp}",
                         f"E ≈ S·(Tгор − Tхол), S ≈ {s_uv} мкВ/°C; на {tmax} °C: {fmt(umax, 'В')}; "
                         f"усиление до АЦП K = 0,9·Uоп/Emax = {gain:.0f}; 1 °C = {s_uv} мкВ — смещение ОУ 1 мВ даёт "
                         f"{1000 / s_uv:.0f} °C ошибки",
                         "обязательна компенсация холодного спая (датчик температуры у клемм), удлинительный провод "
                         "того же типа и полярности; ОУ — с малым смещением (прецизионный или специализированный)")

    def mcu_checks(self):
        c = self.c
        mref, mname, md = self._mcu()
        if not mref:
            return
        vcc = md["vcc"]
        vref = self.S["adc_vref_V"] or vcc
        pins = c.pins(mref)
        # развязка
        vccs = c.mcu_pins("vcc")
        dec = [r for n in vccs for r in c.neighbors(n) if c.comps[r].get("type") == "C"
               and any(x.upper() in core.GROUND_NAMES for x in c.pins(r).values())
               and (self._val(r) or 0) <= 1e-6]
        n_vcc = sum(1 for k in pins if re.search(r"^(VCC|VDD|AVCC|VDDA)", str(k).upper()))
        self.add("средний" if len(dec) < max(1, n_vcc) else "информация", [mref] + dec, f"{mref}: развязка питания",
                 f"керамических 100 нФ у выводов питания: {len(dec)}, выводов VCC/VDD: {n_vcc}",
                 "правило: по 100 нФ на каждый вывод питания вплотную + 4,7–10 мкФ на плату; без них МК сбоит "
                 "от помех и сбрасывается" if len(dec) < max(1, n_vcc) else "в норме")
        # входы АЦП: напряжение и сопротивление источника
        for k, n in pins.items():
            if not re.search(r"ADC|AIN|^A\d|AN\d", str(k).upper()):
                continue
            u = self.U.get(n)
            rs = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R"]
            caps = [r for r in c.neighbors(n) if c.comps[r].get("type") == "C"]
            Rsrc = min((self._val(r) or 1e9) for r in rs) if rs else None
            parts_f = []
            lvl = "информация"
            concl = []
            if u is not None:
                parts_f.append(f"U({n}) = {u:.2f} В (узловые потенциалы)")
                if u > vcc + 0.3:
                    lvl = "высокий"
                    self.mark(mref, 1.5)
                    concl.append(f"выше VCC + 0,3 В — откроются защитные диоды, вход выгорит или исказит соседние каналы")
                elif u > vref:
                    lvl = "средний"
                    concl.append(f"выше Uоп {vref:g} В — АЦП в насыщении")
            if Rsrc and Rsrc < 1e9:
                bits = md.get("adc_bits", 12)
                tsmp = Rsrc * md.get("c_adc", 10e-12) * math.log(2 ** (bits + 2))
                parts_f.append(f"Rист = {fmt(Rsrc, 'Ом')}; время выборки t ≥ R·Cвыб·ln(2^(n+2)) = "
                               f"{fmt(Rsrc, 'Ом')}·{fmt(md.get('c_adc', 10e-12), 'Ф')}·{math.log(2 ** (bits + 2)):.1f} = {fmt(tsmp, 'с')}")
                if Rsrc > md.get("adc_rmax", 10e3) and not caps:
                    lvl = "средний" if lvl == "информация" else lvl
                    concl.append(f"сопротивление источника больше рекомендуемых {fmt(md.get('adc_rmax', 10e3), 'Ом')} "
                                 "и нет конденсатора на входе — показания занижены и зависят от соседнего канала")
            for cr in caps:
                Cv = self._val(cr)
                if Cv and Rsrc:
                    parts_f.append(f"RC-фильтр: fср = 1/(2πRC) = {fmt(1 / (2 * math.pi * Rsrc * Cv), 'Гц')}, τ = {fmt(Rsrc * Cv, 'с')}")
            if md.get("adc_lin") and u and u > md["adc_lin"]:
                concl.append(f"у {mname} АЦП нелинеен выше ≈{md['adc_lin']} В")
            if parts_f:
                self.add(lvl, [mref] + rs + caps, f"{mref}.{k}: вход АЦП", "; ".join(parts_f), "; ".join(concl) or "в норме")
        # светодиоды
        for ref, comp in c.comps.items():
            if comp.get("type") != "LED":
                continue
            p = c.pins(ref)
            rr = [r for n in p.values() for r in c.neighbors(n) if c.comps[r].get("type") == "R"]
            if not rr:
                continue
            R = self._val(rr[0])
            vf = 2.0 if not re.search(r"син|бел|blue|white", str(comp.get("value", "")) + str(comp.get("note", "")), re.I) else 3.0
            src = vcc
            for n in c.pins(rr[0]).values():
                if c.net_v(n):
                    src = c.net_v(n)
            from_pin = any(n in c.pins(mref).values() for n in c.pins(rr[0]).values())
            I = max(0.0, (src - vf) / R) if R else 0
            lvl = "информация"
            concl = "в норме"
            if from_pin and I > md["i_pin"]:
                lvl = "высокий" if I > md["i_abs"] else "средний"
                self.mark(mref, I / md["i_abs"])
                concl = (f"ток вывода больше рекомендуемых {fmt(md['i_pin'], 'А')} для {mname}"
                         + (" и предельных — порт выгорит" if I > md["i_abs"] else " — уровни уплывают, МК греется"))
            self.add(lvl, [ref, rr[0]], f"{ref}: ток светодиода",
                     f"I = (U − Uпр)/R = ({src:g} − {vf:g})/{fmt(R, 'Ом')} = {fmt(I, 'А')}", concl)
        # кварц и конденсаторы нагрузки
        xt = c.mcu_pins("xtal")
        for ref, comp in c.comps.items():
            if comp.get("type") != "XTAL":
                continue
            caps = [r for n in c.pins(ref).values() for r in c.neighbors(n) if c.comps[r].get("type") == "C"]
            vals = [self._val(r) for r in caps if self._val(r)]
            m = re.search(r"CL\s*=?\s*(\d+(?:[.,]\d+)?)\s*p", str(comp.get("value", "")) + str(comp.get("note", "")), re.I)
            cl_need = float(m.group(1).replace(",", ".")) * 1e-12 if m else 18e-12
            if len(vals) >= 2:
                c1, c2 = vals[:2]
                cl = c1 * c2 / (c1 + c2) + 4e-12
                ok = abs(cl - cl_need) / cl_need < 0.3
                self.add("информация" if ok else "средний", [ref] + caps[:2], f"{ref}: нагрузка кварца",
                         f"CL = C1·C2/(C1+C2) + Cпар = {fmt(c1, 'Ф')}·{fmt(c2, 'Ф')}/(…) + 4 пФ = {fmt(cl, 'Ф')}; "
                         f"нужно {fmt(cl_need, 'Ф')}" + ("" if m else " (не указано, принято типовое 18 пФ)"),
                         "в норме" if ok else "частота уходит, генератор может не запускаться на холоде; "
                         f"C1 = C2 ≈ 2·(CL − 4 пФ) = {fmt(2 * (cl_need - 4e-12), 'Ф')}")
            elif not xt & set(c.pins(ref).values()):
                pass
        # I²C и сброс
        for n in c.mcu_pins("i2c"):
            pu = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R"
                  and any(x in c.mcu_pins("vcc") or (c.net_v(x) or 0) > 1 for x in c.pins(r).values())]
            Cb, fs = self.S["i2c_bus_F"], self.S["i2c_speed_Hz"]
            tr_max = 1e-6 if fs <= 100e3 else 300e-9
            if not pu:
                self.add("средний", [mref], f"Шина {n}: нет подтягивающего резистора",
                         "I²C — открытый сток, без подтяжки уровня «1» нет", "связи не будет или будет случайной")
                continue
            R = self._val(pu[0])
            tr = 0.8473 * R * Cb
            rmin = (vcc - 0.4) / 3e-3
            ok = tr <= tr_max and R >= rmin
            self.add("информация" if ok else "средний", pu[:1], f"Шина {n}: подтяжка {fmt(R, 'Ом')}",
                     f"tфронта = 0,8473·R·Cшины = 0,8473·{fmt(R, 'Ом')}·{fmt(Cb, 'Ф')} = {fmt(tr, 'с')} "
                     f"(норма ≤ {fmt(tr_max, 'с')} для {fmt(fs, 'Гц')}); Rмин = (VCC − 0,4)/3 мА = {fmt(rmin, 'Ом')}",
                     "в норме" if ok else "фронты затянуты или ток выше 3 мА — ошибки связи")
        for n in c.mcu_pins("rst"):
            rr = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R"]
            cc = [r for r in c.neighbors(n) if c.comps[r].get("type") == "C"]
            if rr and cc:
                tau = self._val(rr[0]) * self._val(cc[0])
                self.add("информация", rr[:1] + cc[:1], f"{mref}: цепь сброса",
                         f"τ = R·C = {fmt(tau, 'с')}", "нужно больше времени нарастания питания (обычно ≥ 1–10 мс)"
                         if tau < 1e-3 else "в норме")
            elif not rr:
                self.add("низкий", [mref], f"{mref}: вывод сброса без подтяжки",
                         "RESET «висит»", "помехи сбрасывают МК; ставят 10 кОм к VCC и 100 нФ на землю")

    # ---------- обратная связь: энкодеры, тахогенераторы, датчики Холла, делители ОС ----------

    def _divider_below(self, node):
        """Резисторы делителя от узла: верхний (к источнику) и нижний (к земле)."""
        c = self.c
        bot = [r for r in c.neighbors(node) if c.comps[r].get("type") == "R"
               and any(n.upper() in core.GROUND_NAMES for n in c.pins(r).values())]
        return bot

    def feedback(self):
        c = self.c
        S = self.S
        rpm = S["motor_rpm_max"]
        ccab = S["cable_m"] * S["cable_pF_per_m"] * 1e-12
        mref, mname, md = self._mcu()
        vref = S["adc_vref_V"] or (md or {}).get("vcc", 3.3)
        for ref, comp in c.comps.items():
            t = comp.get("type")
            vs = str(comp.get("value", "")) + " " + str(comp.get("note", ""))
            p = c.pins(ref)
            if t == "ENC":
                m = re.search(r"(\d+)\s*(?:имп|PPR|ppr|линий|i/r)", vs)
                ppr = int(m.group(1)) if m else S["encoder_ppr"]
                sig = [n for k, n in p.items() if re.fullmatch(r"[ABZ]", str(k).upper())]
                if not ppr:
                    self.add("информация", [ref], f"{ref}: энкодер", "число импульсов не задано",
                             "укажите PPR в номинале («1024 имп») или encoder_ppr в physics.json")
                    continue
                f = ppr * rpm / 60
                form = (f"f = PPR·n/60 = {ppr}·{rpm}/60 = {fmt(f, 'Гц')}; в квадратуре ×4 = {fmt(4 * f, 'Гц')} отсчётов/с; "
                        f"Cкабеля = {S['cable_m']} м·{S['cable_pF_per_m']} пФ/м = {fmt(ccab, 'Ф')}")
                lvl, concl = "информация", []
                diff = any(re.search(r"-$|N$|/", str(k)) for k in p)
                for n in sig:
                    pu = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R" and r != ref
                          and any((c.net_v(x) or 0) > 1 for x in c.pins(r).values())]
                    cf = [r for r in c.neighbors(n) if c.comps[r].get("type") == "C"]
                    if pu:
                        R = self._val(pu[0])
                        tau = R * (ccab + sum(self._val(x) or 0 for x in cf))
                        flim = 1 / (2 * 2.2 * tau) / 2.5          # фронт ≤ 20 % полупериода
                        form += f"; {n}: τ = Rподт·C = {fmt(R, 'Ом')}·{fmt(ccab, 'Ф')} = {fmt(tau, 'с')}, fмакс ≈ {fmt(flim, 'Гц')}"
                        if flim < f:
                            lvl = "средний"
                            self.mark(pu[0], 1.0)
                            concl.append(f"открытый коллектор с подтяжкой {fmt(R, 'Ом')} не тянет {fmt(f, 'Гц')} по "
                                         f"кабелю {S['cable_m']} м — фронты заваливаются, импульсы теряются (ошибка позиции, "
                                         "скачки скорости); уменьшить Rподт, взять двухтактный/RS-422 выход")
                    elif cf:
                        R2 = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R"]
                        if R2:
                            fc = 1 / (2 * math.pi * self._val(R2[0]) * self._val(cf[0]))
                            form += f"; фильтр {n}: fср = {fmt(fc, 'Гц')}"
                            if fc < 3 * f:
                                lvl = "средний"
                                concl.append(f"фильтр на {n} режет частоту энкодера (fср < 3·f) — пропуск импульсов на высоких оборотах")
                if diff:
                    term = any(c.comps[r].get("type") == "R" and len(set(c.pins(r).values()) & set(p.values())) == 2
                               for n in p.values() for r in c.neighbors(n))
                    if not term and S["cable_m"] > 5:
                        concl.append("дифференциальный выход без терминатора 120 Ом на приёмнике — отражения на длинном кабеле")
                self.add(lvl, [ref], f"{ref}: энкодер {ppr} имп/об", form, "; ".join(dict.fromkeys(concl)) or "в норме")
            elif t == "TACHO":
                m = re.search(r"(\d+(?:[.,]\d+)?)\s*[VВ]\s*/\s*(\d+)\s*(?:об|rpm)", vs, re.I)
                if not m:
                    self.add("информация", [ref], f"{ref}: тахогенератор", "крутизна не указана",
                             "укажите в номинале, например «20V/1000rpm»")
                    continue
                ke = float(m.group(1).replace(",", ".")) / float(m.group(2))
                umax = ke * rpm
                fr = S["tacho_ripple_per_rev"] * rpm / 60
                form = f"U = Kт·n = {ke * 1000:g} В/1000 об/мин · {rpm} = {umax:.1f} В; пульсации f = k·n/60 = {fmt(fr, 'Гц')}"
                lvl, concl = "информация", []
                for n in p.values():
                    if n.upper() in core.GROUND_NAMES:
                        continue
                    tops = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R"]
                    for rt in tops:
                        mid = [x for x in c.pins(rt).values() if x != n][0]
                        bot = self._divider_below(mid)
                        if not bot:
                            continue
                        Rt, Rb = self._val(rt), self._val(bot[0])
                        k = Rb / (Rt + Rb)
                        u = umax * k
                        form += f"; делитель {rt}/{bot[0]}: K = Rн/(Rв+Rн) = {k:.4f} → {u:.2f} В"
                        if u > vref:
                            lvl = "высокий"
                            self.mark(rt, 1.2)
                            concl.append(f"на максимальных оборотах {u:.2f} В > Uоп {vref:g} В — контур «ослепнет» и не ограничит скорость")
                        caps = [x for x in c.neighbors(mid) if c.comps[x].get("type") == "C"]
                        if caps:
                            Rth = Rt * Rb / (Rt + Rb)
                            fc = 1 / (2 * math.pi * Rth * self._val(caps[0]))
                            form += f"; фильтр fср = 1/(2π·(Rв‖Rн)·C) = {fmt(fc, 'Гц')}"
                            if fc > fr / 3:
                                concl.append("фильтр слабо давит коллекторные пульсации — скорость «дрожит»")
                            bw = S["loop_bw_Hz"]
                            if bw:
                                lag = math.degrees(math.atan(bw / fc))
                                form += f"; запаздывание на частоте среза контура {fmt(bw, 'Гц')}: φ = arctg(fк/fср) = {lag:.0f}°"
                                if lag > 20:
                                    lvl = "средний" if lvl == "информация" else lvl
                                    concl.append("фильтр отнимает запас по фазе — возможны автоколебания скорости")
                        else:
                            concl.append("нет фильтрующего конденсатора — пульсации тахогенератора идут прямо в регулятор")
                        concl.append("при реверсе напряжение меняет знак — на однополярный АЦП нужен сдвиг уровня или защита диодами")
                self.add(lvl, [ref], f"{ref}: тахогенератор", form, "; ".join(dict.fromkeys(concl)) or "в норме")
            elif t == "HALL":
                out = p.get("OUT") or next((n for k, n in p.items() if str(k).upper() in ("OUT", "Q", "SIG")), None)
                if not out:
                    continue
                pu = [r for r in c.neighbors(out) if c.comps[r].get("type") == "R"]
                if not pu:
                    self.add("средний", [ref], f"{ref}: выход датчика без подтяжки",
                             "у датчиков Холла и индуктивных NPN — открытый коллектор",
                             "без резистора к питанию сигнал «плавает» — ложные срабатывания")
        # микросхемы с обратной связью по напряжению
        for ref, comp in c.comps.items():
            v = str(comp.get("value", "")).upper()
            name, dd = parts.lookup(comp.get("value", ""))
            if not dd or "fb_pin" not in dd or "vref" not in dd:
                continue
            vr, pin = dd["vref"], dd["fb_pin"]
            kind = "lm317" if dd["kind"] == "REG_ADJ" else "top"
            p = c.pins(ref)
            fb = next((n for k, n in p.items() if str(k).upper() == pin), None)
            if not fb:
                continue
            rs = [r for r in c.neighbors(fb) if c.comps[r].get("type") == "R"]
            bot = [r for r in rs if any(n.upper() in core.GROUND_NAMES for n in c.pins(r).values())]
            top = [r for r in rs if r not in bot]
            if not top or not bot:
                continue
            Rt, Rb = self._val(top[0]), self._val(bot[0])
            if kind == "lm317":
                vout = vr * (1 + Rb / Rt) + 50e-6 * Rb
                form = f"Uвых = 1,25·(1 + R2/R1) + Iadj·R2 = 1,25·(1 + {fmt(Rb, 'Ом')}/{fmt(Rt, 'Ом')}) + 50 мкА·{fmt(Rb, 'Ом')} = {vout:.2f} В"
            else:
                vout = vr * (1 + Rt / Rb)
                form = f"Uвых = Uоп·(1 + Rв/Rн) = {vr}·(1 + {fmt(Rt, 'Ом')}/{fmt(Rb, 'Ом')}) = {vout:.2f} В"
            for r in top + bot:
                self.c.roles[r] = list(dict.fromkeys(self.c.roles[r] + ["feedback"]))
            outnet = [n for n in c.pins(top[0]).values() if n != fb]
            lbl = label = c.meta.get(outnet[0], {}).get("label", outnet[0]) if outnet else ""
            ex = dg.label_voltage(lbl) if outnet else None
            lvl = "информация"
            concl = ("обрыв верхнего резистора ОС уводит выход в минимум, нижнего — в максимум "
                     "(перенапряжение на нагрузке); их проверяют первыми при «завышенном/плавающем» выходе")
            if ex and abs(ex - vout) / ex > 0.05:
                lvl = "средний"
                concl = f"расчёт {vout:.2f} В расходится с меткой «{lbl}» более чем на 5 % — проверьте номиналы; " + concl
            self.add(lvl, [ref] + top[:1] + bot[:1], f"{ref} {name}: обратная связь по напряжению", form, concl)

    def tvs_checks(self):
        """TVS-диод: рабочее напряжение против нормального уровня сигнала и напряжение
        ограничения против того, что выдерживает защищаемый вход."""
        c = self.c
        mref, mname, md = self._mcu()
        mcu_nets = {n for n in c.pins(mref).values() if n.upper() not in core.GROUND_NAMES
                    and n not in c.mcu_pins("vcc")} if mref else set()
        for ref, comp in c.comps.items():
            name, d = parts.lookup(comp.get("value", ""))
            if not d or d.get("kind") != "TVS" or not d.get("vwm"):
                continue
            nets = [n for n in c.pins(ref).values() if n.upper() not in core.GROUND_NAMES]
            if not nets:
                continue
            n = nets[0]
            form = (f"{comp.get('value')}: Uраб = {d['vwm']:g} В, пробой ≈ {d['vbr']:g} В, ограничение при импульсе "
                    f"≈ {d['vc']:g} В, Pимп = {d['ppk']} Вт (10/1000 мкс)")
            concl, lvl = [], "информация"
            umax = None
            if "burden" in " ".join(r for x in c.neighbors(n) for r in c.roles.get(x, [])):
                rb = [self._val(x) for x in c.neighbors(n) if "burden" in c.roles.get(x, [])]
                umax = 0.021 * rb[0] if rb and rb[0] else None
            if umax and umax > d["vwm"]:
                lvl = "средний"
                concl.append(f"нормальный сигнал до {umax:.2f} В (21 мА·Rн) выше Uраб — у верхней границы TVS даёт утечку и искажает показания")
            # путь до вывода МК: напрямую или через один резистор
            direct = n in mcu_nets
            via = [r for r in c.neighbors(n) if c.comps[r].get("type") == "R" and set(c.pins(r).values()) & mcu_nets]
            if md and (direct or via):
                lim = md["vcc"] + 0.3
                if d["vc"] > lim:
                    if direct:
                        lvl = "высокий"
                        concl.append(f"ограничивает на {d['vc']:g} В, а вход {mname} выдерживает {lim:.1f} В — TVS вход не защитит")
                    else:
                        R = self._val(via[0])
                        i = (d["vc"] - lim) / R if R else 0
                        form += f"; ток в защитный диод МК через {via[0]}: I = (Uогр − (VCC+0,3))/R = ({d['vc']:g} − {lim:.1f})/{fmt(R, 'Ом')} = {fmt(i, 'А')}"
                        if i > 5e-3:
                            lvl = "средний" if lvl == "информация" else lvl
                            concl.append("ток в защитный диод МК больше ~5 мА — нужен резистор больше или диоды Шоттки на питание")
                        else:
                            concl.append("последовательный резистор ограничивает ток во внутренние диоды МК — допустимо")
            self.add(lvl, [ref], f"{ref}: TVS-защита", form, "; ".join(concl) or "в норме")

    def mcu_pin_levels(self):
        """Любой вывод МК (кроме питания) под напряжением выше VCC + 0,3 В — выгорит."""
        mref, mname, md = self._mcu()
        if not mref:
            return
        for k, n in self.c.pins(mref).items():
            if re.search(r"^(VCC|VDD|AVCC|VDDA|VBAT|GND|VSS|ADC|AIN|AN\d|A\d)", str(k).upper()):
                continue
            u = self.U.get(n)
            if u is not None and u > md["vcc"] + 0.3:
                src = [r for r in self.c.neighbors(n) if r != mref]
                self.mark(mref, u / md["vcc"])
                self.add("высокий", [mref] + src, f"{mref}.{k}: перенапряжение на выводе",
                         f"U({n}) = {u:.2f} В > VCC + 0,3 = {md['vcc'] + 0.3:.1f} В",
                         "открываются защитные диоды, ток идёт в питание МК — вывод выгорает, МК сбоит. "
                         "Нужен делитель, оптрон или транзисторный ключ на уровень МК")

    def cap_life(self):
        ta = self.S["ambient_C"]
        L0, tmax = self.S["cap_life_h"], self.S["cap_tmax_C"]
        refs = [r for r, c in self.c.comps.items() if c.get("type") == "C" and "+" in self.c.pins(r)]
        if not refs:
            return
        L = L0 * 2 ** ((tmax - ta) / 10)
        self.add("информация", refs, "Ресурс электролитических конденсаторов",
                 f"L = L₀·2^((Tmax − T)/10) = {L0}·2^(({tmax} − {ta})/10) = {L:,.0f} ч ≈ {L / 8760:.1f} года непрерывной работы".replace(",", " "),
                 "каждые +10 °C делят ресурс вдвое; нагрев током пульсаций сокращает его ещё — электролиты стареют первыми")

    def run(self):
        self.solve_dc()
        self.check_resistors()
        self.check_diodes()
        self.report_dividers()
        caps = self.bus_caps()
        self.precharge(caps)
        self.transformer_supply()
        self.gate_drive()
        self.opamp_stages()
        self.oscillators()
        self.hf_checks()
        self.loop_4_20()
        self.temp_sensors()
        self.mcu_checks()
        self.feedback()
        self.tvs_checks()
        self.mcu_pin_levels()
        self.cap_life()
        order = {"высокий": 0, "средний": 1, "низкий": 2, "информация": 3}
        self.findings.sort(key=lambda f: order[f["level"]])
        return {"findings": self.findings, "stress": self.stress, "settings": self.S,
                "stress_sym": {k: sorted(v) for k, v in self.stress_sym.items()},
                "node_voltages": {n: round(v, 3) for n, v in sorted(self.U.items()) if n not in self.fixed}}


def analyze(project):
    project = Path(project)
    nl = core.load_json(project / "netlist.json")
    settings_path = project / "physics.json"
    if not settings_path.exists():
        settings_path.write_text(json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=1), encoding="utf-8")
    settings = core.load_json(settings_path) or {}
    return Physics(nl, settings).run()


def to_markdown(res):
    md = ["## Расчёты", ""]
    for f in res["findings"]:
        md.append(f"- **[{f['level']}] {f['title']}** {', '.join(f['refs'][:6])}  ")
        md.append(f"  `{f['formula']}`  ")
        if f["conclusion"]:
            md.append(f"  {f['conclusion']}")
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    r = analyze(sys.argv[1])
    print(to_markdown(r))
