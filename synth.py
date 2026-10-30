"""Генератор синтетических схем с точным ответом — «учебник» для анализатора.

Каждая схема рисуется со случайным стилем (толщина линий, размер точек,
диоды с проводом насквозь, толстые обкладки, Т-соединения без точки) и затем
портится как скан: наклон, уменьшение, размытие, шум, JPEG, цвет бумаги.
Ответ — координаты выводов каждой детали и номер цепи каждого вывода.

    python synth.py папка [сколько=5]   — сохранить примеры (png + json) посмотреть глазами
"""
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np

INK = 0


class Sheet:
    def __init__(self, rng, w=1700, h=1200):
        self.rng = rng
        self.W, self.H = w, h
        self.img = np.full((h, w), 255, np.uint8)
        t = rng.choice([2, 2, 3, 3, 4])
        self.t = t
        self.dot_r = max(2, int(round(t * rng.uniform(1.1, 2.2))))   # как в книгах: от мелких до крупных
        self.diode_through = rng.random() < 0.5
        self.thick_caps = rng.random() < 0.5
        self.t_dot_prob = rng.uniform(0.55, 1.0)
        self.font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX])
        self.fscale = rng.uniform(0.55, 0.85) if self.font != cv2.FONT_HERSHEY_PLAIN else rng.uniform(1.1, 1.5)
        self.zigzag = rng.random() < 0.4            # резисторы «по-американски»
        self.comps = []          # {id, type, box, pins:[{name, x, y, bx, by}]}
        self.segs = []           # (x0, y0, x1, y1, net)

    # --- рисование ---
    def line(self, a, b, t=None):
        cv2.line(self.img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), INK, t or self.t, cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1, fill=False):
        cv2.rectangle(self.img, (int(x0), int(y0)), (int(x1), int(y1)), INK, -1 if fill else self.t, cv2.LINE_AA)

    def text(self, s, x, y):
        cv2.putText(self.img, s, (int(x), int(y)), self.font, self.fscale, INK, max(1, self.t - 1), cv2.LINE_AA)

    def text_size(self, s):
        (w, h), _ = cv2.getTextSize(s, self.font, self.fscale, max(1, self.t - 1))
        return w, h


def rot(p, c, vertical):
    """Повернуть точку p вокруг c на 90°, если деталь вертикальная."""
    if not vertical:
        return p
    dx, dy = p[0] - c[0], p[1] - c[1]
    return (c[0] - dy, c[1] + dx)


def two_pin(sh, kind, cx, cy, vertical, cid):
    """Двухвыводная деталь. Возвращает (бокс тела, выводы)."""
    t = sh.t
    c = (cx, cy)
    L = lambda p: rot(p, c, vertical)
    segs = []
    if kind == "R":
        bw, bh, lead = 44, 16, 22
        if sh.zigzag:
            n = 6
            pts = [(cx - bw / 2, cy)] + [(cx - bw / 2 + bw * (i + 0.5) / n, cy + (-1) ** i * bh / 2)
                                         for i in range(n)] + [(cx + bw / 2, cy)]
            for a, b in zip(pts, pts[1:]):
                sh.line(L(a), L(b))
        else:
            body = [(cx - bw / 2, cy - bh / 2), (cx + bw / 2, cy + bh / 2)]
            a, b = L(body[0]), L(body[1])
            sh.rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        ends = [(cx - bw / 2, cy), (cx + bw / 2, cy)]
        outs = [(cx - bw / 2 - lead, cy), (cx + bw / 2 + lead, cy)]
    elif kind == "C":
        gap, pl, lead = 10, 30, 30
        th = t * 3 if sh.thick_caps else t
        for sgn in (-1, 1):
            x = cx + sgn * gap / 2
            sh.line(L((x, cy - pl / 2)), L((x, cy + pl / 2)), th)
        ends = [(cx - gap / 2, cy), (cx + gap / 2, cy)]
        outs = [(cx - gap / 2 - lead, cy), (cx + gap / 2 + lead, cy)]
    elif kind == "D":
        w, h, lead = 20, 22, 26
        tri = np.array([L((cx - w / 2, cy - h / 2)), L((cx - w / 2, cy + h / 2)), L((cx + w / 2, cy))], np.int32)
        cv2.polylines(sh.img, [tri], True, INK, t, cv2.LINE_AA)
        sh.line(L((cx + w / 2, cy - h / 2)), L((cx + w / 2, cy + h / 2)), t + 1)
        if sh.diode_through:
            sh.line(L((cx - w / 2, cy)), L((cx + w / 2, cy)))
        ends = [(cx - w / 2, cy), (cx + w / 2, cy)]
        outs = [(cx - w / 2 - lead, cy), (cx + w / 2 + lead, cy)]
    else:  # "L"
        r, n, lead = 7, 4, 14
        x0 = cx - n * r
        for i in range(n):
            ctr = L((x0 + r + 2 * r * i, cy))
            a0 = 180 if not vertical else 90
            cv2.ellipse(sh.img, (int(ctr[0]), int(ctr[1])), (r, r), 0, a0, a0 + 180, INK, t, cv2.LINE_AA)
        ends = [(x0, cy), (x0 + 2 * r * n, cy)]
        outs = [(x0 - lead, cy), (x0 + 2 * r * n + lead, cy)]
    pins = []
    for k in range(2):
        e, o = L(ends[k]), L(outs[k])
        sh.line(e, o)
        pins.append({"name": str(k + 1), "x": o[0], "y": o[1], "bx": e[0], "by": e[1]})
    xs = [p["bx"] for p in pins] + [cx - 18, cx + 18]
    ys = [p["by"] for p in pins] + [cy - 18, cy + 18]
    box = [min(xs), min(ys), max(xs), max(ys)]
    return box, pins


def transistor(sh, cx, cy, mirror):
    r = 24
    cv2.circle(sh.img, (cx, cy), r, INK, sh.t, cv2.LINE_AA)
    s = -1 if mirror else 1
    bx = cx - s * 8
    sh.line((bx, cy - 13), (bx, cy + 13), sh.t + 1)
    pins = []
    b_out = (cx - s * (r + 22), cy)
    sh.line((bx, cy), b_out)
    pins.append({"name": "B", "x": b_out[0], "y": b_out[1], "bx": cx - s * r, "by": cy})
    cxp = (cx + s * 10, cy - r - 22)
    sh.line((bx, cy - 6), (cx + s * 10, cy - 16))
    sh.line((cx + s * 10, cy - 16), cxp)
    pins.append({"name": "C", "x": cxp[0], "y": cxp[1], "bx": cx + s * 10, "by": cy - r + 2})
    ep = (cx + s * 10, cy + r + 22)
    sh.line((bx, cy + 6), (cx + s * 10, cy + 16))
    tip = np.array([(cx + s * 10, cy + 16), (cx + s * 2, cy + 15), (cx + s * 7, cy + 8)], np.int32)
    cv2.fillPoly(sh.img, [tip], INK, cv2.LINE_AA)
    sh.line((cx + s * 10, cy + 16), ep)
    pins.append({"name": "E", "x": ep[0], "y": ep[1], "bx": cx + s * 10, "by": cy + r - 2})
    return [cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], pins


def chip(sh, cx, cy, n):
    half = n // 2
    w, h = 76, 24 * (half + 1)
    x0, y0 = cx - w // 2, cy - h // 2
    sh.rect(x0, y0, x0 + w, y0 + h)
    if sh.rng.random() < 0.4:                 # отсеки, как в отечественных УГО
        sh.line((x0 + w * 0.3, y0), (x0 + w * 0.3, y0 + h))
    pins = []
    for i in range(n):
        left = i < half
        y = y0 + 24 * ((i % half) + 1)
        xe = x0 if left else x0 + w
        xo = xe - 24 if left else xe + 24
        sh.line((xe, y), (xo, y))
        pins.append({"name": str(i + 1), "x": xo, "y": y, "bx": xe, "by": y})
    return [x0, y0, x0 + w, y0 + h], pins


def opamp(sh, cx, cy, mirror):
    """Треугольник ОУ: входы «−» и «+» слева, выход справа (или зеркально)."""
    w, h = 64, 60
    s_ = -1 if mirror else 1
    xl, xr = cx - s_ * w / 2, cx + s_ * w / 2
    tri = np.array([(xl, cy - h / 2), (xl, cy + h / 2), (xr, cy)], np.int32)
    cv2.polylines(sh.img, [tri], True, INK, sh.t, cv2.LINE_AA)
    sh.line((xl + s_ * 6, cy - 15), (xl + s_ * 14, cy - 15))
    sh.line((xl + s_ * 6, cy + 15), (xl + s_ * 14, cy + 15))
    sh.line((xl + s_ * 10, cy + 11), (xl + s_ * 10, cy + 19))
    pins = []
    for name, y in (("-", cy - 15), ("+", cy + 15)):
        o = (xl - s_ * 24, y)
        sh.line((xl, y), o)
        pins.append({"name": name, "x": o[0], "y": o[1], "bx": xl, "by": y})
    o = (xr + s_ * 24, cy)
    sh.line((xr, cy), o)
    pins.append({"name": "out", "x": o[0], "y": o[1], "bx": xr, "by": cy})
    return [min(xl, xr), cy - h / 2, max(xl, xr), cy + h / 2], pins


# ---------- КИП и микроконтроллеры ----------

def sensor(sh, cx, cy, rng):
    """Датчик по ISA/ГОСТ: круг с буквенным кодом (PT, TT, FT, LT, PDT…) и двумя выводами +/−."""
    r = 26
    cv2.circle(sh.img, (cx, cy), r, INK, sh.t, cv2.LINE_AA)
    if rng.random() < 0.5:                                  # полевой прибор: черта посередине — щитовой
        sh.line((cx - r, cy), (cx + r, cy), max(1, sh.t - 1))
    code = rng.choice(["PT", "TT", "FT", "LT", "PDT", "TE", "PIT", "FIT", "TIC", "AT"])
    tw, th = sh.text_size(code)
    sh.text(code, cx - tw / 2, cy - 4 if rng.random() < 0.5 else cy + th / 2)
    pins = []
    for name, dy in (("+", -1), ("-", 1)):
        e = (cx + r * 0.7, cy + dy * r * 0.7)
        o = (cx + r + 26, cy + dy * r * 0.7)
        sh.line(e, o)
        pins.append({"name": name, "x": o[0], "y": o[1], "bx": e[0], "by": e[1]})
    return [cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], pins


def thermistor(sh, cx, cy, vertical, rng):
    """Терморезистор (PT100/NTC): прямоугольник, перечёркнутый наклонной линией с полочкой, «t°»."""
    box, pins = two_pin(sh, "R", cx, cy, vertical, 0) if not sh.zigzag else two_pin(sh, "R", cx, cy, vertical, 0)
    a = rot((cx - 26, cy + 16), (cx, cy), vertical)
    b = rot((cx + 22, cy - 16), (cx, cy), vertical)
    c = rot((cx + 32, cy - 16), (cx, cy), vertical)
    sh.line(a, b)
    sh.line(b, c)
    lab = rng.choice(["t°", "Pt100", "NTC", "θ"]) if rng.random() < 0.6 else ""
    if lab:
        sh.text(lab.replace("°", "o").replace("θ", "t"), cx + 14, cy + 26)
    return [min(box[0], a[0], c[0]), min(box[1], b[1]), max(box[2], c[0]), max(box[3], a[1])], pins


def thermocouple(sh, cx, cy):
    """Термопара: две линии разного вида сходятся в точку спая."""
    tip = (cx, cy + 22)
    a, b = (cx - 14, cy - 18), (cx + 14, cy - 18)
    sh.line(a, tip)
    sh.line(b, tip, sh.t + 2)                              # второй электрод — толще
    cv2.circle(sh.img, (int(tip[0]), int(tip[1])), max(3, sh.t + 1), INK, -1, cv2.LINE_AA)
    pins = []
    for name, p in (("+", a), ("-", b)):
        o = (p[0], p[1] - 24)
        sh.line(p, o)
        pins.append({"name": name, "x": o[0], "y": o[1], "bx": p[0], "by": p[1]})
    return [cx - 18, cy - 20, cx + 18, cy + 26], pins


def crystal(sh, cx, cy, vertical):
    """Кварцевый резонатор: прямоугольник между двумя пластинами."""
    L = lambda p: rot(p, (cx, cy), vertical)
    a, b = L((cx - 7, cy - 13)), L((cx + 7, cy + 13))
    sh.rect(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
    for x in (cx - 13, cx + 13):
        sh.line(L((x, cy - 16)), L((x, cy + 16)))
    pins = []
    for k, (e, o) in enumerate((((cx - 13, cy), (cx - 40, cy)), ((cx + 13, cy), (cx + 40, cy)))):
        sh.line(L(e), L(o))
        oo, ee = L(o), L(e)
        pins.append({"name": str(k + 1), "x": oo[0], "y": oo[1], "bx": ee[0], "by": ee[1]})
    return [cx - 20, cy - 20, cx + 20, cy + 20], pins


def led(sh, cx, cy, vertical):
    box, pins = two_pin(sh, "D", cx, cy, vertical, 0)
    for k in (0, 1):                                       # две стрелки излучения
        p0 = rot((cx + 2 + 8 * k, cy - 14), (cx, cy), vertical)
        p1 = rot((cx + 12 + 8 * k, cy - 26), (cx, cy), vertical)
        cv2.arrowedLine(sh.img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), INK,
                        max(1, sh.t - 1), cv2.LINE_AA, tipLength=0.4)
    return [box[0], box[1] - 12, box[2] + 10, box[3]], pins


def mcu(sh, cx, cy, rng):
    """Микроконтроллер: большой корпус с подписанными выводами (PA0, ADC1, SDA, VCC…)."""
    n = rng.choice([12, 16, 20])
    half = n // 2
    w, step = 120, 22
    h = step * (half + 1)
    x0, y0 = cx - w // 2, cy - h // 2
    sh.rect(x0, y0, x0 + w, y0 + h)
    name = rng.choice(["STM32F103", "ATmega328", "PIC16F", "ESP32", "MSP430", "AT89C2051"])
    tw, th = sh.text_size(name)
    sh.text(name, cx - tw / 2, y0 - 6)
    names = ["VCC", "GND", "RST", "XTAL1", "XTAL2", "ADC0", "ADC1", "SDA", "SCL", "TX", "RX", "PB0", "PB1",
             "PA0", "PA1", "PWM", "INT0", "MOSI", "MISO", "SCK", "AREF", "PC3"]
    rng.shuffle(names)
    pins = []
    small = sh.fscale * 0.7
    for i in range(n):
        left = i < half
        y = y0 + step * ((i % half) + 1)
        xe = x0 if left else x0 + w
        xo = xe - 24 if left else xe + 24
        sh.line((xe, y), (xo, y))
        lab = names[i % len(names)]
        (lw, lh), _ = cv2.getTextSize(lab, sh.font, small, 1)
        cv2.putText(sh.img, lab, (int(xe + 4 if left else xe - lw - 4), int(y + lh / 2)), sh.font, small, INK, 1,
                    cv2.LINE_AA)
        pins.append({"name": lab, "x": xo, "y": y, "bx": xe, "by": y})
    return [x0, y0, x0 + w, y0 + h], pins


def terminal(sh, cx, cy, rng):
    """Клеммник: столбик кружков/квадратов с номерами — сюда приходят провода от датчиков."""
    n = rng.choice([2, 3, 4])
    pins = []
    sq = rng.random() < 0.5
    for i in range(n):
        y = cy + (i - (n - 1) / 2) * 22
        if sq:
            sh.rect(cx - 7, y - 7, cx + 7, y + 7)
        else:
            cv2.circle(sh.img, (int(cx), int(y)), 7, INK, sh.t, cv2.LINE_AA)
        sh.text(str(i + 1), cx - 26, y + 5)
        o = (cx + 32, y)
        sh.line((cx + 7, y), o)
        pins.append({"name": str(i + 1), "x": o[0], "y": o[1], "bx": cx + 7, "by": y})
    return [cx - 9, cy - n * 11 - 2, cx + 9, cy + n * 11 + 2], pins


# ---------- обратная связь по скорости и положению ----------

def encoder(sh, cx, cy, rng):
    """Энкодер: корпус с диском (окружность с рисками) и выводами A, B, Z, +U, 0V."""
    w, h = 70, 96
    x0, y0 = cx - w // 2, cy - h // 2
    sh.rect(x0, y0, x0 + w, y0 + h)
    r = 18
    cv2.circle(sh.img, (cx - 8, cy), r, INK, sh.t, cv2.LINE_AA)
    for k in range(8):
        a = k * math.pi / 4
        p0 = (cx - 8 + (r - 6) * math.cos(a), cy + (r - 6) * math.sin(a))
        p1 = (cx - 8 + r * math.cos(a), cy + r * math.sin(a))
        sh.line(p0, p1, 1)
    sh.text(rng.choice(["ENC", "BQ", "BR", "ЛИР"]), x0 + 4, y0 - 6)
    names = rng.choice([["A", "B", "Z", "+U", "0V"], ["A", "B", "+5V", "GND"], ["A", "A-", "B", "B-", "+U", "0V"]])
    pins = []
    step = h / (len(names) + 1)
    small = sh.fscale * 0.7
    for i, nm in enumerate(names):
        y = y0 + step * (i + 1)
        sh.line((x0 + w, y), (x0 + w + 24, y))
        cv2.putText(sh.img, nm, (int(x0 + w - 22), int(y + 4)), sh.font, small, INK, 1, cv2.LINE_AA)
        pins.append({"name": nm, "x": x0 + w + 24, "y": y, "bx": x0 + w, "by": y})
    return [x0, y0, x0 + w, y0 + h], pins


def tacho(sh, cx, cy, rng):
    """Тахогенератор: окружность с буквами TG/BR/G и двумя щётками-выводами."""
    r = 26
    cv2.circle(sh.img, (cx, cy), r, INK, sh.t, cv2.LINE_AA)
    lab = rng.choice(["TG", "BR", "G", "ТГ"])
    tw, th = sh.text_size(lab)
    sh.text(lab, cx - tw / 2, cy + th / 2)
    pins = []
    for k, dy in enumerate((-1, 1)):
        e = (cx, cy + dy * r)
        sh.rect(cx - 5, cy + dy * r - (8 if dy < 0 else 0), cx + 5, cy + dy * r + (8 if dy > 0 else 0))
        o = (cx, cy + dy * (r + 30))
        sh.line((cx, cy + dy * (r + 8)), o)
        pins.append({"name": str(k + 1), "x": o[0], "y": o[1], "bx": e[0], "by": e[1]})
    return [cx - r - 2, cy - r - 10, cx + r + 2, cy + r + 10], pins


def hall(sh, cx, cy, rng):
    """Датчик Холла / индуктивный датчик приближения: квадрат с меткой и 3 выводами."""
    s = 40
    x0, y0 = cx - s // 2, cy - s // 2
    sh.rect(x0, y0, x0 + s, y0 + s)
    if rng.random() < 0.5:
        sh.text("H", cx - 6, cy + 6)
    else:                                                   # активная поверхность — ромб
        pts = np.array([(cx, y0 + 6), (x0 + s - 6, cy), (cx, y0 + s - 6), (x0 + 6, cy)], np.int32)
        cv2.polylines(sh.img, [pts], True, INK, max(1, sh.t - 1), cv2.LINE_AA)
    pins = []
    for k, nm in enumerate(("+U", "OUT", "0V")):
        y = y0 + 10 + k * 10
        sh.line((x0 + s, y), (x0 + s + 26, y))
        pins.append({"name": nm, "x": x0 + s + 26, "y": y, "bx": x0 + s, "by": y})
    return [x0, y0, x0 + s, y0 + s], pins


# ---------- схемы управления (ГОСТ 2.755 / IEC 60617) ----------

def ctrl(sh, kind, cx, cy, rng, vertical=False):
    """Элемент схемы управления, выводы слева/справа (или сверху/снизу при vertical).
    kind: NO, NC, BTN_NO, BTN_NC, COIL, LAMP, THERMAL, FUSE, QF."""
    c = (cx, cy)
    L = lambda p: rot(p, c, vertical)
    t = sh.t
    half = {"COIL": 22, "LAMP": 18, "THERMAL": 18, "FUSE": 22}.get(kind, 20)
    a, b = (cx - half, cy), (cx + half, cy)
    if kind in ("NO", "NC", "BTN_NO", "BTN_NC", "QF"):
        sh.line(L((cx - half - 8, cy)), L(a))
        sh.line(L(b), L((cx + half + 8, cy)))
        tip = (cx + half - 2, cy - 14)                       # подвижный контакт — наклонная черта
        sh.line(L(a), L(tip))
        if kind in ("NC", "BTN_NC"):
            sh.line(L((cx + half - 6, cy - 16)), L(b))      # упор нормально замкнутого
        if kind.startswith("BTN"):                          # толкатель кнопки (пунктир) и площадка
            for yy in range(int(cy - 9), int(cy - 30), -5):
                sh.line(L((cx - 2, yy)), L((cx - 2, yy - 2)), 1)
            sh.line(L((cx - 10, cy - 32)), L((cx + 6, cy - 32)))
            sh.line(L((cx - 10, cy - 32)), L((cx - 10, cy - 27)))
            sh.line(L((cx + 6, cy - 32)), L((cx + 6, cy - 27)))
        if kind == "QF":                                     # крестик автомата на неподвижном контакте
            sh.line(L((b[0] - 5, cy - 5)), L((b[0] + 5, cy + 5)))
            sh.line(L((b[0] - 5, cy + 5)), L((b[0] + 5, cy - 5)))
        a, b = (cx - half - 8, cy), (cx + half + 8, cy)
        box = (cx - half - 8, cy - 34, cx + half + 8, cy + 6)
    elif kind == "COIL":
        w, h = 30, 18 if rng.random() < 0.5 else 24
        p0, p1 = L((cx - w / 2, cy - h / 2)), L((cx + w / 2, cy + h / 2))
        sh.rect(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        sh.line(L(a), L((cx - w / 2, cy)))
        sh.line(L((cx + w / 2, cy)), L(b))
        box = (cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2)
    elif kind == "LAMP":
        r = 13
        cv2.circle(sh.img, (int(cx), int(cy)), r, INK, t, cv2.LINE_AA)
        d = r * 0.7
        sh.line((cx - d, cy - d), (cx + d, cy + d))
        sh.line((cx - d, cy + d), (cx + d, cy - d))
        sh.line(L(a), L((cx - r, cy)))
        sh.line(L((cx + r, cy)), L(b))
        box = (cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2)
    elif kind == "THERMAL":                                 # нагреватель теплового реле: прямоугольник со «ступенькой»
        p0, p1 = L((cx - 12, cy - 10)), L((cx + 12, cy + 10))
        sh.rect(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        pts = [a, (cx - 4, cy), (cx - 4, cy - 6), (cx + 4, cy - 6), (cx + 4, cy), b]
        for u, v in zip(pts, pts[1:]):
            sh.line(L(u), L(v))
        box = (cx - 14, cy - 12, cx + 14, cy + 12)
    else:                                                   # FUSE: прямоугольник с линией насквозь
        w, h = 30, 12
        p0, p1 = L((cx - w / 2, cy - h / 2)), L((cx + w / 2, cy + h / 2))
        sh.rect(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
        sh.line(L(a), L(b))
        box = (cx - w / 2 - 2, cy - h / 2 - 2, cx + w / 2 + 2, cy + h / 2 + 2)
    if vertical:
        q0, q1 = L(box[:2]), L(box[2:])
        box = (min(q0[0], q1[0]), min(q0[1], q1[1]), max(q0[0], q1[0]), max(q0[1], q1[1]))
    pa, pb = L(a), L(b)
    return [float(v) for v in box], [{"name": "1", "x": pa[0], "y": pa[1], "bx": pa[0], "by": pa[1]},
                                     {"name": "2", "x": pb[0], "y": pb[1], "bx": pb[0], "by": pb[1]}]


def motor(sh, cx, cy):
    r = 30
    cv2.circle(sh.img, (int(cx), int(cy)), r, INK, sh.t, cv2.LINE_AA)
    tw, th = sh.text_size("M")
    sh.text("M", cx - tw / 2, cy + th / 2)
    pins = []
    for k, dx in enumerate((-14, 0, 14)):
        top = (cx + dx, cy - math.sqrt(r * r - dx * dx))
        o = (cx + dx, cy - r - 26)
        sh.line(top, o)
        pins.append({"name": "UVW"[k], "x": o[0], "y": o[1], "bx": top[0], "by": top[1]})
    return [cx - r - 2, cy - r - 2, cx + r + 2, cy + r + 2], pins


CTRL_TYPES = {"NO": "SW_NO", "NC": "SW_NC", "BTN_NO": "BTN", "BTN_NC": "BTN", "COIL": "COIL", "LAMP": "LAMP",
              "THERMAL": "THERMAL", "FUSE": "FU", "QF": "QF"}


def ladder(sh, rng, level):
    """Лист схемы управления: две шины питания, ступени «контакты → катушка/лампа»,
    самоподхват (параллельный контакт под кнопкой «Пуск»)."""
    W, H = sh.W, sh.H
    xl, xr = 90, W - 90
    y0, y1 = 80, H - 60
    names = rng.choice([("L", "N"), ("+24V", "0V"), ("~220V", "N"), ("L1", "L2")])
    sh.line((xl, y0), (xl, y1))
    sh.line((xr, y0), (xr, y1))
    for x, nm in ((xl, names[0]), (xr, names[1])):
        tw, th = sh.text_size(nm)
        sh.text(nm, x - tw / 2, y0 - 12)
    rail_l, rail_n = 1, 2
    nid = [2]

    def new_net():
        nid[0] += 1
        return nid[0]

    n_rungs = max(1, min(2 + level * 2, int((y1 - y0) / 90)))
    step = (y1 - y0) / (n_rungs + 0.5)
    cid = 0
    for r in range(n_rungs):
        y = int(y0 + step * (r + 0.7))
        k = rng.randint(1, min(4, 1 + level))
        kinds = [rng.choice(["BTN_NC", "NC", "THERMAL", "FUSE"]) if i == 0 and rng.random() < 0.5 else
                 rng.choice(["NO", "NC", "BTN_NO", "BTN_NC", "QF"]) for i in range(k)]
        kinds.append(rng.choice(["COIL", "COIL", "COIL", "LAMP"]))
        xs = [int(xl + (xr - xl) * (i + 1) / (len(kinds) + 1)) for i in range(len(kinds))]
        prev_x, prev_net = xl, rail_l
        elems = []
        for i, (kd, x) in enumerate(zip(kinds, xs)):
            box, pins = ctrl(sh, kd, x, y, rng)
            cid += 1
            net_out = rail_n if i == len(kinds) - 1 else new_net()
            pins[0]["net"], pins[1]["net"] = prev_net, net_out
            sh.line((prev_x, y), (pins[0]["x"], y))
            comp = {"id": f"X{cid}", "type": CTRL_TYPES[kd], "box": box, "pins": pins}
            sh.comps.append(comp)
            elems.append((comp, kd))
            lab = {"COIL": "KM", "LAMP": "HL", "BTN_NO": "SB", "BTN_NC": "SB", "THERMAL": "KK", "FUSE": "FU",
                   "QF": "QF"}.get(kd, rng.choice(["KM", "K", "KT"])) + str(cid)
            sh.text(lab, box[0], box[1] - 4)
            prev_x, prev_net = pins[1]["x"], net_out
        sh.line((prev_x, y), (xr, y))
        if rng.random() < sh.t_dot_prob:
            cv2.circle(sh.img, (xl, y), sh.dot_r, INK, -1, cv2.LINE_AA)
            cv2.circle(sh.img, (xr, y), sh.dot_r, INK, -1, cv2.LINE_AA)
        starters = [c for c, kd in elems if kd == "BTN_NO"]
        if starters and rng.random() < 0.7:
            pl, pr = starters[0]["pins"]
            ya = y + int(step * 0.45)
            xa, xb = pl["x"] - 6, pr["x"] + 6
            box, pins = ctrl(sh, "NO", (xa + xb) / 2, ya, rng)
            cid += 1
            pins[0]["net"], pins[1]["net"] = pl["net"], pr["net"]
            sh.line((xa, y), (xa, ya))
            sh.line((xb, y), (xb, ya))
            sh.line((xa, ya), (pins[0]["x"], ya))
            sh.line((pins[1]["x"], ya), (xb, ya))
            for x in (xa, xb):
                if rng.random() < sh.t_dot_prob:
                    cv2.circle(sh.img, (int(x), y), sh.dot_r, INK, -1, cv2.LINE_AA)
            sh.comps.append({"id": f"X{cid}", "type": "SW_NO", "box": box, "pins": pins})
            sh.text("KM" + str(cid), box[0], ya + 22)
    return cid


def ground(sh, cx, cy):
    top = (cx, cy - 22)
    sh.line(top, (cx, cy))
    for i, w in enumerate((30, 20, 10)):
        sh.line((cx - w / 2, cy + i * 6), (cx + w / 2, cy + i * 6))
    return [cx - 16, cy - 2, cx + 16, cy + 14], [{"name": "1", "x": top[0], "y": top[1], "bx": cx, "by": cy}]


def port(sh, cx, cy, rng):
    cv2.circle(sh.img, (cx, cy), 6, INK, sh.t, cv2.LINE_AA)
    side = rng.choice([-1, 1])
    out = (cx + side * 26, cy)
    sh.line((cx + side * 6, cy), out)
    lab = rng.choice(["IN", "OUT", "+5V", "+12V", "VCC", "X1", "A", "B", "CLK", "EN"])
    tw, th = sh.text_size(lab)
    sh.text(lab, cx - tw - 12 if side > 0 else cx + 12, cy + th // 2)
    return [cx - 8, cy - 8, cx + 8, cy + 8], [{"name": "1", "x": out[0], "y": out[1], "bx": cx + side * 6, "by": cy}]


# ---------- провода ----------

def seg_hits_box(a, b, box, pad=4):
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    return not (x1 < box[0] - pad or x0 > box[2] + pad or y1 < box[1] - pad or y0 > box[3] + pad)


def collinear_overlap(a, b, s, gap=7):
    if a[1] == b[1] and s[1] == s[3] and abs(a[1] - s[1]) < gap:
        return min(max(a[0], b[0]), max(s[0], s[2])) > max(min(a[0], b[0]), min(s[0], s[2])) - gap
    if a[0] == b[0] and s[0] == s[2] and abs(a[0] - s[0]) < gap:
        return min(max(a[1], b[1]), max(s[1], s[3])) > max(min(a[1], b[1]), min(s[1], s[3])) - gap
    return False


def point_on_seg(p, s, gap=7):
    x0, x1 = sorted((s[0], s[2]))
    y0, y1 = sorted((s[1], s[3]))
    return x0 - gap <= p[0] <= x1 + gap and y0 - gap <= p[1] <= y1 + gap


def path_ok(sh, pts, net, own_boxes):
    for a, b in zip(pts, pts[1:]):
        for c in sh.comps:
            if c["box"] in own_boxes:
                continue
            if seg_hits_box(a, b, c["box"]):
                return False
        for s in sh.segs:
            if s[4] == net:
                continue
            if collinear_overlap(a, b, s):
                return False
    foreign_pts = [(p["x"], p["y"]) for c in sh.comps for p in c["pins"] if p.get("net") != net]
    foreign_pts += [(s[0], s[1]) for s in sh.segs if s[4] != net] + [(s[2], s[3]) for s in sh.segs if s[4] != net]
    for a, b in zip(pts, pts[1:]):
        seg = (a[0], a[1], b[0], b[1])
        for q in foreign_pts:
            if point_on_seg(q, seg, 6):
                return False
    for q in pts[1:-1] + [pts[-1]]:
        for s in sh.segs:
            if s[4] != net and point_on_seg(q, s, 6):
                return False
    return True


def nearest_on_tree(sh, p, net):
    best = None
    for s in sh.segs:
        if s[4] != net:
            continue
        x0, x1 = sorted((s[0], s[2]))
        y0, y1 = sorted((s[1], s[3]))
        q = (min(max(p[0], x0), x1), min(max(p[1], y0), y1))
        d = abs(q[0] - p[0]) + abs(q[1] - p[1])
        if best is None or d < best[0]:
            best = (d, q, s)
    return best


def route(sh, p, target, net, own_boxes, rng):
    a = (int(p[0]), int(p[1]))
    b = (int(target[0]), int(target[1]))
    cands = [[a, (b[0], a[1]), b], [a, (a[0], b[1]), b]]
    for _ in range(4):
        mx = int(rng.uniform(min(a[0], b[0]), max(a[0], b[0]) + 1))
        my = int(rng.uniform(min(a[1], b[1]), max(a[1], b[1]) + 1))
        cands += [[a, (mx, a[1]), (mx, b[1]), b], [a, (a[0], my), (b[0], my), b]]
    rng.shuffle(cands[:2])
    for pts in cands:
        pts = [q for i, q in enumerate(pts) if i == 0 or q != pts[i - 1]]
        if path_ok(sh, pts, net, own_boxes):
            return pts
    return None


# ---------- схема целиком ----------

# уровни: размер листа, сетка размещения, заполненность, сила порчи (0 — чистый чертёж … 3 — фото телефоном)
LEVELS = {1: dict(size=(1100, 800), grid=(5, 4), fill=0.45, degrade=0),
          2: dict(size=(1500, 1050), grid=(7, 5), fill=0.55, degrade=1),
          3: dict(size=(1700, 1200), grid=(8, 6), fill=0.62, degrade=1),
          4: dict(size=(2300, 1600), grid=(11, 8), fill=0.72, degrade=2),
          5: dict(size=(2800, 2000), grid=(14, 10), fill=0.78, degrade=3)}


def generate(seed, hard=None, level=None):
    """Схема со своим ответом. level 1…5 — от пары деталей на чистом листе до плотной схемы,
    сфотографированной телефоном. hard=True/False — прежние уровни 4/3."""
    rng = random.Random(seed)
    if level is None:
        level = (4 if hard else 3) if hard is not None else rng.choice([1, 2, 3, 3, 4, 4, 5])
    lv = LEVELS[level]
    sh = Sheet(rng, *lv["size"])
    if level >= 2 and rng.random() < 0.3:                  # лист схемы управления (лестничная разводка)
        ladder(sh, rng, level)
        return _finish(sh, rng, seed, level)
    cols, rows = lv["grid"]
    sx, sy = sh.W / cols, sh.H / rows
    kinds = ["R"] * 6 + ["C"] * 4 + ["D"] * 3 + ["L"] * 2 + ["Q"] * 2 + ["U"] * 2 + ["G"] * 2 + ["P"] * 2 + ["A"] * 2
    if level >= 2 and rng.random() < 0.3:                   # элементы управления вперемешку с электроникой
        kinds += ["W"] * 4 + ["Y"] * 2 + ["F"] * 2 + ["O"]
    if rng.random() < 0.45:                                 # лист КИП / микроконтроллерной платы
        kinds += ["S"] * 4 + ["T"] * 3 + ["K"] * 2 + ["X"] * 2 + ["E"] * 3 + ["M"] * 1 + ["B"] * 3
    if rng.random() < 0.35:                                 # привод: энкодер, тахогенератор, датчики Холла
        kinds += ["N"] * 3 + ["V"] * 2 + ["H"] * 3
    cid = 0
    refs = {"R": "R", "C": "C", "D": "VD", "L": "L", "Q": "VT", "U": "DD", "G": "", "P": "", "A": "DA",
            "S": "BP", "T": "RK", "K": "BK", "X": "ZQ", "E": "HL", "M": "DD", "B": "XT",
            "N": "BQ", "V": "BR", "H": "BH", "W": "K", "Y": "KM", "F": "HL", "O": "M"}
    for r in range(rows):
        for c in range(cols):
            if rng.random() > lv["fill"]:
                continue
            k = rng.choice(kinds)
            cx = int((c + 0.5) * sx + rng.uniform(-sx * 0.15, sx * 0.15))
            cy = int((r + 0.5) * sy + rng.uniform(-sy * 0.15, sy * 0.15))
            if k in "RCDL":
                box, pins = two_pin(sh, k, cx, cy, rng.random() < 0.45, cid)
            elif k == "Q":
                box, pins = transistor(sh, cx, cy, rng.random() < 0.5)
            elif k == "A":
                box, pins = opamp(sh, cx, cy, rng.random() < 0.3)
            elif k == "U":
                box, pins = chip(sh, cx, cy, rng.choice([4, 6, 8]))
            elif k == "G":
                box, pins = ground(sh, cx, cy)
            elif k == "S":
                box, pins = sensor(sh, cx, cy, rng)
            elif k == "T":
                box, pins = thermistor(sh, cx, cy, rng.random() < 0.45, rng)
            elif k == "K":
                box, pins = thermocouple(sh, cx, cy)
            elif k == "X":
                box, pins = crystal(sh, cx, cy, rng.random() < 0.45)
            elif k == "E":
                box, pins = led(sh, cx, cy, rng.random() < 0.45)
            elif k == "M":
                box, pins = mcu(sh, cx, cy, rng)
            elif k == "B":
                box, pins = terminal(sh, cx, cy, rng)
            elif k == "N":
                box, pins = encoder(sh, cx, cy, rng)
            elif k == "V":
                box, pins = tacho(sh, cx, cy, rng)
            elif k == "H":
                box, pins = hall(sh, cx, cy, rng)
            elif k in "WYF":
                kd = {"W": rng.choice(["NO", "NC", "BTN_NO", "BTN_NC", "QF"]), "Y": "COIL",
                      "F": rng.choice(["LAMP", "THERMAL", "FUSE"])}[k]
                box, pins = ctrl(sh, kd, cx, cy, rng, rng.random() < 0.4)
            elif k == "O":
                box, pins = motor(sh, cx, cy)
            else:
                box, pins = port(sh, cx, cy, rng)
            cid += 1
            typ = {"G": "GND", "P": "PORT", "Q": "Q", "U": "U", "A": "OPAMP", "S": "SENSOR", "T": "RT",
                   "K": "TC", "X": "XTAL", "E": "LED", "M": "MCU", "B": "TERM",
                   "N": "ENC", "V": "TACHO", "H": "HALL", "O": "MOTOR"}.get(k, k)
            if k in "WYF":
                typ = CTRL_TYPES[kd]
            comp = {"id": f"X{cid}", "type": typ, "box": [float(v) for v in box], "pins": pins}
            sh.comps.append(comp)
            if refs[k]:                                   # позиционное обозначение рядом
                lab = f"{refs[k]}{cid}"
                tw, th = sh.text_size(lab)
                tx, ty = box[0], box[1] - 6
                if ty - th > 0 and sh.img[int(ty - th):int(ty), int(tx):int(tx + tw)].min() > 128:
                    sh.text(lab, tx, ty)
    # цепи: группы из 2–4 ближайших свободных выводов
    allp = [(c, p) for c in sh.comps for p in c["pins"]]
    rng.shuffle(allp)
    free = list(allp)
    nets = []
    while free:
        c0, p0 = free.pop()
        free.sort(key=lambda cp: abs(cp[1]["x"] - p0["x"]) + abs(cp[1]["y"] - p0["y"]))
        k = rng.choice([1, 1, 1, 2, 2, 3])
        grp = [(c0, p0)]
        for cp in list(free[:6]):
            if len(grp) > k:
                break
            if cp[0] is c0 and len(c0["pins"]) <= 2:
                continue
            grp.append(cp)
            free.remove(cp)
        nets.append(grp)
    net_id = 0
    for grp in nets:
        net_id += 1
        gid = net_id
        for _, p in grp:
            p["net"] = gid
        root = grp[0]
        joined = [root]
        for c, p in grp[1:]:
            own = []                                      # сквозь корпуса — никогда
            tree = nearest_on_tree(sh, (p["x"], p["y"]), gid)
            if tree is None:
                target = (root[1]["x"], root[1]["y"])
            else:
                target = tree[1]
            pts = route(sh, (p["x"], p["y"]), target, gid, own, rng)
            if pts is None:
                net_id += 1
                p["net"] = net_id                         # не проложили — своя цепь
                continue
            for a, b in zip(pts, pts[1:]):
                sh.segs.append((a[0], a[1], b[0], b[1], p["net"]))
                sh.line(a, b)
            if tree is not None:
                q = tree[1]
                s = tree[2]
                interior = not ((abs(q[0] - s[0]) < 2 and abs(q[1] - s[1]) < 2) or
                                (abs(q[0] - s[2]) < 2 and abs(q[1] - s[3]) < 2))
                if interior and rng.random() < sh.t_dot_prob:
                    cv2.circle(sh.img, (int(q[0]), int(q[1])), sh.dot_r, INK, -1, cv2.LINE_AA)
            joined.append((c, p))
    # помехи: пунктирная рамка вокруг группы
    if rng.random() < 0.35:
        x0, y0 = rng.randint(40, sh.W // 2), rng.randint(40, sh.H // 2)
        x1, y1 = x0 + rng.randint(200, 500), y0 + rng.randint(150, 400)
        for x in range(x0, x1, 18):
            sh.line((x, y0), (min(x + 9, x1), y0), 1)
            sh.line((x, y1), (min(x + 9, x1), y1), 1)
        for y in range(y0, y1, 18):
            sh.line((x0, y), (x0, min(y + 9, y1)), 1)
            sh.line((x1, y), (x1, min(y + 9, y1)), 1)
    return _finish(sh, rng, seed, level)


def _pt(M, x, y):
    return cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), M)[0][0]


NOTES = ["Все резисторы 0,25 Вт, если не указано иное", "Конденсаторы на 50 В", "Питание от сети 220 В",
         "Include plug or door sense to prevent operation", "Circuit patent application pending",
         "Схема электрическая принципиальная", "Monter sur radiateur", "Note: verify against specifications",
         "Рис. 3. Узел управления", "Fig. 12 Control panel wiring", "Перемычки установить по таблице"]


def _texts(sh, rng, level):
    """Заголовок и абзац примечаний крупным шрифтом — на реальных листах их полно."""
    if level < 2 or rng.random() > 0.7:
        return
    big = rng.uniform(1.0, 1.6) * (1 if sh.font != cv2.FONT_HERSHEY_PLAIN else 2)
    for _ in range(rng.randint(1, 3)):
        txt = rng.choice(NOTES)
        (tw, th), _ = cv2.getTextSize(txt, sh.font, big, 2)
        for _ in range(20):
            x, y = rng.randint(10, max(11, sh.W - tw - 10)), rng.randint(th + 10, sh.H - 10)
            if sh.img[y - th - 4:y + 6, x - 4:x + tw + 4].min() > 200:
                cv2.putText(sh.img, txt, (x, y), sh.font, big, INK, rng.choice([1, 2, 2]), cv2.LINE_AA)
                break


def _finish(sh, rng, seed, level):
    _texts(sh, rng, level)
    strength = LEVELS[level]["degrade"]
    img, M = degrade(sh.img, rng, strength)
    if strength >= 3 or (strength == 2 and rng.random() < 0.3):
        img, M = photo(img, M, rng)
    truth = {"seed": seed, "level": level, "comps": [], "pins": []}
    for c in sh.comps:
        b = c["box"]
        tc = np.array([_pt(M, x, y) for x, y in ((b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3]))])
        truth["comps"].append({"id": c["id"], "type": c["type"],
                               "box": [float(tc[:, 0].min()), float(tc[:, 1].min()),
                                       float(tc[:, 0].max()), float(tc[:, 1].max())]})
        for p in c["pins"]:
            q, o = _pt(M, p["bx"], p["by"]), _pt(M, p["x"], p["y"])
            truth["pins"].append({"comp": c["id"], "type": c["type"], "pin": p["name"],
                                  "x": float(q[0]), "y": float(q[1]), "ox": float(o[0]), "oy": float(o[1]),
                                  "net": int(p.get("net", 0))})
    return img, truth


def degrade(img, rng, strength=1):
    """Как скан из книги: наклон, уменьшение, размытие, шум, JPEG, бумага.
    strength 0 — почти чистый чертёж. Возвращает (картинка, матрица 3×3)."""
    h, w = img.shape
    if strength == 0:
        f = rng.uniform(0.7, 1.0)
        out = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        return out, np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]], np.float64)
    ang = rng.uniform(-0.8, 0.8)
    f = rng.uniform(0.45, 1.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, f)
    nw, nh = int(w * f), int(h * f)
    M[0, 2] -= (w - nw) / 2
    M[1, 2] -= (h - nh) / 2
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_AREA, borderValue=255)
    if rng.random() < 0.6:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 0.9))
    if rng.random() < 0.3:                       # старая печать: линии расплылись
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    paper = rng.uniform(205, 255)
    out = (out.astype(np.float32) / 255 * (paper - 10) + 10)
    if rng.random() < 0.4:                       # фото: неровный свет и тень
        yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
        a = rng.uniform(0, 2 * math.pi)
        grad = (np.cos(a) * xx / nw + np.sin(a) * yy / nh)
        grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
        out = out * (1 - rng.uniform(0.15, 0.45) * grad)
    out += np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 9), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    q = rng.randint(35, 95)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)
    return out, np.vstack([M, [0, 0, 1]])


def photo(img, M, rng):
    """Фото листа телефоном: лист на столе, перспектива, тёплый свет, тень, виньетка,
    лёгкий расфокус, цветной шум, JPEG. Возвращает цветную картинку и итоговую 3×3."""
    h, w = img.shape
    pad = int(max(h, w) * rng.uniform(0.08, 0.2))
    W, H = w + 2 * pad, h + 2 * pad
    gen = np.random.default_rng(rng.randint(0, 1 << 30))
    base = np.array([rng.uniform(40, 150), rng.uniform(40, 130), rng.uniform(30, 110)], np.float32)
    table = np.ones((H, W, 3), np.float32) * base
    table += cv2.GaussianBlur(gen.normal(0, 25, (H, W)).astype(np.float32), (0, 0), 6)[..., None]
    j = lambda: rng.uniform(-0.06, 0.06) * max(w, h)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    dst = np.array([[pad + j(), pad + j()], [pad + w + j(), pad + j()], [pad + w + j(), pad + h + j()],
                    [pad + j(), pad + h + j()]], np.float32)
    P = cv2.getPerspectiveTransform(src, dst)
    tint = np.array([rng.uniform(0.85, 1.0), rng.uniform(0.9, 1.0), 1.0], np.float32)
    page = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32) * tint
    warped = cv2.warpPerspective(page, P, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), P, (W, H))
    out = table * (1 - mask[..., None]) + warped * mask[..., None]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = rng.uniform(0, W), rng.choice([0, H])
    shade = np.exp(-(((xx - cx) / (0.5 * W)) ** 2 + ((yy - cy) / (0.35 * H)) ** 2))
    out *= (1 - rng.uniform(0.2, 0.5) * shade)[..., None]
    vig = 1 - 0.35 * (((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2
    out *= vig[..., None]
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.4))
    out += gen.normal(0, rng.uniform(3, 8), out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(60, 90)])[1], cv2.IMREAD_COLOR)
    return out, P @ M


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 5):
        img, truth = generate(1000 + i, level=int(sys.argv[3]) if len(sys.argv) > 3 else None)
        cv2.imwrite(str(out / f"synth_{i:03d}.png"), img)
        (out / f"synth_{i:03d}.json").write_text(json.dumps(truth), encoding="utf-8")
        nets = {p["net"] for p in truth["pins"]}
        print(f"synth_{i:03d}: деталей {len(truth['comps'])}, выводов {len(truth['pins'])}, цепей {len(nets)}")
