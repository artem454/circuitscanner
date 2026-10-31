"""Эталонные условные графические обозначения (УГО) — с чем ядро сравнивает найденные символы.

Размеры в миллиметрах по стандартам ЕСКД (в скобках) и в стиле IEC 60617 / ANSI:
  резистор 10×4 (ГОСТ 2.728), конденсатор: обкладки 8 мм, зазор 1,5 мм, электролит — «+» или полая обкладка;
  диод: треугольник 5 мм, черта 5 мм (ГОСТ 2.730); транзистор: окружность ⌀12 мм (у малых ⌀10),
  тиристор — диод с управляющим электродом, симистор — два встречных треугольника с одной чертой;
  контакт: длина 10 мм, угол 30° (ГОСТ 2.755), кнопка — с толкателем, контакт теплового реле — с замком,
  контакт реле времени — с дужкой; катушка реле 12×6 (ГОСТ 2.756); нагреватель теплового реле — «изгиб»;
  ОУ — треугольник (IEC/ANSI) или прямоугольник с «▷» (ГОСТ 2.759); логика ГОСТ 2.743 — прямоугольник
  с функцией (&, 1, =1, DC, CD, TT); триггер Шмитта — знак гистерезиса.

Каждый эталон рисуется с выводами на маленьком листе, лист разбирается тем же анализатором,
и признаки найденного символа сохраняются с меткой класса (python symlib.py [вариантов]).
Так эталон попадает в классификатор ровно в том виде, в каком ядро видит символ на схеме.
"""
import math
import random
import sys

import cv2
import numpy as np

INK = 0


class Sheet:
    def __init__(self, w, h, u, t):
        self.img = np.full((h, w), 255, np.uint8)
        self.u, self.t = u, t                 # пикселей на мм, толщина линии

    def P(self, x, y):
        return int(round(x * self.u)), int(round(y * self.u))

    def line(self, a, b, k=1.0):
        cv2.line(self.img, self.P(*a), self.P(*b), INK, max(1, int(round(self.t * k))), cv2.LINE_AA)

    def poly(self, pts, closed=True, fill=False):
        p = np.array([self.P(*q) for q in pts], np.int32)
        if fill:
            cv2.fillPoly(self.img, [p], INK, cv2.LINE_AA)
        cv2.polylines(self.img, [p], closed, INK, self.t, cv2.LINE_AA)

    def rect(self, x0, y0, x1, y1):
        self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def circle(self, c, r, fill=False):
        cv2.circle(self.img, self.P(*c), int(round(r * self.u)), INK, -1 if fill else self.t, cv2.LINE_AA)

    def arc(self, c, r, a0, a1):
        cv2.ellipse(self.img, self.P(*c), (int(r * self.u), int(r * self.u)), 0, a0, a1, INK, self.t, cv2.LINE_AA)

    def text(self, s, x, y, h=3.0):
        sc = h * self.u / 22
        cv2.putText(self.img, s, self.P(x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, INK, max(1, self.t - 1), cv2.LINE_AA)


# Каждый эталон: f(s, x, y, rng) рисует символ с центром (x, y) мм, выводы горизонтальны;
# возвращает точки выводов (концы тела), к которым дорисуются провода.

def resistor(s, x, y, rng):
    if rng.random() < 0.25:                                  # ANSI: зигзаг
        pts = [(x - 5, y)] + [(x - 5 + 10 * (i + 0.5) / 6, y + (-1) ** i * 1.8) for i in range(6)] + [(x + 5, y)]
        s.poly(pts, closed=False)
    else:
        s.rect(x - 5, y - 2, x + 5, y + 2)
    return [(x - 5, y), (x + 5, y)]


def potentiometer(s, x, y, rng):
    ends = resistor(s, x, y, rng)
    s.line((x, y - 7), (x, y - 2.3))
    s.poly([(x - 1, y - 3.8), (x + 1, y - 3.8), (x, y - 2.2)], fill=True)
    return ends + [(x, y - 7)]


def varistor(s, x, y, rng):
    ends = resistor(s, x, y, rng)
    s.line((x - 6, y + 3), (x + 4, y - 3))
    s.line((x - 6, y + 3), (x - 8, y + 3))
    return ends


def thermistor(s, x, y, rng):
    ends = resistor(s, x, y, rng)
    s.line((x - 6, y + 4), (x + 4, y - 4))
    s.line((x - 6, y + 4), (x - 8, y + 4))
    s.text("t", x + 2, y - 4, 2.5)
    return ends


def capacitor(s, x, y, rng):
    s.line((x - 0.75, y - 4), (x - 0.75, y + 4), 1.3)
    s.line((x + 0.75, y - 4), (x + 0.75, y + 4), 1.3)
    return [(x - 0.75, y), (x + 0.75, y)]


def electrolytic(s, x, y, rng):
    if rng.random() < 0.5:                                   # ГОСТ: «+» и тонкая обкладка
        s.line((x - 0.75, y - 4), (x - 0.75, y + 4), 1.3)
        s.line((x + 0.75, y - 4), (x + 0.75, y + 4), 1.3)
        s.line((x - 3, y - 4), (x - 3, y - 2))
        s.line((x - 4, y - 3), (x - 2, y - 3))
    else:                                                    # IEC: полая обкладка + изогнутая
        s.rect(x - 1.6, y - 4, x - 0.6, y + 4)
        s.arc((x + 4, y), 3.8, 145, 215)
    return [(x - 0.75, y), (x + 0.75, y)]


def inductor(s, x, y, rng):
    n = rng.choice((3, 4))
    r = 1.5
    for i in range(n):
        s.arc((x - n * r + r + 2 * r * i, y), r, 180, 360)
    if rng.random() < 0.3:                                   # с сердечником
        s.line((x - n * r, y - 2.5), (x + n * r, y - 2.5))
    return [(x - n * r, y), (x + n * r, y)]


def _diode(s, x, y):
    s.poly([(x - 2.5, y - 2.5), (x - 2.5, y + 2.5), (x + 2.5, y)])
    s.line((x + 2.5, y - 2.5), (x + 2.5, y + 2.5), 1.2)


def diode(s, x, y, rng):
    _diode(s, x, y)
    return [(x - 2.5, y), (x + 2.5, y)]


def zener(s, x, y, rng):
    _diode(s, x, y)
    s.line((x + 2.5, y - 2.5), (x + 1.3, y - 2.5))
    return [(x - 2.5, y), (x + 2.5, y)]


def led(s, x, y, rng):
    _diode(s, x, y)
    for d in (0, 1.6):
        s.line((x + 0.2 + d, y - 3), (x + 2.2 + d, y - 5.5))
        s.poly([(x + 2.2 + d, y - 5.5), (x + 1.2 + d, y - 5.0), (x + 2.0 + d, y - 4.4)], fill=True)
    return [(x - 2.5, y), (x + 2.5, y)]


def thyristor(s, x, y, rng):
    _diode(s, x, y)
    s.line((x + 1.5, y + 1.2), (x + 3.5, y + 5))              # управляющий электрод со стороны катода
    return [(x - 2.5, y), (x + 2.5, y), (x + 3.5, y + 5)]


def triac(s, x, y, rng):
    s.poly([(x - 2.5, y - 3), (x - 2.5, y + 0), (x + 2.5, y - 1.5)])
    s.poly([(x + 2.5, y + 0), (x + 2.5, y + 3), (x - 2.5, y + 1.5)])
    s.line((x - 2.5, y - 3), (x - 2.5, y + 3), 1.2)
    s.line((x + 2.5, y - 3), (x + 2.5, y + 3), 1.2)
    s.line((x + 2.5, y + 2), (x + 4.5, y + 5))
    return [(x - 2.5, y), (x + 2.5, y), (x + 4.5, y + 5)]


def bjt(s, x, y, rng):
    r = 5 if rng.random() < 0.5 else 6
    if rng.random() < 0.8:
        s.circle((x, y), r)
    s.line((x - 1.5, y - 3), (x - 1.5, y + 3), 1.3)
    s.line((x - 1.5, y - 1), (x + 2.5, y - 4))
    s.line((x - 1.5, y + 1), (x + 2.5, y + 4))
    s.poly([(x + 2.5, y + 4), (x + 0.8, y + 3.6), (x + 1.6, y + 2.5)], fill=True)
    s.line((x + 2.5, y - 4), (x + 2.5, y - r - 1))
    s.line((x + 2.5, y + 4), (x + 2.5, y + r + 1))
    return [(x - r, y), (x + 2.5, y - r - 1), (x + 2.5, y + r + 1)]


def mosfet(s, x, y, rng):
    s.circle((x, y), 6)
    s.line((x - 2.5, y - 3), (x - 2.5, y + 3))
    for dy in (-3, 0, 3):
        s.line((x - 1.5, y + dy - 1), (x - 1.5, y + dy + 1), 1.3)
        s.line((x - 1.5, y + dy), (x + 2.5, y + dy))
    s.line((x + 2.5, y - 3), (x + 2.5, y - 7))
    s.line((x + 2.5, y + 3), (x + 2.5, y + 7))
    s.line((x + 2.5, y), (x + 2.5, y + 3))
    s.poly([(x - 1.4, y), (x + 0.4, y - 0.8), (x + 0.4, y + 0.8)], fill=True)
    return [(x - 6, y), (x + 2.5, y - 7), (x + 2.5, y + 7)]


def optocoupler(s, x, y, rng):
    s.rect(x - 7, y - 6, x + 7, y + 6)
    s.poly([(x - 5.5, y - 2), (x - 2.5, y - 2), (x - 4, y + 1)])
    s.line((x - 5.5, y + 1), (x - 2.5, y + 1), 1.2)
    for d in (-1.2, 1.2):
        s.line((x - 1.5, y + d), (x + 0.8, y + d))
        s.poly([(x + 0.8, y + d), (x - 0.2, y + d - 0.6), (x - 0.2, y + d + 0.6)], fill=True)
    if rng.random() < 0.5:                                   # выход — транзистор
        s.line((x + 2, y - 3), (x + 2, y + 3), 1.3)
        s.line((x + 2, y - 1), (x + 5, y - 4))
        s.line((x + 2, y + 1), (x + 5, y + 4))
    else:                                                    # выход — симистор
        s.poly([(x + 2, y - 3), (x + 5, y - 3), (x + 3.5, y)])
        s.poly([(x + 2, y + 3), (x + 5, y + 3), (x + 3.5, y)])
        s.line((x + 2, y), (x + 5, y), 1.2)
    return [(x - 7, y - 4), (x - 7, y + 4), (x + 7, y - 4), (x + 7, y + 4)]


def contact_no(s, x, y, rng):
    s.line((x - 5, y), (x - 2, y))
    s.line((x - 2, y), (x + 4, y - 3))
    s.line((x + 3, y), (x + 5, y))
    return [(x - 5, y), (x + 5, y)]


def contact_nc(s, x, y, rng):
    s.line((x - 5, y), (x - 2, y))
    s.line((x - 2, y), (x + 4, y + 2))
    s.line((x + 3, y), (x + 3, y + 2.5))
    s.line((x + 3, y), (x + 5, y))
    return [(x - 5, y), (x + 5, y)]


def contact_co(s, x, y, rng):                                # переключающий
    s.line((x - 5, y), (x - 2, y))
    s.line((x - 2, y), (x + 4, y - 2))
    s.line((x + 3, y - 3.5), (x + 5, y - 3.5))
    s.line((x + 3, y + 1.5), (x + 5, y + 1.5))
    s.line((x + 3, y + 1.5), (x + 3, y - 1))
    return [(x - 5, y), (x + 5, y - 3.5), (x + 5, y + 1.5)]


def timer_contact(s, x, y, rng):
    ends = contact_no(s, x, y, rng) if rng.random() < 0.5 else contact_nc(s, x, y, rng)
    s.arc((x + 1, y - 5), 2, 0, 180)
    s.line((x + 1, y - 3), (x + 1, y - 1.5))
    return ends


def thermal_contact(s, x, y, rng):
    ends = contact_nc(s, x, y, rng)
    s.line((x, y - 1), (x, y - 5))
    s.poly([(x - 1.5, y - 5), (x - 1.5, y - 7), (x + 1.5, y - 7), (x + 1.5, y - 5)], closed=False)
    return ends


def button_no(s, x, y, rng):
    ends = contact_no(s, x, y, rng)
    s.line((x + 1, y - 1.5), (x + 1, y - 6))
    s.line((x - 1, y - 6), (x + 3, y - 6))
    s.line((x - 1, y - 6), (x - 1, y - 5))
    s.line((x + 3, y - 6), (x + 3, y - 5))
    return ends


def button_nc(s, x, y, rng):
    ends = contact_nc(s, x, y, rng)
    s.line((x + 1, y + 1), (x + 1, y - 5))
    s.line((x - 1, y - 5), (x + 3, y - 5))
    return ends


def relay_coil(s, x, y, rng):
    s.rect(x - 6, y - 3, x + 6, y + 3)
    if rng.random() < 0.3:
        s.line((x - 6, y + 3), (x + 6, y - 3))               # поляризованное / с особенностью
    return [(x - 6, y), (x + 6, y)]


def thermal_heater(s, x, y, rng):
    s.poly([(x - 5, y), (x - 2, y), (x - 2, y - 3), (x + 2, y - 3), (x + 2, y), (x + 5, y)], closed=False)
    if rng.random() < 0.5:
        s.rect(x - 3.5, y - 4.5, x + 3.5, y + 1.5)
    return [(x - 5, y), (x + 5, y)]


def fuse(s, x, y, rng):
    s.rect(x - 5, y - 2, x + 5, y + 2)
    s.line((x - 5, y), (x + 5, y))
    return [(x - 5, y), (x + 5, y)]


def breaker(s, x, y, rng):
    ends = contact_no(s, x, y, rng)
    s.line((x - 2.7, y - 0.7), (x - 1.3, y + 0.7))
    s.line((x - 2.7, y + 0.7), (x - 1.3, y - 0.7))
    s.rect(x + 0.5, y - 4, x + 2, y - 2.5)
    return ends


def lamp(s, x, y, rng):
    s.circle((x, y), 3)
    s.line((x - 2.1, y - 2.1), (x + 2.1, y + 2.1))
    s.line((x - 2.1, y + 2.1), (x + 2.1, y - 2.1))
    return [(x - 3, y), (x + 3, y)]


def motor(s, x, y, rng):
    s.circle((x, y), 6)
    s.text("M", x - 2.2, y + 1.6, 4.5)
    return [(x - 6, y), (x + 6, y)]


def crystal(s, x, y, rng):
    s.rect(x - 1, y - 3, x + 1, y + 3)
    s.line((x - 2.5, y - 3.5), (x - 2.5, y + 3.5), 1.2)
    s.line((x + 2.5, y - 3.5), (x + 2.5, y + 3.5), 1.2)
    return [(x - 2.5, y), (x + 2.5, y)]


def opamp(s, x, y, rng):
    if rng.random() < 0.6:                                   # IEC/ANSI: треугольник
        s.poly([(x - 6, y - 6), (x - 6, y + 6), (x + 6, y)])
        s.text("-", x - 5, y - 1.6, 3)
        s.text("+", x - 5, y + 4.4, 3)
        return [(x - 6, y - 3), (x - 6, y + 3), (x + 6, y)]
    s.rect(x - 6, y - 7, x + 6, y + 7)                       # ГОСТ 2.759: прямоугольник «▷»
    s.poly([(x - 1.5, y - 5.5), (x - 1.5, y - 2.5), (x + 1.5, y - 4)])
    s.circle((x - 6, y - 3), 0.8)
    return [(x - 6.8, y - 3), (x - 6, y + 3), (x + 6, y)]


def logic_gate(s, x, y, rng):
    """Инвертор / буфер / триггер Шмитта в стиле IEC/ANSI (треугольник, кружок, знак гистерезиса)."""
    s.poly([(x - 4, y - 4), (x - 4, y + 4), (x + 3, y)])
    if rng.random() < 0.7:
        s.circle((x + 3.8, y), 0.8)
    if rng.random() < 0.6:                                   # Шмитт
        s.poly([(x - 3, y + 1), (x - 1.2, y + 1), (x - 1.2, y - 1), (x + 0.2, y - 1)], closed=False)
        s.poly([(x - 2.2, y + 1), (x - 2.2, y - 1), (x - 0.4, y - 1)], closed=False)
    return [(x - 4, y), (x + 4.6, y)]


def ansi_and(s, x, y, rng):
    s.poly([(x - 5, y - 4), (x, y - 4)], closed=False)
    s.poly([(x - 5, y + 4), (x, y + 4)], closed=False)
    s.line((x - 5, y - 4), (x - 5, y + 4))
    s.arc((x, y), 4, -90, 90)
    if rng.random() < 0.5:
        s.circle((x + 4.8, y), 0.8)
    return [(x - 5, y - 2), (x - 5, y + 2), (x + 5.6, y)]


def ground(s, x, y, rng):
    s.line((x, y - 4), (x, y))
    for i, w in enumerate((4, 2.6, 1.2)):
        s.line((x - w / 2 * 2, y + i * 1.2), (x + w / 2 * 2, y + i * 1.2))
    return [(x, y - 4)]


# класс классификатора ← эталоны
LIBRARY = {
    "R": [resistor, potentiometer, varistor], "RT": [thermistor], "C": [capacitor, electrolytic],
    "L": [inductor], "D": [diode, zener], "LED": [led], "VS": [thyristor, triac], "Q": [bjt, mosfet],
    "OC": [optocoupler], "SW_NO": [contact_no, contact_co, timer_contact], "SW_NC": [contact_nc],
    "THERMAL": [thermal_contact, thermal_heater], "BTN": [button_no, button_nc], "COIL": [relay_coil],
    "FU": [fuse], "QF": [breaker], "LAMP": [lamp], "MOTOR": [motor], "XTAL": [crystal],
    "OPAMP": [opamp], "LOGIC": [logic_gate, ansi_and], "GND": [ground],
}


def render(draw, rng):
    """Символ с проводами на маленьком листе. Возвращает (серое изображение, бокс тела в px, поворот)."""
    u = rng.uniform(3.0, 9.0)                                # пикселей на мм: от мелкого скана до крупного листа
    t = max(1, int(round(u * rng.uniform(0.18, 0.45))))      # толщина линии 0,2…0,5 мм
    W = H = 60
    s = Sheet(int(W * u), int(H * u), u, t)
    x, y = W / 2, H / 2
    ends = draw(s, x, y, rng)
    ys, xs = np.nonzero(s.img < 128)
    body = [xs.min(), ys.min(), xs.max(), ys.max()]
    for ex, ey in ends:                                      # выводы: провод до края + иногда точка и ответвление
        dx = -1 if ex < x - 0.5 else 1 if ex > x + 0.5 else 0
        dy = 0 if dx else (-1 if ey < y else 1)
        L = rng.uniform(10, 20)
        end = (ex + dx * L, ey + dy * L)
        s.line((ex, ey), end)
        if rng.random() < 0.3:
            s.circle(end, 0.6, fill=True)
            s.line(end, (end[0] + dy * 8, end[1] + dx * 8))
    img = s.img
    k = rng.choice((0, 1, 2, 3))
    if k:
        h0, w0 = img.shape
        img = np.rot90(img, k).copy()
        for _ in range(k):                                   # поворот бокса на 90° против часовой
            x0, y0, x1, y1 = body
            body = [y0, w0 - 1 - x1, y1, w0 - 1 - x0]
            h0, w0 = w0, h0
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.4:
        noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(4, 14), img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return img, body


def _examples(args):
    import analyzer
    cls, idx, n, seed, params = args
    rng = random.Random(seed)
    draw = LIBRARY[cls][idx]
    X, y, Cr = [], [], []
    for _ in range(n):
        img, body = render(draw, rng)
        res = analyzer.analyze_gray(img, params)
        bx = [(body[0] + body[2]) / 2, (body[1] + body[3]) / 2]
        best, bf = None, None
        for sy, f, cr in zip(res["symbols"], res["_features"], res["_crops"]):
            b = sy["box"]
            inter = max(0, min(b[2], body[2]) - max(b[0], body[0])) * max(0, min(b[3], body[3]) - max(b[1], body[1]))
            area = max(1, (body[2] - body[0]) * (body[3] - body[1]))
            if inter / area > 0.35 and (best is None or inter > best):
                best, bf = inter, (f, cr)
        if bf is not None:
            X.append(bf[0])
            y.append(cls)
            Cr.append(bf[1])
    return X, y, Cr


def build(per_symbol=40, params=None):
    """Эталоны → примеры классификатора (src «ref» в learned/symbols.npz)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    import analyzer
    import learn
    params = params or analyzer.load_params()
    jobs = [(cls, i, per_symbol, 7000 + 97 * k + i, params)
            for k, (cls, fs) in enumerate(LIBRARY.items()) for i in range(len(fs))]
    import symclass
    X, y, Cr = [], [], []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for Xi, yi, ci in pool.map(_examples, jobs):
            X += Xi
            y += yi
            Cr += ci
    total = learn.save_examples(X, y, "ref")
    symclass.save([c for c, _ in Cr], [d for _, d in Cr], y, "ref")
    stats = {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"эталонов: {len(y)} {stats}; всего примеров {total}")
    return stats


if __name__ == "__main__":
    build(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
