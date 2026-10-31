"""Аналитическое распознавание схемы — без ИИ, только геометрия.

Находит провода (горизонтальные и вертикальные отрезки), точки соединений,
прямоугольники (резисторы, корпуса микросхем), остальные символы и надписи,
затем по правилам черчения собирает цепи:
  • конец провода, упёршийся в другой провод (Т-образно или углом), — соединение;
  • пересечение без точки — соединения нет, с точкой — есть;
  • конец провода у символа — вывод этого символа.

Пороги лежат в словаре параметров. learn.py подбирает их на синтетических
схемах с известным ответом и сохраняет в learned/params.json, а классификатор
символов (learned/symbols.npz) учится на синтетике и на проверенных проектах.

Результат — в папке проекта, подпапка analysis/:
  analysis.json      отрезки, точки, символы, надписи, цепи (координаты оригинала)
  nets.png           весь лист, каждая цепь своим цветом и с номером N…
  tiles/*.png        те же фрагменты, что для Claude, но с цветными цепями
  netlist_auto.json  черновой netlist в формате программы

    python analyzer.py папка_проекта
"""
import colorsys
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

import core

LEARNED = Path(__file__).resolve().parent / "learned"

# Все пороги — в долях толщины линии sw (кроме явно оговорённых)
DEFAULTS = {
    "target": 2400,         # до какого размера (px) увеличивать картинку
    "thin_upscale": 1,      # линии в 1 px — разбирать в двойном масштабе (0 — выключить)
    "text_max": 12.0,       # макс. размер буквы
    "min_len": 7.0,         # мин. длина провода
    "tol": 2.0,             # допуск касания: tol·sw + 2
    "dot_d": 2.2,           # диаметр диска для поиска залитых точек
    "dot_min": 2.4,         # мин. диаметр точки
    "dot_max": 8.0,         # макс. диаметр точки
    "rect_fill": 0.03,      # макс. заполнение внутри корпуса (доля)
    "rect_cont": 2,         # сколько продолжений сторон за угол = петля проводов (шт.)
    "plate_thick": 1.8,     # толще — обкладка/полоса, а не провод
    "stroke_len": 2.5,      # штрих символа короче stroke_len·min_len
    "sym_area": 4.0,        # мин. площадь символа: sym_area·sw²
    "sym_grow": 3.0,        # насколько раздувать части символа при склейке
    "pin_dist": 2.0,        # конец провода ближе pin_dist·tol к символу — вывод
}


def load_params(extra=None):
    p = dict(DEFAULTS)
    learned = core.load_json(LEARNED / "params.json")
    if learned:
        p.update({k: v for k, v in learned.get("params", {}).items() if k in DEFAULTS})
    if extra:
        p.update(extra)
    return p


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# ---------- примитивы ----------

def binarize(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def stroke_width(bw):
    """Самая частая длина коротких серий чернил по строкам и столбцам = толщина линии."""
    lengths = []
    for arr in (bw, bw.T):
        a = np.pad((arr > 0).astype(np.int8), ((0, 0), (1, 1)))
        d = np.diff(a, axis=1)
        starts, ends = np.nonzero(d == 1), np.nonzero(d == -1)
        lengths.append(ends[1] - starts[1])
    r = np.concatenate(lengths)
    r = r[(r > 0) & (r < 40)]
    return max(1, int(np.bincount(r).argmax())) if r.size else 2


def boxes_near(a, b, pad):
    """Пересекаются ли прямоугольники (x0,y0,x1,y1) с запасом pad."""
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or a[3] + pad < b[1] or b[3] + pad < a[1])


def box_dist(box, x, y):
    dx = max(box[0] - x, 0, x - box[2])
    dy = max(box[1] - y, 0, y - box[3])
    return math.hypot(dx, dy)


def box_center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


# ---------- надписи ----------

def find_text(ink, sw, P):
    """Отдельно стоящие мелкие компоненты = буквы; склеиваем их в слова."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    tmax = max(P["text_max"] * sw, 24)
    chars = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > tmax or w > tmax * 1.5 or a < max(4, sw * sw):
            continue
        chars.append([x, y, x + w - 1, y + h - 1, i])
    m = len(chars)
    uf = UnionFind(m)
    C = np.array(chars, dtype=float).reshape(-1, 5)
    for i in range(m):
        a = chars[i]
        ha, wa = a[3] - a[1] + 1, a[2] - a[0] + 1
        # кандидаты рядом — чтобы не перебирать все пары
        near = np.nonzero((C[:, 0] <= a[2] + 2 * tmax) & (C[:, 2] >= a[0] - 2 * tmax)
                          & (C[:, 1] <= a[3] + 2 * tmax) & (C[:, 3] >= a[1] - 2 * tmax))[0]
        for j in near:
            if j <= i:
                continue
            b = chars[j]
            hb, wb = b[3] - b[1] + 1, b[2] - b[0] + 1
            vov = min(a[3], b[3]) - max(a[1], b[1])
            hov = min(a[2], b[2]) - max(a[0], b[0])
            gap_x = max(a[0], b[0]) - min(a[2], b[2])
            gap_y = max(a[1], b[1]) - min(a[3], b[3])
            if vov >= 0.4 * min(ha, hb) and gap_x <= 0.8 * max(ha, hb):
                uf.union(i, j)                       # одна строка
            elif (hov >= 0.4 * min(wa, wb) and gap_y <= 0.6 * max(wa, wb)
                  and wa > ha * 0.8 and wb > hb * 0.8):
                uf.union(i, j)                       # повёрнутый текст
    groups = {}
    for i in range(m):
        groups.setdefault(uf.find(i), []).append(chars[i])
    words, labels = [], []
    for g in groups.values():
        x0, y0 = min(c[0] for c in g), min(c[1] for c in g)
        x1, y1 = max(c[2] for c in g), max(c[3] for c in g)
        w, h = x1 - x0 + 1, y1 - y0 + 1
        thin = all(min(c[2] - c[0], c[3] - c[1]) <= 1.6 * sw + 1 for c in g)
        if thin and len(g) >= 3 and max(w, h) > 8 * min(w, h):
            continue                                 # пунктир, а не текст
        if len(g) == 1 and max(w, h) < 2.5 * sw:
            continue                                 # соринка
        words.append({"box": [int(x0), int(y0), int(x1), int(y1)], "chars": len(g),
                      "vertical": bool(h > 1.8 * w and len(g) > 1)})
        labels.extend(c[4] for c in g)
    mask = np.isin(lab, labels).astype(np.uint8) * 255 if labels else np.zeros_like(ink)
    return words, mask


def try_ocr(gray, words):
    """Распознать текст, если установлены Tesseract и pytesseract (rus+eng)."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        return False
    langs = pytesseract.get_languages(config="")
    lang = "+".join(l for l in ("rus", "eng") if l in langs) or None
    for wd in words:
        x0, y0, x1, y1 = wd["box"]
        pad = 6
        crop = gray[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        best = ("", -1.0)
        variants = [crop] if not wd["vertical"] else [cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
                                                      cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)]
        for v in variants:
            d = pytesseract.image_to_data(v, lang=lang, config="--psm 7",
                                          output_type=pytesseract.Output.DICT)
            toks = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
            if toks:
                conf = sum(c for _, c in toks) / len(toks)
                if conf > best[1]:
                    best = (" ".join(t for t, _ in toks), conf)
        wd["text"], wd["conf"] = best[0], round(best[1], 1)
    return True


# ---------- прямоугольники ----------

def edge_continuations(ink, box, others, sw, L):
    """Сколько сторон прямоугольника продолжаются линией за угол (из 8 возможных),
    не считая продолжений вдоль соседних отсеков того же корпуса."""
    x0, y0, x1, y1 = box
    H, W = ink.shape
    top, bot = y0 + sw // 2, y1 - sw // 2
    lef, rig = x0 + sw // 2, x1 - sw // 2
    probes = [
        [(x, top) for x in range(x1 + 2, x1 + L)], [(x, top) for x in range(x0 - L, x0 - 1)],
        [(x, bot) for x in range(x1 + 2, x1 + L)], [(x, bot) for x in range(x0 - L, x0 - 1)],
        [(lef, y) for y in range(y0 - L, y0 - 1)], [(lef, y) for y in range(y1 + 2, y1 + L)],
        [(rig, y) for y in range(y0 - L, y0 - 1)], [(rig, y) for y in range(y1 + 2, y1 + L)],
    ]
    n = 0
    for pts in probes:
        pts = [(x, y) for x, y in pts if 0 <= x < W and 0 <= y < H]
        if len(pts) < L // 2:
            continue
        mx, my = pts[len(pts) // 2]
        if any(o is not box and o[0] - sw <= mx <= o[2] + sw and o[1] - sw <= my <= o[3] + sw
               for o in others):
            continue
        hits = 0
        for x, y in pts:
            ya, yb = max(0, y - sw // 2), min(H, y + sw // 2 + 1)
            xa, xb = max(0, x - sw // 2), min(W, x + sw // 2 + 1)
            hits += ink[ya:yb, xa:xb].any()
        if hits >= 0.85 * len(pts):
            n += 1
    return n


def two_lead_resistors(symbols, pins, W, H):
    """Небольшой корпус-прямоугольник с ровно двумя выводами: выводы на противоположных
    коротких сторонах — резистор (даже если тонкие линии не дали сработать порогу по толщине
    штриха), на середине длинных сторон друг напротив друга — обкладка электролита."""
    by_sym = {}
    for p in pins:
        by_sym.setdefault(p["symbol"], []).append((p["x"], p["y"]))
    for k, sy in enumerate(symbols):
        if sy["kind"] not in ("R", "U") or sy.get("cells", 1) != 1 or len(by_sym.get(k, [])) != 2:
            continue
        x0, y0, x1, y1 = sy["box"]
        w, h = x1 - x0, y1 - y0
        if max(w, h) < 1.6 * min(w, h) or max(w, h) > 8 * min(w, h) or max(w, h) > 0.08 * max(W, H):
            continue
        (ax, ay), (bx, by) = by_sym[k]
        horiz = w >= h                                   # длинная сторона горизонтальна
        u, v = (ax, bx) if horiz else (ay, by)           # вдоль длинной стороны
        lo, hi, span = (x0, x1, w) if horiz else (y0, y1, h)
        a2, b2 = (ay, by) if horiz else (ax, bx)         # поперёк
        lo2, hi2, span2 = (y0, y1, h) if horiz else (x0, x1, w)
        if min(u, v) <= lo + 0.25 * span and max(u, v) >= hi - 0.25 * span:
            sy["kind"] = "R"
        elif (min(a2, b2) <= lo2 + 0.3 * span2 and max(a2, b2) >= hi2 - 0.3 * span2
              and abs(u - v) <= 0.25 * span and abs((u + v) / 2 - (lo + hi) / 2) <= 0.25 * span):
            sy["kind"] = "C"


def find_rects(ink, sw, W, H, P):
    """Пустые прямоугольные «дырки» = корпуса резисторов и микросхем."""
    cnts, hier = cv2.findContours(ink, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = []
    if hier is None:
        return []
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 3 * sw or h < 3 * sw or max(w, h) > 0.3 * max(W, H):
            continue
        if cv2.contourArea(c) < 0.85 * w * h:
            continue
        # внутри корпуса пусто (надписи уже убраны)
        inner = ink[y + 2 * sw:y + h - 2 * sw, x + 2 * sw:x + w - 2 * sw]
        if inner.size and inner.mean() / 255 > P["rect_fill"]:
            continue
        holes.append([x - sw, y - sw, x + w + sw, y + h + sw])
    # петля из проводов тоже даёт «дырку» (пересечение замыкает контур), но её
    # стороны уходят дальше за углы; у корпуса стороны обрываются в углах
    L = int(max(P["min_len"] * sw, 12))
    holes = [b for b in holes if edge_continuations(ink, b, holes, sw, L) < P["rect_cont"]]
    # соседние отсеки одного корпуса склеиваем
    uf = UnionFind(len(holes))
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            if boxes_near(holes[i], holes[j], sw + 1):
                uf.union(i, j)
    groups = {}
    for i, b in enumerate(holes):
        groups.setdefault(uf.find(i), []).append(b)
    rects = []
    for g in groups.values():
        box = [min(b[0] for b in g), min(b[1] for b in g), max(b[2] for b in g), max(b[3] for b in g)]
        w, h = box[2] - box[0], box[3] - box[1]
        small = min(w, h) <= 9 * sw and max(w, h) >= 1.7 * min(w, h)
        rects.append({"box": [int(v) for v in box], "kind": "R" if len(g) == 1 and small else "U",
                      "cells": len(g)})
    return rects


# ---------- провода ----------

def line_segments(mask, horizontal, sw, min_len, P):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    wires, plates = [], []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        length = w if horizontal else h
        if length < min_len:
            continue
        thick = a / length
        if horizontal:
            seg = [x, y + h / 2, x + w - 1, y + h / 2]
        else:
            seg = [x + w / 2, y, x + w / 2, y + h - 1]
        if thick > P["plate_thick"] * sw and length < 15 * sw:
            plates.append([int(x), int(y), int(x + w - 1), int(y + h - 1)])
        else:
            wires.append([float(v) for v in seg])
    return wires, plates


def drop_frame(hw, vw, W, H):
    """Рамка чертежа: горизонтали почти во всю ширину у верхнего/нижнего края."""
    return [s for s in hw if not (s[2] - s[0] > 0.8 * W and (s[1] < 0.08 * H or s[1] > 0.92 * H))]


def drop_frame_v(vw, W, H):
    return [s for s in vw if not (s[3] - s[1] > 0.8 * H and (s[0] < 0.08 * W or s[0] > 0.92 * W))]


def drop_through_text(segs, words, sw):
    """Линия, которая проходит насквозь через несколько надписей, — край листа, сгиб,
    царапина на скане. Провода надписи обходят."""
    if not words:
        return segs
    B = np.array([w["box"] for w in words], dtype=float)
    out = []
    for s in segs:
        x0, y0, x1, y1 = s
        pad = sw
        hit = (B[:, 0] - pad < max(x0, x1)) & (B[:, 2] + pad > min(x0, x1)) & \
              (B[:, 1] - pad < max(y0, y1)) & (B[:, 3] + pad > min(y0, y1))
        if y0 == y1:                                    # горизонталь проходит через середину слова
            inside = hit & (B[:, 1] + 0.25 * (B[:, 3] - B[:, 1]) < y0) & (y0 < B[:, 3] - 0.25 * (B[:, 3] - B[:, 1]))
        else:
            inside = hit & (B[:, 0] + 0.25 * (B[:, 2] - B[:, 0]) < x0) & (x0 < B[:, 2] - 0.25 * (B[:, 2] - B[:, 0]))
        if int(inside.sum()) < 3:
            out.append(s)
    return out


def drop_dashed(segs, axis, sw, min_len):
    """Пунктир: ≥4 коротких отрезка на одной прямой с похожими длинами и промежутками."""
    by_line = {}
    for k, s in enumerate(segs):
        key = round(s[1 - axis] / (2 * sw))            # одна прямая с допуском
        by_line.setdefault(key, []).append(k)
    drop = set()
    for ks in by_line.values():
        ks = sorted(ks, key=lambda k: segs[k][axis])
        run = []
        for k in ks + [None]:
            s = segs[k] if k is not None else None
            if s is not None and segs[k][axis + 2] - segs[k][axis] < 3 * min_len and (
                    not run or 1.5 * sw < s[axis] - segs[run[-1]][axis + 2] < 2 * min_len):
                run.append(k)
                continue
            if len(run) >= 4:
                lens = [segs[r][axis + 2] - segs[r][axis] for r in run]
                if max(lens) < 2.5 * min(lens):
                    drop.update(run)
            run = [k] if s is not None and segs[k][axis + 2] - segs[k][axis] < 3 * min_len else []
    return [s for k, s in enumerate(segs) if k not in drop]


def clip_segments(segs, boxes, keep_len):
    """Разрезать провода, проходящие сквозь символы: остаются куски снаружи.
    Короткий остаток у символа — это его ножка, поэтому порог низкий."""
    out = []
    for s in segs:
        pieces = [s]
        for b in boxes:
            nxt = []
            for p in pieces:
                x0, y0, x1, y1 = p
                if y0 == y1 and b[1] <= y0 <= b[3] and x0 < b[2] and x1 > b[0]:
                    if x0 < b[0]:
                        nxt.append([x0, y0, b[0], y0])
                    if x1 > b[2]:
                        nxt.append([b[2], y0, x1, y0])
                elif x0 == x1 and b[0] <= x0 <= b[2] and y0 < b[3] and y1 > b[1]:
                    if y0 < b[1]:
                        nxt.append([x0, y0, x0, b[1]])
                    if y1 > b[3]:
                        nxt.append([x0, b[3], x0, y1])
                else:
                    nxt.append(p)
            pieces = nxt
        out.extend(p for p in pieces if max(p[2] - p[0], p[3] - p[1]) >= keep_len)
    return out


def find_dots(ink, sw, P):
    # диск толще линии: остаются только залитые пятна (точки, стрелки, толстые полосы)
    d = int(P["dot_d"] * sw) + 2
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
    blobs = cv2.morphologyEx(ink, cv2.MORPH_OPEN, k)
    n, _, st, cent = cv2.connectedComponentsWithStats(blobs, connectivity=8)
    dots = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if (max(w, h) > P["dot_max"] * sw or min(w, h) < P["dot_min"] * sw
                or max(w, h) > 1.6 * min(w, h) or a < 0.5 * w * h):
            continue
        dots.append([float(cent[i][0]), float(cent[i][1]), max(w, h) / 2])
    return dots


def connect(segs, dots, tol):
    """Соединения между отрезками по правилам черчения. Возвращает UnionFind,
    множество «занятых» концов (сегмент, 0|1) и множество отрезков, в середину
    которых что-то упёрлось или которые что-то пересекло."""
    n = len(segs)
    uf = UnionFind(n)
    used, hit = set(), set()
    if n == 0:
        return uf, used, hit
    S = np.array(segs, dtype=float)
    horiz = S[:, 1] == S[:, 3]
    D = np.array([d[:2] for d in dots]) if dots else np.zeros((0, 2))

    def near_dot(x, y):
        return D.size and np.min(np.hypot(D[:, 0] - x, D[:, 1] - y)) <= tol * 1.5

    hi, vi = np.nonzero(horiz)[0], np.nonzero(~horiz)[0]
    V = S[vi]
    for i in hi:
        x0, y, x1, _ = S[i]
        cand = np.nonzero((V[:, 0] >= x0 - tol) & (V[:, 0] <= x1 + tol)
                          & (V[:, 1] - tol <= y) & (V[:, 3] + tol >= y))[0]
        for jj in cand:
            j = vi[jj]
            vx, vy0, vy1 = S[j][0], S[j][1], S[j][3]
            h_end = 0 if abs(vx - x0) <= tol else 1 if abs(vx - x1) <= tol else None
            v_end = 0 if abs(y - vy0) <= tol else 1 if abs(y - vy1) <= tol else None
            if h_end is None:
                hit.add(int(i))
            if v_end is None:
                hit.add(int(j))
            if h_end is None and v_end is None and not near_dot(vx, y):
                continue                                   # пересечение без точки
            uf.union(int(i), int(j))
            if h_end is not None:
                used.add((int(i), h_end))
            if v_end is not None:
                used.add((int(j), v_end))
    # продолжения на одной прямой (разрыв из-за точки или надписи)
    for grp, axis in ((hi, 0), (vi, 1)):
        G = S[grp]
        for ia, a in enumerate(grp):
            pa = S[a]
            near = np.nonzero(np.abs(G[:, 1 - axis] - pa[1 - axis]) <= tol / 2)[0]
            for ib in near:
                b = grp[ib]
                if b <= a:
                    continue
                pb = S[b]
                if abs(pa[axis + 2] - pb[axis]) <= tol:
                    uf.union(int(a), int(b)); used.add((int(a), 1)); used.add((int(b), 0))
                elif abs(pb[axis + 2] - pa[axis]) <= tol:
                    uf.union(int(a), int(b)); used.add((int(b), 1)); used.add((int(a), 0))
    return uf, used, hit


# ---------- признаки символов (для классификатора) ----------

FEAT = 24


def symbol_features(ink, box, sw):
    """Вектор признаков символа: картинка 24×24 (размытая, по центру) + форма."""
    x0, y0, x1, y1 = (int(v) for v in box)
    m = sw
    H, W = ink.shape
    crop = ink[max(0, y0 - m):min(H, y1 + m + 1), max(0, x0 - m):min(W, x1 + m + 1)].astype(np.float32)
    h, w = crop.shape
    side = max(h, w, 1)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    sq = cv2.GaussianBlur(sq, (0, 0), max(1.0, side / FEAT / 2))
    img = cv2.resize(sq, (FEAT, FEAT), interpolation=cv2.INTER_AREA).ravel()
    img /= (np.linalg.norm(img) + 1e-6)
    shape = np.array([math.log((w + 1) / (h + 1)), math.log(max(w, h) / max(sw, 1) + 1) / 4], np.float32)
    return np.concatenate([img, shape])


class SymbolModel:
    """k ближайших соседей по сохранённым примерам learned/symbols.npz."""

    def __init__(self, path=LEARNED / "symbols.npz"):
        self.X = self.y = None
        if Path(path).exists():
            d = np.load(path, allow_pickle=False)
            self.X, self.y = d["X"].astype(np.float32), d["y"]

    def ready(self):
        return self.X is not None and len(self.X) >= 10

    def predict(self, f, k=5):
        sims = self.X[:, :-2] @ f[:-2] - 0.5 * np.abs(self.X[:, -2] - f[-2])
        idx = np.argsort(-sims)[:k]
        votes = {}
        for i in idx:
            votes[str(self.y[i])] = votes.get(str(self.y[i]), 0) + max(sims[i], 0)
        lab = max(votes, key=votes.get)
        return lab, round(float(votes[lab] / (sum(votes.values()) + 1e-6)), 2)


# ---------- главное ----------

def analyze_gray(gray0, P=None, ocr=False, progress=None):
    """Разбор серой картинки. Координаты результата — пиксели исходной картинки."""
    P = P or load_params()

    def step(msg):
        if progress:
            progress(msg)

    H0, W0 = gray0.shape
    s = int(max(1, min(4, round(P["target"] / max(H0, W0)))))
    gray = cv2.resize(gray0, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else gray0.copy()
    H, W = gray.shape
    step("бинаризация")
    ink = binarize(gray)
    sw = stroke_width(ink)
    if P.get("thin_upscale", 1) and s == 1 and sw <= 1.2 and max(H0, W0) <= 1.6 * P["target"]:
        # линии в 1 px: короткие выводы между соседними деталями короче минимального
        # провода и теряются — разбираем в двойном масштабе
        s = 2
        gray = cv2.resize(gray0, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        H, W = gray.shape
        ink = binarize(gray)
        sw = stroke_width(ink)
    tol = P["tol"] * sw + 2
    min_len = int(max(P["min_len"] * sw, 12))

    step("надписи")
    words, text_mask = find_text(ink, sw, P)
    work = cv2.subtract(ink, text_mask)

    step("корпуса")
    rects = find_rects(work, sw, W, H, P)
    rect_mask = np.zeros_like(work)
    for r in rects:
        x0, y0, x1, y1 = r["box"]
        cv2.rectangle(rect_mask, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), 255, -1)
    lines_src = cv2.subtract(work, rect_mask)

    step("провода")
    hmask = cv2.morphologyEx(lines_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    vmask = cv2.morphologyEx(lines_src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len)))
    hw, hp = line_segments(hmask, True, sw, min_len, P)
    vw, vp = line_segments(vmask, False, sw, min_len, P)
    dots = find_dots(work, sw, P)

    hw, vw = drop_frame(hw, vw, W, H), drop_frame_v(vw, W, H)
    hw, vw = drop_dashed(hw, 0, sw, min_len), drop_dashed(vw, 1, sw, min_len)
    hw, vw = drop_through_text(hw, words, sw), drop_through_text(vw, words, sw)

    # короткий отрезок со свободными концами, в середину которого упёрся или
    # который пересёк провод, — штрих символа: полоска диода, обкладка, основание
    wires = hw + vw
    _, used0, hit0 = connect(wires, dots, tol)
    stroke_ids = {i for i, sg in enumerate(wires)
                  if (i, 0) not in used0 and (i, 1) not in used0 and i in hit0
                  and max(sg[2] - sg[0], sg[3] - sg[1]) < P["stroke_len"] * min_len}
    strokes = [wires[i] for i in stroke_ids]
    wires = [sg for i, sg in enumerate(wires) if i not in stroke_ids]

    step("символы")
    wire_mask = np.zeros_like(work)
    for x0, y0, x1, y1 in wires:
        cv2.line(wire_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, sw + 2)
    dot_mask = np.zeros_like(work)
    for x, y, r in dots:
        cv2.circle(dot_mask, (int(x), int(y)), int(r + sw + 1), 255, -1)
    resid = cv2.subtract(cv2.subtract(cv2.subtract(lines_src, wire_mask), dot_mask), rect_mask)
    resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for x0, y0, x1, y1 in hp + vp:
        cv2.rectangle(resid, (x0, y0), (x1, y1), 255, -1)
    for x0, y0, x1, y1 in strokes:
        cv2.line(resid, (int(x0), int(y0)), (int(x1), int(y1)), 255, sw + 1)
    g = int(P["sym_grow"] * sw) + 1
    grow = cv2.dilate(resid, cv2.getStructuringElement(cv2.MORPH_RECT, (g, g)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(grow, connectivity=8)
    symbols = [{"box": r["box"], "kind": r["kind"], "cells": r["cells"]} for r in rects]
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        real = int(np.count_nonzero(resid[y:y + h, x:x + w][lab[y:y + h, x:x + w] == i]))
        if real < P["sym_area"] * sw * sw:
            continue
        if max(w, h) > 10 * min(w, h) and min(w, h) < 5 * sw:
            continue                                         # пунктирная линия
        if max(w, h) > 0.35 * max(W, H):
            continue
        symbols.append({"box": [int(x), int(y), int(x + w - 1), int(y + h - 1)], "kind": "?"})

    step("цепи")
    sym_boxes = [sy["box"] for sy in symbols if sy["kind"] == "?"]
    # залитое пятно внутри символа (остриё диода у полоски, стрелка, толстая
    # обкладка) — часть символа, а не точка соединения
    dots = [d for d in dots if not any(b[0] + sw <= d[0] <= b[2] - sw and b[1] + sw <= d[1] <= b[3] - sw
                                       for b in sym_boxes)]
    segs = clip_segments(wires, sym_boxes, tol)
    segs = [[float(v) for v in sg] for sg in segs]
    uf, used, _ = connect(segs, dots, tol)

    # свободные концы → выводы символов или надписи
    pins, ports = [], []
    SB = np.array([sy["box"] for sy in symbols], dtype=float).reshape(-1, 4)
    for i, (x0, y0, x1, y1) in enumerate(segs):
        for end, (x, y) in enumerate(((x0, y0), (x1, y1))):
            if (i, end) in used:
                continue
            if len(SB):
                dx = np.maximum(np.maximum(SB[:, 0] - x, 0), x - SB[:, 2])
                dy = np.maximum(np.maximum(SB[:, 1] - y, 0), y - SB[:, 3])
                dist = np.hypot(dx, dy)
                k = int(np.argmin(dist))
                if dist[k] <= P["pin_dist"] * tol:
                    pins.append({"seg": i, "x": x, "y": y, "symbol": k})
                    continue
            tb = min(((box_dist(wd["box"], x, y), k) for k, wd in enumerate(words)), default=(1e9, -1))
            ports.append({"seg": i, "x": x, "y": y, "text": tb[1] if tb[0] <= 6 * sw else None})

    roots = {}
    for i in range(len(segs)):
        roots.setdefault(uf.find(i), []).append(i)
    # кучка коротких штрихов без выводов в маленькой области — крупная буква/цифра
    pinned = {p["seg"] for p in pins}
    for r, gr in list(roots.items()):
        xs = [v for i in gr for v in (segs[i][0], segs[i][2])]
        ys = [v for i in gr for v in (segs[i][1], segs[i][3])]
        if (not pinned.intersection(gr) and max(xs) - min(xs) < 3 * min_len
                and max(ys) - min(ys) < 3 * min_len):
            del roots[r]
    kept = {i for gr in roots.values() for i in gr}
    ports = [p for p in ports if p["seg"] in kept]
    order = sorted(roots.values(),
                   key=lambda gr: -sum(abs(segs[i][2] - segs[i][0]) + abs(segs[i][3] - segs[i][1]) for i in gr))
    seg_net, nets = {}, []
    for k, gr in enumerate(order):
        for i in gr:
            seg_net[i] = k
        nets.append({"id": f"N{k + 1}", "segments": gr, "pins": [], "labels": [], "dangling": 0})
    for p in pins:
        nets[seg_net[p["seg"]]]["pins"].append(p)
    for p in ports:
        net = nets[seg_net[p["seg"]]]
        if p["text"] is not None:
            net["labels"].append(p["text"])
        else:
            net["dangling"] += 1

    two_lead_resistors(symbols, pins, W, H)
    texts_ok = try_ocr(gray, words) if ocr else False
    feats = []
    for k, sy in enumerate(symbols):
        sy["id"] = f"S{k + 1}"
        sy["near_text"] = sorted((k2 for k2, wd in enumerate(words)
                                  if box_dist(sy["box"], *box_center(wd["box"])) <= 8 * sw),
                                 key=lambda k2: box_dist(sy["box"], *box_center(words[k2]["box"])))[:3]
        feats.append(symbol_features(ink, sy["box"], sw))

    def sc(v):
        return round(v / s, 1)

    res = {
        "scale": s, "stroke": round(sw / s, 2), "ocr": texts_ok, "params": P,
        # схема — почти чисто чёрно-белая; фото платы/прибора — много полутонов
        "midtones": round(float(((gray0 > 70) & (gray0 < 190)).mean()), 3),
        "segments": [[sc(v) for v in sg] for sg in segs],
        "dots": [[sc(x), sc(y)] for x, y, _ in dots],
        "texts": [], "symbols": [], "nets": [],
    }
    for k, wd in enumerate(words):
        res["texts"].append({"id": f"T{k + 1}", "box": [sc(v) for v in wd["box"]], "vertical": wd["vertical"],
                             "text": wd.get("text", ""), "conf": wd.get("conf")})
    for sy in symbols:
        res["symbols"].append({"id": sy["id"], "kind": sy["kind"], "box": [sc(v) for v in sy["box"]],
                               "near_text": [f"T{k + 1}" for k in sy["near_text"]]})
    for net in nets:
        res["nets"].append({
            "id": net["id"], "segments": net["segments"],
            "pins": [{"symbol": symbols[p["symbol"]]["id"], "x": sc(p["x"]), "y": sc(p["y"])} for p in net["pins"]],
            "labels": [f"T{k + 1}" for k in net["labels"]], "dangling": net["dangling"],
        })
    res["_features"] = feats                     # для обучения; в файл не пишется
    return res


def classify(res, model=None):
    """Подписать типы символов обученным классификатором (если он есть)."""
    model = model or SymbolModel()
    if not model.ready():
        return False
    for sy, f in zip(res["symbols"], res["_features"]):
        if sy["kind"] in ("R", "U"):
            continue
        lab, conf = model.predict(f)
        if lab != "OTHER":
            sy["kind"], sy["kind_conf"] = lab, float(conf)
    return True


def analyze(project, ocr=True, progress=None, params=None):
    project = Path(project)
    manifest = core.load_json(project / "manifest.json")
    gray0 = core.imread(project / "clean.png")[:, :, 0]
    res = analyze_gray(gray0, load_params(params), ocr, progress)
    res["classified"] = classify(res)
    cell = manifest["grid"]["cell"]

    def cell_of(x, y):
        return f"{core.col_name(int(x // cell))}{int(y // cell) + 1}"

    for t in res["texts"]:
        t["cell"] = cell_of(*box_center(t["box"]))
    for sy in res["symbols"]:
        sy["cell"] = cell_of(*box_center(sy["box"]))
    for net in res["nets"]:
        for p in net["pins"]:
            p["cell"] = cell_of(p["x"], p["y"])
    if progress:
        progress("картинки")
    render(project, manifest, gray0, res)
    return res


# ---------- картинки и черновой netlist ----------

def palette(n):
    cols = []
    for i in range(max(n, 1)):
        h = (i * 0.618034) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.80 if i % 2 else 0.62)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))
    return cols


def draw(gray0, res, k):
    """Цветная картинка в масштабе k: бледная схема, цепи цветом, символы рамками."""
    base = cv2.resize(gray0, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1 else gray0.copy()
    base = (255 - (255 - base.astype(np.float32)) * 0.35).astype(np.uint8)
    img = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    cols = palette(len(res["nets"]))
    th = max(2, int(round(res["stroke"] * k * 1.6)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    for sy in res["symbols"]:
        x0, y0, x1, y1 = (int(v * k) for v in sy["box"])
        cv2.rectangle(img, (x0, y0), (x1, y1), (200, 120, 0), 1)
        tag = sy["id"] + ("" if sy["kind"] == "?" else ":" + sy["kind"])
        cv2.putText(img, tag, (x0, max(10, y0 - 3)), font, 0.35 * k / 2 + 0.2, (200, 120, 0), 1)
    for ni, net in enumerate(res["nets"]):
        c = cols[ni]
        longest, best = -1, None
        for si in net["segments"]:
            x0, y0, x1, y1 = (v * k for v in res["segments"][si])
            cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), c, th)
            L = abs(x1 - x0) + abs(y1 - y0)
            if L > longest:
                longest, best = L, ((x0 + x1) / 2, (y0 + y1) / 2)
        for p in net["pins"]:
            cv2.circle(img, (int(p["x"] * k), int(p["y"] * k)), th + 2, c, -1)
        if best and (len(net["segments"]) > 1 or net["pins"]):
            fs = 0.3 * k / 2 + 0.25
            (tw, tht), _ = cv2.getTextSize(net["id"], font, fs, 1)
            x, y = int(best[0] - tw / 2), int(best[1] - th - 2)
            cv2.rectangle(img, (x - 1, y - tht - 1), (x + tw + 1, y + 2), (255, 255, 255), -1)
            cv2.putText(img, net["id"], (x, y), font, fs, c, 1, cv2.LINE_AA)
    for x, y in res["dots"]:
        cv2.circle(img, (int(x * k), int(y * k)), th + 1, (0, 0, 0), -1)
    return img


def render(project, manifest, gray0, res):
    out = project / "analysis"
    if (out / "tiles").exists():
        shutil.rmtree(out / "tiles")
    (out / "tiles").mkdir(parents=True)
    grid = manifest["grid"]
    k = 4
    img = draw(gray0, res, k)
    grid_k = dict(grid, cell=grid["cell"] * k, width=grid["width"] * k, height=grid["height"] * k)
    ov = min(1.0, (core.TILE_MAX - 2 * core.MARGIN) / max(img.shape[:2]))
    core.imwrite(out / "nets.png", core.render_region(img, grid_k, 0, 0, grid["cols"], grid["rows"], ov))
    for t in manifest["tiles"]:
        c0, r0, c1, r1 = core.parse_range(t["cells"])
        w = (min(grid["width"], (c1 + 1) * grid["cell"]) - c0 * grid["cell"]) * k
        h = (min(grid["height"], (r1 + 1) * grid["cell"]) - r0 * grid["cell"]) * k
        sc = min(1.0, (core.TILE_MAX - 2 * core.MARGIN) / max(w, h))
        core.imwrite(out / t["file"], core.render_region(img, grid_k, c0, r0, c1 + 1, r1 + 1, sc))
    saved = {k2: v for k2, v in res.items() if not k2.startswith("_")}
    (out / "analysis.json").write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    (out / "netlist_auto.json").write_text(json.dumps(auto_netlist(saved), ensure_ascii=False, indent=1),
                                           encoding="utf-8")


def auto_netlist(res):
    """Черновик в формате программы: символы → компоненты, цепи N… → выводы."""
    texts = {t["id"]: t for t in res["texts"]}
    # класс символа → тип детали netlist
    known = {"R": "R", "C": "C", "L": "L", "D": "D", "Z": "Z", "Q": "Q", "M": "M", "U": "U", "J": "J",
             "OPAMP": "U", "MCU": "MCU", "SENSOR": "SENSOR", "RT": "RT", "TC": "TC", "XTAL": "XTAL", "LED": "LED",
             "TERM": "TERM", "ENC": "ENC", "TACHO": "TACHO", "HALL": "HALL", "PORT": "J",
             "SW_NO": "SW", "SW_NC": "SW", "BTN": "BTN", "COIL": "K", "LAMP": "LAMP", "THERMAL": "SW",
             "FU": "F", "QF": "QF", "MOTOR": "MOTOR"}
    comps = {}
    for sy in res["symbols"]:
        comps[sy["id"]] = {"ref": sy["id"], "type": known.get(sy["kind"], "X"),
                           "value": "", "cell": sy.get("cell", ""), "pins": {}, "confidence": "low"}
        near = [texts[t]["text"] or t for t in sy["near_text"]]
        if near:
            comps[sy["id"]]["note"] = "надписи рядом: " + ", ".join(near)
    for net in res["nets"]:
        for p in net["pins"]:
            c = comps[p["symbol"]]
            c["pins"][f"p{len(c['pins']) + 1}"] = net["id"]
    nets = {}
    for net in res["nets"]:
        meta = {}
        if net["labels"]:
            meta["label"] = ", ".join(texts[t]["text"] or t for t in net["labels"])
            meta["port"] = True
        if net["dangling"]:
            meta["port"] = True
        if meta:
            nets[net["id"]] = meta
    return {"title": "Черновик анализатора (без ИИ)",
            "components": [c for c in comps.values() if c["pins"]], "nets": nets}


def summary(res):
    real = [n for n in res["nets"] if len(n["pins"]) + len(n["labels"]) >= 2]
    warn = ""
    mt = res.get("midtones", 0)
    # пороги — по реальным фото: схемы с телефона и синьки до ≈0,21, фото техники от ≈0,23
    if mt > 0.22:
        warn = (f"; ВНИМАНИЕ: {mt * 100:.0f} % полутонов — похоже на фото платы или прибора, "
                "а не на схему; разбор без ИИ здесь ненадёжен")
    elif mt > 0.12:
        warn = f"; низкий контраст ({mt * 100:.0f} % полутонов: синька или фото) — разбор без ИИ менее надёжен"
    if len(res["texts"]) > 2.5 * max(len(res["segments"]), 1) and len(res["segments"]) < 120:
        warn += "; похоже на страницу текста, а не на схему"
    return (f"символов {len(res['symbols'])}, проводов {len(res['segments'])}, точек {len(res['dots'])}, "
            f"надписей {len(res['texts'])}, цепей {len(real)} (с ≥2 выводами/метками), "
            f"OCR {'есть' if res['ocr'] else 'нет'}, классификатор {'есть' if res.get('classified') else 'нет'}" + warn)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    r = analyze(sys.argv[1], progress=lambda m: print("…", m))
    print(summary(r))
