"""Собственное распознавание надписей схем — без внешних программ.

Буквы рисуются шрифтами Windows (Arial, Times, Consolas, Calibri, ГОСТ тип A/B…)
и шрифтами OpenCV, с наклоном, толщиной, размытием и шумом; классификатор —
ближайшие соседи по картинке 16×16 и пропорциям. Слово собирается из букв,
затем исправляется по правилам обозначений схем (R12, VT3, 2,2k, 100n, +5V):
в обозначении сначала буквы, потом цифры; в номинале цифры, потом множитель.

    python ocr.py train             — обучить (learned/ocr.npz)
    python ocr.py test              — точность на отложенных образцах
"""
import math
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "learned" / "ocr.npz"
FONTS = Path("C:/Windows/Fonts")
FONT_FILES = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "ARIALN.TTF", "times.ttf", "timesi.ttf", "cour.ttf",
              "consola.ttf", "calibri.ttf", "tahoma.ttf", "verdana.ttf", "segoeui.ttf", "GOST_A.TTF",
              "GOST_B.TTF", "GOST_AU.ttf", "GOST_BU.ttf", "isocpeur.ttf", "micross.ttf", "lucon.ttf"]
CHARS = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghkmnprstuvxyz+-.,/()=~*") + ["μ", "Ω"]
# у многих букв нижний и верхний регистр не различить по одной форме — объединяем
MERGE = {"c": "C", "o": "O", "s": "S", "u": "U", "v": "V", "w": "W", "x": "X", "z": "Z", "p": "P", "k": "k"}
N = 16


def _glyph_pil(ch, font_path, size):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f = ImageFont.truetype(str(font_path), size)
    except OSError:
        return None
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size // 2, size // 2), ch, fill=255, font=f)
    return np.array(img)


def _glyph_cv(ch, face, scale, th):
    if not ch.isascii():
        return None
    img = np.zeros((90, 90), np.uint8)
    cv2.putText(img, ch, (20, 65), face, scale, 255, th, cv2.LINE_AA)
    return img


def _augment(g, rng):
    """Наклон (курсив), поворот, толщина, размытие, шум, перешаг дискретизации."""
    h, w = g.shape
    sh = rng.uniform(-0.25, 0.25)
    M = np.float32([[1, sh, -sh * h / 2], [0, 1, 0]])
    ang = rng.uniform(-4, 4)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    g = cv2.warpAffine(g, M, (w, h))
    g = cv2.warpAffine(g, R, (w, h))
    k = rng.choice([0, 0, 1, -1])
    if k > 0:
        g = cv2.dilate(g, np.ones((2, 2), np.uint8))
    elif k < 0:
        g = cv2.erode(g, np.ones((2, 2), np.uint8))
    s = rng.choice([rng.uniform(0.12, 0.3), rng.uniform(0.25, 0.8)])   # мелкий кегль, как на скане и фото
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        g = cv2.GaussianBlur(g, (0, 0), rng.uniform(0.3, 1.2))
    g = g.astype(np.float32) + np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, rng.uniform(0, 25), g.shape)
    return np.clip(g, 0, 255).astype(np.uint8)


def features(ink):
    """ink — 0/255, буква белым. Картинка 16×16 по рамке буквы + пропорции."""
    ys, xs = np.nonzero(ink > 100)
    if len(xs) == 0:
        return None
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    v = cv2.resize(sq, (N, N), interpolation=cv2.INTER_AREA).ravel()
    v /= np.linalg.norm(v) + 1e-6
    fill = float((crop > 100).mean())
    return np.concatenate([v, [math.log((w + 1) / (h + 1)) * 0.6, fill * 0.5]]).astype(np.float32)


def train(n_aug=10, seed=1):
    rng = random.Random(seed)
    X, y, H = [], [], []                              # H — высота буквы относительно «прописной»
    fonts = [FONTS / f for f in FONT_FILES if (FONTS / f).exists()]
    faces = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX]
    for ch in CHARS:
        bases = []
        for f in fonts:
            for size in (28, 40):
                g = _glyph_pil(ch, f, size)
                if g is not None and g.max() > 0:
                    bases.append(g)
        for face in faces:
            g = _glyph_cv(ch, face, 1.6 if face != cv2.FONT_HERSHEY_PLAIN else 3.0, 2)
            if g is not None:
                bases.append(g)
        for g in bases:
            for _ in range(n_aug):
                a = _augment(g, rng)
                _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                f = features(a)
                if f is not None:
                    X.append(f)
                    y.append(MERGE.get(ch, ch))
    MODEL.parent.mkdir(exist_ok=True)
    np.savez_compressed(MODEL, X=np.array(X, np.float16), y=np.array(y))
    return len(X), len(fonts)


EXTRA = HERE / "learned" / "ocr_extra.npz"


class OCR:
    def __init__(self):
        d = np.load(MODEL, allow_pickle=False)
        X, y = [d["X"].astype(np.float32)], [d["y"]]
        if EXTRA.exists():                              # буквы, вырезанные из настоящих и учебных схем
            e = np.load(EXTRA, allow_pickle=False)
            X.append(e["X"].astype(np.float32))
            y.append(e["y"])
        self.X, self.y = np.concatenate(X), np.concatenate(y)

    def char(self, ink, k=5):
        f = features(ink)
        if f is None:
            return "", 0.0, []
        s = self.X[:, :-2] @ f[:-2] - np.abs(self.X[:, -2] - f[-2]) - 0.5 * np.abs(self.X[:, -1] - f[-1])
        idx = np.argpartition(-s, k)[:k]
        votes = {}
        for i in idx:
            votes[str(self.y[i])] = votes.get(str(self.y[i]), 0) + max(float(s[i]), 0.01)
        ranked = sorted(votes.items(), key=lambda kv: -kv[1])
        tot = sum(votes.values())
        return ranked[0][0], ranked[0][1] / tot, [c for c, _ in ranked]

    @staticmethod
    def _split_wide(boxes, lab, hmax):
        """Слипшиеся буквы: компонент шире ~0,9 высоты строки режем по минимумам
        вертикальной проекции на куски примерно квадратной ширины."""
        out = []
        for b in boxes:
            x0, y0, x1, y1, ids = b
            w = x1 - x0
            if w < 1.2 * hmax or (y1 - y0) < 0.6 * hmax:
                out.append(b)
                continue
            m = np.isin(lab[y0:y1, x0:x1], ids)
            col = m.sum(0).astype(float)
            n = max(2, int(round(w / (0.62 * hmax))))
            cuts = []
            for j in range(1, n):
                c = int(j * w / n)
                lo, hi = max(1, c - int(0.2 * hmax)), min(w - 1, c + int(0.2 * hmax))
                if hi > lo:
                    c = lo + int(np.argmin(col[lo:hi]))
                cuts.append(c)
            edges = [0] + cuts + [w]
            for a, c in zip(edges, edges[1:]):
                if c - a < 2:
                    continue
                sub = m[:, a:c]
                ys = np.nonzero(sub.any(1))[0]
                if len(ys):
                    out.append([x0 + a, y0 + ys[0], x0 + c, y0 + ys[-1] + 1, ids, (a, c)])
        return out

    def word(self, gray, box, vertical=False, glyphs=None):
        """Распознать надпись в рамке box (x0,y0,x1,y1) серой картинки.
        glyphs — список, куда складываются маски букв (для обучения на схемах)."""
        x0, y0, x1, y1 = (int(v) for v in box)
        pad = 3
        crop = gray[max(0, y0 - pad):y1 + pad + 1, max(0, x0 - pad):x1 + pad + 1]
        if crop.size == 0:
            return "", 0.0
        if vertical:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        # мелкий текст — увеличить до ~30 пикселей по высоте, иначе буквы слипаются при пороге
        k = max(1.0, min(5.0, 30.0 / max(crop.shape[0], 1)))
        if k > 1.2:
            crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        _, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
        comps = [i for i in range(1, n) if st[i, 4] >= 3]
        if not comps:
            return "", 0.0
        hmax = max(st[i, 3] for i in comps)
        # склеить части одной буквы, стоящие друг над другом (i, j, «:»), и отбросить пыль
        boxes = []
        for i in sorted(comps, key=lambda i: st[i, 0]):
            x, y, w, h, a = st[i]
            if h < 0.2 * hmax and w < 0.2 * hmax and a < 6:
                continue
            if boxes and x < boxes[-1][2] - 0.35 * w and (x + w) <= boxes[-1][2] + 0.35 * w:
                b = boxes[-1]
                boxes[-1] = [min(b[0], x), min(b[1], y), max(b[2], x + w), max(b[3], y + h), b[4] + [i]]
            else:
                boxes.append([x, y, x + w, y + h, [i]])
        boxes = self._split_wide(boxes, lab, hmax)
        out, confs = [], []
        prev_r = None
        for b in boxes:
            bx0, by0, bx1, by1, ids = b[:5]
            m = np.isin(lab[by0:by1, bx0:bx1], ids).astype(np.uint8) * 255
            h = by1 - by0
            if prev_r is not None and bx0 - prev_r > 0.6 * hmax:
                out.append(" ")
            if h < 0.35 * hmax and (bx1 - bx0) < 0.5 * hmax:
                # точка/запятая/дефис: по положению и форме
                ch = "-" if (bx1 - bx0) > 1.8 * h and by1 < crop.shape[0] * 0.8 else "."
                out.append(ch)
                confs.append(0.6)
            else:
                if glyphs is not None:
                    glyphs.append(m)
                ch, c, alt = self.char(m)
                if ch in "Pp" and h < 0.75 * hmax:
                    ch = "p"
                out.append(ch)
                confs.append(c)
            prev_r = bx1
        return "".join(out).strip(), float(np.mean(confs)) if confs else 0.0


def learn_word(engine, gray, box, vertical, truth_text):
    """Если надпись разрезалась на столько же букв, сколько в ответе (без точек и пробелов), —
    вернуть пары (признаки, буква) для дообучения."""
    glyphs = []
    engine.word(gray, box, vertical, glyphs)
    want = [c for c in truth_text if c not in " .,-"]
    if len(glyphs) != len(want):
        return []
    out = []
    for m, ch in zip(glyphs, want):
        f = features(m)
        if f is not None:
            out.append((f, MERGE.get(ch, ch)))
    return out


def save_extra(X, y, keep=60000):
    if EXTRA.exists():
        e = np.load(EXTRA, allow_pickle=False)
        X = np.concatenate([e["X"].astype(np.float32), np.array(X, np.float32)])
        y = np.concatenate([e["y"], np.array(y)])
    X, y = np.array(X)[-keep:], np.array(y)[-keep:]
    np.savez_compressed(EXTRA, X=X.astype(np.float16), y=y)
    return len(y)


# ---------- правила обозначений схем ----------

REF_PREFIX = ["DA", "DD", "VD", "VT", "VS", "HL", "SB", "SA", "KM", "KK", "KT", "QF", "FU", "XT", "XS", "XP", "BQ", "BH",
              "BR", "BP", "BK", "RK", "RP", "ZQ", "IC", "TR", "R", "C", "L", "D", "Q", "U", "K", "T", "J", "X", "F",
              "S", "M", "P", "Z"]
TO_DIGIT = {"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "i": "1", "J": "1", "Z": "2", "S": "5",
            "s": "5", "B": "8", "G": "6", "b": "6", "g": "9", "q": "9", "A": "4", "T": "7", "/": "1", "(": "1"}
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G", "4": "A", "7": "T"}
MULT = "pnuμmkKMR"


def fix_ref(t):
    """'RI2' → 'R12', 'VTI' → 'VT1'. None, если на обозначение не похоже."""
    t = t.replace(" ", "").replace(".", "")
    if not 2 <= len(t) <= 6:
        return None
    up = t.upper()
    for p in REF_PREFIX:
        head = "".join(TO_LETTER.get(c, c) for c in up[:len(p)])
        if head == p and len(up) > len(p):
            tail = "".join(TO_DIGIT.get(c, c) for c in t[len(p):])
            tail = tail.replace(" ", "")
            m = re.fullmatch(r"(\d{1,4})([.]\d{1,2})?([A-Za-z])?", tail)
            if m:
                return p + m.group(1) + (m.group(2) or "") + (m.group(3) or "")
    return None


def fix_value(t):
    """'2,2k' '2.2K' '4k7' '100n' '47μ' '1M' '0.22' → нормализованный номинал или None."""
    t = t.replace(" ", "").replace(",", ".")
    if not t or len(t) > 9:
        return None
    t = t.replace("u", "μ").replace("U", "μ")
    head = "".join(TO_DIGIT.get(c, c) if i == 0 else c for i, c in enumerate(t))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([pnμmkKMR])?(\d*)(F|Ω|V|W)?", head)
    if not m:
        m2 = re.fullmatch(r"(\d+)([pnμkKMR])(\d+)", head)        # 4k7
        if not m2:
            return None
        return f"{m2.group(1)}.{m2.group(3)}{m2.group(2).replace('K', 'k')}"
    num, mult, tail, unit = m.groups()
    if tail and mult:
        num = f"{num}.{tail}"
    elif tail:
        return None
    mult = (mult or "").replace("K", "k").replace("R", "")
    return f"{num}{mult}{unit or ''}"


NET_NAMES = re.compile(r"^([+-]?\d{1,3}([.,]\d)?[VВ]|GND|0V|AGND|DGND|VCC|VDD|VEE|VSS|[+-]U\w*|L\d?|N|PE)$", re.I)


def fix_net(t):
    t = t.replace(" ", "")
    if NET_NAMES.match(t):
        return t.upper().replace("В", "V")
    return None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "train":
        n, nf = train()
        print(f"образцов {n}, шрифтов {nf}")
    elif cmd == "test":
        # отложенные образцы: другое зерно, все шрифты
        rng = random.Random(77)
        o = OCR()
        fonts = [FONTS / f for f in FONT_FILES if (FONTS / f).exists()]
        ok = tot = 0
        conf = {}
        for ch in CHARS:
            for f in fonts:
                g = _glyph_pil(ch, f, 34)
                if g is None or g.max() == 0:
                    continue
                a = _augment(g, rng)
                _, a = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                p, _, _ = o.char(a)
                want = MERGE.get(ch, ch)
                tot += 1
                ok += p == want
                if p != want:
                    conf[(want, p)] = conf.get((want, p), 0) + 1
        print(f"точность по буквам: {ok / tot:.3f} ({tot})")
        print("частые путаницы:", sorted(conf.items(), key=lambda kv: -kv[1])[:12])
    else:
        print(__doc__)
