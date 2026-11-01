"""Классификатор символов по форме: перцептрон (numpy) на признаках формы.

Признаки выреза символа (48×48, линии нормализованы по толщине):
  • HOG — гистограммы направлений штрихов по клеткам (диагональ контакта, треугольник диода, дуги катушки);
  • уменьшенная картинка 12×12 — общий силуэт;
  • число замкнутых контуров (окружность транзистора, корпус, треугольник), доля «чернил»,
    касания краёв (куда уходят выводы), симметрия, пропорции и размер в толщинах линии.

Используется как второе мнение к kNN (analyzer.classify): на 13 проверенных листах, когда
обе модели согласны, тип верен в 33 % случаев против 21 % у одного kNN; при несогласии деталь
помечается неуверенной и уходит в вопросы.

Данные — learned/symdata.npz: вырезы, размеры, метка, источник (synth / ref / real:лист),
поэтому признаки и модель можно менять без повторного разбора схем.
    python symclass.py train      — обучить и сохранить learned/symbol_mlp.*
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

LEARNED = Path(__file__).resolve().parent / "learned"
DATA = LEARNED / "symdata.npz"
MODEL = LEARNED / "symbol_mlp.npz"
S = 48


def hog(c, cell=8, bins=9):
    """Гистограммы направлений градиента по клеткам 8×8, нормированные по блокам 2×2 клетки."""
    f = c.astype(np.float32) / 255
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
    b0 = np.floor(ang).astype(int) % bins
    n = S // cell
    H = np.zeros((n, n, bins), np.float32)
    ci = (np.arange(S) // cell)
    np.add.at(H, (ci[:, None].repeat(S, 1), ci[None, :].repeat(S, 0), b0), mag)
    out = []
    for i in range(n - 1):
        for j in range(n - 1):
            v = H[i:i + 2, j:j + 2].ravel()
            out.append(v / (np.linalg.norm(v) + 1e-3))
    return np.concatenate(out)


def crop(ink, box, sw):
    """Вырез символа: квадрат по большей стороне, линии сведены к одной толщине (≈2 px на 48)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    m = max(1, int(sw))
    H, W = ink.shape
    c = ink[max(0, y0 - m):min(H, y1 + m + 1), max(0, x0 - m):min(W, x1 + m + 1)]
    h, w = c.shape
    side = max(h, w, 1)
    sq = np.zeros((side, side), np.uint8)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = c
    k = max(1, int(round(sw * 0.5)))
    if k > 1:                                            # толстые линии тоньше, чтобы масштаб не менял вид
        sq = cv2.erode(sq, np.ones((k, k), np.uint8))
    out = cv2.resize(sq, (S, S), interpolation=cv2.INTER_AREA)
    out = cv2.dilate((out > 40).astype(np.uint8) * 255, np.ones((2, 2), np.uint8))
    return out, np.array([w, h, sw], np.float32)


def features(c, dims):
    w, h, sw = dims
    hogv = hog(c)
    small = cv2.resize(c, (12, 12), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255
    b = (c > 127).astype(np.uint8)
    cnts, hier = cv2.findContours(b, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = sum(1 for i, cn in enumerate(cnts) if hier is not None and hier[0][i][3] >= 0
                and cv2.contourArea(cn) > 6)
    parts = sum(1 for i in range(len(cnts)) if hier is not None and hier[0][i][3] < 0)
    fill = b.mean()
    edges = [b[0].mean(), b[-1].mean(), b[:, 0].mean(), b[:, -1].mean()]
    sym_lr = 1 - np.abs(b - b[:, ::-1]).mean() / (fill + 1e-6) / 2
    sym_tb = 1 - np.abs(b - b[::-1]).mean() / (fill + 1e-6) / 2
    shape = [math.log((w + 1) / (h + 1)), math.log(max(w, h) / max(sw, 1) + 1) / 3, min(holes, 6) / 3,
             min(parts, 8) / 4, fill * 4, *[e * 3 for e in edges], sym_lr, sym_tb]
    return np.concatenate([hogv, small, np.array(shape, np.float32)]).astype(np.float32)


def compact(c, dims):
    """Компактные признаки (≈200): HOG, усреднённый по четвертям и целиком, силуэт 12×12 и форма.
    На реальных листах полный HOG (900) переобучается на синтетику, компактный — нет."""
    f = features(c, dims)
    h = f[:900].reshape(5, 5, 36)
    pooled = [h[:3, :3].mean((0, 1)), h[2:, :3].mean((0, 1)), h[:3, 2:].mean((0, 1)), h[2:, 2:].mean((0, 1)),
              h.mean((0, 1))]
    return np.concatenate(pooled + [f[900:]]).astype(np.float32)


def save(crops, dims, y, src):
    """Добавить вырезы к learned/symdata.npz; примеры того же источника заменяются."""
    if not len(y):
        return 0
    C = np.array(crops, np.uint8)
    D = np.array(dims, np.float32)
    Y = np.array(y)
    R = np.array([src] * len(y))
    if DATA.exists():
        d = np.load(DATA)
        keep = d["src"] != src
        C, D = np.concatenate([d["crops"][keep], C]), np.concatenate([d["dims"][keep], D])
        Y, R = np.concatenate([d["y"][keep], Y]), np.concatenate([d["src"][keep], R])
    LEARNED.mkdir(exist_ok=True)
    np.savez_compressed(DATA, crops=C, dims=D, y=Y, src=R)
    return len(Y)


def _matrix(d, idx):
    return np.stack([features(d["crops"][i], d["dims"][i]) for i in idx])


class MLP:
    """Перцептрон: вход → ReLU → ReLU → softmax; обучение Adam с весами примеров и L2."""

    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, math.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    def forward(self, X):
        hs = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            hs.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return hs

    def proba(self, X):
        z = self.forward(X)[-1]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)

    def fit(self, X, y, w, epochs=40, lr=2e-3, l2=1e-4, batch=256, seed=0):
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t = 0
        for ep in range(epochs):
            order = rng.permutation(len(X))
            for k in range(0, len(X), batch):
                idx = order[k:k + batch]
                hs = self.forward(X[idx])
                z = hs[-1]
                p = np.exp(z - z.max(1, keepdims=True))
                p /= p.sum(1, keepdims=True)
                g = p.copy()
                g[np.arange(len(idx)), y[idx]] -= 1
                g *= (w[idx] / len(idx))[:, None]
                grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
                for i in range(len(self.W) - 1, -1, -1):
                    grads_W[i] = hs[i].T @ g + l2 * self.W[i]
                    grads_b[i] = g.sum(0)
                    if i:
                        g = (g @ self.W[i].T) * (hs[i] > 0)
                t += 1
                for j, (p_, gr) in enumerate(zip(params, grads_W + grads_b)):
                    m[j] = 0.9 * m[j] + 0.1 * gr
                    v[j] = 0.999 * v[j] + 0.001 * gr * gr
                    p_ -= lr * (m[j] / (1 - 0.9 ** t)) / (np.sqrt(v[j] / (1 - 0.999 ** t)) + 1e-8)
        return self


def dataset(exclude_real=False):
    with np.load(DATA) as d:                       # массивы читаются один раз (npz распаковывает при каждом d[...])
        crops, dims, y, src = d["crops"], d["dims"], d["y"].astype(str), d["src"].astype(str)
    idx = np.arange(len(src))
    if exclude_real:
        idx = idx[~np.char.startswith(src, "real:")]
    X = np.stack([compact(crops[i], dims[i]) for i in idx])
    return X, y[idx], src[idx]


def train(exclude_real=False, hidden=(64,), epochs=60, data=None):
    X, y, s = data or dataset(exclude_real)
    classes = sorted(set(y))
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xn = ((X - mu) / sd).astype(np.float32)
    yi = np.array([classes.index(c) for c in y])
    cnt = {c: (y == c).sum() for c in classes}
    w = np.array([(1 / math.sqrt(cnt[lab])) * (1.5 if si.startswith("gen") else 3.0 if not si.startswith("synth") else 1.0)
                  for lab, si in zip(y, s)], np.float32)
    w *= len(w) / w.sum()
    net = MLP([X.shape[1], *hidden, len(classes)]).fit(Xn, yi, w, epochs=epochs)
    trusted = sorted(set(y[[not si.startswith("synth") for si in s]]))
    return net, {"classes": classes, "mu": mu, "sd": sd, "trusted": trusted}


def save_model(net, meta):
    arrs = {f"W{i}": W for i, W in enumerate(net.W)} | {f"b{i}": b for i, b in enumerate(net.b)}
    np.savez_compressed(MODEL, mu=meta["mu"], sd=meta["sd"], classes=np.array(meta["classes"]),
                        trusted=np.array(meta["trusted"]), **arrs)


class Model:
    def __init__(self, net=None, meta=None):
        self.net = net
        if net is not None:
            self.classes, self.trusted = meta["classes"], set(meta["trusted"])
            self.mu, self.sd = meta["mu"], meta["sd"]
        elif MODEL.exists():
            d = np.load(MODEL)
            n = sum(1 for k in d.files if k.startswith("W"))
            self.net = MLP([1])
            self.net.W = [d[f"W{i}"] for i in range(n)]
            self.net.b = [d[f"b{i}"] for i in range(n)]
            self.classes, self.trusted = list(map(str, d["classes"])), set(map(str, d["trusted"]))
            self.mu, self.sd = d["mu"], d["sd"]

    def ready(self):
        return self.net is not None

    def predict(self, c, dims, trusted_only=True):
        f = ((compact(c, dims) - self.mu) / self.sd).astype(np.float32)[None]
        p = self.net.proba(f)[0]
        k = int(np.argmax(p))
        lab = self.classes[k]
        if trusted_only and self.trusted and lab not in self.trusted:
            return "OTHER", float(p[k])
        return lab, float(p[k])


if __name__ == "__main__":
    if sys.argv[1:2] == ["train"]:
        net, meta = train()
        save_model(net, meta)
        print("классов", len(meta["classes"]), "надёжных", len(meta["trusted"]))
