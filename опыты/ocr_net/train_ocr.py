# Сеть читает надпись целиком (CNN + CTC). Обучение: искусственные надписи шрифтами + настоящие с учебных листов.
#   venv312\Scripts\python.exe train_ocr.py [эпох]
# Лучшая эпоха — по настоящим надписям 3 отложенных листов. Тестовые листы здесь не используются.
import json
import math
import random
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent
FONTS = Path("C:/Windows/Fonts")
FONT_FILES = ["arial.ttf", "arialbd.ttf", "ariali.ttf", "ARIALN.TTF", "times.ttf", "timesi.ttf", "timesbd.ttf", "cour.ttf", "courbd.ttf",
              "consola.ttf", "calibri.ttf", "calibrib.ttf", "tahoma.ttf", "tahomabd.ttf", "verdana.ttf", "segoeui.ttf", "GOST_A.TTF",
              "GOST_B.TTF", "GOST_AU.ttf", "GOST_BU.ttf", "isocpeur.ttf", "micross.ttf", "lucon.ttf"]
CYR = "БГДЖЗИЛПФЦЧШЩЫЬЭЮЯбгджзилпфцчшщыьэюя"
ALPHABET = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+-.,/()=~*:μΩ" + CYR
IDX = {c: i + 1 for i, c in enumerate(ALPHABET)}        # 0 — пустой знак CTC
LOOK = str.maketrans("АВЕКМНОРСТУХаеокмнорсухтµ", "ABEKMHOPCTYXaeokmhopcyxtμ")   # русские буквы, совпадающие с латинскими
DEV = ("155eb18f", "13f85bc9", "87c06120")
H = 32
E24 = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]
PRE = ["R", "R", "R", "R", "C", "C", "C", "L", "VD", "VT", "D", "Q", "U", "DA", "DD", "T", "XP", "XS", "X", "J", "K", "SA", "SB", "FU", "F", "HL", "BA", "IC", "TR", "RV", "VS", "ZD", "LED"]
PARTS = ["1N4148", "1N4007", "1N5819", "BC547", "BC847", "BC557", "2N2222", "2N3904", "KT315", "KT817", "KT3102E", "IRF840", "IRFZ44", "LM358", "LM317", "NE555", "TL431",
         "TL072", "MC34063", "UC3845", "HER106", "FR107", "PC817", "BTA16", "S8550", "SS14", "78L05", "7805", "AMS1117", "ATmega8", "CD4017", "К155ЛА3", "КТ817А", "КД522", "К561ЛН2", "BZX79-B5V"]
WORDS = ["GND", "VCC", "VDD", "VEE", "IN", "OUT", "IN+", "IN-", "OUT+", "OUT-", "Vin", "Vout", "+U", "-U", "Вход", "Выход", "Нагрузка", "Питание", "Сеть", "Стоп", "Пуск", "ЗАЗЕМЛЕНИЕ", "Audio", "PWR", "LDR", "NTC"]


def ref(r):
    p = r.choice(PRE)
    s = p + str(r.choice([r.randint(1, 9), r.randint(1, 30), r.randint(1, 199)]))
    if p in ("DA", "DD", "U", "K") and r.random() < 0.3:
        s += "." + str(r.randint(1, 8))
    return s


def value(r):
    m = r.choice(E24) * r.choice([1, 10, 100]) / 10
    t = f"{m:g}"
    k = r.random()
    if k < 0.28:
        u = r.choice(["k", "K", "k", "M", "R", "", "Ω", "кОм", "Ом", "МОм", "kΩ"])
        if r.random() < 0.25 and "." in t and u in ("k", "K", "M", "R"):
            return t.replace(".", u)                    # 4k7
        return t + u
    if k < 0.56:
        return t + r.choice(["n", "p", "u", "μ", "nF", "pF", "uF", "μF", "мкФ", "нФ", "пФ", "MF", "мк"])
    if k < 0.64:
        return (t + r.choice(["n", "u", "μ", "uF", "мкФ"])) + r.choice(["/", " ", "x", "X"]) + str(r.choice([16, 25, 35, 50, 63, 100, 250, 400, 630])) + r.choice(["V", "В"])
    if k < 0.74:
        return t + r.choice(["mH", "uH", "μH", "мкГн", "мГн"])
    if k < 0.84:
        return r.choice(["", "+", "-", "~", "±"]).replace("±", "+") + f"{r.choice([3.3, 5, 9, 12, 15, 24, 48, 100, 220, 230, 400]):g}" + r.choice(["V", "В"])
    if k < 0.9:
        return f"{m:g}" + r.choice(["W", "A", "mA", "Hz", "kHz"])
    return f"0.{r.choice(['1', '22', '47', '01', '047', '068', '33'])}" + r.choice(["", "u", "μ"])


def label(r):
    k = r.random()
    if k < 0.34:
        return ref(r)
    if k < 0.66:
        return value(r)
    if k < 0.80:
        return f"{ref(r)} {value(r)}"[:20]
    if k < 0.90:
        return r.choice(PARTS)
    if k < 0.96:
        return r.choice(WORDS)
    return str(r.randint(1, 40))


def clean_label(s):
    s = s.translate(LOOK)
    return "".join(c for c in s if c in IDX)


def to_height(g):
    """Высота H; ширина кратна 4, не меньше H. g — серая, чернила тёмные."""
    h, w = g.shape
    k = H / h
    nw = max(H, min(480, int(round(w * k))))
    g = cv2.resize(g, (nw, H), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    pad = -nw % 4
    if pad:
        g = cv2.copyMakeBorder(g, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
    return g


class Synth(torch.utils.data.IterableDataset):
    def __init__(self, real, n):
        self.real, self.n = real, n

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        r = random.Random(time.time_ns() ^ (info.id if info else 0) * 7919)
        fonts = {}
        files = [FONTS / f for f in FONT_FILES if (FONTS / f).exists()]
        for _ in range(self.n):
            if self.real and r.random() < 0.25:
                g, t = r.choice(self.real)
                g = g.copy()
                if r.random() < 0.6:
                    g = cv2.GaussianBlur(g, (0, 0), r.uniform(0.3, 0.9))
                g = np.clip(g.astype(np.float32) * r.uniform(0.85, 1.05) + r.uniform(-10, 15), 0, 255).astype(np.uint8)
            else:
                t = clean_label(label(r))
                if not t:
                    continue
                g = self.render(r, t, fonts, files)
            yield to_height(g), t

    def render(self, r, t, fonts, files):
        size = r.choice([26, 32, 40])
        f = r.choice(files)
        font = fonts.get((f, size)) or fonts.setdefault((f, size), ImageFont.truetype(str(f), size))
        l, top, rt, b = font.getbbox(t)
        W, Ht = rt - l + 2 * size, b - top + 2 * size
        im = Image.new("L", (W, Ht), 0)
        ImageDraw.Draw(im).text((size - l, size - top), t, fill=255, font=font)
        g = np.array(im)
        sh = r.uniform(-0.25, 0.25)
        g = cv2.warpAffine(g, np.float32([[1, sh, -sh * Ht / 2], [0, 1, 0]]), (W, Ht))
        g = cv2.warpAffine(g, cv2.getRotationMatrix2D((W / 2, Ht / 2), r.uniform(-2.5, 2.5), 1.0), (W, Ht))
        k = r.choice([0, 0, 1, -1])
        if k:
            g = (cv2.dilate if k > 0 else cv2.erode)(g, np.ones((2, 2), np.uint8))
        ys, xs = np.nonzero(g > 60)
        if len(xs) == 0:
            return 255 - g
        m = r.randint(3, 10)
        g = g[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]
        # мелкий кегль скана: высота 8–22 px, затем увеличение, как в разборе
        hs = r.choice([r.uniform(8, 13), r.uniform(10, 22), r.uniform(18, 40)])
        s = hs / g.shape[0]
        small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        ink = r.uniform(0.55, 1.0)
        small = 255 - (small.astype(np.float32) * ink)
        if r.random() < 0.4:
            small = cv2.GaussianBlur(small, (0, 0), r.uniform(0.3, 0.9))
        small = small + np.random.default_rng(r.randint(0, 1 << 30)).normal(0, r.uniform(0, 14), small.shape)
        small = np.clip(small, 0, 255).astype(np.uint8)
        if r.random() < 0.3:
            small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, r.randint(35, 85)])[1], cv2.IMREAD_GRAYSCALE)
        if r.random() < 0.15:                           # кусок провода у края рамки
            x = r.choice([0, small.shape[1] - 1])
            small[:, x] = np.minimum(small[:, x], r.randint(0, 90))
        return small


def collate(items):
    W = max(g.shape[1] for g, _ in items)
    x = np.full((len(items), 1, H, W), 0, np.float32)
    lens, targets, tlens = [], [], []
    for i, (g, t) in enumerate(items):
        x[i, 0, :, :g.shape[1]] = (255 - g.astype(np.float32)) / 255
        lens.append(g.shape[1] // 4)
        ids = [IDX[c] for c in t if c in IDX]
        targets += ids
        tlens.append(len(ids))
    return torch.from_numpy(x), torch.tensor(lens), torch.tensor(targets), torch.tensor(tlens)


def block(a, b, k=3, p=1):
    return [nn.Conv2d(a, b, k, padding=p, bias=False), nn.BatchNorm2d(b), nn.ReLU()]


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(*block(1, 32), nn.MaxPool2d(2), *block(32, 64), nn.MaxPool2d(2), *block(64, 96), *block(96, 96), nn.MaxPool2d((2, 1)),
                                  *block(96, 128), nn.MaxPool2d((2, 1)), *block(128, 128, (2, 3), (0, 1)))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Conv1d(128, 160, 5, padding=2), nn.ReLU(), nn.Conv1d(160, n, 1))

    def forward(self, x):
        return self.head(self.body(x).squeeze(2))


def greedy(lg):
    p = lg.softmax(1).max(1)
    out = []
    for ids, conf in zip(p.indices.cpu().numpy(), p.values.cpu().numpy()):
        s, prev, c = [], 0, []
        for i, q in zip(ids, conf):
            if i and i != prev:
                s.append(ALPHABET[i - 1])
                c.append(q)
            prev = i
        out.append(("".join(s), float(np.mean(c)) if c else 0.0))
    return out


def same(a, b):
    return a.replace(" ", "").translate(LOOK) == b.replace(" ", "").translate(LOOK)


def score(net, data):
    net.eval()
    ok = tot = 0
    with torch.no_grad():
        for g, t in data:
            x = torch.from_numpy((255 - to_height(g).astype(np.float32))[None, None] / 255).cuda()
            ok += same(greedy(net(x))[0][0], t)
            tot += 1
    return ok / max(1, tot)


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    rows = json.loads((ROOT / "real" / "labels.json").read_text(encoding="utf-8"))
    load = lambda r: cv2.imdecode(np.fromfile(str(ROOT / "real" / r["file"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    train = [(load(r), clean_label(r["text"])) for r in rows if not r["test"] and not r["sheet"].startswith(DEV)]
    dev = [(load(r), r["text"]) for r in rows if not r["test"] and r["sheet"].startswith(DEV)]
    base = sum(same(r["old"], r["text"]) for r in rows if not r["test"] and r["sheet"].startswith(DEV)) / max(1, len(dev))
    print("настоящих: учёба", len(train), "отложено", len(dev), f"прежний OCR на отложенных {base:.3f}", flush=True)
    net = Net(len(ALPHABET) + 1).cuda()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    steps = 500
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * steps)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    loader = torch.utils.data.DataLoader(Synth(train, 10 ** 9), batch_size=64, num_workers=6, collate_fn=collate, persistent_workers=True, prefetch_factor=4)
    it = iter(loader)
    best = -1
    for ep in range(1, epochs + 1):
        net.train()
        t0, tot = time.time(), 0.0
        for _ in range(steps):
            x, lens, tg, tl = next(it)
            lg = net(x.cuda()).log_softmax(1).permute(2, 0, 1)
            loss = ctc(lg, tg.cuda(), lens.cuda(), tl.cuda())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            sched.step()
            tot += float(loss)
        acc = score(net, dev)
        print(f"эпоха {ep}: потери {tot / steps:.3f}, отложенные {acc:.3f}, {time.time() - t0:.0f} с", flush=True)
        if acc >= best:
            best = acc
            torch.save(net.state_dict(), ROOT / "best.pt")
    torch.save(net.state_dict(), ROOT / "last.pt")
    print("лучшая", round(best, 3), flush=True)
