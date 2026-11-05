# Сеть символов: тип детали или «не деталь» (OTHER) по вырезу с окружением.
#   venv312\Scripts\python.exe train_sym.py [эпох]
# Лучшая эпоха — по 3 отложенным листам. Источник test здесь не читается.
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CLASSES = json.loads((DATA / "classes.json").read_text(encoding="utf-8"))


def load():
    C, T, Y, S = [], [], [], []
    for f in DATA.glob("*.npz"):
        d = np.load(f)
        src = str(d["src"])
        if src == "test":
            continue
        C.append(d["c"])
        T.append(d["t"])
        Y.append(d["y"])
        S += [src] * len(d["y"])
    return np.concatenate(C), np.concatenate(T), np.concatenate(Y).astype(np.int64), np.array(S)


def block(a, b):
    return [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]


class Net(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.body = nn.Sequential(*block(2, 32), *block(32, 64), *block(64, 96), *block(96, 128))
        self.fc1 = nn.Linear(128 + 3, 192)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(192, n)

    def forward(self, x, t):
        z = self.body(x).mean((2, 3))
        return self.fc2(self.drop(torch.relu(self.fc1(torch.cat([z, t], 1)))))


def augment(x):
    k = np.random.randint(4)
    x = torch.rot90(x, k, (2, 3))
    if np.random.random() < 0.5:
        x = x.flip(3)
    if np.random.random() < 0.5:
        x = x.flip(2)
    s = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.8, 1.15)
    x[:, 0] = (x[:, 0] * s).clamp(0, 1)
    return x


def rate(net, X, T, Y):
    net.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(Y), 512):
            P.append(net((255 - X[i:i + 512].float().cuda()) .div(255) * torch.tensor([1, 0], device="cuda").view(1, 2, 1, 1) + X[i:i + 512].float().cuda().div(255) * torch.tensor([0, 1], device="cuda").view(1, 2, 1, 1), T[i:i + 512].cuda()).softmax(1).cpu())
    P = torch.cat(P)
    p = P.argmax(1).numpy()
    y = Y.numpy()
    other = y == 0
    return float((p == y).mean()), float((p[other] == 0).mean()) if other.any() else 0, float((p[~other] != 0).mean()) if (~other).any() else 0


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    c, t, y, s = load()
    print("примеров", len(y), Counter(s).most_common(), flush=True)
    print({CLASSES[k]: v for k, v in Counter(y.tolist()).most_common()}, flush=True)
    dev = s == "dev"
    Xd, Td, Yd = torch.from_numpy(c[dev]), torch.from_numpy(t[dev]), torch.from_numpy(y[dev])
    tr = ~dev
    Xt, Tt, Yt = torch.from_numpy(c[tr]), torch.from_numpy(t[tr]), torch.from_numpy(y[tr])
    cnt = np.bincount(y[tr], minlength=len(CLASSES)) + 1
    w = 1 / np.sqrt(cnt[y[tr]]) * np.where(s[tr] == "real", 5.0, 1.0)
    sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w), 40000, replacement=True)
    net = Net(len(CLASSES)).cuda()
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3)
    bs = 128
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (40000 // bs))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best = -1
    sign = torch.tensor([1.0, 0.0], device="cuda").view(1, 2, 1, 1)
    for ep in range(1, epochs + 1):
        net.train()
        t0, tot = time.time(), 0.0
        idx = np.fromiter(iter(sampler), np.int64, 40000)
        for i in range(0, len(idx) - bs + 1, bs):
            b = idx[i:i + bs]
            x = Xt[b].float().cuda() / 255
            x = x * (1 - sign) + (1 - x) * sign                # чернила — большие значения
            x = augment(x)
            loss = loss_fn(net(x, Tt[b].cuda()), Yt[b].cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        acc, oth, part = rate(net, Xd, Td, Yd)
        score = acc + 0.5 * oth
        print(f"эпоха {ep}: потери {tot / (len(idx) // bs):.3f}, отложенные: тип {acc:.3f}, мусор пойман {oth:.3f}, детали сохранены {part:.3f}, {time.time() - t0:.0f} с", flush=True)
        if score >= best:
            best = score
            torch.save(net.state_dict(), ROOT / "best.pt")
    print("лучшая", round(best, 3), flush=True)
