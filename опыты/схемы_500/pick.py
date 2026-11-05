# Кандидаты на ручную проверку: средняя сложность, читаемые подписи, разные темы. Рисует обзор с номерами.
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "code")
import cv2
import numpy as np
import core

PR = Path("C:/Users/Artem/Downloads/схемы_импульсные_проекты")
R = [json.loads(x) for x in open("итог_none.jsonl", encoding="utf-8")]
src = {m["file"][:-4]: m for m in json.load(open("C:/Users/Artem/Downloads/схемы_импульсные/источники.json", encoding="utf-8"))["files"]}
cands = defaultdict(list)
for r in R:
    n = r.get("деталей", 0)
    if not 30 <= n <= 90:
        continue
    named = r["с обозначением"] / n
    g = core.imread(PR / r["project"] / "clean.png")
    h, w = g.shape[:2]
    if named < 0.3 or not 800 <= max(h, w) <= 2300:
        continue
    q = src.get(r["project"], {}).get("query", "без запроса")
    cands[q].append((named, r["project"], n, w, h))
pick = []
for q, L in cands.items():
    for x in sorted(L, reverse=True)[:2]:
        pick.append((*x, q))
pick.sort(key=lambda t: -t[0])
pick = pick[:48]
json.dump(pick, open("кандидаты.json", "w", encoding="utf-8"), ensure_ascii=False)
cells = []
for i, (nm, p, n, w, h, q) in enumerate(pick):
    g = core.imread(PR / p / "clean.png")
    k = 300 / max(g.shape[:2])
    t = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    c = np.full((230, 310, 3), 255, np.uint8)
    c[20:20 + t.shape[0], 5:5 + t.shape[1]] = t[:210] if t.shape[0] <= 210 else t[:210]
    cv2.putText(c, f"{i} n{n} {nm:.2f} {w}x{h}", (4, 14), 0, 0.42, (0, 0, 200), 1, cv2.LINE_AA)
    cells.append(c)
while len(cells) % 6:
    cells.append(np.full((230, 310, 3), 255, np.uint8))
core.imwrite("кандидаты.png", np.vstack([np.hstack(cells[i:i + 6]) for i in range(0, len(cells), 6)]))
print(len(pick))
