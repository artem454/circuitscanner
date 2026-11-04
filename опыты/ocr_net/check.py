# Настоящие надписи: прежний OCR и сеть (прогон numpy из ocr.py). python check.py <dev|test> <npz>
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main")
import cv2
import numpy as np
import ocr

ROOT = Path(__file__).parent
ocr.NET = Path(sys.argv[2])
net = ocr.NetReader()
DEV = ("155eb18f", "13f85bc9", "87c06120")
LOOK = str.maketrans("АВЕКМНОРСТУХаеокмнорсухтµ", "ABEKMHOPCTYXaeokmhopcyxtμ")
same = lambda a, b: a.replace(" ", "").translate(LOOK) == b.replace(" ", "").translate(LOOK)
rows = json.loads((ROOT / "real" / "labels.json").read_text(encoding="utf-8"))
R = [r for r in rows if (r["test"] if sys.argv[1] == "test" else (not r["test"] and r["sheet"].startswith(DEV)))]
ok_new = ok_old = 0
bad = []
t0 = time.time()
for r in R:
    g = cv2.imdecode(np.fromfile(str(ROOT / "real" / r["file"]), np.uint8), cv2.IMREAD_GRAYSCALE)
    s, c = net.read(g)
    ok_new += same(s, r["text"])
    ok_old += same(r["old"], r["text"])
    if not same(s, r["text"]):
        bad.append((r["text"], s, round(c, 2), r["old"]))
print(sys.argv[1], "надписей", len(R), "прежний", ok_old, "сеть", ok_new, f"{(time.time() - t0) / len(R) * 1000:.0f} мс на надпись")
print(bad[:30])
