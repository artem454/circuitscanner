# Помогает ли chaiNNer читать мелкие надписи: 440 настоящих надписей, как есть и вдвое мельче, через обе модели.
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main")
import cv2
import numpy as np
import core
import enhance
import ocr

REAL = Path("C:/Users/Artem/scanner_ocr/real")
W = Path("C:/Users/Artem/chaiNNer/work/crops")
LOOK = str.maketrans("АВЕКМНОРСТУХаеокмнорсухтµ", "ABEKMHOPCTYXaeokmhopcyxtμ")
same = lambda a, b: a.replace(" ", "").replace(",", ".").translate(LOOK) == b.replace(" ", "").replace(",", ".").translate(LOOK)

rows = json.loads((REAL / "labels.json").read_text(encoding="utf-8"))
W.mkdir(exist_ok=True)
names = []
for r in rows:
    g = core.imread(REAL / r["file"])
    for tag, im in (("o", g), ("d", cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))):
        n = f"{Path(r['file']).stem}_{tag}"
        core.imwrite(W / f"{n}.png", im)
        names.append(n)
for key in ("text2hd", "ultrasharp"):
    print(key, "секунд", round(sum(enhance.run_model(key, [(W / f"{n}.png", W / f"{n}__{key}.png") for n in names]))), flush=True)
net = ocr.NetReader()
res = {}
for r in rows:
    st = Path(r["file"]).stem
    for tag in "od":
        out = {}
        for key in ("", "text2hd", "ultrasharp"):
            g = core.imread(W / (f"{st}_{tag}__{key}.png" if key else f"{st}_{tag}.png"))[:, :, 0]
            out[key] = net.read(g)
        ok = {k: same(v[0], r["text"]) for k, v in out.items()}
        for k, v in ok.items():
            res.setdefault((tag, k), [0, 0])
            res[(tag, k)][0] += v
            res[(tag, k)][1] += 1
        # второе мнение: совпало ли чтение оригинала и улучшенной
        agree = out[""][0].replace(" ", "") == out["ultrasharp"][0].replace(" ", "")
        res.setdefault((tag, "согласны и верно"), [0, 0])
        res.setdefault((tag, "не согласны и неверно"), [0, 0])
        res[(tag, "согласны и верно")][0] += agree and ok[""]
        res[(tag, "согласны и верно")][1] += agree
        res[(tag, "не согласны и неверно")][0] += (not agree) and not ok[""]
        res[(tag, "не согласны и неверно")][1] += not agree
for k, v in res.items():
    print(k, v, round(v[0] / max(1, v[1]), 3))
shutil.rmtree(W)
