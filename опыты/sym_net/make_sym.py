# Примеры символов с окружением: синтетика, проверенные листы, самоподпись по обозначениям на 1500 схемах.
#   python make_sym.py   — пишет data/*.npz; файлы проектов и learned/ не меняет. Тестовые листы — отдельным источником.
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SCANNER = "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main"
sys.path.insert(0, SCANNER)
import cv2
import numpy as np
import analyzer
import autonet
import core
import learn
import ocr as ocrmod
import reflabels
import synth

ROOT = Path(__file__).parent
OUT = ROOT / "data"
CLASSES = ["OTHER", "R", "C", "L", "D", "LED", "Q", "VS", "U", "OPAMP", "LOGIC", "OC", "COIL", "SW_NO", "BTN", "LAMP", "FU", "TERM", "PORT", "GND", "MOTOR", "RT", "XTAL"]
IDX = {c: i for i, c in enumerate(CLASSES)}
DEV = ("155eb18f", "13f85bc9", "87c06120")
S = 64
MERGE = {"SW_NC": "SW_NO", "THERMAL": "SW_NO", "QF": "SW_NO", "MCU": "U", "J": "TERM"}
engine = None


def crop(g, box):
    """Символ с окружением: квадрат в 1,8 размера, второй канал — где сама рамка."""
    x0, y0, x1, y1 = [float(v) for v in box]
    w, h = x1 - x0, y1 - y0
    side = int(max(w, h) * 1.8 + 8)
    cx, cy = int((x0 + x1) / 2), int((y0 + y1) / 2)
    pad = side
    big = cv2.copyMakeBorder(g, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    a, b = cy + pad - side // 2, cx + pad - side // 2
    c = big[a:a + side, b:b + side]
    k = S / side
    c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    m = np.zeros((S, S), np.uint8)
    m[max(0, int((y0 - cy) * k + S / 2)):int((y1 - cy) * k + S / 2) + 1, max(0, int((x0 - cx) * k + S / 2)):int((x1 - cx) * k + S / 2) + 1] = 255
    return np.stack([c, m])


def table(res, sy, npins):
    """Числа о символе: выводы, пропорция, размер в толщинах линии."""
    b = sy["box"]
    w, h = b[2] - b[0] + 1, b[3] - b[1] + 1
    st = max(res.get("stroke", 2), 0.5)
    return [min(npins, 8) / 4, float(np.log(w / h)), float(np.log(max(w, h) / st)) / 3]


def pins_count(res):
    n = {}
    for net in res["nets"]:
        for p in net["pins"]:
            n[p["symbol"]] = n.get(p["symbol"], 0) + 1
    return n


def parse(g):
    res = analyzer.analyze_gray(g, analyzer.load_params(), False)
    res["classified"] = analyzer.classify(res)
    return res


def from_truth(res, g, boxes, src, sheet):
    """Символ на детали ответа — её класс; не на детали или мелкий обломок — OTHER."""
    pins = pins_count(res)
    C, T, Y = [], [], []
    for sy in res["symbols"]:
        cx, cy = (sy["box"][0] + sy["box"][2]) / 2, (sy["box"][1] + sy["box"][3]) / 2
        best = None
        for b, cls in boxes:
            if b[0] - 3 <= cx <= b[2] + 3 and b[1] - 3 <= cy <= b[3] + 3:
                area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
                if best is None or area < best[1]:
                    best = (cls, area)
        cls = "OTHER"
        if best:
            sa = (sy["box"][2] - sy["box"][0]) * (sy["box"][3] - sy["box"][1])
            cls = best[0] if sa >= 0.15 * best[1] else "OTHER"
        cls = MERGE.get(cls, cls)
        if cls in IDX:
            C.append(crop(g, sy["box"]))
            T.append(table(res, sy, pins.get(sy["id"], 0)))
            Y.append(IDX[cls])
    return C, T, Y


def save(name, C, T, Y, src):
    if Y:
        np.savez_compressed(OUT / f"{name}.npz", c=np.array(C, np.uint8), t=np.array(T, np.float32), y=np.array(Y, np.int16), src=np.array(src))


def do_synth(seed):
    try:
        img, truth = synth.generate(seed)
        g, truth = learn.prep(img, truth)
        res = parse(g)
        boxes = [(c["box"], c["type"]) for c in truth["comps"]]
        save(f"s{seed}", *from_truth(res, g, boxes, "synth", ""), "synth")
    except Exception as e:
        print("синтетика", seed, e, flush=True)


def do_real(a):
    path, src = a
    p = Path(path)
    nl = core.load_json(p / "netlist.json")
    g = core.imread(p / "clean.png")[:, :, 0]
    res = parse(g)
    REAL = dict(learn.TYPE_MAP)
    boxes = []
    for c in nl["components"]:
        cls = REAL.get(c.get("type", ""), None)
        for b in (c.get("section_boxes") or {"": c.get("box")}).values():
            if b and cls:
                boxes.append((b, cls))
    for s in nl.get("power_symbols", []):
        boxes.append((s["box"], "GND" if s.get("type") == "GND" else "PORT"))
    save(f"r_{src}_{p.name[:10]}", *from_truth(res, g, boxes, src, p.name), src)


def do_self(path):
    """Самоподпись: однозначная пара «обозначение — символ»; уверенные надписи и обломки — OTHER."""
    global engine
    engine = engine or ocrmod.OCR()
    p = Path(path)
    try:
        g = core.imread(p)[:, :, 0]
        if max(g.shape) > 3000:
            return
        res = parse(g)
        autonet.read_texts(res, g, engine)
        autonet.drop_false_symbols(res, g, engine)
        pins = pins_count(res)
        texts = [t for t in res["texts"] if t.get("conf", 0) >= 0.9]
        C, T, Y = [], [], []
        pairs = reflabels.match(res["symbols"], texts, res.get("stroke", 2))
        rng = random.Random(hash(str(p)) & 0xffff)
        rng.shuffle(pairs)
        for si, ref, cls in pairs[:50]:
            cls = MERGE.get(cls, cls)
            if cls in IDX:
                sy = res["symbols"][si]
                C.append(crop(g, sy["box"]))
                T.append(table(res, sy, pins.get(sy["id"], 0)))
                Y.append(IDX[cls])
        for sy in res["symbols"]:
            if sy.get("junk"):
                C.append(crop(g, sy["box"]))
                T.append(table(res, sy, pins.get(sy["id"], 0)))
                Y.append(0)
        save(f"p_{abs(hash(str(p))) % 10 ** 10}", C, T, Y, "self")
    except Exception as e:
        print("схема", p.name, e, flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    tests = [str(p) for p in learn.test_sheets()]
    real = []
    for p in [str(x) for x in learn.registered()] + tests:
        nl = core.load_json(Path(p) / "netlist.json")
        if nl and learn.truth_from_netlist(nl) and (Path(p) / "source.png").exists():
            src = "test" if p in tests else "dev" if Path(p).name.startswith(DEV) else "real"
            real.append((p, src))
    real = list(dict.fromkeys(real))
    PR = Path("C:/Users/Artem/Downloads/схемы_импульсные_проекты")
    pool_list = [x for x in Path("C:/Users/Artem/scanner_yolo/pseudo_list.txt").read_text(encoding="utf-8").split("\n") if x]
    random.Random(2).shuffle(pool_list)
    many = [str(p / "clean.png") for p in sorted(PR.iterdir()) if (p / "clean.png").exists()] + pool_list[:1000]
    with ProcessPoolExecutor(10) as pool:
        list(pool.map(do_real, real))
        print("проверенные листы", len(real), flush=True)
        list(pool.map(do_synth, range(610000, 610500), chunksize=4))
        print("синтетика", flush=True)
        for i, _ in enumerate(pool.map(do_self, many, chunksize=4), 1):
            if i % 200 == 0:
                print("схем", i, flush=True)
    (OUT / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")
    print("готово", flush=True)
