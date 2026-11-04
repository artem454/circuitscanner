# 500 схем: отбор → chaiNNer → все инструменты сканера, два варианта (без улучшения и с ним).
#   python run_all.py select   — отобрать схемы и подготовить проекты
#   python run_all.py enhance <ключ> — прогнать через chaiNNer (кеш в проектах)
#   python run_all.py tools <ключ|none> — все инструменты; продолжает после остановки
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
SRC = Path("C:/Users/Artem/Downloads/схемы_импульсные")
PR = Path("C:/Users/Artem/Downloads/схемы_импульсные_проекты")
LIST = ROOT / "отобрано.json"


def screen(f):
    """Похожа ли картинка на схему: светлый фон, мало цвета, есть провода и символы."""
    import cv2
    import numpy as np
    import analyzer
    import core
    try:
        img = core.imread(f)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color = float((hsv[:, :, 1] > 90).mean())
        g = core.clean(img)
        H, W = g.shape
        if max(H, W) > 3000:
            k = 3000 / max(H, W)
            g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        dark = float((g < 128).mean())
        res = analyzer.analyze_gray(g, analyzer.load_params(), False)
        return {"file": f.name, "color": round(color, 3), "dark": round(dark, 3), "wires": len(res["segments"]), "symbols": len(res["symbols"]),
                "dots": len(res.get("dots", [])), "texts": len(res["texts"]), "mp": round(H * W / 1e6, 2)}
    except Exception as e:
        return {"file": f.name, "err": str(e)[:100]}


def is_schematic(s):
    return ("err" not in s and s["color"] < 0.35 and 0.008 < s["dark"] < 0.3 and s["wires"] >= 25 and 6 <= s["symbols"] <= 400 and s["texts"] >= 5)


def one(a):
    d, key = a
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "sheet.py"), str(d), key], capture_output=True, encoding="utf-8", errors="replace", timeout=420, env=env)
        line = next((x for x in r.stdout.splitlines() if x.startswith("РЕЗУЛЬТАТ ")), None)
        return json.loads(line[10:]) if line else {"project": d.name, "key": key, "падение": r.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"project": d.name, "key": key, "зависло": round(time.time() - t0)}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        from concurrent.futures import ProcessPoolExecutor
        files = sorted(SRC.glob("*.png"))
        with ProcessPoolExecutor(10) as pool:
            rows = list(pool.map(screen, files, chunksize=4))
        ok = [s for s in rows if is_schematic(s)]
        ok.sort(key=lambda s: -s["wires"] * (1 if s["mp"] < 8 else 0.3))
        (ROOT / "отбор.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        LIST.write_text(json.dumps([s["file"] for s in ok[:500]], ensure_ascii=False), encoding="utf-8")
        print("всего", len(rows), "похоже на схему", len(ok), "мегапикселей у отобранных", round(sum(s["mp"] for s in ok[:500])))
    elif cmd == "prepare":
        import core
        PR.mkdir(exist_ok=True)
        for f in json.loads(LIST.read_text(encoding="utf-8")):
            p = PR / Path(f).stem
            if not (p / "manifest.json").exists():
                core.prepare(SRC / f, p)
        print("проектов", len(list(PR.iterdir())))
    elif cmd == "enhance":
        import enhance
        import core
        key = sys.argv[2]
        import cv2
        import numpy as np
        todo = []
        for p in (PR / Path(f).stem for f in json.loads(LIST.read_text(encoding="utf-8"))):
            if (p / f"enhanced_{key}.png").exists():
                continue
            h, w = cv2.imdecode(np.fromfile(str(p / "clean.png"), np.uint8), cv2.IMREAD_GRAYSCALE).shape
            if h * w <= 1_500_000:                     # крупным листам хватает своего разрешения
                todo.append((h * w, p))
        todo = [p for _, p in sorted(todo, key=lambda t: t[0])]
        t0 = time.time()
        for i in range(0, len(todo), 20):
            part = todo[i:i + 20]
            enhance.run_model(key, [(p / "source.png", p / f"raw_{key}.png") for p in part])
            for p in part:
                core.imwrite(p / f"enhanced_{key}.png", core.clean(core.imread(p / f"raw_{key}.png")))
                (p / f"raw_{key}.png").unlink()
            print(f"{i + len(part)}/{len(todo)} за {time.time() - t0:.0f} с", flush=True)
    else:
        key = sys.argv[2]
        res = ROOT / f"итог_{key}.jsonl"
        done = {json.loads(x)["project"] for x in res.read_text(encoding="utf-8").splitlines()} if res.exists() else set()
        todo = [(PR / Path(f).stem, key) for f in json.loads(LIST.read_text(encoding="utf-8")) if Path(f).stem not in done]
        print("осталось", len(todo), flush=True)
        with ThreadPoolExecutor(10) as pool, res.open("a", encoding="utf-8") as out:
            for i, r in enumerate(pool.map(one, todo), 1):
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                out.flush()
                if i % 25 == 0:
                    print(i, flush=True)
        print("готово", flush=True)
