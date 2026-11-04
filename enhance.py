"""Необязательное улучшение картинки нейросетью перед поиском элементов и линий.

Улучшенная копия нужна только геометрии: analyze_gray ищет по ней символы и провода, а надписи и
номиналы читаются по оригиналу (модели уверенно искажают мелкий текст: R8 → R6, 20N60 → 20NC0).
Модели работают в Python из chaiNNer: в среде сканера PyTorch нет. Папку можно задать переменной
CHAINNER_DIR. Результат кешируется в проекте, координаты всегда считаются по clean.png.
"""
import os
import subprocess
from pathlib import Path
import core

HERE = Path(__file__).resolve().parent
CHAINNER = Path(os.environ.get("CHAINNER_DIR", r"C:\Users\Artem\chaiNNer"))
PYTHON = CHAINNER / "python" / "python" / "python.exe"
MODELS = {                                             # файл модели, размер тайла (под видеопамять)
    "ultrasharp": ("models/candidates/4x-UltraSharpV2.safetensors", 256),
    "text2hd": ("models/candidates/2x_Text2HD_v.1-RealPLKSR.pth", 512),
}
TITLES = {"": "нет", "ultrasharp": "UltraSharpV2", "text2hd": "Text2HD"}


def key_of(title):
    return next(k for k, t in TITLES.items() if t == title)


def available(key):
    return key in MODELS and PYTHON.exists() and (CHAINNER / MODELS[key][0]).exists()


def run_model(key, pairs):
    """Прогнать картинки через модель одним процессом: [(вход, выход)] → [секунды на картинку]."""
    if not available(key):
        raise FileNotFoundError(f"нет модели {key} или Python в {CHAINNER}")
    rel, tile = MODELS[key]
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    secs = []
    for i in range(0, len(pairs), 60):                 # длина командной строки Windows ограничена
        args = [str(PYTHON), str(HERE / "enhance_worker.py"), str(CHAINNER / rel), str(tile)]
        args += [str(p) for pair in pairs[i:i + 60] for p in pair]
        r = subprocess.run(args, capture_output=True, encoding="utf-8", errors="replace", env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode:
            raise RuntimeError(r.stderr[-800:])
        secs += [float(line.split("\t")[2]) for line in r.stdout.splitlines() if line.count("\t") == 2]
    return secs


def enhanced_gray(project, key):
    """Серая улучшенная копия листа (в целое число раз больше clean.png). Считается один раз."""
    project = Path(project)
    cache = project / f"enhanced_{key}.png"
    source = project / "source.png"
    if not cache.exists() or cache.stat().st_mtime < source.stat().st_mtime:
        run_model(key, [(source, cache)])
        core.imwrite(cache, core.clean(core.imread(cache)))
    return core.imread(cache)[:, :, 0]
