# Подготовка схем для распознавания мелких знаков: запускать Python-ом chaiNNer.
#   C:\Users\Artem\chaiNNer\python\python\python.exe мелкие_знаки.py <папка с картинками> <папка результата> [ultrasharp|text2hd]
# Для каждой картинки пишет увеличенную копию <имя>_x4.png (x2 для Text2HD). Копия нужна для поиска элементов и линий
# и как второе мнение при чтении подписей. Сами подписи читаются по оригиналу: модели уверенно искажают мелкий текст.
# Сравнение на 440 настоящих надписях: прямое чтение копии чуть хуже оригинала (79,5 % против 80,7 %), но расхождение
# двух чтений указывает на ошибку в 83 % случаев.
import sys
import time
from pathlib import Path

SCANNER = r"C:\Users\Artem\OneDrive\Desktop\Сканер электросхем\--main"
sys.path.insert(0, SCANNER)
import spandrel
import torch
from enhance_worker import imread, imwrite, upscale

MODELS = {"ultrasharp": (r"C:\Users\Artem\chaiNNer\models\candidates\4x-UltraSharpV2.safetensors", 256),
          "text2hd": (r"C:\Users\Artem\chaiNNer\models\candidates\2x_Text2HD_v.1-RealPLKSR.pth", 512)}

if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    path, tile = MODELS[sys.argv[3] if len(sys.argv) > 3 else "ultrasharp"]
    dst.mkdir(parents=True, exist_ok=True)
    d = spandrel.ModelLoader().load_from_file(path)
    model = d.model.eval().cuda()
    files = [f for f in sorted(src.iterdir()) if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".webp")]
    for i, f in enumerate(files, 1):
        out = dst / f"{f.stem}_x{d.scale}.png"
        if out.exists():
            continue
        t0, size = time.time(), tile
        img = imread(f)
        while True:
            try:
                res = upscale(model, d.scale, img, size)
                break
            except torch.cuda.OutOfMemoryError:             # видеопамять занята другой программой
                torch.cuda.empty_cache()
                size -= 64
                if size < 128:
                    raise
        imwrite(out, res)
        print(f"{i}/{len(files)} {f.name}: {time.time() - t0:.1f} с", flush=True)
