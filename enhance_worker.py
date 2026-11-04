# Запускается Python-ом из chaiNNer (там PyTorch и spandrel), сканер вызывает его через enhance.py:
#   python.exe enhance_worker.py <модель> <размер тайла> <вход1> <выход1> [<вход2> <выход2> ...]
# Вывод в stdout — по строке на картинку: путь выхода, масштаб, секунды.
import sys
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import spandrel


def imread(path):
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)     # путь с кириллицей


def imwrite(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(path)
    buf.tofile(str(path))


def starts(n, tile, step):
    out = list(range(0, max(1, n - tile), step))
    if out[-1] + tile < n:
        out.append(n - tile)
    return out


def upscale(model, scale, bgr, tile, overlap=16):
    """Картинка режется на тайлы; у соседних тайлов оставляем середину перекрытия, чтобы не было швов."""
    h, w = bgr.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h or pad_w:
        bgr = cv2.copyMakeBorder(bgr, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
    H, W = bgr.shape[:2]
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255
    out = np.zeros((H * scale, W * scale, 3), np.uint8)
    step = tile - overlap
    ys, xs = starts(H, tile, step), starts(W, tile, step)
    half = overlap // 2
    for y in ys:
        for x in xs:
            t = torch.from_numpy(rgb[y:y + tile, x:x + tile].transpose(2, 0, 1).copy())[None].cuda()
            with torch.inference_mode():
                r = model(t).clamp(0, 1)[0].cpu().numpy()
            r = (r.transpose(1, 2, 0)[:, :, ::-1] * 255 + 0.5).astype(np.uint8)
            top = 0 if y == ys[0] else half
            left = 0 if x == xs[0] else half
            out[(y + top) * scale:(y + tile) * scale, (x + left) * scale:(x + tile) * scale] = \
                r[top * scale:, left * scale:]
    return out[:h * scale, :w * scale]


if __name__ == "__main__":
    path, tile = sys.argv[1], int(sys.argv[2])
    files = sys.argv[3:]
    d = spandrel.ModelLoader().load_from_file(path)
    model = d.model.eval().cuda()
    for src, dst in zip(files[::2], files[1::2]):
        t0 = time.time()
        img = imread(src)
        size = tile
        while True:
            try:
                out = upscale(model, d.scale, img, size)
                break
            except torch.cuda.OutOfMemoryError:             # видеопамять занята другой программой
                torch.cuda.empty_cache()
                size -= 64
                if size < 128:
                    raise
        imwrite(dst, out)
        print(f"{Path(dst).name}\t{d.scale}\t{time.time() - t0:.1f}", flush=True)
