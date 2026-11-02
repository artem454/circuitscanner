"""Эталоны условных обозначений из PDF-справочников (ГОСТ, плакаты, книги).

    python pdfsyms.py файл.pdf [ещё.pdf …]        — извлечь, показать сводку и добавить в обучение

Как читается: страница растрируется (pypdfium2), из PDF берутся строки текста с координатами.
Символ — связная группа штрихов, не похожая на текст; ему подписывается ближайшая строка слева
в той же строке таблицы (а если там «а) односторонний» — ближайшая выше с ключевым словом).
Название переводится в тип по ключевым словам («стабилитрон» → Z, «транзистор полевой» → M…).
Каждый найденный символ потом размножается (масштаб, поворот, толщина, шум) и проходит через
тот же анализатор, что и схема, — в классификатор попадает ровно то, что ядро видит на схемах.
"""
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np

# название → тип (порядок важен: частные раньше общих)
KEYWORDS = [
    (r"оптопар|оптрон|оптосимистор|оптотиристор", "OC"), (r"фототранзистор|фотодиод|фоторезистор|фототиристор", "PHOTO"),
    (r"светодиод|излучающ", "LED"), (r"варикап", "VC"), (r"стабилитрон|ограничител|стабистор|супрессор", "Z"),
    (r"симистор|триак", "VS"), (r"тиристор|динистор|тринистор", "VS"),
    (r"mosfet|полев|мдп|моп", "M"), (r"igbt|биполярн\w* с изолир", "IGBT"), (r"транзистор", "Q"),
    (r"диодный мост|выпрямительный мост", "BR"), (r"диод", "D"),
    (r"терморезистор|термистор|позистор", "RT"), (r"варистор", "RV"),
    (r"потенциометр|резистор\w* (подстроечн|переменн|регулир)", "RP"), (r"резистор", "R"),
    (r"конденсатор\w* (переменн|подстроечн)", "CV"), (r"конденсатор", "C"),
    (r"трансформатор", "T"), (r"дроссел|индуктивн|катушк\w* индукт", "L"),
    (r"кварц|пьезоэлектрическ|резонатор", "XTAL"), (r"операционн\w* усилител", "OPAMP"),
    (r"обмотк\w* реле|реле|катушк\w* (контактора|пускателя|электромагнитн)", "K"),
    (r"предохранител", "F"), (r"кнопк", "BTN"), (r"выключател|переключател|контакт", "SW"),
    (r"лампа", "LAMP"), (r"двигател|электродвигател", "MOTOR"),
    (r"громкоговор|динамик|звон|телефон", "BA"), (r"микрофон", "BM"),
    (r"батаре|аккумулят|гальваническ", "GB"), (r"амперметр|вольтметр|ваттметр", "METER"),
    (r"заземлен|корпус|массой|общий провод", "GND"), (r"положительн\w* напряжен|питани", "PORT"),
    (r"антенн", "WA"), (r"разрядник", "FV"), (r"генератор|источник", "G"),
]


def name_type(text):
    t = text.lower().replace("ё", "е")
    for pat, typ in KEYWORDS:
        if re.search(pat, t):
            return typ
    return None


def _text_lines(page, scale, H):
    """Строки текста страницы: (x0, y0, x1, y1 в пикселях растра, текст)."""
    tp = page.get_textpage()
    out = []
    for i in range(tp.count_rects()):
        l, b, r, t = tp.get_rect(i)
        txt = tp.get_text_bounded(l, b, r, t).strip()
        if txt:
            out.append((l * scale, H - t * scale, r * scale, H - b * scale, txt))
    return out


def extract(pdf, scale=3.0, pages=None):
    """Список символов: {"img": серый вырез, "type", "name", "page"}."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(str(pdf))
    out = []
    for pi in (pages or range(len(doc))):
        page = doc[pi]
        img = np.array(page.render(scale=scale).to_pil().convert("L"))
        H, W = img.shape
        lines = _text_lines(page, scale, H)
        if not lines:
            continue
        ink = (img < 140).astype(np.uint8) * 255
        for x0, y0, x1, y1, _ in lines:                      # текст убрать — остаются графика и линии таблиц
            ink[max(0, int(y0) - 2):int(y1) + 3, max(0, int(x0) - 2):int(x1) + 3] = 0
        long_h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(W * 0.15), 1)))
        long_v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(H * 0.08))))
        sym = cv2.subtract(ink, cv2.bitwise_or(long_h, long_v))   # без линий таблиц
        grow = cv2.dilate(sym, np.ones((int(6 * scale), int(6 * scale)), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(grow, 8)
        for k in range(1, n):
            x, y, w, h, area = st[k]
            real = int(np.count_nonzero(sym[y:y + h, x:x + w][lab[y:y + h, x:x + w] == k]))
            if max(w, h) < 12 * scale or max(w, h) > 0.4 * W or real < 30 * scale:
                continue
            cy = y + h / 2
            # своя колонка названий: ближайшая слева строка текста (по правому краю), её левый край — граница колонки
            left = [ln for ln in lines if ln[2] < x and x - ln[2] < 0.45 * W]
            if not left:
                continue
            near_x2 = max(ln[2] for ln in left if abs((ln[1] + ln[3]) / 2 - cy) < 60 * scale) \
                if any(abs((ln[1] + ln[3]) / 2 - cy) < 60 * scale for ln in left) else None
            if near_x2 is None:
                continue
            col = [ln for ln in left if ln[2] > near_x2 - 0.3 * W and ln[2] <= near_x2 + 2]
            col_x0 = min(ln[0] for ln in col if abs((ln[1] + ln[3]) / 2 - cy) < 60 * scale)
            col = [ln for ln in left if ln[0] >= col_x0 - 6 * scale]
            row = [ln for ln in col if ln[1] - 4 * scale <= cy <= ln[3] + 4 * scale]
            name, typ = None, None
            for ln in sorted(row, key=lambda ln: abs((ln[1] + ln[3]) / 2 - cy)):
                typ = name_type(ln[4])
                if typ:
                    name = ln[4]
                    break
            if not typ:                                       # «а) односторонний» — выше, в той же колонке
                for ln in sorted((ln for ln in col if ln[3] <= cy + 2 * scale and cy - ln[3] < 40 * scale),
                                 key=lambda ln: cy - ln[3]):
                    typ = name_type(ln[4])
                    if typ:
                        name = ln[4]
                        break
            if not typ:
                continue
            crop = 255 - sym[max(0, y - 4):y + h + 4, max(0, x - 4):x + w + 4]
            if _textlike(crop):
                continue
            out.append({"img": crop, "type": typ, "name": name, "page": pi + 1})
    return out


def _textlike(crop):
    """Курсивное «или», цифры, одиночные стрелки и пунктир — не символы детали."""
    b = (crop < 128).astype(np.uint8)
    h, w = b.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    comps = [st[k] for k in range(1, n) if st[k][4] > 3]
    if not comps:
        return True
    tallest = max(c[3] for c in comps)
    widest = max(c[2] for c in comps)
    if len(comps) >= 3 and tallest < 0.75 * h and widest < 0.45 * w and h < 0.6 * w:
        return True                                      # несколько отдельных «букв» в строку
    if h < 0.22 * w and len(comps) >= 3:
        return True                                      # пунктир
    return False


def _render(item, rng):
    """Символ на листе с проводами-выводами в случайном масштабе, повороте, толщине, с шумом."""
    g = item["img"]
    size = rng.uniform(40, 160)
    k = size / max(g.shape)
    g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    t = rng.choice((0, 0, 1, 2))
    if t:
        ker = np.ones((t, t), np.uint8)
        g = cv2.erode(g, ker) if rng.random() < 0.7 else cv2.dilate(g, ker)
    pad = int(size)
    canvas = np.full((g.shape[0] + 2 * pad, g.shape[1] + 2 * pad), 255, np.uint8)
    canvas[pad:pad + g.shape[0], pad:pad + g.shape[1]] = np.minimum(canvas[pad:pad + g.shape[0], pad:pad + g.shape[1]], g)
    body = [pad, pad, pad + g.shape[1], pad + g.shape[0]]
    if item.get("body"):                                   # тело символа без дорисованных выводов (libsyms)
        bx0, by0, bx1, by1 = item["body"]
        body = [int(pad + bx0 * k), int(pad + by0 * k), int(pad + bx1 * k), int(pad + by1 * k)]
    rot = rng.choice((0, 1, 2, 3))
    if rot:
        h0, w0 = canvas.shape
        canvas = np.rot90(canvas, rot).copy()
        for _ in range(rot):
            x0, y0, x1, y1 = body
            body = [y0, w0 - 1 - x1, y1, w0 - 1 - x0]
            h0, w0 = w0, h0
    if rng.random() < 0.5:
        canvas = cv2.GaussianBlur(canvas, (0, 0), rng.uniform(0.3, 1.0))
    return canvas, body


def _examples(args):
    import analyzer
    import learn
    items, per, seed, params = args
    rng = random.Random(seed)
    X, y, Cr = [], [], []
    for it in items:
        cls = learn.TYPE_MAP.get(it["type"], it["type"])
        for _ in range(per):
            img, body = _render(it, rng)
            res = analyzer.analyze_gray(img, params)
            best = None
            for sy, f, cr in zip(res["symbols"], res["_features"], res["_crops"]):
                b = sy["box"]
                inter = max(0, min(b[2], body[2]) - max(b[0], body[0])) * max(0, min(b[3], body[3]) - max(b[1], body[1]))
                area = max(1, (body[2] - body[0]) * (body[3] - body[1]))
                if inter / area > 0.3 and (best is None or inter > best[0]):
                    best = (inter, f, cr)
            if best:
                X.append(best[1])
                y.append(cls)
                Cr.append(best[2])
    return X, y, Cr


def add_to_training(items, per=12, src="pdf"):
    """Эталоны из PDF → примеры обоих классификаторов (источник «pdf», заменяет прежние)."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    import analyzer
    import learn
    import symclass
    params = analyzer.load_params()
    chunks = [items[i::8] for i in range(8)]
    X, y, Cr = [], [], []
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for Xi, yi, ci in pool.map(_examples, [(c, per, 500 + i, params) for i, c in enumerate(chunks) if c]):
            X += Xi
            y += yi
            Cr += ci
    learn.save_examples(X, y, src)
    symclass.save([c for c, _ in Cr], [d for _, d in Cr], y, src)
    return {k: int(v) for k, v in zip(*np.unique(y, return_counts=True))}


def contact_sheet(items, path, cols=10):
    tiles = []
    for it in items:
        g = it["img"]
        k = 110 / max(g.shape)
        g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        t = np.full((140, 130, 3), 255, np.uint8)
        t[5:5 + g.shape[0], 5:5 + g.shape[1]] = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        cv2.putText(t, it["type"], (4, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        tiles.append(t)
    while len(tiles) % cols:
        tiles.append(np.full((140, 130, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows) if rows else np.full((10, 10, 3), 255, np.uint8))


if __name__ == "__main__":
    allit = []
    for f in sys.argv[1:]:
        it = extract(f)
        print(Path(f).name, "символов:", len(it))
        allit += it
    from collections import Counter
    print(Counter(i["type"] for i in allit))
