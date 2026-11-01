"""Обучение на правке пользователя: «это VT3, транзистор» или «это не деталь».

Правка сразу становится проверенным примером:
  • вырез символа с правильным классом — в learned/symdata.npz (модель формы) и learned/symbols.npz (kNN);
    kNN учитывает пример немедленно, модель формы переобучается в фоне (≈10–20 с);
  • если рядом с символом есть надпись с тем же числом букв, что в обозначении, — буквы идут в дообучение OCR;
  • повторная правка той же детали заменяет её прежний пример (источник user:проект:обозначение).
"""
import json
import threading
from pathlib import Path

import numpy as np

import analyzer
import core
import learn
import ocr as ocrmod
import symclass

NOT_PART = "НЕ ДЕТАЛЬ"
_retrain = {"thread": None, "again": False}


def _ink_and_stroke(project):
    gray = core.imread(Path(project) / "clean.png")[:, :, 0]
    ink = analyzer.binarize(gray)
    res = core.load_json(Path(project) / "analysis" / "analysis.json") or {}
    sw = res.get("stroke") or analyzer.stroke_width(ink)
    return gray, ink, max(1, int(round(float(sw)))), res


def teach(project, comp, new_type, new_ref=None, old_ref=None):
    """Запомнить правильный класс символа детали comp (нужен её «box» из netlist без ИИ).
    Возвращает строку-отчёт о том, чему научились."""
    box = [int(round(v)) for v in comp["box"]] if comp.get("box") else None
    if not box:
        return "у детали нет координат символа (netlist не из «Netlist без ИИ») — запомнено только в netlist"
    project = Path(project)
    gray, ink, sw, res = _ink_and_stroke(project)
    cls = "OTHER" if new_type == NOT_PART else learn.TYPE_MAP.get(new_type, "OTHER")
    src = f"user:{project.name}:{old_ref or comp.get('ref')}"
    crop, dims = symclass.crop(ink, box, sw)
    symclass.save([crop], [dims], [cls], src)
    learn.save_examples([analyzer.symbol_features(ink, box, sw)], [cls], src)
    analyzer._MODEL.clear()                                  # kNN перечитает примеры при следующем разборе
    done = [f"символ запомнен как «{cls}»"]
    if new_ref and new_ref != old_ref and new_type != NOT_PART:
        got = learn_ref_glyphs(gray, res, box, new_ref)
        if got:
            done.append(f"буквы обозначения «{new_ref}» — в чтение надписей ({got})")
    retrain_async()
    return "; ".join(done)


def learn_ref_glyphs(gray, res, box, ref):
    """Ближайшая к символу надпись с тем же числом букв — образец начертания для OCR."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    size = max(box[2] - box[0], box[3] - box[1])
    texts = sorted(res.get("texts", []), key=lambda t: analyzer.box_dist(box, *analyzer.box_center(t["box"])))
    eng = ocrmod.OCR()
    for t in texts[:4]:
        if analyzer.box_dist(box, *analyzer.box_center(t["box"])) > 2.5 * size:
            break
        pairs = ocrmod.learn_word(eng, gray, t["box"], t.get("vertical", False), ref)
        if pairs:
            ocrmod.save_extra([p[0] for p in pairs], [p[1] for p in pairs])
            return len(pairs)
    return 0


def retrain_async(done=None):
    """Переобучить модель формы в фоне; правки, пришедшие во время обучения, — следующим заходом."""
    if _retrain["thread"] and _retrain["thread"].is_alive():
        _retrain["again"] = True
        return

    def work():
        while True:
            _retrain["again"] = False
            net, meta = symclass.train()
            symclass.save_model(net, meta)
            analyzer._MODEL.clear()
            if not _retrain["again"]:
                break
        if done:
            done()
    _retrain["thread"] = threading.Thread(target=work, daemon=True)
    _retrain["thread"].start()


def apply_edit(nl, ref, new_ref, new_type, new_value):
    """Поправить netlist: переименовать/сменить тип/номинал или удалить «не деталь»."""
    comps = nl.get("components", [])
    c = next((x for x in comps if x.get("ref") == ref), None)
    if c is None:
        raise KeyError(ref)
    if new_type == NOT_PART:
        comps.remove(c)
        return None
    if new_ref and new_ref != ref and any(x.get("ref") == new_ref for x in comps):
        raise ValueError(f"обозначение {new_ref} уже есть в netlist")
    c["ref"] = new_ref or ref
    c["type"] = new_type
    c["value"] = new_value
    c.pop("confidence", None)
    c["checked"] = True                                      # проверено человеком
    return c


def save_netlist(project, nl):
    p = Path(project) / "netlist.json"
    p.write_text(json.dumps(nl, ensure_ascii=False, indent=1), encoding="utf-8")
