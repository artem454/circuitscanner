# Сканер против проверенных сложных листов. python measure.py
import json
import sys
from pathlib import Path
sys.path.insert(0, "C:/Users/Artem/OneDrive/Desktop/Сканер электросхем/--main/опыты")
import enhance_eval as e
from enhance_eval import *

OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
for d in sorted(OUT.iterdir()):
    nl = core.load_json(d / "netlist.json")
    if not nl.get("verified"):
        continue
    r = one((str(d), True, "как есть", ""))
    print(d.name, f"деталей {r['n']}, найдено {r['found']}, ложных {r['extra']}, тип {r['type']}/{r['found']}, обозначение {r['ref']}/{r['ref_n']}, номинал {r['val']}/{r['val_n']}, F1 цепей {r['f1']:.3f}")
