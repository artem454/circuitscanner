# 15 схем на ручную проверку: своя папка, черновик netlist.json из текущего сканера, список в ЛИСТЫ_на_проверку.md.
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, "code")
import autonet
import core

PR = Path("C:/Users/Artem/Downloads/схемы_импульсные_проекты")
OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
ITEMS = [(7, "обратноходовый_RadioStorage", True), (8, "обратноходовый_UC3842", False), (20, "обратноходовый_UC3845", False), (34, "повышающий_TL494", True),
         (28, "понижающий_LM5116", False), (13, "повышающий_250Вт", False), (47, "повышающий_UC3843", False), (46, "дежурный_блок_телевизора", True),
         (6, "полумост_IR2153", False), (15, "полумост_КР1156", False), (18, "полумост_IR2111", False), (4, "двухтактный_блок", False),
         (26, "инвертор_H_мост", False), (38, "усилитель_класс_D", False), (3, "усилитель_на_транзисторах", False)]
pick = json.load(open("кандидаты.json", encoding="utf-8"))
OUT.mkdir(exist_ok=True)
lines = ["", "## Сложные схемы (импульсные источники), 5 ноября 2026", "", f"Папка: `{OUT}`. Черновик netlist.json уже собран: кнопку «Netlist без ИИ» нажимать не нужно.", ""]
for i, (idx, name, test) in enumerate(ITEMS, 1):
    project = pick[idx][1]
    d = OUT / f"{i:02d}_{name}"
    if not d.exists():
        shutil.copytree(PR / project, d, ignore=shutil.ignore_patterns("enhanced_*", "raw_*", "netlist*", "pcb", "faults.json", "diagnosis.*", "physics.json"))
    nl = autonet.run(d, overwrite=True)
    C = nl["components"]
    g = core.imread(d / "clean.png")
    low = sum(c.get("confidence") == "low" for c in C)
    print(i, name, "деталей", len(C), "сомнительных", low, "подписанных", sum("?" not in c["ref"] for c in C), "земли/питание", len(nl.get("power_symbols", [])), dict(Counter(c["type"] for c in C).most_common(5)), flush=True)
    lines += [f"{i}. `{d}`" + (" — ТЕСТ" if test else "") + "  ", f"   деталей в черновике {len(C)}, сомнительных {low}, размер {g.shape[1]}×{g.shape[0]}"]
Path("список.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
