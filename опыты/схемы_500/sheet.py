# Один лист через все инструменты сканера. python sheet.py <проект> <ключ улучшения или none>
# Код берётся из копии scanner_smps/code: learned/ настоящего сканера не меняется.
import json
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

CODE = Path(__file__).parent / "code"
sys.path.insert(0, str(CODE))
import autonet
import checkplan  # noqa: F401
import core
import diagnose
import faults
import pcb
import physics

faults.learn_stats = lambda *a: None                  # параллельные процессы не пишут в общий файл статистики
project, key = Path(sys.argv[1]), sys.argv[2]
enh = "" if key == "none" else key
out = {"project": project.name, "key": key}
if enh and not (project / f"enhanced_{enh}.png").exists():
    print("РЕЗУЛЬТАТ " + json.dumps(dict(out, пропущен=True), ensure_ascii=False))
    sys.exit()


def tool(name, fn):
    t0 = time.time()
    try:
        r = fn()
        out[name] = {"ok": True, "s": round(time.time() - t0, 2)}
        return r
    except Exception as e:
        kind = "отказ" if isinstance(e, (ValueError, FileNotFoundError)) else "сбой"
        out[name] = {"ok": False, "kind": kind, "s": round(time.time() - t0, 2), "err": f"{type(e).__name__}: {e}"[:200],
                     "where": traceback.format_exc().strip().splitlines()[-3][:160]}
        return None


nl = tool("разбор", lambda: autonet.run(project, overwrite=True, enhance=enh))
if nl:
    C = nl["components"]
    out["деталей"] = len(C)
    out["типы"] = dict(Counter(c["type"] for c in C).most_common())
    out["с обозначением"] = sum("?" not in c["ref"] for c in C)
    out["с номиналом"] = sum(bool(c.get("value")) for c in C)
    out["сомнительных"] = sum(c.get("confidence") == "low" for c in C)
    out["подпись под вопросом"] = sum(bool(c.get("doubtful")) for c in C)
    out["неопознанных"] = sum(c["type"] == "X" for c in C)
    out["цепей"] = len({n for c in C for n in c["pins"].values()})
    ph = tool("физика", lambda: physics.analyze(project))
    if ph:
        out["находок физики"] = len(ph.get("findings", []))
    dg = tool("диагностика", lambda: diagnose.diagnose(project, []))
    if dg:
        out["кандидатов"] = len(dg.get("candidates", []))
        out["рисков"] = len(dg.get("risks", []))
        out["шагов плана"] = len((dg.get("plan") or {}).get("steps", []))
    fl = tool("неисправности", lambda: faults.simulate(project))
    if fl:
        out["вариантов отказа"] = len(fl)
    tool("плата", lambda: pcb.make(project, force=False, tries=1))
    if (project / "netlist.json").exists():
        shutil.copyfile(project / "netlist.json", project / f"netlist_{key}.json")
print("РЕЗУЛЬТАТ " + json.dumps(out, ensure_ascii=False))
