# Проверка инструмента «плата»: копия netlist без пометки черновика, расчёт во временной папке.
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
TMP = ROOT / "pcbtmp"
PR = Path("C:/Users/Artem/Downloads/схемы_импульсные_проекты")


def child(p):
    sys.path.insert(0, str(ROOT / "code"))
    import pcb
    t0 = time.time()
    out = {"project": p.name}
    try:
        r = pcb.make(p, tries=1)
        out.update(ok=True, s=round(time.time() - t0, 1), mode=r.get("mode", "board"), keys={k: r[k] for k in ("width", "height", "unrouted", "routed", "nets", "wires") if k in r})
    except Exception as e:
        out.update(ok=False, s=round(time.time() - t0, 1), err=f"{type(e).__name__}: {e}"[:160], where=traceback.format_exc().strip().splitlines()[-3][:130])
    print("РЕЗУЛЬТАТ " + json.dumps(out, ensure_ascii=False))


def one(name):
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    nl = json.loads((PR / name / "netlist.json").read_text(encoding="utf-8"))
    nl.pop("generated", None)
    (d / "netlist.json").write_text(json.dumps(nl, ensure_ascii=False), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    try:
        r = subprocess.run([sys.executable, "-X", "utf8", __file__, str(d)], capture_output=True, encoding="utf-8", errors="replace", timeout=300, env=env)
        line = next((x for x in r.stdout.splitlines() if x.startswith("РЕЗУЛЬТАТ ")), None)
        res = json.loads(line[10:]) if line else {"project": name, "ok": False, "err": "падение: " + r.stderr[-160:]}
    except subprocess.TimeoutExpired:
        res = {"project": name, "ok": False, "err": "дольше 300 с"}
    shutil.rmtree(d, ignore_errors=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(Path(sys.argv[1]))
    else:
        names = sorted(p.name for p in PR.iterdir() if (p / "netlist.json").exists())
        with ThreadPoolExecutor(8) as pool, open(ROOT / "итог_плата.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(pool.map(one, names), 1):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(i, flush=True)
        print("готово", flush=True)
