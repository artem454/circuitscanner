# Стабилитроны на непроверенных листах: выводы A и K вместо 1 и 2 (1 был стороной катода).
import json
import subprocess
import sys
from pathlib import Path

for n in range(2, 16):
    f = Path(f"{n:02d}.json")
    d = json.load(open(f, encoding="utf-8"))
    changed = 0
    for c in d["comps"]:
        if c[1] == "Z" and set(c[4]) == {"1", "2"}:
            c[4] = {"K": c[4]["1"], "A": c[4]["2"]}
            changed += 1
    if changed:
        f.write_text(json.dumps(d, ensure_ascii=False, indent=0), encoding="utf-8")
        r = subprocess.run([sys.executable, "-X", "utf8", "apply.py", str(n)], capture_output=True, encoding="utf-8", errors="replace")
        err = [x for x in r.stdout.splitlines() if x.startswith("ошибки проверки") or x.startswith("ФИЗИКА")]
        print(n, "стабилитронов", changed, err)
