import json
from pathlib import Path
OUT = Path("C:/Users/Artem/Downloads/схемы_на_проверку")
T = N = P = S = K = M = 0
for d in sorted(OUT.iterdir()):
    a = json.load(open(d / "netlist.json", encoding="utf-8"))
    b = json.load(open(d / "netlist_scanner.json", encoding="utf-8"))
    n, s = len(a["components"]), len(a.get("power_symbols", []))
    nets = len({x for c in a["components"] for x in c["pins"].values()})
    ar, br = {c["ref"] for c in a["components"]}, {c["ref"] for c in b["components"]}
    vals = {c["ref"]: c.get("value") for c in b["components"]}
    same = sum(1 for c in a["components"] if vals.get(c["ref"]) and vals[c["ref"]].split("/")[0] == str(c.get("value")).split("/")[0])
    T += n; N += nets; P += s; S += len(ar & br); K += len(b["components"]); M += same
    print(d.name[:32], "деталей", n, "цепей", nets, "знаков", s, "в черновике", len(b["components"]), "совпало обозначений", len(ar & br), "номиналов", same)
print("всего деталей", T, "цепей", N, "знаков", P, "| в черновиках", K, "совпало обозначений", S, "номиналов", M)
