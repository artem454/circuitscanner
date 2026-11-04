# Схемы импульсных преобразователей и сложных устройств из Яндекс Картинок (адреса, разрешённые robots.txt: source=morda).
#   python download.py [сколько]  — докачивает, повторный запуск продолжает. При капче останавливается.
import hashlib
import html
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

OUT = Path("C:/Users/Artem/Downloads/схемы_импульсные")
ROOT = Path(__file__).parent
META = OUT / "источники.json"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"
STOCK = ("istockphoto", "shutterstock", "alamy", "dreamstime", "depositphotos", "123rf", "gettyimages", "freepik", "vectorstock", "adobe.com")
Q = ["схема импульсного блока питания", "flyback converter schematic", "схема обратноходового преобразователя", "buck converter schematic", "boost converter schematic",
     "схема понижающего преобразователя", "схема повышающего преобразователя", "half bridge smps schematic", "full bridge converter schematic", "LLC resonant converter schematic",
     "схема полумостового преобразователя", "UC3842 схема блока питания", "UC3845 flyback schematic", "TL494 схема преобразователя", "SG3525 inverter schematic", "IR2153 схема блока питания",
     "TOP223 схема", "TNY264 schematic", "VIPer22 схема", "LNK364 schematic", "NCP1200 схема", "схема блока питания ATX", "ATX power supply schematic", "схема блока питания ноутбука",
     "laptop adapter schematic", "схема блока питания телевизора", "TV power supply schematic", "схема зарядного устройства импульсного", "схема сварочного инвертора", "welding inverter schematic",
     "схема ИБП", "UPS inverter schematic", "схема инвертора 12 220", "DC-DC converter schematic", "схема драйвера светодиодов импульсного", "LED driver schematic smps", "схема лабораторного блока питания импульсного",
     "схема частотного преобразователя", "VFD inverter schematic", "схема усилителя мощности сложная", "class D amplifier schematic", "схема блока питания LCD монитора", "LCD monitor power board schematic",
     "схема солнечного контроллера MPPT", "MPPT charge controller schematic", "PFC boost converter schematic", "схема корректора мощности"]


def get(url, timeout=20, limit=12_000_000):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ru,en"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(limit + 1)
    return None if len(data) > limit else data


robots = {}


def allowed(url):
    p = urllib.parse.urlsplit(url)
    host = p.scheme + "://" + p.netloc
    if host not in robots:
        rp = urllib.robotparser.RobotFileParser()
        try:
            rp.parse(get(host + "/robots.txt", 5).decode("utf-8", "ignore").splitlines())
        except Exception:
            rp = None                                  # файла нет — ограничений нет
        robots[host] = rp
    return robots[host] is None or robots[host].can_fetch("*", url)


def dhash(g):
    s = cv2.resize(g, (17, 16), interpolation=cv2.INTER_AREA)
    return int("".join("1" if v else "0" for v in (s[:, 1:] > s[:, :-1]).ravel()), 2)


def main(target):
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads(META.read_text(encoding="utf-8")) if META.exists() else {"files": [], "bad": [], "done": []}
    seen = {m["url"] for m in meta["files"]} | set(meta["bad"])
    hashes = [int(m["hash"]) for m in meta["files"]]
    bad_host = {}
    for page in range(0, 12):
        for q in Q:
            key = f"{q}|{page}"
            if key in meta["done"] or len(meta["files"]) >= target:
                continue
            url = "https://yandex.ru/images/search?" + urllib.parse.urlencode({"text": q, "source": "morda", "p": page})
            try:
                text = get(url).decode("utf-8", "ignore")
            except Exception as e:
                print("поиск:", e, flush=True)
                time.sleep(30)
                continue
            if "showcaptcha" in text.lower() or "smartcaptcha" in text.lower():
                print("КАПЧА: остановка, скачано", len(meta["files"]), flush=True)
                META.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
                return
            urls = list(dict.fromkeys(html.unescape(u) for u in re.findall(r'img_href&quot;:&quot;(.*?)&quot;', text)))
            got = 0
            fresh = []
            for u in urls:
                u = u.replace("\\/", "/")
                host = urllib.parse.urlsplit(u).netloc.lower()
                if u in seen or any(s in host for s in STOCK) or bad_host.get(host, 0) >= 3:
                    continue
                seen.add(u)
                fresh.append(u)

            def fetch(u):
                try:
                    if not allowed(u):
                        return u, None, False
                    data = get(u, 10)
                    return u, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None, False
                except Exception:
                    return u, None, True

            with ThreadPoolExecutor(8) as pool:
                fetched = list(pool.map(fetch, fresh))
            for u, img, failed in fetched:
                host = urllib.parse.urlsplit(u).netloc.lower()
                if failed:
                    bad_host[host] = bad_host.get(host, 0) + 1
                    meta["bad"].append(u)
                    continue
                if img is None or min(img.shape[:2]) < 300 or max(img.shape[:2]) < 500:
                    meta["bad"].append(u)
                    continue
                h = dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                if any(bin(h ^ o).count("1") <= 8 for o in hashes):
                    continue
                name = f"{len(meta['files']) + 1:04d}_{hashlib.md5(u.encode()).hexdigest()[:8]}.png"
                cv2.imencode(".png", img)[1].tofile(str(OUT / name))
                meta["files"].append({"file": name, "url": u, "query": q, "hash": str(h), "w": img.shape[1], "h": img.shape[0]})
                hashes.append(h)
                got += 1
            meta["done"].append(key)
            META.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            print(f"{q} стр.{page}: ссылок {len(urls)}, новых {got}, всего {len(meta['files'])}", flush=True)
            time.sleep(random.uniform(4, 8))
    print("итог", len(meta["files"]), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 900)
