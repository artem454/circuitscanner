"""Мост приложения к ассистенту (Claude Code, Codex): что происходит в окне — по факту.

Приложение пишет в папку session/ рядом с программой:
  log.jsonl   — журнал: действия, сообщения строки состояния, ошибки с полным traceback;
  state.json  — текущее состояние: проект, выбранная деталь, сводка netlist, последнее сообщение;
  screen.png  — снимок окна (по запросу);
  inbox/      — команды от ассистента (их кладёт mcp_server.py), outbox/ — результаты.
Команды выполняются в окне приложения из белого списка (ACTIONS), ничего постороннего запустить нельзя.
"""
import json
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

SESSION = Path(__file__).resolve().parent / "session"
LOG = SESSION / "log.jsonl"
STATE = SESSION / "state.json"
SCREEN = SESSION / "screen.png"
INBOX = SESSION / "inbox"
OUTBOX = SESSION / "outbox"
MAX_LOG = 5000                                       # строк журнала держим, старые обрезаются

ACTIONS = {
    "state": "текущее состояние (то же, что state.json)",
    "screenshot": "снимок окна приложения → session/screen.png",
    "open_project": "открыть проект: args.path — папка *_project",
    "select": "выделить деталь в списке и на схеме: args.ref",
    "run_analysis": "«Анализ без ИИ»",
    "run_autonet": "«Netlist без ИИ»",
    "run_pcb": "«Печатная плата»",
    "diagnose": "диагностика: args.text — неисправность словами («не запускается»)",
    "status": "показать сообщение в строке состояния: args.text",
}


def log(kind, text, **data):
    SESSION.mkdir(exist_ok=True)
    row = {"t": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "text": str(text)[:4000], **data}
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tail(n=60):
    if not LOG.exists():
        return []
    lines = LOG.read_text(encoding="utf-8").splitlines()
    if len(lines) > MAX_LOG:                         # обрезать журнал
        LOG.write_text("\n".join(lines[-MAX_LOG // 2:]) + "\n", encoding="utf-8")
    return [json.loads(l) for l in lines[-n:] if l.strip()]


class Bridge:
    """Подключается к окну App: журнал, перехват ошибок, состояние, очередь команд."""

    def __init__(self, app):
        self.app = app
        INBOX.mkdir(parents=True, exist_ok=True)
        OUTBOX.mkdir(parents=True, exist_ok=True)
        self._last_state = None
        app.status.trace_add("write", lambda *a: log("status", app.status.get()))
        app.report_callback_exception = self._tk_error
        threading.excepthook = lambda a: log("error", f"{a.exc_type.__name__}: {a.exc_value}",
                                             trace="".join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
        log("start", "приложение запущено", python=sys.version.split()[0])

    def _tk_error(self, exc, val, tb):
        log("error", f"{exc.__name__}: {val}", trace="".join(traceback.format_exception(exc, val, tb)))
        self.app.status.set(f"Ошибка: {exc.__name__}: {val} (подробности — session/log.jsonl)")

    def action(self, name, **info):
        log("action", name, **info)

    # ---------- состояние ----------

    def state(self):
        a = self.app
        comps = a.netlist.get("components", []) if a.netlist else []
        sel = a.tv_comp.selection() if hasattr(a, "tv_comp") else ()
        st = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "project": str(a.project.resolve()) if a.project else None,
            "window": a.title(),
            "status": a.status.get(),
            "tab": a.tabs.tab(a.tabs.select(), "text") if a.tabs.tabs() else None,
            "selected_component": sel[0] if sel else None,
            "highlight": sorted(a.hl_refs),
            "netlist": {
                "generated": a.netlist.get("generated") if a.netlist else None,
                "components": len(comps),
                "doubtful": sum(1 for c in comps if c.get("confidence") == "low"),
                "checked_by_user": sum(1 for c in comps if c.get("checked")),
                "nets": len(getattr(a, "nets", {}) or {}),
                "issues": [f"[{i['level']}] {i.get('ref') or i.get('cell')}: {i['text']}" for i in (a.issues or [])][:20]
                if getattr(a, "issues", None) else [],
            },
            "open_windows": [w.title() for w in a.winfo_children() if w.winfo_class() == "Toplevel"],
            "files": sorted(p.name for p in a.project.iterdir()) if a.project and a.project.exists() else [],
        }
        return st

    def write_state(self):
        try:
            st = self.state()
        except Exception as e:                       # окно ещё строится
            st = {"error": str(e)}
        key = json.dumps({k: v for k, v in st.items() if k != "time"}, ensure_ascii=False, sort_keys=True)
        if key != self._last_state:
            STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
            self._last_state = key

    def screenshot(self):
        """Снимок именно окна программы, даже если его перекрывают другие окна (Windows PrintWindow)."""
        a = self.app
        a.update_idletasks()
        img = None
        if sys.platform == "win32":
            try:
                img = _print_window(int(a.wm_frame(), 16))
            except Exception as e:
                log("error", f"PrintWindow: {e}")
        if img is None:                                  # запасной путь: область экрана
            from PIL import ImageGrab
            x, y = a.winfo_rootx(), a.winfo_rooty()
            img = ImageGrab.grab(bbox=(x, y, x + a.winfo_width(), y + a.winfo_height()), all_screens=True)
        img.save(SCREEN)
        return str(SCREEN)

    # ---------- команды от ассистента ----------

    def poll(self):
        self.write_state()
        for f in sorted(INBOX.glob("*.json")):
            try:
                cmd = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            f.unlink(missing_ok=True)
            res = self.run(cmd.get("action"), cmd.get("args") or {})
            (OUTBOX / f.name).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")

    def run(self, name, args):
        a = self.app
        log("assistant", f"команда {name}", args=args)
        try:
            if name not in ACTIONS:
                return {"ok": False, "error": f"нет такого действия; доступны: {', '.join(ACTIONS)}"}
            if name == "state":
                return {"ok": True, "state": self.state()}
            if name == "screenshot":
                return {"ok": True, "path": self.screenshot()}
            if name == "open_project":
                a.open_project(Path(args["path"]))
            elif name == "select":
                ref = args["ref"]
                if ref not in getattr(a, "comps", {}):
                    return {"ok": False, "error": f"нет детали {ref}"}
                a.tabs.select(0)
                a.tv_comp.selection_set(ref)
                a.tv_comp.see(ref)
                a.on_comp()
            elif name == "run_analysis":
                a.run_analysis()
            elif name == "run_autonet":
                a.run_autonet()
            elif name == "run_pcb":
                a.run_pcb()
            elif name == "diagnose":
                a.var_desc.set(args.get("text", ""))
                a.run_diag()
            elif name == "status":
                a.status.set(str(args.get("text", "")))
            a.update_idletasks()
            return {"ok": True, "state": self.state(), "note": "долгие действия идут в фоне — следите за state и log"}
        except Exception as e:
            log("error", f"команда {name}: {type(e).__name__}: {e}", trace=traceback.format_exc())
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _print_window(hwnd):
    """Содержимое окна hwnd через PrintWindow (PW_RENDERFULLCONTENT) — без перекрывающих окон."""
    import ctypes
    from ctypes import wintypes
    from PIL import Image
    u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
    rect = wintypes.RECT()
    u32.GetWindowRect(hwnd, ctypes.byref(rect))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    hdc = u32.GetWindowDC(hwnd)
    mdc = g32.CreateCompatibleDC(hdc)
    bmp = g32.CreateCompatibleBitmap(hdc, w, h)
    g32.SelectObject(mdc, bmp)
    try:
        if not u32.PrintWindow(hwnd, mdc, 2):
            return None

        class BMI(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]
        bmi = BMI(ctypes.sizeof(BMI), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        buf = ctypes.create_string_buffer(w * h * 4)
        g32.GetDIBits(mdc, bmp, 0, h, buf, ctypes.byref(bmi), 0)
        return Image.frombuffer("RGB", (w, h), buf, "raw", "BGRX", 0, 1)
    finally:
        g32.DeleteObject(bmp)
        g32.DeleteDC(mdc)
        u32.ReleaseDC(hwnd, hdc)


def send(action, args=None, timeout=20.0):
    """Со стороны ассистента: положить команду в очередь и дождаться ответа приложения."""
    INBOX.mkdir(parents=True, exist_ok=True)
    cid = f"{time.time():.6f}".replace(".", "_") + ".json"
    tmp = INBOX / (cid + ".tmp")
    tmp.write_text(json.dumps({"action": action, "args": args or {}}, ensure_ascii=False), encoding="utf-8")
    tmp.rename(INBOX / cid)
    out = OUTBOX / cid
    t0 = time.time()
    while time.time() - t0 < timeout:
        if out.exists():
            res = json.loads(out.read_text(encoding="utf-8"))
            out.unlink(missing_ok=True)
            return res
        time.sleep(0.2)
    (INBOX / cid).unlink(missing_ok=True)
    return {"ok": False, "error": "приложение не ответило — оно запущено? (команды обрабатываются раз в 1,5 с)"}
