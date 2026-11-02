"""MCP-сервер «Сканер электросхем»: Claude Code или Codex видит и направляет приложение.

Запускается ассистентом сам (stdio, JSON-RPC 2.0, по строке на сообщение), зависимостей нет.
Подключение — кнопкой «Ассистент» в приложении или вручную:
  Claude Code:  файл .mcp.json в папке программы (создаёт кнопка)
  Codex:        ~/.codex/config.toml →  [mcp_servers.schematic-scanner] command = "python", args = ["…/mcp_server.py"]

Инструменты:
  app_state      — что сейчас в окне: проект, выбранная деталь, сводка netlist, сообщение, открытые окна;
  app_log        — журнал действий и ошибок (с traceback);
  app_screenshot — снимок окна приложения (картинка);
  app_action     — действие в приложении из белого списка (открыть проект, анализ, netlist без ИИ, плата,
                   выделить деталь, диагностика);
  project_file   — прочитать файл текущего проекта (netlist.json, diagnosis.md, pcb/report.md…).
"""
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge  # noqa: E402

TOOLS = [
    {"name": "app_state", "description": "Текущее состояние окна «Сканер электросхем»: проект, вкладка, выбранная деталь, "
                                         "сводка netlist (деталей, сомнительных, проверенных), замечания, открытые окна.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "app_log", "description": "Журнал приложения: действия, сообщения строки состояния, ошибки с traceback.",
     "inputSchema": {"type": "object", "properties": {"lines": {"type": "integer", "description": "сколько последних записей (по умолчанию 60)"},
                                                      "errors_only": {"type": "boolean"}}}},
    {"name": "app_screenshot", "description": "Снимок окна приложения — увидеть то же, что видит пользователь.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "app_action", "description": "Выполнить действие в приложении. Доступные действия: " +
                                          "; ".join(f"{k} — {v}" for k, v in bridge.ACTIONS.items()),
     "inputSchema": {"type": "object", "properties": {"action": {"type": "string", "enum": list(bridge.ACTIONS)},
                                                      "args": {"type": "object"}}, "required": ["action"]}},
    {"name": "project_file", "description": "Прочитать текстовый файл текущего проекта (путь относительно папки проекта, "
                                            "например netlist.json, diagnosis.md, pcb/report.md, analysis/netlist_auto.json).",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
]


def text(s):
    return {"content": [{"type": "text", "text": s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, indent=1)}]}


def call(name, args):
    if name == "app_state":
        res = bridge.send("state", timeout=5)
        if not res.get("ok") and bridge.STATE.exists():      # приложение закрыто — последнее записанное состояние
            return text({"note": "приложение не отвечает; последнее сохранённое состояние",
                         "state": json.loads(bridge.STATE.read_text(encoding="utf-8"))})
        return text(res)
    if name == "app_log":
        rows = bridge.tail(int(args.get("lines") or 60) * (5 if args.get("errors_only") else 1))
        if args.get("errors_only"):
            rows = [r for r in rows if r.get("kind") == "error"]
        return text("\n".join(f"{r['t']} [{r['kind']}] {r['text']}" + (f"\n{r['trace']}" if r.get("trace") else "")
                              for r in rows) or "журнал пуст")
    if name == "app_screenshot":
        res = bridge.send("screenshot", timeout=10)
        if not res.get("ok"):
            return text(res)
        data = base64.b64encode(Path(res["path"]).read_bytes()).decode()
        return {"content": [{"type": "image", "data": data, "mimeType": "image/png"}]}
    if name == "app_action":
        return text(bridge.send(args["action"], args.get("args") or {}, timeout=30))
    if name == "project_file":
        st = bridge.send("state", timeout=5)
        proj = (st.get("state") or {}).get("project") if st.get("ok") else None
        if not proj and bridge.STATE.exists():
            proj = json.loads(bridge.STATE.read_text(encoding="utf-8")).get("project")
        if not proj:
            return text("в приложении не открыт проект")
        root = Path(proj).resolve()
        p = (root / args["path"]).resolve()
        if root not in p.parents and p != root:
            return text("можно читать только файлы внутри папки проекта")
        if not p.exists():
            return text(f"нет файла {args['path']}; в проекте: {', '.join(sorted(x.name for x in root.iterdir()))}")
        return text(p.read_text(encoding="utf-8", errors="replace")[:200000])
    raise ValueError(f"нет инструмента {name}")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2024-11-05"),
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "schematic-scanner", "version": "1.0"},
                  "instructions": "Сканер электросхем: смотрите app_state и app_log, чтобы понять, что происходит "
                                  "у пользователя; app_screenshot — увидеть окно; app_action — действовать в приложении."}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params", {})
        try:
            result = call(p.get("name"), p.get("arguments") or {})
        except Exception as e:
            result = {"content": [{"type": "text", "text": f"ошибка: {type(e).__name__}: {e}"}], "isError": True}
    elif method == "ping":
        result = {}
    elif mid is None:                                    # уведомления (initialized и т. п.) — без ответа
        return None
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"метод {method} не поддерживается"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    stdin = open(sys.stdin.fileno(), "r", encoding="utf-8", newline="\n", closefd=False)
    stdout = open(sys.stdout.fileno(), "w", encoding="utf-8", newline="\n", closefd=False)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        out = handle(msg)
        if out is not None:
            stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stdout.flush()


if __name__ == "__main__":
    main()
