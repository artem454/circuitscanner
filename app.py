"""Schematic Helper — полуавтомат «картинка схемы → netlist» вместе с Claude.

Порядок работы:
  1. «Открыть картинку» — программа чистит её, кладёт сетку и режет на фрагменты.
  2. «Задание для Claude» — копирует текст; вставь его в Claude Code.
  3. Claude пишет netlist.json в папку проекта — программа подхватывает его сама.
  4. Проверяешь: щёлкай компоненты/цепи — они подсвечиваются на картинке.
     Правый клик по клетке — оставить вопрос или поправку для Claude (notes.json).
"""
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

from PIL import Image, ImageTk

import analyzer
import core
import diagnose

HERE = Path(__file__).resolve().parent

LEVEL_RU = {"error": "ошибка", "warn": "внимание", "question": "вопрос", "info": "инфо"}
LEVEL_COLOR = {"error": "#c62828", "warn": "#b26a00", "question": "#6a1b9a", "info": "#555"}


class App(tk.Tk):
    def __init__(self, project=None):
        super().__init__()
        self.title("Schematic Helper — схема → netlist")
        self.geometry("1500x900")
        self.minsize(1000, 600)

        self.project = None
        self.manifest = None
        self.netlist = {}
        self.notes = []
        self.nl_mtime = None
        self.img = None            # PIL, чистая картинка в пикселях оригинала
        self.view_bg = "source"    # source | clean | analysis
        self.analysis = None       # результат analyzer.analyze
        self.diag = None
        self.fault_rank = []
        self.train_proc = None
        self.zoom = 1.0
        self.off = [0.0, 0.0]
        self.hl_refs = set()       # подсвеченные компоненты
        self.hl_cells = set()      # подсвеченные клетки (col, row)
        self.drag = None
        self._photo = None
        self._need_fit = False

        self._build_ui()
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        if project:
            self.open_project(project)
        self.after(1500, self._poll)

    # ---------- интерфейс ----------

    def _build_ui(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Treeview", rowheight=22)

        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side="top", fill="x")
        ttk.Button(bar, text="Открыть картинку…", command=self.open_image).pack(side="left")
        ttk.Button(bar, text="Открыть проект…", command=self.ask_project).pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Label(bar, text="Клеток по длинной стороне:").pack(side="left")
        self.var_cells = tk.IntVar(value=12)
        ttk.Spinbox(bar, from_=4, to=40, width=4, textvariable=self.var_cells).pack(side="left")
        ttk.Label(bar, text="  Клеток во фрагменте:").pack(side="left")
        self.var_tile = tk.IntVar(value=3)
        ttk.Spinbox(bar, from_=1, to=8, width=3, textvariable=self.var_tile).pack(side="left")
        self.var_deskew = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Выпрямлять наклон", variable=self.var_deskew).pack(side="left", padx=6)
        ttk.Button(bar, text="Пересобрать фрагменты", command=self.rebuild).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(bar, text="Анализ без ИИ", command=self.run_analysis).pack(side="left", padx=4)
        ttk.Button(bar, text="Netlist без ИИ", command=self.run_autonet).pack(side="left")
        ttk.Button(bar, text="Задание для Claude", command=self.copy_task).pack(side="left")
        ttk.Button(bar, text="Экспорт", command=self.export).pack(side="left", padx=4)
        ttk.Button(bar, text="Папка проекта", command=self.open_folder).pack(side="left")
        ttk.Button(bar, text="Дообучить", command=self.train).pack(side="left", padx=4)

        self.status = tk.StringVar(value="Откройте картинку схемы или папку проекта.")
        ttk.Label(self, textvariable=self.status, anchor="w", relief="sunken",
                  padding=(6, 2)).pack(side="bottom", fill="x")

        pan = ttk.PanedWindow(self, orient="horizontal")
        pan.pack(fill="both", expand=True)

        left = ttk.Frame(pan)
        vbar = ttk.Frame(left)
        vbar.pack(side="top", fill="x")
        ttk.Button(vbar, text="Вписать", command=self.fit).pack(side="left", padx=2, pady=2)
        ttk.Button(vbar, text="Оригинал / чистая / цепи", command=self.toggle_bg).pack(side="left", padx=2)
        ttk.Button(vbar, text="Снять подсветку", command=self.clear_hl).pack(side="left", padx=2)
        self.var_grid = tk.BooleanVar(value=True)
        self.var_marks = tk.BooleanVar(value=True)
        ttk.Checkbutton(vbar, text="Сетка", variable=self.var_grid, command=self.redraw).pack(side="left", padx=6)
        ttk.Checkbutton(vbar, text="Метки компонентов", variable=self.var_marks,
                        command=self.redraw).pack(side="left")
        ttk.Label(vbar, text="   колесо — масштаб, тянуть — двигать, правый клик — вопрос Claude",
                  foreground="#777").pack(side="left")
        self.canvas = tk.Canvas(left, bg="#808080", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        pan.add(left, weight=3)

        right = ttk.Frame(pan)
        pan.add(right, weight=1)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self.tv_comp = self._tree(self.tabs, "Компоненты", [("ref", "Обозн.", 70), ("type", "Тип", 50),
                                                            ("value", "Номинал", 120), ("cell", "Клетка", 55)])
        self.tv_net = self._tree(self.tabs, "Цепи", [("net", "Цепь", 130), ("n", "Выв.", 40),
                                                     ("label", "Метка", 120)])
        self.tv_issue = self._tree(self.tabs, "Проверка", [("lvl", "Уровень", 75), ("where", "Где", 70),
                                                           ("text", "Текст", 300)])
        self.tv_block = self._tree(self.tabs, "Блоки", [("name", "Блок", 160), ("cells", "Клетки", 70),
                                                        ("n", "Комп.", 45)])
        self.tv_note = self._tree(self.tabs, "Мои вопросы", [("cell", "Клетка", 60), ("text", "Текст", 330)])
        ttk.Button(self.tv_note.master, text="Удалить выбранный вопрос",
                   command=self.delete_note).pack(side="bottom", fill="x", before=self.tv_note)
        self._build_diag_tab()

        for lvl, color in LEVEL_COLOR.items():
            self.tv_issue.tag_configure(lvl, foreground=color)
        self.tv_comp.tag_configure("low", foreground="#b26a00")

        ttk.Label(right, text="Подробности", padding=(4, 4, 0, 0)).pack(anchor="w")
        self.detail = tk.Text(right, height=13, wrap="word", font=("Consolas", 10))
        self.detail.pack(fill="x")
        self.detail.tag_configure("h", font=("Consolas", 10, "bold"))

        self.tv_comp.bind("<<TreeviewSelect>>", self.on_comp)
        self.tv_net.bind("<<TreeviewSelect>>", self.on_net)
        self.tv_issue.bind("<<TreeviewSelect>>", self.on_issue)
        self.tv_block.bind("<<TreeviewSelect>>", self.on_block)
        self.tv_note.bind("<<TreeviewSelect>>", self.on_note)

        c = self.canvas
        c.bind("<MouseWheel>", self.on_wheel)
        c.bind("<ButtonPress-1>", self.on_press)
        c.bind("<B1-Motion>", self.on_motion)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_right)

    def _build_diag_tab(self):
        frame = ttk.Frame(self.tabs)
        self.tabs.add(frame, text="Диагностика")
        box = ttk.LabelFrame(frame, text="Симптомы", padding=4)
        box.pack(fill="x")
        self.sym_vars = {}
        for i, (key, text) in enumerate(diagnose.SYMPTOMS.items()):
            v = tk.BooleanVar(value=False)
            self.sym_vars[key] = v
            ttk.Checkbutton(box, text=text, variable=v).grid(row=i // 2, column=i % 2, sticky="w")
        drow = ttk.Frame(frame)
        drow.pack(fill="x", pady=2)
        ttk.Label(drow, text="Неисправность словами:").pack(side="left")
        self.var_desc = tk.StringVar()
        ent = ttk.Entry(drow, textvariable=self.var_desc)
        ent.pack(side="left", fill="x", expand=True, padx=4)
        ent.bind("<Return>", lambda e: self.run_diag())
        row = ttk.Frame(frame)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Найти причины", command=self.run_diag).pack(side="left")
        ttk.Button(row, text="Деталь неисправна", command=lambda: self.record_repair("faulty")).pack(side="left", padx=4)
        ttk.Button(row, text="Деталь исправна", command=lambda: self.record_repair("ok")).pack(side="left")
        ttk.Button(row, text="Условия (physics.json)", command=self.open_physics).pack(side="left", padx=4)
        mrow = ttk.Frame(frame)
        mrow.pack(fill="x", pady=2)
        ttk.Label(mrow, text="Измерения:").pack(side="left")
        self.var_meas = tk.StringVar()
        ttk.Entry(mrow, textvariable=self.var_meas).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(mrow, text="Поломка по измерениям", command=self.run_faults).pack(side="left")
        ttk.Label(frame, text="формат: цепь=вольты; …  например  ADC0N=0; AI1=0,02; +24V=23,8",
                  foreground="#777").pack(anchor="w")
        prow = ttk.Frame(frame)
        prow.pack(fill="x", pady=2)
        ttk.Button(prow, text="Найти деталь в справочнике…", command=self.find_part).pack(side="left")
        ttk.Button(prow, text="Импорт даташита (PDF)…", command=self.import_datasheet).pack(side="left", padx=4)
        inner = ttk.Frame(frame)
        inner.pack(fill="both", expand=True)
        tv = ttk.Treeview(inner, columns=("p", "ref", "what"), show="headings", selectmode="browse")
        for key, head, width in (("p", "%", 50), ("ref", "Деталь", 80), ("what", "Отказ / слабое место", 300)):
            tv.heading(key, text=head)
            tv.column(key, width=width, stretch=key == "what")
        sb = ttk.Scrollbar(inner, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tv.pack(fill="both", expand=True)
        for lvl, color in (("высокий", "#c62828"), ("средний", "#b26a00"), ("низкий", "#555"), ("информация", "#555")):
            tv.tag_configure(lvl, foreground=color)
        tv.tag_configure("stage", foreground="#1565c0")
        tv.bind("<<TreeviewSelect>>", self.on_diag)
        self.tv_diag = tv

    def run_diag(self):
        if not self.project or not self.netlist:
            return messagebox.showinfo("Диагностика", "Нужен netlist.json: сначала распознайте схему.")
        for k in diagnose.parse_symptoms(self.var_desc.get()):   # «не запускается» → отметить симптом
            self.sym_vars[k].set(True)
        syms = [k for k, v in self.sym_vars.items() if v.get()]
        try:
            self.diag = diagnose.diagnose(self.project, syms)
        except Exception as e:
            return messagebox.showerror("Диагностика", str(e))
        tv = self.tv_diag
        tv.delete(*tv.get_children())
        plan = self.diag.get("plan") or {}
        if plan.get("focus"):
            tv.insert("", "end", iid="pf", tags=("stage",),
                      values=("", "Где", "Вероятное место: " + ", ".join(
                          f"{a['ref']} ({a['cell'] or '?'})" for a in plan["focus"][:5])))
        stage = None
        for i, st in enumerate(plan.get("steps", [])):
            if st["stage"] != stage:
                stage = st["stage"]
                tv.insert("", "end", iid=f"ps{i}", tags=("stage",), values=("", "", f"▶ {stage}"))
            tv.insert("", "end", iid=f"p{i}", values=(f"{i + 1}.", ", ".join(st["refs"][:3]), st["what"]))
        for i, r in enumerate(self.diag["risks"]):
            if r["level"] == "информация":
                continue
            tv.insert("", "end", iid=f"r{i}", tags=(r["level"],),
                      values=("⚠", ", ".join(r["refs"][:3]), f"[{r['level']}] {r['text']}"))
        for i, c in enumerate(self.diag["candidates"]):
            tv.insert("", "end", iid=f"c{i}", values=(f"{c['p'] * 100:.1f}", c["ref"], c["mode"]))
        self.status.set(f"Диагностика: {len(self.diag['candidates'])} кандидатов, "
                        f"слабых мест {len(self.diag['risks'])}, учтено ремонтов {self.diag['repairs_used']}. "
                        f"Отчёт с формулами — diagnosis.md")

    def run_faults(self):
        """Поиск поломки по измеренным напряжениям (моделирование неисправностей)."""
        if not self.project or not self.netlist:
            return messagebox.showinfo("Измерения", "Нужен netlist.json: сначала распознайте схему.")
        meas = {}
        for part in re.split(r"[;\n]+", self.var_meas.get()):
            k, _, v = part.partition("=")
            try:
                meas[k.strip()] = float(v.strip().replace(",", "."))
            except ValueError:
                pass
        if not meas:
            return messagebox.showinfo("Измерения", "Введите, например: ADC0N=0; AI1=0,02")
        import faults
        import physics
        fl = faults.simulate(self.project)
        unknown = [n for n in meas if not any(n in f["U"] for f in fl)]
        ranked = faults.by_measurements(fl, meas)
        base = physics.Physics(self.netlist, core.load_json(self.project / "physics.json") or {})
        base.solve_dc()
        tests = faults.next_test_points(ranked, meas, base.fixed)
        self.fault_rank = ranked
        tv = self.tv_diag
        tv.delete(*tv.get_children())
        for i, r in enumerate(ranked[:15]):
            tv.insert("", "end", iid=f"f{i}", values=(f"{r['p'] * 100:.1f}", r["ref"] or "—",
                                                     r["desc"] if r["ref"] else "схема исправна"))
        self._set_detail([(faults.report(ranked, tests, meas) + "\n", None)] +
                         ([(f"\nНет таких цепей в схеме: {', '.join(unknown)}\n", "h")] if unknown else []))
        self.status.set(f"Смоделировано поломок: {len(fl) - 1}. Главный кандидат: "
                        f"{ranked[0]['ref'] or 'исправна'} {ranked[0]['desc']} ({ranked[0]['p'] * 100:.0f} %)")

    def find_part(self):
        import parts
        q = simpledialog.askstring("Справочник", "Деталь (как в номинале на схеме), например IRFP460:", parent=self)
        if not q:
            return
        name, d = parts.lookup(q)
        if not d:
            return self._set_detail([(f"{q}: нет в справочнике. Добавьте её импортом даташита.\n", "h")])
        rows = [(f"{name}{' (ваша)' if d.get('_user') else ''}\n", "h")]
        rows += [(f"  {k:10} {v}\n", None) for k, v in d.items() if k not in ("pins", "_user")]
        self._set_detail(rows)

    def import_datasheet(self):
        import parts
        path = filedialog.askopenfilename(title="Даташит", filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        try:
            name, p, where = parts.import_pdf(path)
        except SystemExit as e:
            return messagebox.showerror("Импорт", str(e))
        except Exception as e:
            return messagebox.showerror("Импорт", f"Не удалось прочитать PDF: {e}")
        rows = [(f"{name} — добавлено в learned/parts_user.json\n", "h")]
        rows += [(f"  {k:8} = {p[k]:g}   ← {where[k]}\n", None) for k in where]
        rows.append(("\nПроверьте значения по даташиту: таблицы PDF распознаются не идеально.\n", None))
        self._set_detail(rows)

    def on_diag(self, _=None):
        sel = self.tv_diag.selection()
        if not sel or not self.diag and not sel[0].startswith("f"):
            return
        if sel[0].startswith("f"):
            r = self.fault_rank[int(sel[0][1:])]
            refs = [r["ref"]] if r["ref"] else []
            self.hl_refs, self.hl_cells = set(refs), self._cells_of(refs)
            parts_ = [(f"{r['ref'] or 'исправна'} — {r['desc']} ({r['p'] * 100:.1f} %)\n", "h")]
            parts_ += [(f"  {n}: модель {us:.2f} В, измерено {um:.2f} В\n", None) for n, us, um in r["detail"]]
            if r.get("symptoms"):
                parts_.append(("Ожидаемые симптомы: " + ", ".join(diagnose.SYMPTOMS.get(s, s) for s in r["symptoms"]) + "\n", None))
            self._set_detail(parts_)
            return self._focus_cells()
        if sel[0].startswith("p"):
            plan = self.diag.get("plan") or {}
            if sel[0] == "pf" or sel[0].startswith("ps"):
                refs = [a["ref"] for a in plan.get("focus", [])]
                parts = [("Где искать\n", "h")] + [(f"> {f}\n\n", None) for f in plan.get("first", [])]
                parts += [(f"• {a['ref']} {a['value']} [{a['cell'] or '?'}] — {a['p'] * 100:.0f} %: "
                           f"{', '.join(dict.fromkeys(a['modes']))}\n", None) for a in plan.get("focus", [])]
            else:
                st = plan["steps"][int(sel[0][1:])]
                refs = st["refs"]
                parts = [(f"{st['stage']}: {st['what']}\n", "h"),
                         (f"Место: {st['cell'] or '—'}\n\n", None),
                         (f"Норма / что ожидать:\n{st['expect']}\n\n", None)]
                if st["why"]:
                    parts.append((f"Почему: {st['why']}\n", None))
            self.hl_refs, self.hl_cells = set(refs), self._cells_of(refs)
            self._set_detail(parts)
            return self._focus_cells()
        kind, i = sel[0][0], int(sel[0][1:])
        if kind == "r":
            r = self.diag["risks"][i]
            refs = r["refs"]
            parts = [(f"[{r['level']}] {r['text']}\n", "h"), (", ".join(refs) + "\n\n", None), (r["why"] + "\n", None)]
        else:
            c = self.diag["candidates"][i]
            refs = [c["ref"]]
            parts = [(f"{c['ref']} {c['value']} — {c['mode']} ({c['p'] * 100:.1f} %)\n", "h"),
                     ("роль: " + ", ".join(c["roles"]) + "\n", None)]
            parts += [(f"• {w}\n", None) for w in c["why"]]
            parts.append((f"\nКак проверить: {c['check']}\n", None))
        self.hl_refs = set(refs)
        self.hl_cells = self._cells_of(refs)
        self._set_detail(parts)
        self._focus_cells()

    def record_repair(self, result):
        sel = self.tv_diag.selection()
        if not sel or not sel[0].startswith("c"):
            return messagebox.showinfo("Ремонт", "Выберите деталь в списке кандидатов.")
        c = self.diag["candidates"][int(sel[0][1:])]
        syms = [k for k, v in self.sym_vars.items() if v.get()]
        diagnose.record(self.project, c["ref"], result, syms)
        self.status.set(f"Записано: {c['ref']} — {'неисправна' if result == 'faulty' else 'исправна'}. "
                        f"Модель учтёт это в следующих диагнозах.")
        self.run_diag()

    def open_physics(self):
        if self.project:
            p = self.project / "physics.json"
            if not p.exists():
                import physics
                p.write_text(json.dumps(physics.DEFAULT_SETTINGS, ensure_ascii=False, indent=1), encoding="utf-8")
            os.startfile(p)

    # ---------- анализ без ИИ и обучение ----------

    def run_analysis(self, quiet=False):
        if not self.project:
            return
        self.status.set("Анализ без ИИ: провода, точки, символы, цепи…")

        def work():
            try:
                res = analyzer.analyze(self.project, ocr=True,
                                       progress=lambda m: self.after(0, self.status.set, f"Анализ: {m}…"))
                self.after(0, self._analysis_done, res)
            except Exception as e:
                self.after(0, self.status.set, f"Анализ не удался: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _analysis_done(self, res):
        self.analysis = res
        self.status.set("Анализ без ИИ: " + analyzer.summary(res) +
                        ". Цепи — кнопка «Оригинал / чистая / цепи»; Claude получит их подсказкой.")
        if self.view_bg == "analysis":
            self._load_bg()
            self.redraw()

    def run_autonet(self):
        """Собрать netlist самостоятельно (OCR + разбор). Проверенный netlist.json не затирается."""
        if not self.project:
            return messagebox.showinfo("Нет проекта", "Сначала откройте картинку.")
        self.status.set("Netlist без ИИ: разбор и чтение надписей…")

        def work():
            try:
                import autonet
                nl = autonet.run(self.project, progress=lambda m: self.after(0, self.status.set, f"Netlist без ИИ: {m}…"))
                self.after(0, self._autonet_done, nl)
            except Exception as e:
                self.after(0, self.status.set, f"Netlist без ИИ не удался: {e}")
        threading.Thread(target=work, daemon=True).start()

    def _autonet_done(self, nl):
        n = len(nl["components"])
        named = sum(1 for c in nl["components"] if "?" not in c["ref"])
        vals = sum(1 for c in nl["components"] if c["value"])
        if nl.get("_file") == "netlist.json":
            self.load_netlist()
            where = "netlist.json — открыт во вкладках"
        else:
            where = "netlist_autonet.json (проверенный netlist.json не тронут)"
        self.status.set(f"Netlist без ИИ: деталей {n}, с прочитанным обозначением {named}, с номиналом {vals} → {where}. "
                        "Это черновик: сверяйте по картинке или отдайте Claude на доработку.")

    def train(self, project=None, quiet=False):
        """Дообучение в отдельном процессе, чтобы окно не зависало."""
        if self.train_proc and self.train_proc.poll() is None:
            if not quiet:
                self.status.set("Обучение уже идёт…")
            return
        args = ["add", str(project)] if project else ["auto", "60", "20"]
        (HERE / "learned").mkdir(exist_ok=True)
        log = open(HERE / "learned" / "last_train.log", "w", encoding="utf-8")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.train_proc = subprocess.Popen([sys.executable, str(HERE / "learn.py")] + args, cwd=HERE,
                                           stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        self.status.set("Обучение: " + ("добавляю проверенную схему в примеры…" if project else
                                        "проекты → классификатор символов → параметры (несколько минут)…"))
        self.after(2000, self._train_wait)

    def _train_wait(self):
        if self.train_proc and self.train_proc.poll() is None:
            return self.after(2000, self._train_wait)
        try:
            tail = (HERE / "learned" / "last_train.log").read_text(encoding="utf-8").replace("\r", "\n")
            last = [l for l in tail.splitlines() if l.strip()][-1:]
        except Exception:
            last = []
        self.status.set("Обучение завершено. " + (last[0] if last else ""))

    def _tree(self, nb, title, cols):
        frame = ttk.Frame(nb)
        tv = ttk.Treeview(frame, columns=[k for k, _, _ in cols], show="headings", selectmode="browse")
        for key, head, width in cols:
            tv.heading(key, text=head)
            tv.column(key, width=width, stretch=key in ("text", "value", "label", "name"))
        sb = ttk.Scrollbar(frame, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        tv.pack(fill="both", expand=True)
        nb.add(frame, text=title)
        return tv

    # ---------- проект ----------

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Картинка схемы",
            filetypes=[("Картинки", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.gif"), ("Все", "*.*")])
        if path:
            self._prepare(path, None)

    def rebuild(self):
        if not self.project:
            return messagebox.showinfo("Нет проекта", "Сначала откройте картинку.")
        orig = next(self.project.glob("original.*"), None)
        if orig:
            self._prepare(orig, self.project)

    def _prepare(self, image, project):
        self.status.set("Подготовка картинки…")
        cells, tile, deskew = self.var_cells.get(), self.var_tile.get(), self.var_deskew.get()

        def work():
            try:
                p = core.prepare(image, project, cells, tile, deskew,
                                 progress=lambda k, n: self.after(0, self.status.set, f"Фрагменты: {k}/{n}"))
                self.after(0, self.open_project, p)
                self.after(100, self.run_analysis)          # сразу и анализ без ИИ
            except Exception as e:
                self.after(0, messagebox.showerror, "Ошибка подготовки", str(e))
        threading.Thread(target=work, daemon=True).start()

    def ask_project(self):
        path = filedialog.askdirectory(title="Папка проекта (где manifest.json)")
        if path:
            self.open_project(path)

    def open_project(self, path):
        path = Path(path)
        manifest = core.load_json(path / "manifest.json")
        if not manifest:
            return messagebox.showerror("Не проект", f"В {path} нет manifest.json")
        self.project, self.manifest = path, manifest
        self._load_bg()
        self.nl_mtime = None
        self.load_netlist()
        self.load_notes()
        self.title(f"Schematic Helper — {path.name}")
        g = manifest["grid"]
        self.status.set(f"Проект {path}  |  сетка {g['cols']}×{g['rows']}, фрагментов: "
                        f"{len(manifest['tiles'])}  |  netlist: "
                        f"{'есть' if self.netlist else 'ещё нет — нажмите «Задание для Claude»'}")
        self._need_fit = True
        if self.canvas.winfo_width() > 50:
            self.fit()

    def _load_bg(self):
        if self.view_bg == "analysis":
            res = self.analysis or core.load_json(self.project / "analysis" / "analysis.json")
            if res:
                gray = core.imread(self.project / "clean.png")[:, :, 0]
                bgr = analyzer.draw(gray, res, 1)
                self.img = Image.fromarray(bgr[:, :, ::-1])
                return
            self.view_bg = "source"
        name = "source.png" if self.view_bg == "source" else "clean.png"
        self.img = Image.open(self.project / name).convert("RGB")

    def toggle_bg(self):
        if self.project:
            order = ["source", "clean", "analysis"]
            self.view_bg = order[(order.index(self.view_bg) + 1) % 3]
            if self.view_bg == "analysis" and not (self.analysis or (self.project / "analysis" / "analysis.json").exists()):
                self.status.set("Цепей ещё нет — нажмите «Анализ без ИИ».")
                self.view_bg = "source"
            self._load_bg()
            self.redraw()

    def load_netlist(self):
        p = self.project / "netlist.json"
        if not p.exists():
            self.netlist, self.nl_mtime = {}, None
        else:
            self.nl_mtime = p.stat().st_mtime
            try:
                self.netlist = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                self.status.set(f"netlist.json с ошибкой: {e}")
                return
        self.comps = {c.get("ref", "?"): c for c in self.netlist.get("components", [])}
        self.nets = core.net_index(self.netlist)
        self.issues = core.check(self.netlist, self.manifest) if self.netlist else []
        self._fill_lists()
        self.redraw()

    def load_notes(self):
        self.notes = core.load_json(self.project / "notes.json", []) or []
        tv = self.tv_note
        tv.delete(*tv.get_children())
        for i, n in enumerate(self.notes):
            tv.insert("", "end", iid=str(i), values=(n.get("cell", ""), n.get("text", "")))
        self.redraw()

    def save_notes(self):
        (self.project / "notes.json").write_text(json.dumps(self.notes, ensure_ascii=False, indent=1),
                                                 encoding="utf-8")
        self.load_notes()

    def _fill_lists(self):
        for tv in (self.tv_comp, self.tv_net, self.tv_issue, self.tv_block):
            tv.delete(*tv.get_children())
        for ref, c in sorted(self.comps.items(), key=lambda kv: _natural(kv[0])):
            self.tv_comp.insert("", "end", iid=ref, values=(ref, c.get("type", ""), c.get("value", ""),
                                                           c.get("cell", "")),
                                tags=("low",) if c.get("confidence") == "low" else ())
        meta = self.netlist.get("nets", {})
        for net, conns in sorted(self.nets.items(), key=lambda kv: _natural(kv[0])):
            self.tv_net.insert("", "end", iid=net, values=(net, len(conns), meta.get(net, {}).get("label", "")))
        for i, it in enumerate(self.issues):
            self.tv_issue.insert("", "end", iid=str(i), tags=(it["level"],),
                                 values=(LEVEL_RU.get(it["level"], it["level"]),
                                         it["ref"] or it["cell"], it["text"]))
        for i, b in enumerate(self.netlist.get("blocks", [])):
            cells = b.get("cells", "")
            cells = ":".join(cells) if isinstance(cells, list) else cells
            self.tv_block.insert("", "end", iid=str(i), values=(b.get("name", ""), cells, len(b.get("refs", []))))
        cnt = {lvl: sum(1 for i in self.issues if i["level"] == lvl) for lvl in LEVEL_COLOR}
        self.tabs.tab(2, text=f"Проверка: {cnt['error']} ош. / {cnt['warn']} вн. / {cnt['question']} ?"
                      if self.issues else "Проверка")

    def _poll(self):
        if self.project:
            p = self.project / "netlist.json"
            m = p.stat().st_mtime if p.exists() else None
            if m != self.nl_mtime:
                self.load_netlist()
                if m:
                    self.status.set(f"netlist.json обновлён {datetime.now():%H:%M:%S}: "
                                    f"{len(self.comps)} компонентов, {len(self.nets)} цепей, "
                                    f"замечаний: {len(self.issues)}")
                    # схема без ошибок — сама становится учебным примером для анализатора
                    if self.netlist and not any(i["level"] == "error" for i in self.issues):
                        self.after(3000, lambda: self.train(self.project, quiet=True))
        self.after(1500, self._poll)

    def copy_task(self):
        if not self.project:
            return messagebox.showinfo("Нет проекта", "Сначала откройте картинку.")
        text = (f"Разбери схему в папке {self.project.resolve()} — прочитай там CLAUDE_TASK.md "
                f"и сделай по нему netlist.json.")
        if (self.project / "analysis" / "analysis.json").exists():
            text += (" Там же в analysis/ лежит разбор без ИИ (цветные цепи в analysis/tiles, черновик "
                     "netlist_auto.json) — используй как подсказку, но сверяй с картинкой.")
        if self.notes:
            text += " В notes.json есть мои вопросы и поправки — учти их."
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status.set("Задание скопировано — вставьте его в Claude Code.")

    def export(self):
        if not self.project:
            return
        try:
            out = core.export_all(self.project)
        except Exception as e:
            return messagebox.showerror("Экспорт", str(e))
        self.status.set(f"Экспорт: {out} (schematic.cir, bom.csv, summary.md)")
        os.startfile(out)

    def open_folder(self):
        if self.project:
            os.startfile(self.project)

    # ---------- холст ----------

    def _on_canvas_resize(self, e):
        if self._need_fit and e.width > 50:
            self.fit()
        else:
            self.redraw()

    def fit(self):
        if not self.img:
            return
        self._need_fit = False
        cw, ch = max(self.canvas.winfo_width(), 50), max(self.canvas.winfo_height(), 50)
        w, h = self.img.size
        self.zoom = min(cw / w, ch / h) * 0.97
        self.off = [(cw - w * self.zoom) / 2, (ch - h * self.zoom) / 2]
        self.redraw()

    def to_screen(self, x, y):
        return x * self.zoom + self.off[0], y * self.zoom + self.off[1]

    def to_image(self, sx, sy):
        return (sx - self.off[0]) / self.zoom, (sy - self.off[1]) / self.zoom

    def cell_at(self, sx, sy):
        if not self.manifest:
            return None
        g = self.manifest["grid"]
        x, y = self.to_image(sx, sy)
        c, r = int(x // g["cell"]), int(y // g["cell"])
        if 0 <= c < g["cols"] and 0 <= r < g["rows"] and x >= 0 and y >= 0:
            return c, r
        return None

    def cell_rect(self, c, r):
        g = self.manifest["grid"]
        s = g["cell"]
        x0, y0 = self.to_screen(c * s, r * s)
        x1, y1 = self.to_screen(min(g["width"], (c + 1) * s), min(g["height"], (r + 1) * s))
        return x0, y0, x1, y1

    def redraw(self):
        cv = self.canvas
        cv.delete("all")
        if not self.img:
            return
        cw, ch = cv.winfo_width(), cv.winfo_height()
        w, h = self.img.size
        # рисуем только видимую часть картинки
        ix0, iy0 = self.to_image(0, 0)
        ix1, iy1 = self.to_image(cw, ch)
        ix0, iy0 = max(0, int(ix0)), max(0, int(iy0))
        ix1, iy1 = min(w, int(ix1) + 1), min(h, int(iy1) + 1)
        if ix1 > ix0 and iy1 > iy0:
            crop = self.img.crop((ix0, iy0, ix1, iy1))
            sw, sh = max(1, round((ix1 - ix0) * self.zoom)), max(1, round((iy1 - iy0) * self.zoom))
            res = Image.LANCZOS if self.zoom < 1 else Image.BICUBIC
            self._photo = ImageTk.PhotoImage(crop.resize((sw, sh), res))
            sx, sy = self.to_screen(ix0, iy0)
            cv.create_image(round(sx), round(sy), image=self._photo, anchor="nw")

        g = self.manifest["grid"]
        for (c, r) in self.hl_cells:
            cv.create_rectangle(*self.cell_rect(c, r), fill="#ffd54f", stipple="gray25", outline="#ff8f00", width=2)

        if self.var_grid.get():
            fs = max(7, min(14, int(g["cell"] * self.zoom / 7)))
            for c in range(g["cols"] + 1):
                x, _ = self.to_screen(min(c * g["cell"], g["width"]), 0)
                cv.create_line(x, *self.to_screen(0, 0)[1:], x, self.to_screen(0, g["height"])[1],
                               fill="#e53935", dash=(2, 3))
            for r in range(g["rows"] + 1):
                _, y = self.to_screen(0, min(r * g["cell"], g["height"]))
                cv.create_line(self.to_screen(0, 0)[0], y, self.to_screen(g["width"], 0)[0], y,
                               fill="#e53935", dash=(2, 3))
            for c in range(g["cols"]):
                for r in range(g["rows"]):
                    x0, y0, _, _ = self.cell_rect(c, r)
                    cv.create_text(x0 + 3, y0 + 2, text=f"{core.col_name(c)}{r + 1}", anchor="nw",
                                   fill="#c62828", font=("Segoe UI", fs))

        if self.var_marks.get() and self.comps:
            by_cell = {}
            for ref, comp in self.comps.items():
                pc = core.parse_cell(comp.get("cell", ""))
                if pc:
                    by_cell.setdefault(pc, []).append(ref)
            fs = max(7, min(13, int(g["cell"] * self.zoom / 9)))
            line_h = fs * 1.6
            for (c, r), refs in by_cell.items():
                x0, y0, x1, y1 = self.cell_rect(c, r)
                fit_n = max(1, int((y1 - y0 - line_h) // line_h))
                refs.sort(key=_natural)
                shown = refs if len(refs) <= fit_n else refs[:fit_n - 1] + [f"+{len(refs) - fit_n + 1}"]
                for i, ref in enumerate(shown):
                    hl = ref in self.hl_refs
                    low = self.comps.get(ref, {}).get("confidence") == "low"
                    color = "#d50000" if hl else ("#e65100" if low else "#1565c0")
                    t = cv.create_text(x1 - 3, y0 + line_h * (i + 1), text=ref, anchor="e", fill="white",
                                       font=("Segoe UI", fs, "bold"))
                    bb = cv.bbox(t)
                    cv.create_rectangle(bb[0] - 2, bb[1], bb[2] + 2, bb[3], fill=color, outline="")
                    cv.tag_raise(t)

        for n in self.notes:
            pc = core.parse_cell(n.get("cell", ""))
            if pc:
                x0, y0, x1, y1 = self.cell_rect(*pc)
                cv.create_oval(x0 + 3, y1 - 21, x0 + 21, y1 - 3, fill="#6a1b9a", outline="white")
                cv.create_text(x0 + 12, y1 - 12, text="?", fill="white", font=("Segoe UI", 10, "bold"))

    def on_wheel(self, e):
        if not self.img:
            return
        k = 1.2 if e.delta > 0 else 1 / 1.2
        new = min(12.0, max(0.05, self.zoom * k))
        ix, iy = self.to_image(e.x, e.y)
        self.zoom = new
        self.off = [e.x - ix * new, e.y - iy * new]
        self.redraw()

    def on_press(self, e):
        self.drag = (e.x, e.y, self.off[0], self.off[1], False)

    def on_motion(self, e):
        if not self.drag:
            return
        x, y, ox, oy, moved = self.drag
        if moved or abs(e.x - x) + abs(e.y - y) > 4:
            self.drag = (x, y, ox, oy, True)
            self.off = [ox + e.x - x, oy + e.y - y]
            self.redraw()

    def on_release(self, e):
        if self.drag and not self.drag[4]:
            cell = self.cell_at(e.x, e.y)
            if cell:
                self.show_cell(cell)
        self.drag = None

    def on_right(self, e):
        cell = self.cell_at(e.x, e.y)
        if not cell:
            return
        name = f"{core.col_name(cell[0])}{cell[1] + 1}"
        m = tk.Menu(self, tearoff=0)
        m.add_command(label=f"Вопрос / поправка для Claude по {name}…", command=lambda: self.add_note(name))
        m.add_command(label=f"Что в клетке {name}", command=lambda: self.show_cell(cell))
        m.tk_popup(e.x_root, e.y_root)

    def add_note(self, cell):
        text = simpledialog.askstring("Вопрос для Claude", f"Клетка {cell}.\nНапример: «здесь провода "
                                      f"не соединены», «R5 = 4.7k, а не 47k», «что делает этот узел?»",
                                      parent=self)
        if text:
            self.notes.append({"cell": cell, "text": text, "created": datetime.now().isoformat(timespec="seconds")})
            self.save_notes()
            self.status.set(f"Вопрос по {cell} сохранён в notes.json")

    def delete_note(self):
        sel = self.tv_note.selection()
        if sel:
            del self.notes[int(sel[0])]
            self.save_notes()

    # ---------- выбор и подсветка ----------

    def clear_hl(self):
        self.hl_refs, self.hl_cells = set(), set()
        self.redraw()

    def _set_detail(self, parts):
        d = self.detail
        d.delete("1.0", "end")
        for text, tag in parts:
            d.insert("end", text, tag)

    def _cells_of(self, refs):
        out = set()
        for r in refs:
            pc = core.parse_cell(self.comps.get(r, {}).get("cell", ""))
            if pc:
                out.add(pc)
        return out

    def _comp_text(self, ref):
        c = self.comps.get(ref, {})
        parts = [(f"{ref}  {c.get('type', '')} {core.TYPES.get(c.get('type', ''), '')}  "
                  f"{c.get('value', '')}  [{c.get('cell', '')}]\n", "h")]
        for pin, net in (c.get("pins") or {}).items():
            others = [f"{r}.{p}" for r, p in self.nets.get(net, []) if r != ref]
            parts.append((f"  {pin:>3} → {net}:  {', '.join(others) or '—'}\n", None))
        if c.get("note"):
            parts.append((f"  примечание: {c['note']}\n", None))
        if c.get("confidence") == "low":
            parts.append(("  уверенность низкая — проверьте по картинке\n", None))
        return parts

    def on_comp(self, _=None):
        sel = self.tv_comp.selection()
        if not sel:
            return
        ref = sel[0]
        self.hl_refs = {ref}
        self.hl_cells = self._cells_of([ref])
        self._set_detail(self._comp_text(ref))
        self._focus_cells()

    def on_net(self, _=None):
        sel = self.tv_net.selection()
        if not sel:
            return
        net = sel[0]
        conns = self.nets.get(net, [])
        self.hl_refs = {r for r, _ in conns}
        self.hl_cells = self._cells_of(self.hl_refs)
        meta = self.netlist.get("nets", {}).get(net, {})
        parts = [(f"Цепь {net}" + (f" ({meta['label']})" if meta.get("label") else "") + "\n", "h")]
        for r, p in conns:
            c = self.comps.get(r, {})
            parts.append((f"  {r}.{p}  {c.get('type', '')} {c.get('value', '')}  [{c.get('cell', '')}]\n", None))
        self._set_detail(parts)
        self._focus_cells()

    def on_issue(self, _=None):
        sel = self.tv_issue.selection()
        if not sel:
            return
        it = self.issues[int(sel[0])]
        self.hl_refs = {it["ref"]} if it["ref"] else set()
        pc = core.parse_cell(it["cell"])
        self.hl_cells = {pc} if pc else self._cells_of(self.hl_refs)
        parts = [(f"[{LEVEL_RU.get(it['level'], it['level'])}] {it['text']}\n\n", "h")]
        if it["ref"] in self.comps:
            parts += self._comp_text(it["ref"])
        self._set_detail(parts)
        self._focus_cells()

    def on_block(self, _=None):
        sel = self.tv_block.selection()
        if not sel:
            return
        b = self.netlist.get("blocks", [])[int(sel[0])]
        rng = core.parse_range(b.get("cells", ""))
        self.hl_refs = set(b.get("refs", []))
        self.hl_cells = ({(c, r) for c in range(rng[0], rng[2] + 1) for r in range(rng[1], rng[3] + 1)}
                         if rng else self._cells_of(self.hl_refs))
        self._set_detail([(b.get("name", "") + "\n", "h"), (b.get("description", "") + "\n\n", None),
                          ("Компоненты: " + ", ".join(b.get("refs", [])) + "\n", None)])
        self._focus_cells()

    def on_note(self, _=None):
        sel = self.tv_note.selection()
        if not sel:
            return
        n = self.notes[int(sel[0])]
        pc = core.parse_cell(n.get("cell", ""))
        self.hl_refs, self.hl_cells = set(), {pc} if pc else set()
        self._set_detail([(f"Вопрос по {n.get('cell', '')}\n", "h"), (n.get("text", "") + "\n", None)])
        self._focus_cells()

    def show_cell(self, cell):
        name = f"{core.col_name(cell[0])}{cell[1] + 1}"
        refs = [r for r, c in self.comps.items() if core.parse_cell(c.get("cell", "")) == cell]
        self.hl_cells, self.hl_refs = {cell}, set(refs)
        parts = [(f"Клетка {name}\n", "h")]
        for r in sorted(refs, key=_natural):
            parts += self._comp_text(r)
        notes = [n["text"] for n in self.notes if n.get("cell", "").upper() == name]
        parts += [(f"  ? {t}\n", None) for t in notes]
        if not refs and not notes:
            parts.append(("  компонентов не отмечено\n", None))
        self._set_detail(parts)
        self.redraw()

    def _focus_cells(self):
        """Если подсвеченное не видно — сдвинуть картинку к нему."""
        if not self.hl_cells:
            return self.redraw()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        rects = [self.cell_rect(c, r) for c, r in self.hl_cells]
        x0, y0 = min(r[0] for r in rects), min(r[1] for r in rects)
        x1, y1 = max(r[2] for r in rects), max(r[3] for r in rects)
        if x1 < 0 or y1 < 0 or x0 > cw or y0 > ch:
            self.off[0] += cw / 2 - (x0 + x1) / 2
            self.off[1] += ch / 2 - (y0 + y1) / 2
        self.redraw()


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
