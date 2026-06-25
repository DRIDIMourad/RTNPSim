from __future__ import annotations

import json
import math
import platform
import queue
import subprocess
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

import yaml

from npsim_gui.exporters import export_project_files, run_generated_script
from npsim_gui.scenario import (
    CRITICITIES,
    LAYER_KINDS,
    PRECISIONS,
    SUPPORTS,
    default_arch_settings,
    normalize_task_table,
    periods_to_simulation_time,
    scenario_examples,
    tick_cycles_from_arch,
    cycles_to_ticks,
    scenario_single_cnn,
    scenario_two_cnn,
)

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"

APP_TITLE = "NPsim"
APP_SUBTITLE = "Real-time CPU/NPU simulator"
COLORS = {
    "bg": "#f4f7fb",
    "surface": "#ffffff",
    "surface_alt": "#eef3f8",
    "border": "#d8e0ea",
    "text": "#1f2937",
    "muted": "#5b677a",
    "primary": "#2563eb",
    "primary_hover": "#1d4ed8",
    "success": "#047857",
    "success_hover": "#065f46",
    "warning": "#b45309",
    "danger": "#b91c1c",
}

TASK_COLUMNS = [
    "name",
    "cnn_id",
    "kind",
    "support",
    "precision",
    "period",
    "deadline",
    "priority",
    "phase",
    "wcet_int8",
    "wcet_fp16",
    "wcet_fp32",
    "preds",
    "criticity",
    "size",
]

COLUMN_LABELS = {
    "name": "Name",
    "cnn_id": "CNN",
    "kind": "Type",
    "support": "Support",
    "precision": "Initial Precision",
    "period": "Period (tick)",
    "deadline": "Deadline (tick)",
    "priority": "Priority",
    "phase": "Phase (tick)",
    "wcet_int8": "WCET INT8 (tick)",
    "wcet_fp16": "WCET FP16 (tick)",
    "wcet_fp32": "WCET FP32 (tick)",
    "preds": "Predecessors",
    "criticity": "Criticality",
    "size": "Size (Ki elems)",
}


class ScrollableFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class TaskDialog(tk.Toplevel):
    def __init__(self, parent: tk.Widget, title: str, initial: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(parent)
        self.title(title)
        self.result: Optional[Dict[str, Any]] = None
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        row = initial or {
            "name": "NEW_LAYER",
            "cnn_id": "CNN_A",
            "kind": "generic",
            "support": "CPU/NPU",
            "period": 100,
            "deadline": 85,
            "priority": 0,
            "phase": 0,
            "precision": "INT8",
            "wcet_int8": 20,
            "wcet_fp16": 30,
            "wcet_fp32": 45,
            "preds": "-",
            "criticity": "LO",
            "size": 128,
        }

        self.vars: Dict[str, tk.Variable] = {}
        form = ttk.Frame(self, padding=14)
        form.grid(row=0, column=0, sticky="nsew")

        def add_text(r: int, key: str, label: str, width: int = 26) -> None:
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            var = tk.StringVar(value=str(row.get(key, "")))
            self.vars[key] = var
            ttk.Entry(form, textvariable=var, width=width).grid(row=r, column=1, sticky="ew", pady=4)

        def add_int(r: int, key: str, label: str, minimum: int = 0) -> None:
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            var = tk.IntVar(value=int(row.get(key, minimum)))
            self.vars[key] = var
            ttk.Spinbox(form, textvariable=var, from_=minimum, to=10_000_000, increment=1, width=24).grid(
                row=r, column=1, sticky="ew", pady=4
            )

        def add_combo(r: int, key: str, label: str, values: List[str]) -> None:
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=4)
            var = tk.StringVar(value=str(row.get(key, values[0])))
            self.vars[key] = var
            cb = ttk.Combobox(form, textvariable=var, values=values, state="readonly", width=24)
            cb.grid(row=r, column=1, sticky="ew", pady=4)

        add_text(0, "name", "Task name")
        add_text(1, "cnn_id", "CNN ID")
        add_combo(2, "kind", "Task kind", LAYER_KINDS)
        add_combo(3, "support", "Hardware support", SUPPORTS)
        add_combo(4, "precision", "Initial precision", PRECISIONS)
        add_int(5, "period", "Period (tick)", 1)
        add_int(6, "deadline", "Relative deadline (tick)", 1)
        add_int(7, "priority", "Priority", 0)
        add_int(8, "phase", "Phase (tick)", 0)
        add_int(9, "wcet_int8", "WCET INT8 (tick)", 0)
        add_int(10, "wcet_fp16", "WCET FP16 (tick)", 0)
        add_int(11, "wcet_fp32", "WCET FP32 (tick)", 0)
        add_text(12, "preds", "Predecessors: - or A,B")
        add_combo(13, "criticity", "Criticality", CRITICITIES)
        add_int(14, "size", "Size (Ki elems, ignored for generic)", 1)

        help_txt = (
            "Generic kind = WCET-only task: duration comes only from the selected WCET. "
            "Precision rule: NPU execution is INT8 only; CPU execution is FP16 or FP32. "
            "CPU/NPU lets the mapping algorithm choose the resource."
        )
        ttk.Label(form, text=help_txt, wraplength=420, foreground="#555").grid(row=15, column=0, columnspan=2, sticky="w", pady=(8, 4))

        btns = ttk.Frame(form)
        btns.grid(row=16, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(btns, text="OK", command=self._on_ok).pack(side="right")

        self.bind("<Return>", lambda _e: self._on_ok())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.update_idletasks()
        x = parent.winfo_rootx() + max(0, (parent.winfo_width() - self.winfo_width()) // 2)
        y = parent.winfo_rooty() + 50
        self.geometry(f"+{x}+{y}")
        self.wait_window(self)

    def _on_ok(self) -> None:
        try:
            data = {
                "name": self.vars["name"].get().strip().replace(" ", "_"),
                "cnn_id": self.vars["cnn_id"].get().strip().replace(" ", "_"),
                "kind": self.vars["kind"].get(),
                "support": self.vars["support"].get(),
                "precision": self.vars["precision"].get(),
                "period": int(self.vars["period"].get()),
                "deadline": int(self.vars["deadline"].get()),
                "priority": int(self.vars["priority"].get()),
                "phase": int(self.vars["phase"].get()),
                "wcet_int8": int(self.vars["wcet_int8"].get()),
                "wcet_fp16": int(self.vars["wcet_fp16"].get()),
                "wcet_fp32": int(self.vars["wcet_fp32"].get()),
                "preds": self.vars["preds"].get().strip() or "-",
                "criticity": self.vars["criticity"].get(),
                "size": int(self.vars["size"].get()),
            }
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Invalid value", f"A numeric field is invalid.\n\n{exc}", parent=self)
            return
        normalized = normalize_task_table([data])[0]
        self.result = normalized
        self.destroy()


class NPsimTkApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(f"{APP_TITLE} - {APP_SUBTITLE}")
        self.geometry("1380x900")
        self.minsize(1120, 720)

        self.scenario_examples = scenario_examples()
        self.tasks: List[Dict[str, Any]] = normalize_task_table(self.scenario_examples[0]["tasks"])
        self.arch: Dict[str, Any] = default_arch_settings()
        self.generated_paths: Optional[Dict[str, Path]] = None
        self.run_result: Optional[Dict[str, Any]] = None
        self.result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.gantt_path: Optional[Path] = None
        self.run_controls: List[tk.Widget] = []

        self._build_style()
        self._build_variables()
        self._build_layout()
        self._bind_shortcuts()
        self._refresh_task_table()

    def _build_style(self) -> None:
        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(size=10)
            heading_font = tkfont.nametofont("TkHeadingFont")
            heading_font.configure(size=10, weight="bold")
        except Exception:
            pass

        self.configure(bg=COLORS["bg"])
        self.style = ttk.Style(self)
        try:
            if platform.system() == "Darwin":
                self.style.theme_use("aqua")
            elif "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        try:
            self.style.configure(".", background=COLORS["bg"], foreground=COLORS["text"])
            self.style.configure("TFrame", background=COLORS["bg"])
            self.style.configure("Panel.TFrame", background=COLORS["surface"])
            self.style.configure("AppBar.TFrame", background=COLORS["surface"], relief="flat")
            self.style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
            self.style.configure("Muted.TLabel", background=COLORS["bg"], foreground=COLORS["muted"])
            self.style.configure("Title.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("TkDefaultFont", 16, "bold"))
            self.style.configure("Subtitle.TLabel", background=COLORS["surface"], foreground=COLORS["muted"])
            self.style.configure("Card.TFrame", background=COLORS["surface"], relief="solid", borderwidth=1)
            self.style.configure("CardTitle.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("TkDefaultFont", 9))
            self.style.configure("CardValue.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("TkDefaultFont", 16, "bold"))
            self.style.configure("TLabelframe", background=COLORS["bg"], bordercolor=COLORS["border"])
            self.style.configure("TLabelframe.Label", background=COLORS["bg"], foreground=COLORS["text"], font=("TkDefaultFont", 10, "bold"))
            self.style.configure("TNotebook", background=COLORS["bg"], borderwidth=0)
            self.style.configure("TNotebook.Tab", padding=(12, 7))
            self.style.configure("TButton", padding=(10, 6))
            self.style.configure("Primary.TButton", padding=(12, 7), foreground="white", background=COLORS["primary"])
            self.style.map("Primary.TButton", background=[("active", COLORS["primary_hover"]), ("disabled", COLORS["border"])])
            self.style.configure("Treeview", rowheight=28, bordercolor=COLORS["border"], fieldbackground=COLORS["surface"], background=COLORS["surface"], foreground=COLORS["text"])
            self.style.configure("Treeview.Heading", padding=(6, 8), font=("TkDefaultFont", 10, "bold"))
            self.style.map("Treeview", background=[("selected", COLORS["primary"])], foreground=[("selected", "white")])
        except Exception:
            # Some native themes ignore custom colors. The GUI remains functional.
            pass

    def _build_variables(self) -> None:
        first_example = self.scenario_examples[0] if self.scenario_examples else {}
        self.horizon_ticks_var = tk.IntVar(value=int(first_example.get("horizon_ticks", first_example.get("simulation_time", 1))))
        self.npu_size_var = tk.IntVar(value=int(first_example.get("npu_size", 1)))
        self.runtime_policy_var = tk.StringVar(value=str(first_example.get("runtime_policy", "edf")))
        self.mapping_policy_var = tk.StringVar(value=str(first_example.get("mapping_policy", "offline_df")))
        self.switch_cost_var = tk.IntVar(value=int(first_example.get("switch_cost", 0)))
        self.switch_cost_mode_var = tk.StringVar(value=str(first_example.get("switch_cost_mode", "per_cnn")))
        self.max_configs_var = tk.IntVar(value=int(first_example.get("max_configs_per_cnn", 512)))
        self.gantt_time_unit_var = tk.StringVar(value=str(first_example.get("gantt_time_unit", "tick")))
        self.task_filter_var = tk.StringVar(value="")
        self.task_count_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")
        self.active_example_var = tk.StringVar(value=f"Scenario: {self.scenario_examples[0]['label']}")
        self.summary_vars: Dict[str, tk.StringVar] = {
            "tasks": tk.StringVar(value="0"),
            "cnns": tk.StringVar(value="0"),
            "cpu": tk.StringVar(value="0"),
            "hybrid": tk.StringVar(value="0"),
            "npu": tk.StringVar(value="0"),
        }

        self.arch_vars: Dict[str, tk.Variable] = {}
        for key, value in self.arch.items():
            if isinstance(value, bool):
                self.arch_vars[key] = tk.BooleanVar(value=value)
            elif isinstance(value, int):
                self.arch_vars[key] = tk.IntVar(value=value)
            elif isinstance(value, list):
                continue
            else:
                self.arch_vars[key] = tk.StringVar(value=str(value))
        self.precision_vars = {p: tk.BooleanVar(value=(p == "INT8")) for p in PRECISIONS}
        if "npu_count" in self.arch_vars:
            self.arch_vars["npu_count"].trace_add("write", lambda *_args: self._update_summary() if hasattr(self, "task_tree") else None)

    def _build_layout(self) -> None:
        self._build_menu()

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        self._build_app_bar(outer)

        root = ttk.PanedWindow(outer, orient="horizontal")
        root.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(root, width=400)
        right = ttk.Frame(root)
        root.add(left, weight=0)
        root.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)
        self._build_status_bar(outer)

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Import tasks...", command=self._import_tasks_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Import architecture...", command=self._import_arch_file)
        file_menu.add_separator()
        file_menu.add_command(label="▶ RUN", command=self._run_script, accelerator="Ctrl+R")
        file_menu.add_separator()
        file_menu.add_command(label="Open workspace", command=self._open_workspace)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menubar, tearoff=False)
        edit_menu.add_command(label="Add task", command=self._add_task, accelerator="Ctrl+N")
        edit_menu.add_command(label="Edit task", command=self._edit_task, accelerator="Enter")
        edit_menu.add_command(label="Delete task", command=self._delete_task, accelerator="Del")
        menubar.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Aide", menu=help_menu)
        self.configure(menu=menubar)

    def _build_app_bar(self, parent: ttk.Frame) -> None:
        bar = ttk.Frame(parent, style="AppBar.TFrame", padding=(14, 10))
        bar.pack(fill="x", padx=10, pady=10)
        bar.grid_columnconfigure(0, weight=1)

        title_box = ttk.Frame(bar, style="AppBar.TFrame")
        title_box.grid(row=0, column=0, sticky="ew")
        ttk.Label(title_box, text=APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, textvariable=self.active_example_var, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))

        actions = ttk.Frame(bar, style="AppBar.TFrame")
        actions.grid(row=0, column=1, sticky="e")
        run_btn = self._make_run_button(actions, compact=True)
        run_btn.pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Workspace", command=self._open_workspace).pack(side="left")
        self.run_controls.append(run_btn)

    def _build_status_bar(self, parent: ttk.Frame) -> None:
        status = ttk.Frame(parent, padding=(12, 6))
        status.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(status, textvariable=self.status_var, style="Muted.TLabel").pack(side="left")
        ttk.Label(status, text="Shortcuts: Ctrl+R RUN · Enter to edit", style="Muted.TLabel").pack(side="right")

    def _bind_shortcuts(self) -> None:
        self.bind_all("<Control-o>", lambda _e: self._import_tasks_file())
        self.bind_all("<Control-r>", lambda _e: self._run_script())
        self.bind_all("<Control-n>", lambda _e: self._add_task())
        self.bind_all("<Delete>", lambda _e: self._delete_task())
        self.bind_all("<Return>", lambda _e: self._edit_task() if getattr(self, "focus_get", lambda: None)() == getattr(self, "task_tree", None) else None)

    def _show_about(self) -> None:
        messagebox.showinfo(
            "About NPsim",
            f"{APP_TITLE}\n{APP_SUBTITLE}\n\nTkinter interface to generate, run, and analyze CPU/NPU scenarios.",
            parent=self,
        )

    def _make_run_button(self, parent: tk.Widget, compact: bool = False) -> tk.Button:
        font_size = 11 if compact else 13
        padx = 16 if compact else 24
        pady = 7 if compact else 12
        return tk.Button(
            parent,
            text="▶ RUN",
            command=self._run_script,
            bg=COLORS["success"],
            fg="white",
            activebackground=COLORS["success_hover"],
            activeforeground="white",
            disabledforeground="#d1d5db",
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            font=("TkDefaultFont", font_size, "bold"),
            padx=padx,
            pady=pady,
        )

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        general = ScrollableFrame(nb)
        arch = ScrollableFrame(nb)
        memory = ScrollableFrame(nb)
        nb.add(general, text="General")
        nb.add(arch, text="Architecture")
        nb.add(memory, text="Memory")

        self._build_general_tab(general.inner)
        self._build_arch_tab(arch.inner)
        self._build_memory_tab(memory.inner)

    def _labeled_spin(self, parent: ttk.Frame, label: str, var: tk.Variable, row: int, minimum: int = 0, increment: int = 1) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        ttk.Spinbox(parent, textvariable=var, from_=minimum, to=10_000_000, increment=increment, width=14).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        parent.grid_columnconfigure(1, weight=1)

    def _labeled_combo(self, parent: ttk.Frame, label: str, var: tk.Variable, values: List[str], row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=5)
        ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=18).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        parent.grid_columnconfigure(1, weight=1)

    def _section(self, parent: ttk.Frame, title: str, row: int) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=8)
        frame.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(0, weight=1)
        return frame

    def _build_general_tab(self, parent: ttk.Frame) -> None:
        sc = self._section(parent, "Built-in examples", 0)
        ttk.Label(
            sc,
            text="Choose a scenario from the list, then click Load.",
            wraplength=320,
            foreground="#444",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 6))

        list_frame = ttk.Frame(sc)
        list_frame.grid(row=1, column=0, sticky="nsew")
        self.example_listbox = tk.Listbox(list_frame, height=8, exportselection=False)
        example_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.example_listbox.yview)
        self.example_listbox.configure(yscrollcommand=example_scroll.set)
        self.example_listbox.grid(row=0, column=0, sticky="nsew")
        example_scroll.grid(row=0, column=1, sticky="ns")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(0, weight=1)
        sc.grid_columnconfigure(0, weight=1)

        for ex in self.scenario_examples:
            self.example_listbox.insert("end", ex["label"])
        self.example_listbox.selection_set(0)
        self.example_listbox.bind("<<ListboxSelect>>", lambda _e: self._update_example_description())

        self.example_desc_var = tk.StringVar(value=self.scenario_examples[0]["description"])
        ttk.Label(sc, textvariable=self.example_desc_var, wraplength=320, foreground="#555").grid(
            row=2, column=0, sticky="ew", pady=(6, 4)
        )
        ttk.Button(sc, text="Load selected example", command=self._load_selected_example).grid(
            row=3, column=0, sticky="ew", pady=(2, 0)
        )

        params = self._section(parent, "Simulation settings", 1)
        self._labeled_spin(params, "Simulation horizon (tick)", self.horizon_ticks_var, 0, minimum=1, increment=10)
        self._labeled_combo(params, "Runtime policy", self.runtime_policy_var, ["np_fp", "edf", "rm"], 1)
        self._labeled_combo(params, "Mapping policy", self.mapping_policy_var, ["online_df", "online_qf", "offline_df", "offline_qf", "file"], 2)
        self._labeled_spin(params, "Switch cost (tick)", self.switch_cost_var, 3, minimum=0, increment=1)
        self._labeled_combo(params, "Switch cost mode", self.switch_cost_mode_var, ["per_cnn", "per_layer"], 4)
        self._labeled_spin(params, "Max testable configurations per CNN", self.max_configs_var, 5, minimum=8, increment=16)
        self._labeled_combo(params, "Gantt X-axis unit", self.gantt_time_unit_var, ["tick", "cycles", "ns", "us", "ms", "s"], 6)

        files = self._section(parent, "Generated files", 2)
        self.files_var = tk.StringVar(value="No file generated yet.")
        ttk.Label(files, textvariable=self.files_var, wraplength=330, foreground="#555").grid(row=0, column=0, sticky="ew")

    def _build_arch_tab(self, parent: ttk.Frame) -> None:
        mapping = self._section(parent, "Resources", 0)
        self._labeled_spin(mapping, "NPU count", self.arch_vars["npu_count"], 0, minimum=1, increment=1)

        clocks = self._section(parent, "Clocks / frequencies", 1)
        self._labeled_spin(clocks, "Reference frequency (MHz)", self.arch_vars["reference_frequency_mhz"], 0, minimum=1, increment=50)
        self._labeled_spin(clocks, "CPU frequency (MHz)", self.arch_vars["cpu_frequency_mhz"], 1, minimum=1, increment=50)
        self._labeled_spin(clocks, "NPU frequency (MHz)", self.arch_vars["npu_frequency_mhz"], 2, minimum=1, increment=50)
        self._labeled_spin(clocks, "Interconnect frequency (MHz)", self.arch_vars["interconnect_frequency_mhz"], 3, minimum=1, increment=50)
        self._labeled_spin(clocks, "Cycles per tick", self.arch_vars["tick_cycles"], 4, minimum=1, increment=100)

        comm = self._section(parent, "Interconnect", 2)
        self._labeled_combo(comm, "Bus CPU/NPU", self.arch_vars["cpu_npu_mode"], ["separate_full_duplex", "shared_half_duplex"], 0)
        self._labeled_spin(comm, "CPU→NPU setup", self.arch_vars["cpu_to_npu_setup"], 1, minimum=0)
        self._labeled_spin(comm, "CPU→NPU / flit 64B", self.arch_vars["cpu_to_npu_per_unit"], 2, minimum=0)
        self._labeled_spin(comm, "NPU→CPU setup", self.arch_vars["npu_to_cpu_setup"], 3, minimum=0)
        self._labeled_spin(comm, "NPU→CPU / flit 64B", self.arch_vars["npu_to_cpu_per_unit"], 4, minimum=0)
        self._labeled_spin(comm, "Shared setup", self.arch_vars["shared_setup"], 5, minimum=0)
        self._labeled_spin(comm, "Shared / flit 64B", self.arch_vars["shared_per_unit"], 6, minimum=0)

        noc = self._section(parent, "NoC NPU↔NPU", 3)
        self._labeled_combo(noc, "Topologie", self.arch_vars["noc_topology"], ["mesh", "ring", "full"], 0)
        self._labeled_combo(noc, "Arbitrage", self.arch_vars["arb_policy"], ["fixed_priority", "round_robin"], 1)
        self._labeled_spin(noc, "NoC setup", self.arch_vars["noc_setup"], 2, minimum=0)
        self._labeled_spin(noc, "NoC / flit 64B", self.arch_vars["noc_per_unit"], 3, minimum=0)
        self._labeled_spin(noc, "Router latency", self.arch_vars["router_latency"], 4, minimum=0)
        self._labeled_spin(noc, "Quantum RR", self.arch_vars["noc_rr_quantum_units"], 5, minimum=1)

        npu = self._section(parent, "NPU / Systolic Array", 4)
        self._labeled_spin(npu, "SA rows", self.arch_vars["sa_rows"], 0, minimum=1)
        self._labeled_spin(npu, "SA cols", self.arch_vars["sa_cols"], 1, minimum=1)
        self._labeled_spin(npu, "Systolic arrays", self.arch_vars["systolic_arrays"], 2, minimum=1)
        self._labeled_spin(npu, "Setup NPU", self.arch_vars["npu_setup_cycles"], 3, minimum=0)
        ttk.Label(npu, text="NPU precision").grid(row=4, column=0, sticky="w", padx=8, pady=5)
        ttk.Label(npu, text="INT8 only", foreground="#555").grid(row=4, column=1, sticky="w", padx=8, pady=5)

        cpu = self._section(parent, "CPU", 5)
        self._labeled_spin(cpu, "Setup CPU", self.arch_vars["cpu_setup_cycles"], 0, minimum=0)
        self._labeled_spin(cpu, "SIMD MAC / cycle", self.arch_vars["simd_mac_per_tick"], 1, minimum=1)
        self._labeled_spin(cpu, "Pack elems / cycle", self.arch_vars["pack_elements_per_tick"], 2, minimum=1)
        self._labeled_spin(cpu, "Epilogue elems / cycle", self.arch_vars["epilogue_elements_per_tick"], 3, minimum=1)
        self._labeled_spin(cpu, "Vector elems / cycle", self.arch_vars["vector_elements_per_tick"], 4, minimum=1)

    def _build_memory_tab(self, parent: ttk.Frame) -> None:
        npu_mem = self._section(parent, "NPU memory / local DRAM", 0)
        self._labeled_spin(npu_mem, "NPU local DRAM capacity (KiB)", self.arch_vars["local_dram_capacity_kb"], 0, minimum=1, increment=256)
        self._labeled_spin(npu_mem, "NPU local DRAM latency", self.arch_vars["local_dram_latency"], 1, minimum=0)
        self._labeled_spin(
            npu_mem,
            "Local DRAM bandwidth (bytes/cycle)",
            self.arch_vars["local_dram_bandwidth_bytes_per_tick"],
            2,
            minimum=1,
            increment=64,
        )
        self._labeled_spin(npu_mem, "Setup DMA", self.arch_vars["dma_setup"], 3, minimum=0)
        self._labeled_spin(
            npu_mem,
            "DMA bandwidth (bytes/cycle)",
            self.arch_vars["dma_bandwidth_bytes_per_tick"],
            4,
            minimum=1,
            increment=64,
        )
        self._labeled_spin(npu_mem, "NPU vector lanes", self.arch_vars["vector_lanes"], 5, minimum=1)
        self._labeled_spin(npu_mem, "NPU vector setup", self.arch_vars["vector_setup"], 6, minimum=0)

        cpu_mem = self._section(parent, "CPU memory / cache", 1)
        self._labeled_spin(cpu_mem, "CPU cache capacity (KiB)", self.arch_vars["cache_capacity_kb"], 0, minimum=1, increment=64)
        self._labeled_spin(cpu_mem, "CPU cache latency", self.arch_vars["cache_latency"], 1, minimum=0)
        self._labeled_spin(
            cpu_mem,
            "CPU cache bandwidth (bytes/cycle)",
            self.arch_vars["cache_bandwidth_bytes_per_tick"],
            2,
            minimum=1,
            increment=64,
        )
        self._labeled_spin(
            cpu_mem,
            "CPU read-modify-write bandwidth",
            self.arch_vars["rmw_bandwidth_bytes_per_tick"],
            3,
            minimum=1,
            increment=64,
        )

        note = ttk.Label(
            parent,
            text=(
                "Generic tasks are WCET-only: their duration is exactly the selected WCET. "
                "Other kinds keep the hardware model: capacity, latency, bandwidth, DMA, SIMD, SA, and NoC."
            ),
            wraplength=340,
            foreground="#555",
        )
        note.grid(row=2, column=0, sticky="ew", padx=12, pady=10)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        top = ttk.LabelFrame(parent, text="Tasks", padding=6)
        top.pack(fill="both", expand=True)

        table_frame = ttk.Frame(top)
        table_frame.pack(fill="both", expand=True)
        self.task_tree = ttk.Treeview(table_frame, columns=TASK_COLUMNS, show="headings", selectmode="browse", height=12)
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.task_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.task_tree.xview)
        self.task_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.task_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        widths = {
            "name": 110,
            "cnn_id": 80,
            "kind": 90,
            "support": 130,
            "precision": 80,
            "period": 70,
            "deadline": 80,
            "priority": 70,
            "phase": 60,
            "wcet_int8": 80,
            "wcet_fp16": 80,
            "wcet_fp32": 80,
            "preds": 120,
            "criticity": 70,
            "size": 55,
        }
        for col in TASK_COLUMNS:
            self.task_tree.heading(col, text=COLUMN_LABELS[col])
            self.task_tree.column(col, width=widths.get(col, 90), anchor="center", stretch=False)
        self.task_tree.tag_configure("stripe", background=COLORS["surface_alt"])
        self.task_tree.tag_configure("crit_HI", foreground=COLORS["danger"])
        self.task_tree.tag_configure("crit_LO", foreground=COLORS["text"])
        self.task_tree.bind("<Double-1>", lambda _e: self._edit_task())
        self.task_tree.bind("<Button-3>", self._show_task_context_menu)

        self.task_menu = tk.Menu(self, tearoff=False)
        self.task_menu.add_command(label="Edit", command=self._edit_task)
        self.task_menu.add_command(label="Delete", command=self._delete_task)
        self.task_menu.add_separator()
        self.task_menu.add_command(label="Copy row", command=self._copy_selected_task)

        buttons = ttk.Frame(top)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="Add", command=self._add_task).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Edit", command=self._edit_task).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Delete", command=self._delete_task).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Move up", command=lambda: self._move_task(-1)).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Move down", command=lambda: self._move_task(1)).pack(side="left", padx=(0, 6))

        bottom = ttk.Frame(parent)
        bottom.pack(fill="both", expand=True, pady=(6, 0))

        report_frame = ttk.LabelFrame(bottom, text="Report and execution output", padding=8)
        report_frame.pack(fill="both", expand=True)

        report_actions = ttk.Frame(report_frame)
        report_actions.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Button(report_actions, text="Show Gantt", command=self._show_gantt).pack(side="left", padx=(0, 6))
        ttk.Button(report_actions, text="Copy output", command=self._copy_output).pack(side="left", padx=(0, 6))
        ttk.Button(report_actions, text="Clear output", command=self._clear_output).pack(side="left", padx=(0, 10))
        self.gantt_status_var = tk.StringVar(value="No Gantt generated.")
        ttk.Label(report_actions, textvariable=self.gantt_status_var, foreground="#555").pack(side="left")

        self.output_text = tk.Text(report_frame, height=18, wrap="word", borderwidth=1, relief="solid", bg=COLORS["surface"], fg=COLORS["text"])
        self.output_text.configure(state="disabled")
        output_scroll = ttk.Scrollbar(report_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=output_scroll.set)
        self.output_text.grid(row=1, column=0, sticky="nsew")
        output_scroll.grid(row=1, column=1, sticky="ns")
        report_frame.grid_rowconfigure(1, weight=1)
        report_frame.grid_columnconfigure(0, weight=1)

    def _summary_card(self, parent: ttk.Frame, title: str, value_var: tk.StringVar) -> ttk.Frame:
        card = ttk.Frame(parent, style="Card.TFrame", padding=(12, 10))
        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, textvariable=value_var, style="CardValue.TLabel").pack(anchor="w", pady=(3, 0))
        return card

    def _update_summary(self, visible_count: Optional[int] = None) -> None:
        normalized = normalize_task_table(self.tasks)
        cnn_count = len({str(row.get("cnn_id", "")) for row in normalized if row.get("cnn_id")})
        cpu_only = sum(1 for row in normalized if str(row.get("support", "")).lower().startswith("cpu seulement"))
        hybrid = sum(1 for row in normalized if "npu" in str(row.get("support", "")).lower())
        self.summary_vars["tasks"].set(str(len(normalized)))
        self.summary_vars["cnns"].set(str(cnn_count))
        self.summary_vars["cpu"].set(str(cpu_only))
        self.summary_vars["hybrid"].set(str(hybrid))
        self.summary_vars["npu"].set(str(self.arch_vars.get("npu_count", self.npu_size_var).get()))
        if visible_count is None or visible_count == len(normalized):
            self.task_count_var.set(f"{len(normalized)} task(s)")
        else:
            self.task_count_var.set(f"{visible_count} / {len(normalized)} task(s)")

    def _show_task_context_menu(self, event: tk.Event) -> None:
        row_id = self.task_tree.identify_row(event.y)
        if row_id:
            self.task_tree.selection_set(row_id)
            self.task_menu.tk_popup(event.x_root, event.y_root)

    def _copy_selected_task(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        row = normalize_task_table(self.tasks)[idx]
        text = "	".join(str(row.get(col, "")) for col in TASK_COLUMNS)
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set(f"Row copied: {row.get('name', '')}")

    def _duplicate_task(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Selection", "Select a task to duplicate.", parent=self)
            return
        row = dict(self.tasks[idx])
        base_name = str(row.get("name", "LAYER"))
        existing = {str(task.get("name", "")) for task in self.tasks}
        candidate = f"{base_name}_copy"
        suffix = 2
        while candidate in existing:
            candidate = f"{base_name}_copy{suffix}"
            suffix += 1
        row["name"] = candidate
        row["priority"] = int(row.get("priority", idx)) + 1
        self.tasks.insert(idx + 1, row)
        self.tasks = normalize_task_table(self.tasks)
        self.generated_paths = None
        self.run_result = None
        self.gantt_path = None
        self._refresh_task_table()
        self.task_tree.selection_set(str(idx + 1))
        self.status_var.set(f"Task duplicated: {candidate}")

    def _set_running_state(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for button in self.run_controls:
            try:
                button.configure(state=state)
            except Exception:
                pass

    def _clear_output(self) -> None:
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")
        self.status_var.set("Output cleared.")

    def _copy_output(self) -> None:
        text = self.output_text.get("1.0", "end-1c")
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Output copied to the clipboard.")

    def _copy_text_to_clipboard(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.status_var.set("Path copied to the clipboard.")

    def _infer_kind_from_name(self, name: str) -> str:
        # Do not infer hardware layer kind from the task name. A task named
        # "conv", "pool", "nms", etc. must remain generic unless the user
        # explicitly chooses a non-generic kind in the Type field.
        return "generic"

    def _import_tasks_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Import a task list",
            filetypes=[("Task list", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            imported_tasks: List[Dict[str, Any]] = []
            period_count: Optional[int] = None
            horizon_ticks: Optional[int] = None
            sim_time_cycles: Optional[int] = None
            sim_time_value: Optional[int] = None
            npu_size: Optional[int] = None
            time_unit = "cycles"
            tick_cycles = tick_cycles_from_arch(self._collect_arch())

            def to_gui_ticks(value: int, *, minimum: int = 0) -> int:
                if time_unit == "tick":
                    return max(minimum, int(value))
                return cycles_to_ticks(int(value), tick_cycles, minimum=minimum)

            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith("#"):
                        parts = line[1:].strip().split()
                        if not parts:
                            continue
                        key = parts[0].upper()
                        if len(parts) >= 2 and key == "TIME_UNIT":
                            time_unit = "tick" if parts[1].lower() in {"tick", "ticks"} else "cycles"
                        elif len(parts) >= 2 and key in {"TICK_CYCLES", "CYCLES_PER_TICK"}:
                            tick_cycles = max(1, int(float(parts[1])))
                            if "tick_cycles" in self.arch_vars:
                                self.arch_vars["tick_cycles"].set(tick_cycles)
                        elif len(parts) >= 2 and key == "PERIODS":
                            period_count = int(parts[1])
                        elif len(parts) >= 2 and key in {"HORIZON_TICKS", "SIMULATION_TICKS"}:
                            horizon_ticks = int(parts[1])
                        elif len(parts) >= 2 and key in {"SIMULATION_TIME_CYCLES", "HORIZON_CYCLES"}:
                            sim_time_cycles = int(parts[1])
                        elif len(parts) >= 2 and key == "SIMULATION_TIME":
                            sim_time_value = int(parts[1])
                        elif len(parts) >= 2 and key == "NPU_SIZE":
                            npu_size = int(parts[1])
                        continue
                    parts = line.split()
                    if len(parts) < 14:
                        continue
                    name = parts[0]
                    if len(parts) >= 15 and parts[1].upper() not in {"CPU", "NPU"}:
                        kind = parts[1].lower()
                        resource = parts[2].upper()
                        offset = 1
                    else:
                        kind = self._infer_kind_from_name(name)
                        resource = parts[1].upper()
                        offset = 0
                    imported_tasks.append({
                        "name": name,
                        "cnn_id": parts[11 + offset],
                        "kind": kind,
                        "support": "CPU only" if resource == "CPU" else "CPU/NPU",
                        "period": to_gui_ticks(int(parts[2 + offset]), minimum=1),
                        "wcet_int8": to_gui_ticks(int(parts[3 + offset]), minimum=0),
                        "wcet_fp16": to_gui_ticks(int(parts[4 + offset]), minimum=0),
                        "wcet_fp32": to_gui_ticks(int(parts[5 + offset]), minimum=0),
                        "deadline": to_gui_ticks(int(parts[6 + offset]), minimum=1),
                        "priority": int(parts[7 + offset]),
                        "phase": to_gui_ticks(int(parts[8 + offset]), minimum=0),
                        "precision": parts[9 + offset].upper(),
                        "preds": parts[10 + offset],
                        "criticity": parts[12 + offset].upper(),
                        "size": int(parts[13 + offset]),
                    })
            if not imported_tasks:
                messagebox.showwarning("Import", "No valid task found in this file.", parent=self)
                return
            self.tasks = normalize_task_table(imported_tasks)
            if horizon_ticks is not None:
                self.horizon_ticks_var.set(max(1, int(horizon_ticks)))
            elif sim_time_cycles is not None:
                self.horizon_ticks_var.set(cycles_to_ticks(sim_time_cycles, tick_cycles, minimum=1))
            elif sim_time_value is not None:
                sim_ticks = sim_time_value if time_unit == "tick" else cycles_to_ticks(sim_time_value, tick_cycles, minimum=1)
                self.horizon_ticks_var.set(max(1, int(sim_ticks)))
            elif period_count is not None:
                # Compatibility with older files that expressed the horizon in periods.
                self.horizon_ticks_var.set(periods_to_simulation_time(self.tasks, max(1, int(period_count))))
            if npu_size is not None and "npu_count" in self.arch_vars:
                self.arch_vars["npu_count"].set(max(1, npu_size))
            self.generated_paths = None
            self.run_result = None
            self.gantt_path = None
            self._refresh_task_table()
            self.status_var.set(f"Task list imported: {Path(path).name}")
            self._append_output(f"[INFO] Task list imported: {path} — GUI unit=tick, 1 tick={tick_cycles} internal cycles\n")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Import error", f"Could not import the task list.\n\n{exc}", parent=self)

    def _import_arch_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Import an architecture",
            filetypes=[("Architecture", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            updates = default_arch_settings()

            resources = data.get("resources", data.get("mapping", {})) or {}
            updates["cpu_name"] = str(resources.get("cpu_name", updates["cpu_name"]))
            npu_nodes = resources.get("npu_nodes", []) or []
            if isinstance(npu_nodes, list) and npu_nodes:
                updates["npu_count"] = len(npu_nodes)

            clock = data.get("clock", data.get("clocks", data.get("timing", {}))) or {}
            if "reference_frequency_mhz" in clock or "frequency_mhz" in clock:
                updates["reference_frequency_mhz"] = clock.get("reference_frequency_mhz", clock.get("frequency_mhz"))
            for src_key in ("cpu_frequency_mhz", "npu_frequency_mhz", "interconnect_frequency_mhz", "tick_cycles", "cycles_per_tick"):
                if src_key in clock:
                    updates["tick_cycles" if src_key == "cycles_per_tick" else src_key] = clock[src_key]

            communications = data.get("communications", {}) or {}
            cpu_npu = communications.get("cpu_npu", {}) or {}
            noc = communications.get("noc", {}) or {}
            key_map = {
                "mode": "cpu_npu_mode",
                "cpu_to_npu_setup": "cpu_to_npu_setup",
                "cpu_to_npu_per_unit": "cpu_to_npu_per_unit",
                "npu_to_cpu_setup": "npu_to_cpu_setup",
                "npu_to_cpu_per_unit": "npu_to_cpu_per_unit",
                "shared_setup": "shared_setup",
                "shared_per_unit": "shared_per_unit",
            }
            for src_key, dst_key in key_map.items():
                if src_key in cpu_npu:
                    updates[dst_key] = cpu_npu[src_key]
            noc_map = {
                "topology": "noc_topology",
                "setup": "noc_setup",
                "per_unit": "noc_per_unit",
                "router_latency": "router_latency",
                "arb_policy": "arb_policy",
                "rr_quantum_units": "noc_rr_quantum_units",
            }
            for src_key, dst_key in noc_map.items():
                if src_key in noc:
                    updates[dst_key] = noc[src_key]

            npu = data.get("npu", {}) or {}
            npu_map = {
                "sa_rows": "sa_rows",
                "sa_cols": "sa_cols",
                "systolic_arrays": "systolic_arrays",
                "setup_cycles": "npu_setup_cycles",
                "precision_modes": "precision_modes",
                "local_dram_capacity_kb": "local_dram_capacity_kb",
                "local_dram_latency": "local_dram_latency",
                "local_dram_bandwidth_bytes_per_cycle": "local_dram_bandwidth_bytes_per_tick",
                "local_dram_bandwidth_bytes_per_tick": "local_dram_bandwidth_bytes_per_tick",
                "dma_setup": "dma_setup",
                "dma_bandwidth_bytes_per_cycle": "dma_bandwidth_bytes_per_tick",
                "dma_bandwidth_bytes_per_tick": "dma_bandwidth_bytes_per_tick",
                "vector_lanes": "vector_lanes",
                "vector_setup": "vector_setup",
            }
            for src_key, dst_key in npu_map.items():
                if src_key in npu:
                    updates[dst_key] = npu[src_key]
            if "local_dram_capacity_bytes" in npu and "local_dram_capacity_kb" not in npu:
                updates["local_dram_capacity_kb"] = max(1, int(npu["local_dram_capacity_bytes"]) // 1024)

            cpu = data.get("cpu", {}) or {}
            cpu_map = {
                "setup_cycles": "cpu_setup_cycles",
                "cache_capacity_kb": "cache_capacity_kb",
                "cache_latency": "cache_latency",
                "cache_bandwidth_bytes_per_cycle": "cache_bandwidth_bytes_per_tick",
                "cache_bandwidth_bytes_per_tick": "cache_bandwidth_bytes_per_tick",
                "simd_mac_per_cycle": "simd_mac_per_tick",
                "simd_mac_per_tick": "simd_mac_per_tick",
                "pack_elements_per_cycle": "pack_elements_per_tick",
                "pack_elements_per_tick": "pack_elements_per_tick",
                "epilogue_elements_per_cycle": "epilogue_elements_per_tick",
                "epilogue_elements_per_tick": "epilogue_elements_per_tick",
                "vector_elements_per_cycle": "vector_elements_per_tick",
                "vector_elements_per_tick": "vector_elements_per_tick",
                "rmw_bandwidth_bytes_per_cycle": "rmw_bandwidth_bytes_per_tick",
                "rmw_bandwidth_bytes_per_tick": "rmw_bandwidth_bytes_per_tick",
            }
            for src_key, dst_key in cpu_map.items():
                if src_key in cpu:
                    updates[dst_key] = cpu[src_key]
            if "cache_capacity_bytes" in cpu and "cache_capacity_kb" not in cpu:
                updates["cache_capacity_kb"] = max(1, int(cpu["cache_capacity_bytes"]) // 1024)

            self._apply_arch_updates(updates)
            self.npu_size_var.set(int(updates.get("npu_count", 1)))
            self.generated_paths = None
            self.run_result = None
            self.gantt_path = None
            self.status_var.set(f"Architecture imported: {Path(path).name}")
            self._append_output(f"[INFO] Architecture imported: {path}\n")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Import error", f"Could not import the architecture.\n\n{exc}", parent=self)

    def _selected_example_index(self) -> int:
        selected = self.example_listbox.curselection() if hasattr(self, "example_listbox") else ()
        if not selected:
            return 0
        return int(selected[0])

    def _update_example_description(self) -> None:
        idx = self._selected_example_index()
        self.example_desc_var.set(self.scenario_examples[idx]["description"])

    def _apply_arch_updates(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key == "precision_modes":
                for p, var in self.precision_vars.items():
                    var.set(p == "INT8")
            elif key in self.arch_vars:
                self.arch_vars[key].set(value)

    def _load_selected_example(self) -> None:
        idx = self._selected_example_index()
        example = self.scenario_examples[idx]
        self.tasks = normalize_task_table(example["tasks"])
        self.horizon_ticks_var.set(int(example.get("horizon_ticks", example.get("simulation_time", 1000))))
        # Reset architecture before applying the selected example.
        defaults = default_arch_settings()
        defaults["npu_count"] = int(example.get("npu_size", example.get("arch_updates", {}).get("npu_count", defaults.get("npu_count", 1))))
        for key, value in defaults.items():
            if key == "precision_modes":
                for p, var in self.precision_vars.items():
                    var.set(p == "INT8")
            elif key in self.arch_vars:
                self.arch_vars[key].set(value)
        self.npu_size_var.set(int(defaults.get("npu_count", 1)))
        self._apply_arch_updates(example.get("arch_updates", {}))
        if "runtime_policy" in example:
            self.runtime_policy_var.set(example["runtime_policy"])
        if "mapping_policy" in example:
            self.mapping_policy_var.set(example["mapping_policy"])
        if "switch_cost" in example:
            self.switch_cost_var.set(int(example["switch_cost"]))
        if "switch_cost_mode" in example:
            self.switch_cost_mode_var.set(example["switch_cost_mode"])
        if "max_configs_per_cnn" in example:
            self.max_configs_var.set(int(example["max_configs_per_cnn"]))
        if "gantt_time_unit" in example:
            self.gantt_time_unit_var.set(str(example["gantt_time_unit"]))
        self.active_example_var.set(f"Scenario: {example['label']}")
        self._invalidate_generated_state()
        self._refresh_task_table()
        self.status_var.set(f"Example loaded: {example['label']}")

    def _refresh_task_table(self) -> None:
        if not hasattr(self, "task_tree"):
            return
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)

        normalized = normalize_task_table(self.tasks)
        # The task filter UI was removed to maximize useful vertical space.
        # Keep the variable for compatibility, but always show the complete table.
        if hasattr(self, "task_filter_var"):
            self.task_filter_var.set("")
        visible_count = 0
        for idx, row in enumerate(normalized):
            tags = []
            if visible_count % 2 == 1:
                tags.append("stripe")
            criticity = str(row.get("criticity", "")).upper()
            if criticity:
                tags.append(f"crit_{criticity}")
            self.task_tree.insert("", "end", iid=str(idx), values=[row.get(c, "") for c in TASK_COLUMNS], tags=tuple(tags))
            visible_count += 1
        self._update_summary(visible_count)

    def _invalidate_generated_state(self) -> None:
        self.generated_paths = None
        self.run_result = None
        self.gantt_path = None
        if hasattr(self, "gantt_status_var"):
            self.gantt_status_var.set("No Gantt generated.")

    def _selected_index(self) -> Optional[int]:
        selected = self.task_tree.selection()
        if not selected:
            return None
        try:
            return int(selected[0])
        except ValueError:
            return None

    def _add_task(self) -> None:
        default_priority = len(self.tasks)
        default_pred = self.tasks[-1]["name"] if self.tasks else "-"
        initial = {
            "name": f"LAYER_{len(self.tasks) + 1}",
            "cnn_id": self.tasks[-1].get("cnn_id", "CNN_A") if self.tasks else "CNN_A",
            "kind": "generic",
            "support": "CPU/NPU",
            "period": 100,
            "deadline": 85,
            "priority": default_priority,
            "phase": 0,
            "precision": "INT8",
            "wcet_int8": 20,
            "wcet_fp16": 30,
            "wcet_fp32": 45,
            "preds": default_pred,
            "criticity": "LO",
            "size": 1,
        }
        dlg = TaskDialog(self, "Add task", initial)
        if dlg.result:
            self.tasks.append(dlg.result)
            self.tasks = normalize_task_table(self.tasks)
            self._invalidate_generated_state()
            self._refresh_task_table()

    def _edit_task(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Selection", "Select a task to edit.", parent=self)
            return
        dlg = TaskDialog(self, "Edit task", self.tasks[idx])
        if dlg.result:
            self.tasks[idx] = dlg.result
            self.tasks = normalize_task_table(self.tasks)
            self._invalidate_generated_state()
            self._refresh_task_table()
            self.task_tree.selection_set(str(idx))

    def _delete_task(self) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Selection", "Select a task to delete.", parent=self)
            return
        row = self.tasks[idx]
        if messagebox.askyesno("Delete", f"Delete {row['name']}?", parent=self):
            del self.tasks[idx]
            self.tasks = normalize_task_table(self.tasks)
            self._invalidate_generated_state()
            self._refresh_task_table()

    def _move_task(self, delta: int) -> None:
        idx = self._selected_index()
        if idx is None:
            messagebox.showinfo("Selection", "Select a task to move.", parent=self)
            return
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= len(self.tasks):
            return
        self.tasks[idx], self.tasks[new_idx] = self.tasks[new_idx], self.tasks[idx]
        self._invalidate_generated_state()
        self._refresh_task_table()
        self.task_tree.selection_set(str(new_idx))

    def _collect_arch(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        for key, var in self.arch_vars.items():
            data[key] = var.get()
        data["precision_modes"] = ["INT8"]
        return data

    def _generate_files(self) -> None:
        try:
            self.tasks = normalize_task_table(self.tasks)
            self.arch = self._collect_arch()
            self.generated_paths = export_project_files(
                WORKSPACE,
                tasks=self.tasks,
                simulation_time=int(self.horizon_ticks_var.get()),
                npu_size=int(self.arch_vars["npu_count"].get()),
                arch=self.arch,
                runtime_policy=self.runtime_policy_var.get(),
                mapping_policy=self.mapping_policy_var.get(),
                switch_cost=int(self.switch_cost_var.get()),
                switch_cost_mode=self.switch_cost_mode_var.get(),
                max_configs_per_cnn=int(self.max_configs_var.get()),
                gantt_time_unit=self.gantt_time_unit_var.get(),
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Generation error", str(exc), parent=self)
            return
        self._refresh_task_table()
        horizon_ticks = max(1, int(self.horizon_ticks_var.get()))
        tick_cycles = tick_cycles_from_arch(self.arch)
        sim_cycles = horizon_ticks * tick_cycles
        self.status_var.set(f"Files generated for a {horizon_ticks}-tick horizon in {WORKSPACE}")
        self.files_var.set(
            f"Generated: tasks.txt, arch.yaml, scenario.yaml, run.sh\n"
            f"Horizon GUI : {horizon_ticks} ticks ({sim_cycles} internal cycles)\n"
            f"Tick : 1 tick = {tick_cycles} internal cycles\n"
            f"Dossier : {WORKSPACE}\n"
            "Raw file contents are not displayed in the GUI."
        )
        self._append_output(f"[INFO] Files generated in {WORKSPACE} — horizon={horizon_ticks} ticks, cycles={sim_cycles}\n")

    def _run_script(self) -> None:
        # Always regenerate before running so the simulation uses the latest GUI state.
        self._generate_files()
        if self.generated_paths is None:
            return
        self._set_running_state(True)
        self.status_var.set("run.sh is executing...")
        self._append_output("\n[INFO] Executing run.sh...\n")
        thread = threading.Thread(target=self._run_worker, daemon=True)
        thread.start()
        self.after(150, self._poll_run_result)

    def _run_worker(self) -> None:
        try:
            result = run_generated_script(WORKSPACE, timeout=240)
        except subprocess.TimeoutExpired as exc:
            result = {"returncode": -1, "stdout": exc.stdout or "", "stderr": f"Timeout: {exc}", "log": None, "gantt": None}
        except Exception as exc:  # noqa: BLE001
            result = {"returncode": -1, "stdout": "", "stderr": str(exc), "log": None, "gantt": None}
        self.result_queue.put(result)

    def _poll_run_result(self) -> None:
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(150, self._poll_run_result)
            return
        self.run_result = result
        self._set_running_state(False)
        self._display_result(result)

    def _display_result(self, result: Dict[str, Any]) -> None:
        rc = result.get("returncode")
        self.status_var.set("Simulation finished." if rc == 0 else "Simulation finished with errors.")
        self._append_output(f"[INFO] Code retour : {rc}\n")
        if result.get("stdout"):
            self._append_output("\n=== STDOUT ===\n" + str(result["stdout"]) + "\n")
        if result.get("stderr"):
            self._append_output("\n=== STDERR ===\n" + str(result["stderr"]) + "\n")
        log_path = result.get("log")
        if log_path and Path(log_path).exists():
            txt = Path(log_path).read_text(encoding="utf-8")
            self._append_output("\n=== RAPPORT NPsim ===\n" + txt + "\n")
            self.files_var.set((self.files_var.get() + f"\nReport: {log_path}").strip())
        gantt_path = result.get("gantt")
        if gantt_path and Path(gantt_path).exists():
            self.gantt_path = Path(gantt_path)
            self.gantt_status_var.set(f"Gantt ready: {self.gantt_path.name}")
            self.files_var.set((self.files_var.get() + f"\nGantt : {gantt_path}").strip())
            self._show_gantt(show_info_if_missing=False)
        else:
            self.gantt_path = None
            self.gantt_status_var.set("No Gantt generated.")
        if rc == 0:
            messagebox.showinfo("Simulation finished", "The report and Gantt were generated.", parent=self)
        else:
            messagebox.showwarning("Simulation finished with errors", "Check the execution output.", parent=self)

    def _show_gantt(self, show_info_if_missing: bool = True) -> None:
        if self.gantt_path is None or not self.gantt_path.exists():
            if show_info_if_missing:
                messagebox.showinfo("Gantt", "Aucun Gantt disponible.", parent=self)
            return

        metadata_path = self.gantt_path.with_suffix(".json")
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self._show_interactive_gantt(metadata, metadata_path)
                return
            except Exception as exc:  # noqa: BLE001
                messagebox.showwarning(
                    "Gantt interactif",
                    f"The interactive Gantt data is unreadable. Showing the static PNG.\n\n{exc}",
                    parent=self,
                )

        self._show_static_gantt_image(show_info_if_missing=show_info_if_missing)

    def _show_static_gantt_image(self, show_info_if_missing: bool = True) -> None:
        if self.gantt_path is None or not self.gantt_path.exists():
            if show_info_if_missing:
                messagebox.showinfo("Gantt", "Aucun Gantt disponible.", parent=self)
            return

        if self.gantt_path.suffix.lower() not in {".png", ".gif", ".ppm", ".pgm"}:
            messagebox.showinfo(
                "Gantt",
                "The generated Gantt is not an image that NPsim can display. "
                f"File available: {self.gantt_path}",
                parent=self,
            )
            return

        try:
            image = tk.PhotoImage(file=str(self.gantt_path))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Gantt", f"Could not display the Gantt inside NPsim: {exc}", parent=self)
            return

        win = tk.Toplevel(self)
        win.title(f"NPsim — Gantt — {self.gantt_path.name}")
        win.geometry("1100x720")
        win.minsize(720, 420)
        win.configure(bg=COLORS["bg"])

        toolbar = ttk.Frame(win, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=f"Gantt : {self.gantt_path.name}").pack(side="left")
        ttk.Button(toolbar, text="Copier le chemin", command=lambda: self._copy_text_to_clipboard(str(self.gantt_path))).pack(side="right")

        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground=COLORS["border"])
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        canvas.create_image(0, 0, anchor="nw", image=image)
        canvas.configure(scrollregion=(0, 0, image.width(), image.height()))
        canvas.image = image  # type: ignore[attr-defined]

        def _on_mousewheel(event: tk.Event) -> None:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            canvas.yview_scroll(delta * 3, "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", lambda _e: canvas.yview_scroll(-3, "units"))
        canvas.bind("<Button-5>", lambda _e: canvas.yview_scroll(3, "units"))

    def _show_interactive_gantt(self, metadata: Dict[str, Any], metadata_path: Path) -> None:
        lanes = list(metadata.get("lanes") or [])
        compute_items = list(metadata.get("compute") or [])
        message_items = list(metadata.get("messages") or [])
        if not lanes or (not compute_items and not message_items):
            self._show_static_gantt_image(show_info_if_missing=False)
            return

        win = tk.Toplevel(self)
        title_name = self.gantt_path.name if self.gantt_path else metadata_path.name
        win.title(f"NPsim — Interactive Gantt — {title_name}")
        win.geometry("1180x760")
        win.minsize(780, 460)
        win.configure(bg=COLORS["bg"])

        state: Dict[str, Any] = {"zoom": 1.0, "show_dag": True}
        zoom_var = tk.StringVar(value="100%")
        dag_button_var = tk.StringVar(value="Hide DAG")
        hint_var = tk.StringVar(value="Hover over a bar or DAG arrow to read details. Ctrl + wheel = zoom, Shift + wheel = horizontal scroll.")

        toolbar = ttk.Frame(win, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text=f"Gantt interactif : {title_name}").pack(side="left")
        ttk.Button(toolbar, text="-", width=3, command=lambda: _set_zoom(state["zoom"] / 1.25)).pack(side="left", padx=(12, 2))
        ttk.Button(toolbar, text="+", width=3, command=lambda: _set_zoom(state["zoom"] * 1.25)).pack(side="left", padx=2)
        ttk.Button(toolbar, text="100%", command=lambda: _set_zoom(1.0)).pack(side="left", padx=(6, 2))
        ttk.Button(toolbar, text="Ajuster", command=lambda: _set_zoom(1.0)).pack(side="left", padx=2)
        ttk.Label(toolbar, textvariable=zoom_var).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, textvariable=dag_button_var, command=lambda: _toggle_dag()).pack(side="left", padx=(14, 2))
        ttk.Button(toolbar, text="Copier le chemin", command=lambda: self._copy_text_to_clipboard(str(metadata_path))).pack(side="right")

        ttk.Label(win, textvariable=hint_var, style="Muted.TLabel").pack(anchor="w", padx=12, pady=(0, 6))

        canvas_frame = ttk.Frame(win)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, highlightbackground=COLORS["border"])
        xscroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=canvas.xview)
        yscroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        tooltip: Dict[str, Any] = {"window": None, "label": None}

        def _show_tooltip(event: tk.Event, text: str) -> None:
            if tooltip["window"] is None:
                tw = tk.Toplevel(win)
                tw.withdraw()
                tw.overrideredirect(True)
                tw.configure(bg="#111827")
                label = tk.Label(
                    tw,
                    text=text,
                    justify="left",
                    background="#111827",
                    foreground="white",
                    relief="solid",
                    borderwidth=1,
                    padx=9,
                    pady=7,
                    font=("TkDefaultFont", 9),
                )
                label.pack()
                tooltip["window"] = tw
                tooltip["label"] = label
            tooltip["label"].configure(text=text)
            tooltip["window"].geometry(f"+{event.x_root + 14}+{event.y_root + 14}")
            tooltip["window"].deiconify()

        def _move_tooltip(event: tk.Event) -> None:
            if tooltip["window"] is not None:
                tooltip["window"].geometry(f"+{event.x_root + 14}+{event.y_root + 14}")

        def _hide_tooltip(_event: tk.Event | None = None) -> None:
            try:
                if tooltip["window"] is not None and tooltip["window"].winfo_exists():
                    tooltip["window"].withdraw()
            except tk.TclError:
                pass

        def _fmt(value: Any) -> str:
            if value is None:
                return "-"
            if isinstance(value, float):
                if abs(value - round(value)) < 1e-9:
                    return str(int(round(value)))
                return f"{value:.6g}"
            return str(value)

        x_unit = str(metadata.get("x_unit") or "cycles")

        def _compute_tooltip(item: Dict[str, Any]) -> str:
            lines = [
                f"Task: {item.get('task', item.get('id', '-'))}",
                f"CNN / instance : {item.get('cnn_id', '-')}[{item.get('instance', '-')}]",
                f"Ressource : {item.get('resource', '-')}",
                f"Support / precision: {item.get('support', '-')} / {item.get('precision', '-')}",
                f"Kind / criticality: {item.get('layer_kind', '-')} / {item.get('criticity', '-')}",
                f"Start → finish: {_fmt(item.get('start'))} → {_fmt(item.get('finish'))} {x_unit}",
                f"Duration: {_fmt(item.get('duration'))} {x_unit} ({_fmt(item.get('duration_cycles'))} internal cycles)",
                f"Release / deadline : {_fmt(item.get('release'))} / {_fmt(item.get('deadline'))} {x_unit}",
                f"Priority: {item.get('priority', '-')}",
            ]
            if item.get("miss"):
                lines.append("Status: deadline missed")
            else:
                lines.append("Status: deadline met")
            if item.get("changed"):
                action = item.get("df_action_label") or item.get("df_action") or "mapping changed"
                lines.append(f"Adaptation: P{item.get('df_period')} — {action}")
            preds = item.get("preds") or []
            if preds:
                lines.append(f"Predecessors: {', '.join(map(str, preds))}")
            stages = item.get("stages") or []
            if stages:
                stage_text = ", ".join(f"{s.get('label')}={_fmt(s.get('duration_cycles'))}" for s in stages[:5])
                if len(stages) > 5:
                    stage_text += ", ..."
                lines.append(f"Internal stages (cycles): {stage_text}")
            return "\n".join(lines)

        def _message_tooltip(item: Dict[str, Any]) -> str:
            route = item.get("route_nodes") or []
            lines = [
                f"Communication : {item.get('id', '-')}",
                f"Medium: {item.get('medium', item.get('resource', '-'))}",
                f"Source → destination : {item.get('pred_job_id', '-')} → {item.get('dst_job_id', '-')}",
                f"Resources: {item.get('src_resource', '-')} → {item.get('dst_resource', '-')}",
                f"Start → finish: {_fmt(item.get('start'))} → {_fmt(item.get('finish'))} {x_unit}",
                f"Duration: {_fmt(item.get('duration'))} {x_unit} ({_fmt(item.get('duration_cycles'))} internal cycles)",
                f"Payload: {item.get('payload_units', '-')} units",
            ]
            if route:
                lines.append(f"Route : {' → '.join(map(str, route))}")
            return "\n".join(lines)

        def _dependency_tooltip(dep: Dict[str, Any], items_by_id: Dict[str, Dict[str, Any]]) -> str:
            pred_id = str(dep.get("pred_job_id", "-"))
            dst_id = str(dep.get("dst_job_id", "-"))
            pred = items_by_id.get(pred_id, {})
            dst = items_by_id.get(dst_id, {})
            kind = "via communication" if dep.get("crossed_medium") else "locale"
            lines = [
                "DAG dependency",
                f"Type : {kind}",
                f"Predecessor: {pred.get('task', pred_id)}",
                f"Successeur : {dst.get('task', dst_id)}",
                f"Pred. finish → succ. start: {_fmt(pred.get('finish'))} → {_fmt(dst.get('start'))} {x_unit}",
            ]
            if pred.get("resource") or dst.get("resource"):
                lines.append(f"Resources: {pred.get('resource', '-')} → {dst.get('resource', '-')}")
            return "\n".join(lines)

        def _nice_step(raw: float) -> float:
            if raw <= 0:
                return 1.0
            exp = 10 ** int(math.floor(math.log10(raw)))
            frac = raw / exp
            if frac <= 1:
                nice = 1
            elif frac <= 2:
                nice = 2
            elif frac <= 5:
                nice = 5
            else:
                nice = 10
            return nice * exp

        palette = [
            "#2563eb", "#059669", "#d97706", "#7c3aed", "#dc2626",
            "#0891b2", "#65a30d", "#c026d3", "#4f46e5", "#ea580c",
        ]
        color_by_task: Dict[str, str] = {}

        def _color_for(task: str) -> str:
            if task not in color_by_task:
                color_by_task[task] = palette[len(color_by_task) % len(palette)]
            return color_by_task[task]

        def _bind_tooltip(item_id: int, text: str) -> None:
            canvas.tag_bind(item_id, "<Enter>", lambda event, t=text: _show_tooltip(event, t))
            canvas.tag_bind(item_id, "<Motion>", _move_tooltip)
            canvas.tag_bind(item_id, "<Leave>", _hide_tooltip)

        def _draw() -> None:
            x0, x1 = canvas.xview()
            y0, y1 = canvas.yview()
            canvas.delete("all")
            color_by_task.clear()
            zoom = float(state["zoom"])
            zoom_var.set(f"{int(round(zoom * 100))}%")
            canvas_width = max(canvas.winfo_width(), 900)
            left = 155
            right = 36
            # Dedicated header band for period/adaptation labels.
            # Labels such as "P0 / NPU first" must not be drawn inside the
            # first CPU/NPU resource lane, where task bars can hide them.
            top = 122
            header_top = 58
            axis_y = top - 12
            period_label_y = header_top + 6
            row_h = max(34, int(46 * min(zoom, 1.65)))
            lane_h = row_h * len(lanes)
            bottom = 64
            max_finish = float(metadata.get("max_finish") or 0)
            span = max(max_finish, 1.0)
            pad = max(span * 0.04, 1.0)
            total_span = span + pad
            base_timeline_w = max(canvas_width - left - right, 820)
            timeline_w = int(base_timeline_w * zoom)
            px_per_unit = timeline_w / total_span if total_span else 1.0
            total_w = left + timeline_w + right
            total_h = top + lane_h + bottom

            def x(value: Any) -> float:
                try:
                    return left + float(value) * px_per_unit
                except Exception:
                    return left

            canvas.create_text(16, 18, text="NPsim Gantt", anchor="nw", fill=COLORS["text"], font=("TkDefaultFont", 14, "bold"))
            canvas.create_text(16, 42, text=str(metadata.get("x_label") or "Time"), anchor="nw", fill=COLORS["muted"], font=("TkDefaultFont", 9))

            # Header band reserved for period/adaptation labels. It prevents
            # period names from being covered by CPU/NPU task bars.
            canvas.create_rectangle(left, header_top, total_w - right, top, fill="#f8fafc", outline="#e5e7eb")
            canvas.create_text(left - 12, header_top + 10, text="Periods", anchor="ne", fill=COLORS["muted"], font=("TkDefaultFont", 8, "bold"))

            tick_step = _nice_step(total_span / max(6, min(14, int(canvas_width / 95))))
            tick = 0.0
            while tick <= total_span + 1e-9:
                xx = x(tick)
                canvas.create_line(xx, axis_y, xx, top + lane_h, fill="#e5e7eb")
                canvas.create_text(xx, axis_y - 3, text=_fmt(tick), anchor="s", fill="#4b5563", font=("TkDefaultFont", 8))
                tick += tick_step

            for idx, lane in enumerate(lanes):
                y_mid = top + idx * row_h + row_h / 2
                y_top = top + idx * row_h
                fill = "#f8fafc" if idx % 2 == 0 else "#ffffff"
                canvas.create_rectangle(0, y_top, total_w, y_top + row_h, fill=fill, outline="")
                canvas.create_line(left, y_top, total_w - right, y_top, fill="#e5e7eb")
                canvas.create_text(left - 12, y_mid, text=str(lane), anchor="e", fill=COLORS["text"], font=("TkDefaultFont", 9, "bold"))
            canvas.create_line(left, top, left, top + lane_h, fill="#cbd5e1")
            canvas.create_line(left, top + lane_h, total_w - right, top + lane_h, fill="#cbd5e1")

            markers = metadata.get("markers") or []
            for i, marker in enumerate(markers):
                rel = x(marker.get("release", 0))
                dead = x(marker.get("deadline", 0))
                canvas.create_line(rel, top - 10, rel, top + lane_h, fill="#6b7280", dash=(), width=1)
                canvas.create_line(dead, top - 10, dead, top + lane_h, fill="#ef4444", dash=(5, 3), width=1)
                if i < 12:
                    label = f"R {marker.get('cnn_id')}[{marker.get('instance')}]"
                    canvas.create_text(rel - 2, top + lane_h + 4, text=label, anchor="ne", angle=90, fill="#4b5563", font=("TkDefaultFont", 7))
                    label = f"D {marker.get('cnn_id')}[{marker.get('instance')}]"
                    canvas.create_text(dead + 2, top + lane_h + 4, text=label, anchor="nw", angle=90, fill="#ef4444", font=("TkDefaultFont", 7))

            for period in metadata.get("periods") or []:
                xx = x(period.get("release", 0))
                canvas.create_line(xx, header_top, xx, top + lane_h, fill="#334155", dash=(2, 3), width=1)
                action = str(period.get("action_label", "") or "").strip()
                label = f"P{period.get('period')}" + (f"\n{action}" if action else "")
                text_id = canvas.create_text(
                    xx + 5,
                    period_label_y,
                    text=label,
                    anchor="nw",
                    fill="#334155",
                    font=("TkDefaultFont", 8),
                    width=132,
                )
                bbox = canvas.bbox(text_id)
                if bbox:
                    pad_box = 2
                    canvas.create_rectangle(
                        bbox[0] - pad_box, bbox[1] - pad_box, bbox[2] + pad_box, bbox[3] + pad_box,
                        fill="white", outline="#cbd5e1", width=1,
                    )
                    canvas.tag_raise(text_id)

            lane_index = {lane: idx for idx, lane in enumerate(lanes)}
            font_size = max(7, min(10, int(8.5 * min(zoom, 1.35))))

            compute_positions: Dict[str, Dict[str, float]] = {}
            items_by_id: Dict[str, Dict[str, Any]] = {}
            for item in compute_items:
                lane = item.get("resource")
                if lane not in lane_index:
                    continue
                item_id = str(item.get("id", ""))
                if not item_id:
                    continue
                y_mid = top + lane_index[lane] * row_h + row_h / 2
                x_start = x(item.get("start", 0))
                x_finish = max(x(item.get("finish", 0)), x_start + 2)
                compute_positions[item_id] = {"start_x": x_start, "finish_x": x_finish, "y": y_mid}
                items_by_id[item_id] = item

            dependencies = list(metadata.get("dependencies") or [])
            max_dag_edges = 260
            if state.get("show_dag", True) and dependencies:
                hidden_count = max(0, len(dependencies) - max_dag_edges)
                for dep in dependencies[:max_dag_edges]:
                    pred_id = str(dep.get("pred_job_id", ""))
                    dst_id = str(dep.get("dst_job_id", ""))
                    pred_pos = compute_positions.get(pred_id)
                    dst_pos = compute_positions.get(dst_id)
                    if not pred_pos or not dst_pos:
                        continue
                    x_a = pred_pos["finish_x"]
                    y_a = pred_pos["y"]
                    x_b = dst_pos["start_x"]
                    y_b = dst_pos["y"]
                    crossed = bool(dep.get("crossed_medium"))
                    line_color = "#334155" if not crossed else "#64748b"
                    line_dash = () if not crossed else (5, 3)
                    line_width = 1.15 if zoom >= 0.8 else 0.9
                    bus_offset = row_h * (0.34 if abs(y_a - y_b) < 1 else 0.18)
                    mid_x = (x_a + x_b) / 2
                    # Elbow line: keeps same-lane DAG edges visible instead of hiding them under the bars.
                    points = [x_a, y_a, x_a + 6, y_a + bus_offset, mid_x, y_a + bus_offset, mid_x, y_b - bus_offset, x_b - 6, y_b - bus_offset, x_b, y_b]
                    edge = canvas.create_line(
                        *points,
                        fill=line_color,
                        width=line_width,
                        dash=line_dash,
                        arrow="last",
                        arrowshape=(8, 10, 4),
                        smooth=True,
                        tags=("dag_edge",),
                    )
                    _bind_tooltip(edge, _dependency_tooltip(dep, items_by_id))
                if hidden_count:
                    canvas.create_text(
                        total_w - right - 8,
                        top - 48,
                        text=f"DAG: {hidden_count} hidden dependency/dependencies — zoom or filtering recommended",
                        anchor="ne",
                        fill="#92400e",
                        font=("TkDefaultFont", 8),
                    )

            for item in message_items:
                lane = item.get("resource") or item.get("medium")
                if lane not in lane_index:
                    continue
                y_mid = top + lane_index[lane] * row_h + row_h / 2
                y1 = y_mid - row_h * 0.17
                y2 = y_mid + row_h * 0.17
                x1 = x(item.get("start", 0))
                x2 = max(x(item.get("finish", 0)), x1 + 2)
                rect = canvas.create_rectangle(x1, y1, x2, y2, fill="#d1d5db", outline="#111827", width=1, stipple="gray25")
                _bind_tooltip(rect, _message_tooltip(item))
                if x2 - x1 > 72:
                    txt = canvas.create_text((x1 + x2) / 2, y_mid, text=item.get("label", item.get("id", "")), fill="#111827", font=("TkDefaultFont", max(7, font_size - 1)))
                    _bind_tooltip(txt, _message_tooltip(item))

            for item in compute_items:
                lane = item.get("resource")
                if lane not in lane_index:
                    continue
                y_mid = top + lane_index[lane] * row_h + row_h / 2
                y1 = y_mid - row_h * 0.25
                y2 = y_mid + row_h * 0.25
                x1 = x(item.get("start", 0))
                x2 = max(x(item.get("finish", 0)), x1 + 2)
                fill = _color_for(str(item.get("task", item.get("id", ""))))
                outline = "#dc2626" if item.get("miss") else ("#f59e0b" if item.get("changed") else "#111827")
                width = 3 if item.get("miss") else (2 if item.get("changed") else 1)
                rect = canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline, width=width)
                tooltip_text = _compute_tooltip(item)
                _bind_tooltip(rect, tooltip_text)
                if x2 - x1 > 62:
                    label = str(item.get("label", item.get("task", "")))
                    txt = canvas.create_text((x1 + x2) / 2, y_mid, text=label, fill="white", font=("TkDefaultFont", font_size), justify="center")
                    _bind_tooltip(txt, tooltip_text)

            canvas.configure(scrollregion=(0, 0, total_w, total_h))
            canvas.xview_moveto(x0)
            canvas.yview_moveto(y0)

        def _toggle_dag() -> None:
            state["show_dag"] = not bool(state.get("show_dag", True))
            dag_button_var.set("Hide DAG" if state["show_dag"] else "Show DAG")
            _draw()

        def _set_zoom(value: float) -> None:
            state["zoom"] = max(0.35, min(5.0, float(value)))
            _draw()

        def _on_mousewheel(event: tk.Event) -> str | None:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            if getattr(event, "state", 0) & 0x0004:  # Ctrl
                _set_zoom(state["zoom"] * (1.12 if delta < 0 else 1 / 1.12))
                return "break"
            if getattr(event, "state", 0) & 0x0001:  # Shift
                canvas.xview_scroll(delta * 4, "units")
                return "break"
            canvas.yview_scroll(delta * 3, "units")
            return None

        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", lambda _e: canvas.yview_scroll(-3, "units"))
        canvas.bind("<Button-5>", lambda _e: canvas.yview_scroll(3, "units"))
        canvas.bind("<Configure>", lambda _e: _draw())
        win.bind("<Destroy>", lambda _e: _hide_tooltip())
        _draw()

    # Backward-compatible alias for older callbacks/tests.
    def _open_gantt_png(self, show_info_if_missing: bool = True) -> None:
        self._show_gantt(show_info_if_missing=show_info_if_missing)

    def _append_output(self, text: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def _open_workspace(self) -> None:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        try:
            webbrowser.open(WORKSPACE.resolve().as_uri())
        except Exception:
            folder = filedialog.askdirectory(initialdir=str(WORKSPACE.resolve()), title="Workspace")
            if folder:
                webbrowser.open(Path(folder).resolve().as_uri())


def main() -> None:
    app = NPsimTkApp()
    app.mainloop()


if __name__ == "__main__":
    main()
