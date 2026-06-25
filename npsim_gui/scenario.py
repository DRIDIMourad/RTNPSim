from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

PRECISIONS = ["INT8", "FP16", "FP32"]
NPU_PRECISIONS = ["INT8"]
CPU_PRECISIONS = ["FP16", "FP32"]
CRITICITIES = ["HI", "LO"]
SUPPORTS = ["CPU only", "CPU/NPU", "NPU preferred"]

# GUI time scale. The simulator still writes and runs cycles internally.
TICK_CYCLES_DEFAULT = 10000


def tick_cycles_from_arch(arch: Dict[str, Any] | None = None) -> int:
    if not arch:
        return TICK_CYCLES_DEFAULT
    return max(1, int(float(arch.get("tick_cycles", arch.get("cycles_per_tick", TICK_CYCLES_DEFAULT)))))


def cycles_to_ticks(value: int | float, tick_cycles: int = TICK_CYCLES_DEFAULT, *, minimum: int = 0) -> int:
    ticks = int(math.ceil(max(0, float(value)) / max(1, int(tick_cycles))))
    return max(minimum, ticks)


def ticks_to_cycles(value: int | float, tick_cycles: int = TICK_CYCLES_DEFAULT, *, minimum: int = 0) -> int:
    cycles = int(max(0, float(value)) * max(1, int(tick_cycles)))
    return max(minimum, cycles)


def task_ticks_to_cycles(task: Dict[str, Any], tick_cycles: int = TICK_CYCLES_DEFAULT) -> Dict[str, Any]:
    row = deepcopy(task)
    for key in ("period", "deadline", "phase", "wcet_int8", "wcet_fp16", "wcet_fp32"):
        row[key] = ticks_to_cycles(row.get(key, 0), tick_cycles, minimum=1 if key in {"period", "deadline"} else 0)
    return row


def tasks_ticks_to_cycles(tasks: Iterable[Dict[str, Any]], tick_cycles: int = TICK_CYCLES_DEFAULT) -> List[Dict[str, Any]]:
    return [task_ticks_to_cycles(t, tick_cycles) for t in tasks]


def task_cycles_to_ticks(task: Dict[str, Any], tick_cycles: int = TICK_CYCLES_DEFAULT) -> Dict[str, Any]:
    row = deepcopy(task)
    for key in ("period", "deadline", "phase", "wcet_int8", "wcet_fp16", "wcet_fp32"):
        row[key] = cycles_to_ticks(row.get(key, 0), tick_cycles, minimum=1 if key in {"period", "deadline"} else 0)
    return row


def tasks_cycles_to_ticks(tasks: Iterable[Dict[str, Any]], tick_cycles: int = TICK_CYCLES_DEFAULT) -> List[Dict[str, Any]]:
    return [task_cycles_to_ticks(t, tick_cycles) for t in tasks]


def dominant_period(tasks: Iterable[Dict[str, Any]]) -> int:
    periods: List[int] = []
    for row in tasks:
        try:
            p = int(row.get("period", 0))
        except Exception:
            p = 0
        if p > 0:
            periods.append(p)
    return min(periods) if periods else 1


def periods_to_simulation_time(tasks: Iterable[Dict[str, Any]], periods: int) -> int:
    return max(1, int(periods)) * dominant_period(tasks)


def simulation_time_to_periods(tasks: Iterable[Dict[str, Any]], simulation_time: int) -> int:
    base = dominant_period(tasks)
    return max(1, int(math.ceil(max(1, int(simulation_time)) / base)))


# Generic is first because it is the default. Generic tasks are WCET-only in the simulator.
#
# The GUI must preserve explicit task kinds from imported tasks.txt files.
# Internal hardware-model kinds such as ``systolic`` and ``depthwise`` are
# therefore first-class values here, not aliases that get collapsed to generic.
LAYER_KINDS = [
    "generic",
    "systolic",
    "depthwise",
    "pool",
    "bn",
    "activation",
    "fuse",
    "nms",
    "control",
    # User-facing aliases kept for manual GUI editing. normalize_task_table()
    # canonicalizes them before export/import.
    "conv",
    "convolution",
    "fc",
    "fully_connected",
    "gemm",
    "matmul",
    "depthwise_conv",
    "dwconv",
    "pointwise_conv",
    "pwconv",
    "pointwise",
    "softmax",
    "postprocess",
    "add",
    "concat",
    "detection_head",
    "identity",
]

LAYER_KIND_ALIASES = {
    "generic": "generic",
    "control": "control",
    "ctrl": "control",
    "systolic": "systolic",
    "conv": "systolic",
    "convolution": "systolic",
    "fc": "systolic",
    "fully_connected": "systolic",
    "gemm": "systolic",
    "matmul": "systolic",
    "pointwise": "systolic",
    "pointwise_conv": "systolic",
    "pwconv": "systolic",
    "detection_head": "systolic",
    "detect": "systolic",
    "head": "systolic",
    "depthwise": "depthwise",
    "depthwise_conv": "depthwise",
    "dwconv": "depthwise",
    "pool": "pool",
    "pooling": "pool",
    "max_pool": "pool",
    "max_pool_2d": "pool",
    "mean": "pool",
    "avg_pool": "pool",
    "bn": "bn",
    "batchnorm": "bn",
    "norm": "bn",
    "activation": "activation",
    "relu": "activation",
    "act": "activation",
    "softmax": "activation",
    "fuse": "fuse",
    "postprocess": "fuse",
    "post": "fuse",
    "add": "fuse",
    "concat": "fuse",
    "identity": "fuse",
    "skip": "fuse",
    "nms": "nms",
}


def _task(
    name: str,
    cnn_id: str,
    kind: str,
    support: str,
    period: int,
    deadline: int,
    priority: int,
    phase: int,
    precision: str,
    wcet: tuple[int, int, int],
    preds: str,
    criticity: str = "LO",
    size: int = 1,
) -> Dict[str, Any]:
    return {
        "name": name,
        "cnn_id": cnn_id,
        "kind": kind,
        "support": support,
        "period": period,
        "deadline": deadline,
        "priority": priority,
        "phase": phase,
        "precision": precision,
        "wcet_int8": wcet[0],
        "wcet_fp16": wcet[1],
        "wcet_fp32": wcet[2],
        "preds": preds,
        "criticity": criticity,
        "size": size,
    }


# Minimal default: a calibrated generic pipeline. Values are in GUI ticks.
DEFAULT_TASKS: List[Dict[str, Any]] = [
    _task("PRE_A", "CNN_A", "generic", "CPU only", 100, 85, 0, 0, "FP32", (0, 8, 8), "-", "HI", 1),
    _task("TASK_A1", "CNN_A", "generic", "CPU/NPU", 100, 85, 1, 0, "INT8", (22, 32, 48), "PRE_A", "LO", 1),
    _task("TASK_A2", "CNN_A", "generic", "CPU/NPU", 100, 85, 2, 0, "INT8", (18, 27, 40), "TASK_A1", "LO", 1),
    _task("POST_A", "CNN_A", "generic", "CPU only", 100, 85, 3, 0, "FP32", (0, 9, 9), "TASK_A2", "HI", 1),
]

TWO_CNN_EXTRA_TASKS: List[Dict[str, Any]] = [
    _task("PRE_B", "CNN_B", "generic", "CPU only", 100, 85, 0, 12, "FP32", (0, 8, 8), "-", "HI", 1),
    _task("TASK_B1", "CNN_B", "generic", "CPU/NPU", 100, 85, 1, 12, "INT8", (21, 31, 46), "PRE_B", "LO", 1),
    _task("TASK_B2", "CNN_B", "generic", "CPU/NPU", 100, 85, 2, 12, "INT8", (19, 28, 42), "TASK_B1", "LO", 1),
    _task("POST_B", "CNN_B", "generic", "CPU only", 100, 85, 3, 12, "FP32", (0, 9, 9), "TASK_B2", "HI", 1),
]


def default_arch_settings() -> Dict[str, Any]:
    """Hardware defaults used by the GUI and examples.

    Generic tasks are WCET-only. Non-generic tasks can still use the hardware
    estimation equations, with finite bandwidth/latency defaults.
    """
    return {
        "cpu_name": "CPU",
        "npu_count": 1,
        "reference_frequency_mhz": 1000,
        "cpu_frequency_mhz": 1000,
        "npu_frequency_mhz": 1000,
        "interconnect_frequency_mhz": 1000,
        "tick_cycles": TICK_CYCLES_DEFAULT,
        "cpu_npu_mode": "separate_full_duplex",
        "cpu_to_npu_setup": 40,
        "cpu_to_npu_per_unit": 1,
        "npu_to_cpu_setup": 40,
        "npu_to_cpu_per_unit": 1,
        "shared_setup": 40,
        "shared_per_unit": 1,
        "noc_topology": "mesh",
        "noc_setup": 30,
        "noc_per_unit": 1,
        "router_latency": 3,
        "arb_policy": "fixed_priority",
        "noc_rr_quantum_units": 64,
        "sa_rows": 16,
        "sa_cols": 16,
        "systolic_arrays": 1,
        "npu_setup_cycles": 20,
        "precision_modes": ["INT8"],
        "local_dram_capacity_kb": 1024,
        "local_dram_latency": 120,
        "local_dram_bandwidth_bytes_per_tick": 16,
        "dma_setup": 120,
        "dma_bandwidth_bytes_per_tick": 16,
        "vector_lanes": 64,
        "vector_setup": 20,
        "cpu_setup_cycles": 1,
        "cache_capacity_kb": 256,
        "cache_latency": 40,
        "cache_bandwidth_bytes_per_tick": 16,
        "simd_mac_per_tick": 8,
        "pack_elements_per_tick": 32,
        "epilogue_elements_per_tick": 32,
        "vector_elements_per_tick": 8,
        "rmw_bandwidth_bytes_per_tick": 16,
    }


def scenario_single_cnn() -> List[Dict[str, Any]]:
    return deepcopy(DEFAULT_TASKS)


def scenario_two_cnn() -> List[Dict[str, Any]]:
    return deepcopy(DEFAULT_TASKS + TWO_CNN_EXTRA_TASKS)


def _int_value(value: Any, default: int = 0, minimum: int | None = None) -> int:
    try:
        if value is None or value == "":
            out = default
        else:
            out = int(value)
    except (TypeError, ValueError):
        out = default
    if minimum is not None:
        out = max(minimum, out)
    return out


def _str_value(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def normalize_task_table(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    seen_names: set[str] = set()
    for idx, raw in enumerate(rows):
        name = _str_value(raw.get("name"), f"TASK_{idx}").replace(" ", "_")
        if name in seen_names:
            name = f"{name}_{idx}"
        seen_names.add(name)

        cnn_id = _str_value(raw.get("cnn_id"), "CNN_A").replace(" ", "_")
        kind_token = _str_value(raw.get("kind"), "generic").lower().replace("-", "_")
        kind = LAYER_KIND_ALIASES.get(kind_token, kind_token if kind_token in LAYER_KINDS else "generic")

        support = _str_value(raw.get("support"), "CPU/NPU")
        support_aliases = {
            "CPU seulement": "CPU only",
            "cpu seulement": "CPU only",
        }
        support = support_aliases.get(support, support)
        if support not in SUPPORTS:
            support = "CPU/NPU"

        precision = _str_value(raw.get("precision"), "INT8").upper()
        if precision not in PRECISIONS:
            precision = "INT8"
        # Names never infer precision. The initial/file mapping precision is
        # clamped only by the hardware support: NPU executes INT8 only; CPU
        # executes FP16/FP32 only. Mapping algorithms still evaluate all legal
        # CPU/NPU choices for CPU/NPU-capable tasks.
        if support == "CPU only" and precision == "INT8":
            precision = "FP32"
        elif support != "CPU only" and precision != "INT8":
            precision = "INT8"

        criticity = _str_value(raw.get("criticity"), "LO").upper()
        if criticity not in CRITICITIES:
            criticity = "LO"

        preds = _str_value(raw.get("preds"), "-")
        preds = preds.replace(" ", "") if preds != "-" else "-"
        if not preds:
            preds = "-"

        tasks.append(
            {
                "name": name,
                "cnn_id": cnn_id,
                "kind": kind,
                "support": support,
                "period": _int_value(raw.get("period"), 100, minimum=1),
                "deadline": _int_value(raw.get("deadline"), 85, minimum=1),
                "priority": _int_value(raw.get("priority"), idx, minimum=0),
                "phase": _int_value(raw.get("phase"), 0, minimum=0),
                "precision": precision,
                "wcet_int8": _int_value(raw.get("wcet_int8"), 1, minimum=0),
                "wcet_fp16": _int_value(raw.get("wcet_fp16"), 1, minimum=0),
                "wcet_fp32": _int_value(raw.get("wcet_fp32"), 1, minimum=0),
                "preds": preds,
                "criticity": criticity,
                "size": _int_value(raw.get("size"), 1, minimum=1),
            }
        )
    return tasks


def _chain(tag: str, cnn_id: str, period: int, deadline: int, phase: int, wcets: list[tuple[int, int, int]], pre: int, post: int) -> List[Dict[str, Any]]:
    tasks = [_task(f"{tag}_PRE", cnn_id, "generic", "CPU only", period, deadline, 0, phase, "FP32", (0, pre, pre), "-", "HI", 1)]
    pred = f"{tag}_PRE"
    for i, wcet in enumerate(wcets, start=1):
        name = f"{tag}_TASK{i}"
        tasks.append(_task(name, cnn_id, "generic", "CPU/NPU", period, deadline, i, phase, "INT8", wcet, pred, "LO", 1))
        pred = name
    tasks.append(_task(f"{tag}_POST", cnn_id, "generic", "CPU only", period, deadline, len(wcets) + 1, phase, "FP32", (0, post, post), pred, "HI", 1))
    return tasks




def road_sign_tiny_dscnn_tasks(
    period: int,
    deadline: int,
    phase: int = 0,
    wcet_profile: str = "base",
) -> List[Dict[str, Any]]:
    """One-CNN GTSRB/Tiny-DSCNN task chain used by the MCXN workflow.

    The operator sequence is intentionally identical to the imported
    npsim_tasks.txt generated from the TFLite graph. Scenarios may change only
    timing parameters: period, deadline, phase and WCET columns.
    """
    base_rows = [
        ("OP000_CONV_2D", "systolic", (37, 0, 114), "-", "HI", 36),
        ("OP001_MAX_POOL_2D", "pool", (1, 0, 2), "OP000_CONV_2D", "LO", 9),
        ("OP002_DEPTHWISE_CONV_2D", "depthwise", (3, 0, 10), "OP001_MAX_POOL_2D", "LO", 9),
        ("OP003_CONV_2D", "systolic", (19, 0, 59), "OP002_DEPTHWISE_CONV_2D", "LO", 18),
        ("OP004_MAX_POOL_2D", "pool", (1, 0, 1), "OP003_CONV_2D", "LO", 5),
        ("OP005_DEPTHWISE_CONV_2D", "depthwise", (2, 0, 6), "OP004_MAX_POOL_2D", "LO", 5),
        ("OP006_CONV_2D", "systolic", (10, 0, 31), "OP005_DEPTHWISE_CONV_2D", "LO", 9),
        ("OP007_MAX_POOL_2D", "pool", (1, 0, 1), "OP006_CONV_2D", "LO", 3),
        ("OP008_DEPTHWISE_CONV_2D", "depthwise", (1, 0, 4), "OP007_MAX_POOL_2D", "LO", 3),
        ("OP009_CONV_2D", "systolic", (4, 0, 14), "OP008_DEPTHWISE_CONV_2D", "LO", 4),
        ("OP010_MAX_POOL_2D", "pool", (1, 0, 1), "OP009_CONV_2D", "LO", 1),
        ("OP011_DEPTHWISE_CONV_2D", "depthwise", (1, 0, 3), "OP010_MAX_POOL_2D", "LO", 1),
        ("OP012_CONV_2D", "systolic", (2, 0, 7), "OP011_DEPTHWISE_CONV_2D", "LO", 2),
        ("OP013_MEAN", "pool", (1, 0, 1), "OP012_CONV_2D", "LO", 1),
        ("OP014_FULLY_CONNECTED", "systolic", (1, 0, 3), "OP013_MEAN", "LO", 1),
        ("OP015_SOFTMAX", "activation", (1, 0, 1), "OP014_FULLY_CONNECTED", "HI", 1),
    ]

    if wcet_profile == "mixed_frontier":
        wcets = [
            (109, 438, 736), (4, 10, 11), (21, 21, 25), (116, 141, 320),
            (2, 2, 2), (13, 18, 41), (12, 41, 175), (3, 7, 8),
            (5, 13, 14), (12, 40, 56), (1, 3, 7), (7, 9, 9),
            (3, 6, 34), (2, 3, 3), (7, 11, 23), (6, 7, 7),
        ]
        base_rows = [(name, kind, wcets[i], preds, crit, size) for i, (name, kind, _w, preds, crit, size) in enumerate(base_rows)]
    elif wcet_profile == "cpu_gap":
        wcets = [
            (45, 75, 180), (3, 5, 9), (8, 14, 30), (24, 45, 95),
            (2, 4, 7), (6, 12, 24), (16, 30, 70), (2, 4, 6),
            (4, 8, 18), (10, 18, 42), (2, 3, 5), (4, 7, 16),
            (5, 9, 24), (2, 3, 5), (4, 8, 18), (2, 3, 5),
        ]
        base_rows = [(name, kind, wcets[i], preds, crit, size) for i, (name, kind, _w, preds, crit, size) in enumerate(base_rows)]
    elif wcet_profile == "npu_pressure":
        wcets = [
            (70, 105, 210), (2, 4, 8), (18, 28, 55), (60, 90, 170),
            (2, 3, 6), (14, 22, 48), (45, 70, 135), (2, 3, 6),
            (10, 16, 34), (28, 44, 90), (1, 2, 5), (8, 12, 28),
            (16, 25, 60), (2, 3, 5), (8, 13, 32), (2, 3, 5),
        ]
        base_rows = [(name, kind, wcets[i], preds, crit, size) for i, (name, kind, _w, preds, crit, size) in enumerate(base_rows)]

    tasks: List[Dict[str, Any]] = []
    for prio, (name, kind, wcet, preds, criticity, size) in enumerate(base_rows):
        tasks.append(
            _task(
                name=name,
                cnn_id="ROAD_SIGN_TINY_DSCNN",
                kind=kind,
                support="CPU/NPU",
                period=period,
                deadline=deadline,
                priority=prio,
                phase=phase,
                precision="INT8",
                wcet=wcet,
                preds=preds,
                criticity=criticity,
                size=size,
            )
        )
    return tasks


def _same_road_sign_onecnn_examples(add, one_cpu_one_npu: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Additional GUI scenarios that keep exactly one CNN stream.

    Old GUI scenarios are preserved. These extra scenarios are meant to test
    whether Offline DF and Offline QF converge or diverge on the same TFLite
    operator chain when only timing values are changed.
    """
    base = road_sign_tiny_dscnn_tasks(period=300, deadline=500, wcet_profile="base")
    mixed = road_sign_tiny_dscnn_tasks(period=450, deadline=400, wcet_profile="mixed_frontier")
    gap = road_sign_tiny_dscnn_tasks(period=450, deadline=390, wcet_profile="cpu_gap")
    pressure = road_sign_tiny_dscnn_tasks(period=700, deadline=610, wcet_profile="npu_pressure")

    return [
        add(
            "11 — ONE CNN RoadSign Tiny-DSCNN — Offline DF baseline",
            "Same single CNN as the generated npsim_tasks.txt. Only timing values are set by the scenario. Offline DF starts from NPU/INT8 and improves quality while deadlines stay valid.",
            base,
            horizon_ticks=1500,
            mapping_policy="offline_df",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "12 — ONE CNN RoadSign Tiny-DSCNN — Offline QF baseline",
            "Same exact single-CNN task table as scenario 11. Offline QF starts from CPU/high quality and searches for the best fixed feasible mapping.",
            base,
            horizon_ticks=1500,
            mapping_policy="offline_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "13 — ONE CNN mixed frontier — Offline DF",
            "One CNN only. Same 16 operators, but with FP16 values enabled and irregular WCETs to create several feasible CPU/NPU quality frontiers.",
            mixed,
            horizon_ticks=1800,
            mapping_policy="offline_df",
            arch_updates=one_cpu_one_npu,
            max_configs=16,
        ),
        add(
            "14 — ONE CNN mixed frontier — Offline QF",
            "Same exact task table as scenario 13, evaluated with Offline QF. This pair checks whether the greedy DF path and the QF global quality search select the same fixed configuration.",
            mixed,
            horizon_ticks=1800,
            mapping_policy="offline_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=16,
        ),
        add(
            "15 — ONE CNN CPU-gap profile — Offline DF",
            "One CNN only. CPU FP32 is intentionally more expensive on most operators, while FP16 creates intermediate quality points. Used to test DF's incremental quality ramp.",
            gap,
            horizon_ticks=1800,
            mapping_policy="offline_df",
            arch_updates=one_cpu_one_npu,
            max_configs=32,
        ),
        add(
            "16 — ONE CNN CPU-gap profile — Offline QF",
            "Same exact task table as scenario 15, evaluated with Offline QF. Compare the selected configuration against scenario 15.",
            gap,
            horizon_ticks=1800,
            mapping_policy="offline_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=32,
        ),
        add(
            "17 — ONE CNN NPU-pressure profile — Offline DF",
            "One CNN only. Convolution and depthwise layers are heavier, so the single NPU becomes the critical resource even without a second CNN stream.",
            pressure,
            horizon_ticks=2100,
            mapping_policy="offline_df",
            arch_updates=one_cpu_one_npu,
            max_configs=32,
        ),
        add(
            "18 — ONE CNN NPU-pressure profile — Offline QF",
            "Same exact task table as scenario 17, evaluated with Offline QF. This keeps the experiment clean: one CNN, same operator chain, only timing profile changes.",
            pressure,
            horizon_ticks=2100,
            mapping_policy="offline_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=32,
        ),
    ]


def scenario_examples() -> List[Dict[str, Any]]:
    """One-CPU / one-NPU MobileNet examples for algorithm comparison.

    All examples use a deliberately simple architecture: one CPU and one NPU.
    Every task is exported as kind=generic. Names such as CONV, DW, PW, HEAD,
    or ALERT are labels only; timing comes from the WCET columns unless the user
    explicitly changes the kind field later in the GUI.
    Timings below are authored in GUI ticks.
    """

    def arch(**updates: Any) -> Dict[str, Any]:
        a = default_arch_settings()
        a.update({
            "npu_count": 1,
            "precision_modes": ["INT8"],
            "tick_cycles": TICK_CYCLES_DEFAULT,
        })
        a.update(updates)
        return a

    def add(
        label: str,
        description: str,
        tasks_ticks: List[Dict[str, Any]],
        horizon_ticks: int,
        mapping_policy: str,
        arch_updates: Dict[str, Any] | None = None,
        runtime_policy: str = "edf",
        switch_ticks: int = 1,
        max_configs: int = 128,
    ) -> Dict[str, Any]:
        hw = arch(**(arch_updates or {}))
        tick_cycles = tick_cycles_from_arch(hw)
        tasks_norm = normalize_task_table(tasks_ticks)
        return {
            "label": label,
            "description": (
                description
                + " Architecture: exactly 1 CPU and 1 NPU. "
                + "Precision rule: NPU uses INT8 only; CPU uses FP16 or FP32. "
                + f"1 tick = {tick_cycles} cycles."
            ),
            "tasks": tasks_norm,
            "horizon_ticks": int(horizon_ticks),
            "simulation_time": int(horizon_ticks),
            "simulation_time_cycles": ticks_to_cycles(horizon_ticks, tick_cycles, minimum=1),
            "tick_cycles": tick_cycles,
            "npu_size": 1,
            "arch_updates": hw,
            "runtime_policy": runtime_policy,
            "mapping_policy": mapping_policy,
            "switch_cost": int(switch_ticks),
            "switch_cost_mode": "per_layer",
            "max_configs_per_cnn": int(max_configs),
            "gantt_time_unit": "tick",
        }

    def mb_chain(
        prefix: str,
        cnn_id: str,
        period: int,
        deadline: int,
        phase: int = 0,
        layers: list[tuple[str, int, int, int]] | None = None,
        pre: int = 6,
        post: int = 6,
    ) -> List[Dict[str, Any]]:
        if layers is None:
            layers = [
                ("EXP", 20, 30, 42),
                ("DW", 18, 27, 38),
                ("PW", 22, 33, 46),
            ]
        tasks = [
            _task(f"{prefix}_PRE", cnn_id, "generic", "CPU only", period, deadline, 0, phase, "FP32", (0, pre, pre), "-", "HI", 1)
        ]
        pred = f"{prefix}_PRE"
        for prio, (suffix, int8, fp16, fp32) in enumerate(layers, start=1):
            name = f"{prefix}_{suffix}"
            tasks.append(
                _task(name, cnn_id, "generic", "CPU/NPU", period, deadline, prio, phase, "INT8", (int8, fp16, fp32), pred, "LO", 1)
            )
            pred = name
        tasks.append(
            _task(f"{prefix}_HEAD", cnn_id, "generic", "CPU only", period, deadline, len(layers) + 1, phase, "FP32", (0, post, post), pred, "HI", 1)
        )
        return tasks

    one_cpu_one_npu = arch()

    # Scenario family A: same single-stream MobileNet workload under the four policies.
    single_mobile = mb_chain(
        "MB_A",
        "MOBILENET_A",
        period=100,
        deadline=95,
        phase=0,
        layers=[("EXP", 20, 30, 42), ("DW", 18, 27, 38), ("PW", 22, 33, 46)],
        pre=6,
        post=6,
    )

    # Scenario family B: two camera streams sharing the same single NPU.
    dual_mobile: List[Dict[str, Any]] = []
    dual_mobile.extend(
        mb_chain(
            "CAM_A",
            "CAMERA_A",
            period=120,
            deadline=105,
            phase=0,
            layers=[("EXP", 28, 32, 44), ("PW", 28, 32, 44)],
            pre=7,
            post=7,
        )
    )
    dual_mobile.extend(
        mb_chain(
            "CAM_B",
            "CAMERA_B",
            period=120,
            deadline=105,
            phase=30,
            layers=[("EXP", 27, 31, 43), ("PW", 27, 31, 43)],
            pre=7,
            post=7,
        )
    )

    # Scenario family C: steady stream plus delayed one-shot MobileNet alert.
    burst_mobile: List[Dict[str, Any]] = []
    burst_mobile.extend(
        mb_chain(
            "MAIN",
            "MOBILENET_MAIN",
            period=100,
            deadline=95,
            phase=0,
            layers=[("CONV", 30, 44, 60)],
            pre=5,
            post=5,
        )
    )
    burst_mobile.extend(
        mb_chain(
            "ALERT",
            "MOBILENET_ALERT",
            period=1000,
            deadline=75,
            phase=250,
            layers=[("CONV", 40, 56, 74)],
            pre=5,
            post=5,
        )
    )

    # Scenario family D: branch/join MobileNet block to show dependencies on a compact DAG.
    branch_mobile = [
        _task("SKIP_PRE", "MOBILENET_SKIP", "generic", "CPU only", 120, 100, 0, 0, "FP32", (0, 6, 6), "-", "HI", 1),
        _task("SKIP_EXP", "MOBILENET_SKIP", "generic", "CPU/NPU", 120, 100, 1, 0, "INT8", (22, 32, 46), "SKIP_PRE", "LO", 1),
        _task("SKIP_DW", "MOBILENET_SKIP", "generic", "CPU/NPU", 120, 100, 1, 0, "INT8", (20, 30, 44), "SKIP_PRE", "LO", 1),
        _task("SKIP_ADD", "MOBILENET_SKIP", "generic", "CPU/NPU", 120, 100, 2, 0, "INT8", (14, 22, 32), "SKIP_EXP,SKIP_DW", "HI", 1),
        _task("SKIP_HEAD", "MOBILENET_SKIP", "generic", "CPU only", 120, 100, 3, 0, "FP32", (0, 7, 7), "SKIP_ADD", "HI", 1),
    ]

    examples = [
        add(
            "01 — 1CPU/1NPU MobileNet cold start — Online DF keeps deadlines",
            "Online DF starts from a safe NPU/INT8-heavy configuration and raises quality only after validation. Expected behavior: no startup deadline debt; quality improves gradually.",
            single_mobile,
            horizon_ticks=500,
            mapping_policy="online_df",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "02 — 1CPU/1NPU MobileNet cold start — Online QF creates debt",
            "Same workload as scenario 01. Online QF starts all CPU/high quality. Expected behavior: P0 is too long, then the backlog causes visible deadline misses while repair is limited to one change per period.",
            single_mobile,
            horizon_ticks=500,
            mapping_policy="online_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "03 — 1CPU/1NPU MobileNet cold start — Offline QF avoids startup transient",
            "Same workload as scenarios 01/02. Offline QF searches CPU/NPU configurations before execution and then runs one fixed mapping. Expected behavior: no P0 transient, with the best feasible fixed quality.",
            single_mobile,
            horizon_ticks=500,
            mapping_policy="offline_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "04 — 1CPU/1NPU MobileNet cold start — Offline DF fixed safety-first plan",
            "Same workload as scenarios 01/02/03. Offline DF starts safe, then raises quality only while the full fixed schedule remains feasible. Expected behavior: deterministic fixed execution with no runtime remapping.",
            single_mobile,
            horizon_ticks=500,
            mapping_policy="offline_df",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "05 — 1CPU/1NPU dual MobileNet cameras — Online DF manages contention",
            "Two staggered MobileNet camera streams share the only NPU. Expected behavior: Online DF avoids startup misses and performs a conservative quality ramp under single-NPU contention.",
            dual_mobile,
            horizon_ticks=600,
            mapping_policy="online_df",
            arch_updates=one_cpu_one_npu,
            max_configs=64,
        ),
        add(
            "06 — 1CPU/1NPU dual MobileNet cameras — Online QF overloads the start",
            "Same workload as scenario 05. Online QF starts high-quality/all-CPU on both streams. Expected behavior: CPU overload creates a large backlog before the online repair catches up.",
            dual_mobile,
            horizon_ticks=600,
            mapping_policy="online_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "07 — 1CPU/1NPU dual MobileNet cameras — Offline QF fixed compromise",
            "Same workload as scenarios 05/06. Offline QF chooses one CPU/NPU split before execution. Expected behavior: deterministic no-remap schedule; less startup risk than Online QF, but no period-by-period adaptation.",
            dual_mobile,
            horizon_ticks=600,
            mapping_policy="offline_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=16,
        ),
        add(
            "08 — 1CPU/1NPU aperiodic MobileNet alert — Online DF handles burst",
            "A steady MobileNet stream is joined by a delayed one-shot alert at tick 250. Expected behavior: Online DF keeps the burst feasible by preserving timing slack and validating changes online.",
            burst_mobile,
            horizon_ticks=420,
            mapping_policy="online_df",
            arch_updates=one_cpu_one_npu,
            max_configs=16,
        ),
        add(
            "09 — 1CPU/1NPU aperiodic MobileNet alert — Online QF misses burst",
            "Same workload as scenario 08. Online QF preserves quality first, leaving too little slack for the delayed alert. Expected behavior: a visible alert-related deadline miss.",
            burst_mobile,
            horizon_ticks=420,
            mapping_policy="online_qf",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
        add(
            "10 — 1CPU/1NPU MobileNet branch/join DAG — Offline DF shows dependencies",
            "A compact MobileNet residual block with a branch/join dependency structure. Expected behavior: Offline DF produces a fixed safe plan and the Gantt highlights the DAG dependencies clearly on one CPU and one NPU.",
            branch_mobile,
            horizon_ticks=480,
            mapping_policy="offline_df",
            arch_updates=one_cpu_one_npu,
            max_configs=128,
        ),
    ]

    examples.extend(_same_road_sign_onecnn_examples(add, one_cpu_one_npu))
    return examples

def architecture_for_yaml(arch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "clock": {
            "time_base": "cycles",
            "user_time_unit": "tick",
            "tick_cycles": tick_cycles_from_arch(arch),
            "reference_frequency_mhz": float(arch.get("reference_frequency_mhz", 1000)),
            "cpu_frequency_mhz": float(arch.get("cpu_frequency_mhz", 1000)),
            "npu_frequency_mhz": float(arch.get("npu_frequency_mhz", 1000)),
            "interconnect_frequency_mhz": float(arch.get("interconnect_frequency_mhz", 1000)),
        },
        "resources": {
            "cpu_name": arch.get("cpu_name", "CPU"),
            "npu_count": int(arch.get("npu_count", 1)),
        },
        "communications": {
            "cpu_npu_mode": arch.get("cpu_npu_mode", "separate_full_duplex"),
            "cpu_to_npu_setup": int(arch.get("cpu_to_npu_setup", 2)),
            "cpu_to_npu_per_unit": int(arch.get("cpu_to_npu_per_unit", 1)),
            "npu_to_cpu_setup": int(arch.get("npu_to_cpu_setup", 2)),
            "npu_to_cpu_per_unit": int(arch.get("npu_to_cpu_per_unit", 1)),
            "shared_setup": int(arch.get("shared_setup", 2)),
            "shared_per_unit": int(arch.get("shared_per_unit", 1)),
            "noc_topology": arch.get("noc_topology", "mesh"),
            "noc_setup": int(arch.get("noc_setup", 1)),
            "noc_per_unit": int(arch.get("noc_per_unit", 1)),
            "router_latency": int(arch.get("router_latency", 1)),
            "arb_policy": arch.get("arb_policy", "fixed_priority"),
            "noc_rr_quantum_units": int(arch.get("noc_rr_quantum_units", 32)),
        },
        "npu": {
            "sa_rows": int(arch.get("sa_rows", 16)),
            "sa_cols": int(arch.get("sa_cols", 16)),
            "systolic_arrays": int(arch.get("systolic_arrays", 1)),
            "setup_cycles": int(arch.get("npu_setup_cycles", 2)),
            "precision_modes": ["INT8"],
            "local_dram_capacity_kb": int(arch.get("local_dram_capacity_kb", 2048)),
            "local_dram_latency": int(arch.get("local_dram_latency", 6)),
            "local_dram_bandwidth_bytes_per_cycle": int(arch.get("local_dram_bandwidth_bytes_per_cycle", arch.get("local_dram_bandwidth_bytes_per_tick", 256))),
            "dma_setup": int(arch.get("dma_setup", 2)),
            "dma_bandwidth_bytes_per_cycle": int(arch.get("dma_bandwidth_bytes_per_cycle", arch.get("dma_bandwidth_bytes_per_tick", 512))),
            "vector_lanes": int(arch.get("vector_lanes", 128)),
            "vector_setup": int(arch.get("vector_setup", 1)),
        },
        "cpu": {
            "setup_cycles": int(arch.get("cpu_setup_cycles", 1)),
            "cache_capacity_kb": int(arch.get("cache_capacity_kb", 512)),
            "cache_latency": int(arch.get("cache_latency", 6)),
            "cache_bandwidth_bytes_per_cycle": int(arch.get("cache_bandwidth_bytes_per_cycle", arch.get("cache_bandwidth_bytes_per_tick", 512))),
            "simd_mac_per_cycle": int(arch.get("simd_mac_per_cycle", arch.get("simd_mac_per_tick", 64))),
            "pack_elements_per_cycle": int(arch.get("pack_elements_per_cycle", arch.get("pack_elements_per_tick", 128))),
            "epilogue_elements_per_cycle": int(arch.get("epilogue_elements_per_cycle", arch.get("epilogue_elements_per_tick", 64))),
            "vector_elements_per_cycle": int(arch.get("vector_elements_per_cycle", arch.get("vector_elements_per_tick", 32))),
            "rmw_bandwidth_bytes_per_cycle": int(arch.get("rmw_bandwidth_bytes_per_cycle", arch.get("rmw_bandwidth_bytes_per_tick", 256))),
        },
    }


def scenario_to_yaml_dict(
    tasks: List[Dict[str, Any]],
    simulation_time: int,
    npu_size: int,
    mapping_policy: str,
    runtime_policy: str,
    switch_cost: int,
    switch_cost_mode: str,
    max_configs_per_cnn: int,
    arch: Dict[str, Any],
    gantt_time_unit: str = "tick",
) -> Dict[str, Any]:
    tick_cycles = tick_cycles_from_arch(arch)
    horizon_ticks = max(1, int(simulation_time))
    simulation_time_cycles = ticks_to_cycles(horizon_ticks, tick_cycles, minimum=1)
    tasks_cycles = normalize_task_table(tasks_ticks_to_cycles(tasks, tick_cycles))
    return {
        "time_unit": "cycles",
        "tasks_time_unit": "cycles",
        "gui_time_unit": "tick",
        "tick_cycles": tick_cycles,
        "horizon_ticks": horizon_ticks,
        "simulation_time": simulation_time_cycles,
        "simulation_time_cycles": simulation_time_cycles,
        "npu_size": npu_size,
        "runtime_policy": runtime_policy,
        "mapping_policy": mapping_policy,
        "switch_cost": switch_cost,
        "switch_cost_mode": switch_cost_mode,
        "max_configs_per_cnn": max_configs_per_cnn,
        "gantt_time_unit": gantt_time_unit,
        "gantt_format": "png",
        "open_gantt": False,
        "tasks": tasks_cycles,
        "architecture": architecture_for_yaml(arch),
    }

# ---------------------------------------------------------------------------
# Corrected MCXN GUI examples override
# ---------------------------------------------------------------------------
# The previous built-in examples were synthetic.  For the corrected MCXN
# package, expose only the real-value Offline DF/QF scenarios used in the report.

def _load_mcxn_tasks_txt(path: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 15:
                continue
            tasks.append({
                "name": p[0],
                "kind": p[1],
                "support": "CPU/NPU" if p[2].upper() == "NPU" else "CPU only",
                "resource": p[2].upper(),
                "period": int(p[3]),
                "wcet_int8": int(p[4]),
                "wcet_fp16": int(p[5]),
                "wcet_fp32": int(p[6]),
                "deadline": int(p[7]),
                "priority": int(p[8]),
                "phase": int(p[9]),
                "precision": p[10],
                "preds": p[11],
                "cnn_id": p[12],
                "criticity": p[13],
                "size": int(p[14]),
            })
    return normalize_task_table(tasks)


def scenario_examples() -> List[Dict[str, Any]]:  # type: ignore[override]
    """Corrected MCXN and synthetic divergence examples shown by the GUI.

    Examples 01-06 use the latest MCXN measurements and the measured mapping
    profiles. Examples 07-10 are synthetic one-CPU/one-NPU cases designed so
    Offline DF and Offline QF do not converge to the same final configuration.
    """
    package_root = Path(__file__).resolve().parents[1]
    examples_dir = package_root / "examples"
    specs = [
        {"dir": "ex01_mcxn_tight_10ms_offline_df", "label": "Tight 10 ms — Offline DF", "policy": "offline_df", "horizon": 400, "deadline": 100, "max_configs": 512, "description": "Corrected MCXN real-value scenario. Input tasks start all NPU/INT8. Offline DF selects among measured operator-class mapping profiles."},
        {"dir": "ex02_mcxn_tight_10ms_offline_qf", "label": "Tight 10 ms — Offline QF", "policy": "offline_qf", "horizon": 400, "deadline": 100, "max_configs": 512, "description": "Corrected MCXN real-value scenario. Same task table as the previous example, but Offline QF starts from maximum numerical quality before searching for a fixed feasible mapping."},
        {"dir": "ex03_mcxn_medium_120ms_offline_df", "label": "Medium 120 ms — Offline DF", "policy": "offline_df", "horizon": 1600, "deadline": 1200, "max_configs": 512, "description": "Corrected MCXN real-value scenario. Offline DF selects the best feasible measured mapping for the 120 ms deadline."},
        {"dir": "ex04_mcxn_medium_120ms_offline_qf", "label": "Medium 120 ms — Offline QF", "policy": "offline_qf", "horizon": 1600, "deadline": 1200, "max_configs": 512, "description": "Corrected MCXN real-value scenario. Same task table as the previous example, evaluated with Offline QF."},
        {"dir": "ex05_mcxn_loose_140ms_offline_df", "label": "Loose 140 ms — Offline DF", "policy": "offline_df", "horizon": 1800, "deadline": 1400, "max_configs": 512, "description": "Corrected MCXN real-value scenario. Offline DF can safely raise quality up to CPU FP32 under the relaxed 140 ms deadline."},
        {"dir": "ex06_mcxn_loose_140ms_offline_qf", "label": "Loose 140 ms — Offline QF", "policy": "offline_qf", "horizon": 1800, "deadline": 1400, "max_configs": 512, "description": "Corrected MCXN real-value scenario. Same task table as the previous example, evaluated with Offline QF."},
        {"dir": "ex07_divergence_two_streams_offline_df", "label": "DIVERGENCE two streams — Offline DF", "policy": "offline_df", "horizon": 400, "deadline": 167, "max_configs": 64, "description": "Synthetic two-stream contention scenario built so Offline DF and Offline QF do NOT converge. Offline DF greedily improves from the NPU/INT8 baseline and ends at CNN_A:V2 + CNN_B:V14."},
        {"dir": "ex08_divergence_two_streams_offline_qf", "label": "DIVERGENCE two streams — Offline QF", "policy": "offline_qf", "horizon": 400, "deadline": 167, "max_configs": 64, "description": "Same exact task table as the previous example. Offline QF performs a quality-first global feasible search and ends at CNN_A:V1 + CNN_B:V16, a different final configuration."},
        {"dir": "ex09_divergence_short_pair_offline_df", "label": "DIVERGENCE short pair — Offline DF", "policy": "offline_df", "horizon": 400, "deadline": 110, "max_configs": 64, "description": "Compact three-layer synthetic case. Offline DF ends at CNN_A:V2 + CNN_B:V5 with about 53.67% weighted numerical fidelity."},
        {"dir": "ex10_divergence_short_pair_offline_qf", "label": "DIVERGENCE short pair — Offline QF", "policy": "offline_qf", "horizon": 400, "deadline": 110, "max_configs": 64, "description": "Same exact task table as the previous example. Offline QF ends at CNN_A:V3 + CNN_B:V1 with about 53.76% weighted numerical fidelity, showing a visible difference from Offline DF."},
    ]
    hw = default_arch_settings()
    hw.update({
        "npu_count": 1,
        "reference_frequency_mhz": 150.0,
        "cpu_frequency_mhz": 150.0,
        "npu_frequency_mhz": 150.0,
        "interconnect_frequency_mhz": 150.0,
        "tick_cycles": 15000,
        "precision_modes": ["INT8"],
    })
    out: List[Dict[str, Any]] = []
    for spec in specs:
        dirname = spec["dir"]
        label = spec["label"]
        mapping_policy = spec["policy"]
        horizon = int(spec["horizon"])
        task_path = examples_dir / dirname / "tasks.txt"
        tasks = _load_mcxn_tasks_txt(str(task_path)) if task_path.exists() else DEFAULT_TASKS
        out.append({
            "label": label,
            "description": (
                spec["description"]
                + " Precision score is the weighted numerical bit-width score, not prediction accuracy. "
                + "1 tick = 15000 cycles = 0.1 ms at 150 MHz. Architecture: 1 CPU + 1 NPU."
            ),
            "tasks": tasks,
            "horizon_ticks": horizon,
            "simulation_time": horizon,
            "simulation_time_cycles": ticks_to_cycles(horizon, 15000, minimum=1),
            "tick_cycles": 15000,
            "npu_size": 1,
            "arch_updates": hw,
            "runtime_policy": "edf",
            "mapping_policy": mapping_policy,
            "switch_cost": 0,
            "switch_cost_mode": "per_cnn",
            "max_configs_per_cnn": int(spec.get("max_configs", 512)),
            "gantt_time_unit": "tick",
        })
    return out
