from __future__ import annotations
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

from .models import ArchConfig, ClockConfig, MappingConfig, InterconnectConfig, SystolicArrayConfig, CPUConfig, ComputeJob, Task

# Internal layer kinds used by npu_model.py.
#
# Important: GUI/YAML tasks may provide a user-facing `kind` field such as
# `conv`, `detection_head`, `nms`, etc. That explicit kind is the only source
# used to select a hardware model. Task names are labels only: a task named
# "conv" stays generic when kind is missing or explicitly set to generic.
USER_KIND_ALIASES = {
    "control": "control",
    "ctrl": "control",
    "conv": "systolic",
    "convolution": "systolic",
    "backbone": "systolic",
    "fc": "systolic",
    "fully_connected": "systolic",
    "gemm": "systolic",
    "matmul": "systolic",
    "pwconv": "systolic",
    "pointwise": "systolic",
    "pointwise_conv": "systolic",
    "detect": "systolic",
    "head": "systolic",
    "detection_head": "systolic",
    "dwconv": "depthwise",
    "depthwise": "depthwise",
    "depthwise_conv": "depthwise",
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
    "fuse": "fuse",
    "concat": "fuse",
    "merge": "fuse",
    "add": "fuse",
    "nms": "fuse",
    "skip": "fuse",
    "identity": "fuse",
    "softmax": "activation",
    "postprocess": "fuse",
    "post": "fuse",
    "generic": "generic",
    "systolic": "systolic",
}



def _arch_number(section: dict, new_key: str, old_key: str | None, default):
    if new_key in section:
        return section[new_key]
    if old_key and old_key in section:
        return section[old_key]
    return default

LAYER_PATTERNS = []  # Names are labels only; no automatic hardware-kind inference.


def normalize_layer_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    token = str(kind).strip().lower().replace("-", "_")
    if not token or token in {"-", "none", "null"}:
        return None
    return USER_KIND_ALIASES.get(token)


def infer_layer_kind_from_name(name: str) -> str:
    # Legacy files without a kind column now default to generic as well.
    # The user can opt into conv/pool/bn/etc. only through an explicit kind.
    return "generic"




def normalize_execution_precision(resource: str, precision: str, wcet_fp16: int, wcet_fp32: int) -> str:
    """Clamp file/initial precision to the hardware precision rule.

    CPU executions may use FP16 or FP32. NPU executions are INT8 only.
    Mapping algorithms apply the same rule when generating candidate options;
    this helper protects direct file-based execution as well.
    """
    res = (resource or "NPU").upper()
    p = (precision or "INT8").upper()
    if res == "CPU":
        if p in {"FP16", "FP32"}:
            return p
        return "FP32" if int(wcet_fp32) > 0 else "FP16"
    return "INT8"

def parse_tasks_file(path: str) -> Tuple[List[Task], Optional[int], Optional[int]]:
    """Parse tasks.txt and convert user timing units to internal cycles.

    Current GUI exports write task period/WCET/deadline/phase and horizon in
    internal cycles, plus tick metadata for display:
        #TIME_UNIT cycles
        #TICK_CYCLES 100000
        #HORIZON_TICKS 334
        #SIMULATION_TIME 33400000
    Legacy files without TIME_UNIT are treated as cycles. Older tick-based
    files remain supported via #TIME_UNIT tick.
    """
    tasks: List[Task] = []
    sim_time_value: Optional[int] = None
    sim_time_cycles: Optional[int] = None
    horizon_ticks: Optional[int] = None
    period_count: Optional[int] = None
    npu_size: Optional[int] = None
    time_unit = "cycles"
    tick_cycles = 1

    def to_cycles(value: int) -> int:
        if str(time_unit).lower() == "tick":
            return int(value) * max(1, int(tick_cycles))
        return int(value)

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line[1:].strip().split()
                if not parts:
                    continue
                key = parts[0].upper()
                if len(parts) >= 2 and key == "TIME_UNIT":
                    unit = parts[1].strip().lower()
                    time_unit = "tick" if unit in {"tick", "ticks"} else "cycles"
                elif len(parts) >= 2 and key in {"TICK_CYCLES", "CYCLES_PER_TICK"}:
                    tick_cycles = max(1, int(float(parts[1])))
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

            # Supported formats:
            #   legacy: name resource period wcet_int8 wcet_fp16 wcet_fp32 ... size
            #   new:    name kind resource period wcet_int8 wcet_fp16 wcet_fp32 ... size
            # The explicit kind is authoritative. If the kind column is absent
            # or unknown, default to generic; task names are labels only.
            if len(parts) >= 15 and parts[1].upper() not in {"CPU", "NPU"}:
                explicit_kind = parts[1]
                resource = parts[2].upper()
                offset = 1
            else:
                explicit_kind = None
                resource = parts[1].upper()
                offset = 0

            period = to_cycles(int(parts[2 + offset]))
            wcet_int8 = to_cycles(int(parts[3 + offset]))
            wcet_fp16 = to_cycles(int(parts[4 + offset]))
            wcet_fp32 = to_cycles(int(parts[5 + offset]))
            deadline = to_cycles(int(parts[6 + offset]))
            priority = int(parts[7 + offset])
            phase = to_cycles(int(parts[8 + offset]))
            precision = normalize_execution_precision(resource, parts[9 + offset], wcet_fp16, wcet_fp32)
            preds_field = parts[10 + offset]
            cnn_id = parts[11 + offset]
            criticity = parts[12 + offset].upper()
            size = int(parts[13 + offset])
            preds = [] if preds_field in ("-", "none", "None") else [p.strip() for p in preds_field.split(",") if p.strip()]
            layer_kind = normalize_layer_kind(explicit_kind) or "generic"
            tasks.append(Task(
                name=name,
                resource=resource,
                period=period,
                wcet_int8=wcet_int8,
                wcet_fp16=wcet_fp16,
                wcet_fp32=wcet_fp32,
                deadline=deadline,
                priority=priority,
                phase=phase,
                precision=precision,
                preds=preds,
                cnn_id=cnn_id,
                criticity=criticity,
                size=size,
                layer_kind=layer_kind,
                input_time_unit=time_unit,
                tick_cycles=max(1, int(tick_cycles)),
            ))
    if period_count is not None:
        base_period = min((t.period for t in tasks if t.period > 0), default=1)
        sim_time_cycles = max(1, int(period_count)) * base_period
    elif horizon_ticks is not None:
        sim_time_cycles = int(horizon_ticks) * max(1, int(tick_cycles))
    elif sim_time_cycles is None and sim_time_value is not None:
        sim_time_cycles = to_cycles(int(sim_time_value))
    return tasks, sim_time_cycles, npu_size

def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b) if a and b else 0


def compute_hyperperiod(tasks: List[Task]) -> int:
    periods = [t.period for t in tasks if t.period > 0]
    if not periods:
        return 0
    hp = periods[0]
    for p in periods[1:]:
        hp = lcm(hp, p)
    return hp


def generate_jobs(tasks: List[Task], horizon_ticks: int) -> List[ComputeJob]:
    jobs: List[ComputeJob] = []
    for t in tasks:
        release = t.phase
        instance = 0
        wcet = t.wcet()
        while release < horizon_ticks:
            jobs.append(ComputeJob(task=t, instance=instance, release=release, deadline=release + t.deadline, wcet=wcet))
            release += t.period
            instance += 1
    jobs.sort(key=lambda j: (j.release, j.task.priority, j.task.name, j.instance))
    return jobs


def default_arch_config() -> ArchConfig:
    return ArchConfig()


def load_arch_config(path: Optional[str]) -> ArchConfig:
    cfg = default_arch_config()
    if not path:
        return cfg
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    # New architecture files describe available hardware resources only.
    # Old files with a mapping section are still accepted for compatibility,
    # but manual task_to_npu placement is ignored by Deadline-First; unmapped NPU jobs are distributed across available NPUs.
    m = data.get("resources", data.get("mapping", {}))
    c = data.get("communications", {})
    npu = data.get("npu", {})
    cpu = data.get("cpu", {})
    clock = data.get("clock", data.get("clocks", data.get("timing", {}))) or {}

    cfg.clock = ClockConfig(
        time_base=str(clock.get("time_base", cfg.clock.time_base)),
        reference_frequency_mhz=float(clock.get("reference_frequency_mhz", clock.get("frequency_mhz", cfg.clock.reference_frequency_mhz))),
        cpu_frequency_mhz=float(clock.get("cpu_frequency_mhz", cfg.clock.cpu_frequency_mhz)),
        npu_frequency_mhz=float(clock.get("npu_frequency_mhz", cfg.clock.npu_frequency_mhz)),
        interconnect_frequency_mhz=float(clock.get("interconnect_frequency_mhz", cfg.clock.interconnect_frequency_mhz)),
        tick_cycles=max(1, int(float(clock.get("tick_cycles", clock.get("cycles_per_tick", cfg.clock.tick_cycles))))),
    )

    cfg.mapping = MappingConfig(
        cpu_name=str(m.get("cpu_name", cfg.mapping.cpu_name)),
        npu_nodes=list(m.get("npu_nodes", cfg.mapping.npu_nodes)),
        task_to_npu={},
        round_robin_unmapped=True,
    )

    cpu_npu = c.get("cpu_npu", {})
    noc = c.get("noc", {})
    cfg.interconnect = InterconnectConfig(
        cpu_npu_mode=str(cpu_npu.get("mode", cfg.interconnect.cpu_npu_mode)),
        cpu_to_npu_setup=int(cpu_npu.get("cpu_to_npu_setup", cfg.interconnect.cpu_to_npu_setup)),
        cpu_to_npu_per_unit=int(cpu_npu.get("cpu_to_npu_per_unit", cfg.interconnect.cpu_to_npu_per_unit)),
        npu_to_cpu_setup=int(cpu_npu.get("npu_to_cpu_setup", cfg.interconnect.npu_to_cpu_setup)),
        npu_to_cpu_per_unit=int(cpu_npu.get("npu_to_cpu_per_unit", cfg.interconnect.npu_to_cpu_per_unit)),
        shared_setup=int(cpu_npu.get("shared_setup", cfg.interconnect.shared_setup)),
        shared_per_unit=int(cpu_npu.get("shared_per_unit", cfg.interconnect.shared_per_unit)),
        noc_topology=str(noc.get("topology", cfg.interconnect.noc_topology)),
        noc_setup=int(noc.get("setup", cfg.interconnect.noc_setup)),
        noc_per_unit=int(noc.get("per_unit", cfg.interconnect.noc_per_unit)),
        router_latency=int(noc.get("router_latency", cfg.interconnect.router_latency)),
        arb_policy=str(noc.get("arb_policy", cfg.interconnect.arb_policy)),
        noc_rr_quantum_units=int(noc.get("rr_quantum_units", cfg.interconnect.noc_rr_quantum_units)),
    )

    cfg.npu = SystolicArrayConfig(
        rows=int(npu.get("sa_rows", cfg.npu.rows)),
        cols=int(npu.get("sa_cols", cfg.npu.cols)),
        arrays=int(npu.get("systolic_arrays", cfg.npu.arrays)),
        setup_cycles=int(npu.get("setup_cycles", cfg.npu.setup_cycles)),
        precision_modes=("INT8",),
        local_dram_capacity_kb=int(npu.get("local_dram_capacity_kb", cfg.npu.local_dram_capacity_kb)),
        local_dram_latency=int(npu.get("local_dram_latency", cfg.npu.local_dram_latency)),
        local_dram_bandwidth_bytes_per_tick=int(_arch_number(npu, "local_dram_bandwidth_bytes_per_cycle", "local_dram_bandwidth_bytes_per_tick", cfg.npu.local_dram_bandwidth_bytes_per_tick)),
        dma_setup=int(npu.get("dma_setup", cfg.npu.dma_setup)),
        dma_bandwidth_bytes_per_tick=int(_arch_number(npu, "dma_bandwidth_bytes_per_cycle", "dma_bandwidth_bytes_per_tick", cfg.npu.dma_bandwidth_bytes_per_tick)),
        vector_lanes=int(npu.get("vector_lanes", cfg.npu.vector_lanes)),
        vector_setup=int(npu.get("vector_setup", cfg.npu.vector_setup)),
    )

    cfg.cpu = CPUConfig(
        setup_cycles=int(cpu.get("setup_cycles", cfg.cpu.setup_cycles)),
        cache_capacity_kb=int(cpu.get("cache_capacity_kb", cfg.cpu.cache_capacity_kb)),
        cache_latency=int(cpu.get("cache_latency", cfg.cpu.cache_latency)),
        cache_bandwidth_bytes_per_tick=int(_arch_number(cpu, "cache_bandwidth_bytes_per_cycle", "cache_bandwidth_bytes_per_tick", cfg.cpu.cache_bandwidth_bytes_per_tick)),
        simd_mac_per_tick=int(_arch_number(cpu, "simd_mac_per_cycle", "simd_mac_per_tick", cfg.cpu.simd_mac_per_tick)),
        pack_elements_per_tick=int(_arch_number(cpu, "pack_elements_per_cycle", "pack_elements_per_tick", cfg.cpu.pack_elements_per_tick)),
        epilogue_elements_per_tick=int(_arch_number(cpu, "epilogue_elements_per_cycle", "epilogue_elements_per_tick", cfg.cpu.epilogue_elements_per_tick)),
        vector_elements_per_tick=int(_arch_number(cpu, "vector_elements_per_cycle", "vector_elements_per_tick", cfg.cpu.vector_elements_per_tick)),
        rmw_bandwidth_bytes_per_tick=int(_arch_number(cpu, "rmw_bandwidth_bytes_per_cycle", "rmw_bandwidth_bytes_per_tick", cfg.cpu.rmw_bandwidth_bytes_per_tick)),
    )
    return cfg
