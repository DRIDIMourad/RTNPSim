from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PRECISION_SCALES = {"INT8": 1, "FP16": 2, "FP32": 4}


@dataclass
class Task:
    name: str
    resource: str
    period: int
    wcet_int8: int
    wcet_fp16: int
    wcet_fp32: int
    deadline: int
    priority: int
    phase: int
    precision: str
    preds: List[str]
    cnn_id: str
    criticity: str
    size: int
    layer_kind: str = "generic"
    # Extra runtime penalty injected by Deadline-First when a layer changes
    # resource/precision configuration. By default it applies to instance 0
    # for offline mapping; online DF can set switch_instance to a later period.
    switch_cost: int = 0
    switch_instance: Optional[int] = 0

    # Metadata used only for reports/Gantt when Deadline-First runs online
    # period by period. It does not affect the execution model.
    df_period: int = -1
    df_config: str = ""
    df_action: str = ""
    df_changed: bool = False

    # Input-file timing metadata. Tasks are converted to internal reference cycles
    # by the parser; this keeps the original user unit available for reports/Gantt.
    input_time_unit: str = "cycles"
    tick_cycles: int = 10000

    def wcet(self) -> int:
        p = self.precision.upper()
        if p == "INT8":
            return self.wcet_int8
        if p == "FP16":
            return self.wcet_fp16
        return self.wcet_fp32

    def precision_scale(self) -> int:
        return PRECISION_SCALES.get(self.precision.upper(), 1)


@dataclass
class StageSegment:
    label: str
    duration: int
    start: Optional[int] = None
    finish: Optional[int] = None
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class MessageSegment:
    start: int
    finish: int
    payload_units: int
    cost: int
    route_links: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class ComputeJob:
    task: Task
    instance: int
    release: int
    deadline: int
    wcet: int
    mapped_resource: str = ""
    start: Optional[int] = None
    finish: Optional[int] = None
    chain_root_release: Optional[int] = None
    local_pred_ids: List[str] = field(default_factory=list)
    incoming_msg_ids: List[str] = field(default_factory=list)
    direct_predecessors: List[str] = field(default_factory=list)
    stages: List[StageSegment] = field(default_factory=list)
    inferred_payload_units: int = 0
    notes: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        return f"{self.task.name}[{self.instance}]"

    @property
    def response_time(self) -> Optional[int]:
        return None if self.finish is None else self.finish - self.release


@dataclass
class MessageJob:
    msg_id: str
    pred_job_id: str
    dst_job_id: str
    src_resource: str
    dst_resource: str
    medium: str
    payload_units: int
    cost: int
    priority: int
    deadline: int
    period: int
    release: Optional[int] = None
    start: Optional[int] = None
    finish: Optional[int] = None
    route_nodes: List[str] = field(default_factory=list)
    route_links: List[Tuple[str, str]] = field(default_factory=list)
    priority_key: Tuple = field(default_factory=tuple)
    notes: List[str] = field(default_factory=list)
    remaining_payload_units: int = 0
    transfer_segments: List[MessageSegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.remaining_payload_units <= 0:
            self.remaining_payload_units = self.payload_units

    @property
    def response_time(self) -> Optional[int]:
        if self.release is None or self.finish is None:
            return None
        return self.finish - self.release




@dataclass
class ClockConfig:
    # All simulated durations are counted in reference cycles.
    # Frequencies are used for display/conversion only unless a future
    # architecture model explicitly scales per-resource timings.
    time_base: str = "cycles"
    reference_frequency_mhz: float = 1000.0
    cpu_frequency_mhz: float = 1000.0
    npu_frequency_mhz: float = 1000.0
    interconnect_frequency_mhz: float = 1000.0
    # User-facing tick size for compact task files/Gantt display.
    # Internal simulation still uses reference cycles.
    tick_cycles: int = 10000


@dataclass
class SystolicArrayConfig:
    rows: int = 16
    cols: int = 16
    arrays: int = 1
    setup_cycles: int = 20
    precision_modes: Tuple[str, ...] = ("INT8",)
    local_dram_capacity_kb: int = 1024
    local_dram_latency: int = 120
    local_dram_bandwidth_bytes_per_tick: int = 16
    dma_setup: int = 120
    dma_bandwidth_bytes_per_tick: int = 16
    vector_lanes: int = 64
    vector_setup: int = 20


@dataclass
class CPUConfig:
    setup_cycles: int = 1
    cache_capacity_kb: int = 256
    cache_latency: int = 40
    cache_bandwidth_bytes_per_tick: int = 16
    simd_mac_per_tick: int = 8
    pack_elements_per_tick: int = 32
    epilogue_elements_per_tick: int = 32
    vector_elements_per_tick: int = 8
    rmw_bandwidth_bytes_per_tick: int = 16


@dataclass
class InterconnectConfig:
    cpu_npu_mode: str = "separate_full_duplex"
    cpu_to_npu_setup: int = 40
    cpu_to_npu_per_unit: int = 1
    npu_to_cpu_setup: int = 40
    npu_to_cpu_per_unit: int = 1
    shared_setup: int = 40
    shared_per_unit: int = 1
    noc_topology: str = "mesh"
    noc_setup: int = 30
    noc_per_unit: int = 1
    router_latency: int = 3
    arb_policy: str = "fixed_priority"
    noc_rr_quantum_units: int = 64


@dataclass
class MappingConfig:
    cpu_name: str = "CPU"
    npu_nodes: List[str] = field(default_factory=lambda: ["NPU0"])
    task_to_npu: Dict[str, str] = field(default_factory=dict)
    round_robin_unmapped: bool = True


@dataclass
class ArchConfig:
    clock: ClockConfig = field(default_factory=ClockConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    interconnect: InterconnectConfig = field(default_factory=InterconnectConfig)
    npu: SystolicArrayConfig = field(default_factory=SystolicArrayConfig)
    cpu: CPUConfig = field(default_factory=CPUConfig)


@dataclass
class SimulationResult:
    compute_jobs: List[ComputeJob]
    message_jobs: List[MessageJob]
    stats: Dict[str, float]
