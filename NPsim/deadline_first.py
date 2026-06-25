from __future__ import annotations

from collections import defaultdict
import math
import re
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import ArchConfig, ComputeJob, SimulationResult, Task
from .npu_model import build_cpu_stages, build_npu_stages
from .parser import generate_jobs
from .simulator import simulate, validate_schedule


PRECISION_ORDER: Tuple[str, ...] = ("INT8", "FP16", "FP32")
# Numerical bit-width used by the quality objective.
# This is not classification accuracy. It implements:
# EffectivePrecision = 100 * sum(w_i * b_i) / (32 * sum(w_i)),
# with w_i taken from the CPU/FP32 WCET of each layer.
PRECISION_BITWIDTH: Dict[str, int] = {"INT8": 8, "FP16": 16, "FP32": 32}
CPU_EXECUTION_PRECISIONS: Tuple[str, ...] = ("FP16", "FP32")
NPU_EXECUTION_PRECISIONS: Tuple[str, ...] = ("INT8",)


def _scale_profile_cycles(cycles: int, reference_mhz: float, local_mhz: float) -> int:
    cycles = int(cycles)
    if cycles <= 0:
        return 0
    if reference_mhz <= 0 or local_mhz <= 0:
        return cycles
    return max(1, int(math.ceil(cycles * reference_mhz / local_mhz)))


@dataclass(frozen=True)
class LayerOption:
    """One executable choice for one CNN layer."""

    task_name: str
    resource: str
    precision: str
    cost: int
    accuracy_score: float

    @property
    def key(self) -> Tuple[str, str]:
        return (self.resource.upper(), self.precision.upper())


@dataclass(frozen=True)
class CNNConfig:
    """A CNN-level configuration: resource/precision choice for every layer."""

    cnn_id: str
    name: str
    options: Tuple[LayerOption, ...]
    cost: int
    accuracy_score: float

    @property
    def by_task(self) -> Dict[str, LayerOption]:
        return {o.task_name: o for o in self.options}

    def differs_from(self, other: "CNNConfig") -> bool:
        mine = self.by_task
        theirs = other.by_task
        if mine.keys() != theirs.keys():
            return True
        return any(mine[k].key != theirs[k].key for k in mine)

    def changed_layers_from(self, other: "CNNConfig") -> List[str]:
        mine = self.by_task
        theirs = other.by_task
        changed = []
        for k in mine:
            if k not in theirs or mine[k].key != theirs[k].key:
                changed.append(k)
        return changed


@dataclass
class MappingEvaluation:
    selection: Dict[str, CNNConfig]
    result: SimulationResult
    issues: List[str]
    deadline_misses: List[str]
    unfinished: List[str]
    total_accuracy_score: float
    total_nominal_cost: int
    worst_lateness: int
    sum_lateness: int

    @property
    def feasible(self) -> bool:
        return not self.deadline_misses and not self.unfinished and not self.issues


@dataclass
class CandidateTrace:
    phase: str
    cnn_id: str
    from_config: str
    to_config: str
    delta_accuracy: float
    delta_cost: int
    ratio: float
    accepted: bool
    miss_count: int
    worst_lateness: int
    note: str = ""


@dataclass
class PeriodStep:
    """Configuration actually executed by one periodic instance in online DF."""

    period_index: int
    selection: Dict[str, CNNConfig]
    action: str
    changed_cnn: str = ""
    from_config: str = ""
    to_config: str = ""
    accepted: bool = True
    miss_count: int = 0
    worst_lateness: int = 0
    note: str = ""


@dataclass
class DeadlineFirstMappingResult:
    tasks: List[Task]
    selection: Dict[str, CNNConfig]
    initial_selection: Dict[str, CNNConfig]
    configs_by_cnn: Dict[str, List[CNNConfig]]
    evaluation: MappingEvaluation
    trace: List[CandidateTrace] = field(default_factory=list)
    repair_used: bool = False
    global_search_used: bool = False
    schedulable: bool = True
    normalization_accuracy_score: float = 1.0
    period_steps: List[PeriodStep] = field(default_factory=list)

    @property
    def normalized_accuracy_percent(self) -> float:
        if self.normalization_accuracy_score <= 0:
            return 0.0
        return 100.0 * self.evaluation.total_accuracy_score / self.normalization_accuracy_score


def _available_precisions(task: Task) -> List[str]:
    vals = {
        "INT8": task.wcet_int8,
        "FP16": task.wcet_fp16,
        "FP32": task.wcet_fp32,
    }
    return [p for p in PRECISION_ORDER if vals.get(p, 0) > 0]


def _task_supports_npu(task: Task) -> bool:
    # In the input format, resource=CPU means the layer is mandatory CPU
    # (e.g. Softmax, LayerNorm, control code, unsupported operator). A layer
    # marked NPU can still be mapped back to CPU by the mapper.
    return task.resource.upper() != "CPU"


def _accuracy_score(task: Task, precision: str, resource: str) -> float:
    """Weighted numerical precision objective.

    The mapper does not use prediction accuracy. Instead, it maximizes the
    effective numerical precision used in the report:

        score = sum_i(w_i * b_i)

    where b_i is the execution bit-width (32 for FP32, 16 for FP16, 8 for
    INT8), and w_i is the layer workload. For the MCXN-calibrated files, the
    best available workload proxy is the CPU/FP32 WCET column, because it was
    calibrated from measured board execution. Normalizing by the all-FP32
    score gives 100% for CPU/FP32 and 25% for full INT8.
    """
    bitwidth = PRECISION_BITWIDTH.get(precision.upper(), 8)
    weight = int(task.wcet_fp32) if int(task.wcet_fp32) > 0 else max(1, int(task.size))
    return float(max(1, weight) * bitwidth)


def _profile_cost(task: Task, resource: str, precision: str, arch: ArchConfig) -> int:
    profiled = replace(task, resource=resource.upper(), precision=precision.upper(), switch_cost=0)
    job = ComputeJob(
        task=profiled,
        instance=0,
        release=0,
        deadline=profiled.deadline,
        wcet=profiled.wcet(),
        mapped_resource=arch.mapping.cpu_name if resource.upper() == "CPU" else (arch.mapping.npu_nodes[0] if arch.mapping.npu_nodes else "NPU0"),
    )
    if resource.upper() == "CPU":
        stages, _ = build_cpu_stages(job, arch.cpu, arch.npu)
        local_mhz = arch.clock.cpu_frequency_mhz
    else:
        stages, _ = build_npu_stages(job, arch.npu)
        local_mhz = arch.clock.npu_frequency_mhz
    local_cycles = sum(int(s.duration) for s in stages)
    return max(1, _scale_profile_cycles(local_cycles, arch.clock.reference_frequency_mhz, local_mhz))


def _layer_options(task: Task, arch: ArchConfig) -> List[LayerOption]:
    options: List[LayerOption] = []
    precisions = _available_precisions(task)
    npu_precisions = {p.upper() for p in arch.npu.precision_modes}

    # Hardware precision rule, enforced by the mapper rather than inferred from
    # names: CPU can execute FP16 or FP32, while every NPU execution is INT8.
    # This applies to generic WCET-only tasks and to non-generic hardware-model
    # tasks. The INT8 column is therefore the accelerator profile; FP16/FP32 are
    # CPU profiles.
    cpu_precisions = [p for p in precisions if p.upper() in CPU_EXECUTION_PRECISIONS]
    legal_npu_precisions = [p for p in precisions if p.upper() in NPU_EXECUTION_PRECISIONS and p.upper() in npu_precisions]

    for precision in cpu_precisions:
        options.append(
            LayerOption(
                task_name=task.name,
                resource="CPU",
                precision=precision,
                cost=_profile_cost(task, "CPU", precision, arch),
                accuracy_score=_accuracy_score(task, precision, "CPU"),
            )
        )

    if _task_supports_npu(task):
        for precision in legal_npu_precisions:
            options.append(
                LayerOption(
                    task_name=task.name,
                    resource="NPU",
                    precision=precision,
                    cost=_profile_cost(task, "NPU", precision, arch),
                    accuracy_score=_accuracy_score(task, precision, "NPU"),
                )
            )

    if not options:
        # Defensive fallback for malformed input. Preserve the hardware rule:
        # CPU never falls back to INT8, and NPU never falls back to FP16/FP32.
        if task.resource.upper() == "CPU" or not _task_supports_npu(task):
            precision = "FP32" if task.wcet_fp32 > 0 else "FP16"
            resource = "CPU"
        else:
            precision = "INT8"
            resource = "NPU"
        options.append(
            LayerOption(
                task_name=task.name,
                resource=resource,
                precision=precision,
                cost=_profile_cost(task, resource, precision, arch),
                accuracy_score=_accuracy_score(task, precision, resource),
            )
        )

    # Remove duplicate resource/precision pairs, keeping the cheapest cost.
    dedup: Dict[Tuple[str, str], LayerOption] = {}
    for opt in options:
        prev = dedup.get(opt.key)
        if prev is None or opt.cost < prev.cost:
            dedup[opt.key] = opt
    return sorted(dedup.values(), key=lambda o: (o.cost, -o.accuracy_score, o.resource, o.precision))

def _prune_layer_options(options: List[LayerOption]) -> List[LayerOption]:
    # Keep non-dominated options but do NOT let a CPU option eliminate an NPU
    # option, or vice versa. A resource that is slower in isolation can still
    # be useful online because it reduces contention on the other resource.
    kept: List[LayerOption] = []
    for opt in options:
        dominated = False
        for other in options:
            if other is opt or other.resource.upper() != opt.resource.upper():
                continue
            if other.cost <= opt.cost and other.accuracy_score >= opt.accuracy_score:
                if other.cost < opt.cost or other.accuracy_score > opt.accuracy_score:
                    dominated = True
                    break
        if not dominated:
            kept.append(opt)
    return sorted(kept, key=lambda o: (o.cost, -o.accuracy_score, o.resource, o.precision))


def _prune_cnn_configs(configs: List[CNNConfig], max_configs: int) -> List[CNNConfig]:
    # Pareto prune by nominal compute cost and accuracy score, while preserving
    # different CPU/NPU placement signatures. A mapping with higher nominal
    # compute cost may reduce online interference and become feasible.
    unique: Dict[Tuple[Tuple[str, str, str], ...], CNNConfig] = {}
    for cfg in configs:
        key = tuple((o.task_name, o.resource, o.precision) for o in cfg.options)
        prev = unique.get(key)
        if prev is None or (cfg.cost, -cfg.accuracy_score) < (prev.cost, -prev.accuracy_score):
            unique[key] = cfg

    grouped: Dict[Tuple[Tuple[str, str], ...], List[CNNConfig]] = defaultdict(list)
    for cfg in unique.values():
        resource_signature = tuple((o.task_name, o.resource.upper()) for o in cfg.options)
        grouped[resource_signature].append(cfg)

    kept: List[CNNConfig] = []
    for arr in grouped.values():
        arr.sort(key=lambda c: (c.cost, -c.accuracy_score, c.name))
        best_accuracy = float("-inf")
        for cfg in arr:
            if cfg.accuracy_score > best_accuracy:
                kept.append(cfg)
                best_accuracy = cfg.accuracy_score

    kept.sort(key=lambda c: (c.cost, -c.accuracy_score, c.name))
    if len(kept) <= max_configs:
        return _rename_configs(kept)

    # If the resource-aware frontier is still large, preserve cheap mappings,
    # accurate mappings and good accuracy/cost compromises.
    fastest = kept[: max(1, max_configs // 3)]
    most_accurate = sorted(kept, key=lambda c: (-c.accuracy_score, c.cost))[: max(1, max_configs // 3)]
    middle_budget = max_configs - len({id(x) for x in fastest + most_accurate})
    middle = sorted(kept, key=lambda c: (-(c.accuracy_score / max(1, c.cost)), c.cost))[: max(0, middle_budget)]
    selected_by_key: Dict[Tuple[Tuple[str, str, str], ...], CNNConfig] = {}
    for cfg in fastest + most_accurate + middle:
        key = tuple((o.task_name, o.resource, o.precision) for o in cfg.options)
        selected_by_key[key] = cfg
    selected = sorted(selected_by_key.values(), key=lambda c: (c.cost, -c.accuracy_score, c.name))[:max_configs]
    return _rename_configs(selected)


def _rename_configs(configs: List[CNNConfig]) -> List[CNNConfig]:
    out: List[CNNConfig] = []
    for idx, cfg in enumerate(sorted(configs, key=lambda c: (c.cost, -c.accuracy_score, c.name)), start=1):
        out.append(CNNConfig(cnn_id=cfg.cnn_id, name=f"V{idx}", options=cfg.options, cost=cfg.cost, accuracy_score=cfg.accuracy_score))
    return out




def _mapping_group_name(task: Task) -> str:
    """Return the mapping group for a layer.

    The MCXN/eIQ experiments convert operator classes, not arbitrary individual
    layer instances. For example, the mapping `npu_conv_only` maps all CONV_2D
    instances together. Grouping by operator type prevents NPsim from selecting
    unrealistic configurations such as moving only one convolution to NPU.
    """
    name = task.name.upper()
    m = re.match(r"OP\d+_(.+)", name)
    if m:
        return m.group(1)
    return name

def _group_options(group_tasks: List[Task], arch: ArchConfig) -> List[Tuple[Tuple[LayerOption, ...], int, float]]:
    """Build legal group-level options.

    Every task in the same TFLite operator class receives the same
    resource/precision choice. The cost and precision score are summed over all
    tasks in the group.
    """
    per_task: List[List[LayerOption]] = [_prune_layer_options(_layer_options(t, arch)) for t in group_tasks]
    common_keys = set(o.key for o in per_task[0]) if per_task else set()
    for opts in per_task[1:]:
        common_keys &= set(o.key for o in opts)
    out: List[Tuple[Tuple[LayerOption, ...], int, float]] = []
    for key in sorted(common_keys, key=lambda k: (0 if k == ("NPU", "INT8") else 1, k)):
        selected: List[LayerOption] = []
        for opts in per_task:
            match = [o for o in opts if o.key == key]
            if not match:
                selected = []
                break
            selected.append(match[0])
        if selected:
            out.append((tuple(selected), sum(o.cost for o in selected), sum(o.accuracy_score for o in selected)))
    return out



MCXN_ALLOWED_NPU_PROFILES = {
    frozenset(),
    frozenset({"FULLY_CONNECTED"}),
    frozenset({"CONV_2D"}),
    frozenset({"CONV_2D", "FULLY_CONNECTED"}),
    frozenset({"CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED"}),
    # Measured intermediate profile: all convolutions, all depthwise
    # convolutions and the final fully connected layer are mapped to NPU/INT8.
    # Pooling, MEAN and SOFTMAX remain on CPU/FP32.
    frozenset({"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}),
    frozenset({"CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"}),
}

# Real measured MCXN947 inference times from the new uploaded package.
# Values are converted to reference cycles at 150 MHz. NPsim still uses the
# original layer-level tasks for Gantt display, but a selected profile is scaled
# to this measured whole-CNN cost during materialization.
MCXN_PROFILE_REAL_COST_CYCLES = {
    frozenset(): 18_504_000,  # CPU FP32, 123.36 ms
    frozenset({"FULLY_CONNECTED"}): 19_989_136,  # NPU FC, 133.2609 ms
    frozenset({"CONV_2D"}): 11_141_086,  # NPU CONV, 74.2739 ms
    frozenset({"CONV_2D", "FULLY_CONNECTED"}): 11_126_945,  # NPU CONV+FC, 74.1796 ms
    frozenset({"CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED"}): 10_490_345,  # NPU CONV+POOL+FC, 69.9356 ms
    frozenset({"CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED"}): 7_947_000,  # NPU CONV+DW+FC, 52.98 ms
    frozenset({"CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"}): 513_286,  # all forced, 3.4219 ms
}

def _mcxn_profile_signature(cfg: CNNConfig) -> frozenset[str]:
    groups = set()
    for opt in cfg.options:
        if opt.resource.upper() == "NPU":
            # Derive group from the task name in the same way as _mapping_group_name.
            name = opt.task_name.upper()
            m = re.match(r"OP\d+_(.+)", name)
            groups.add(m.group(1) if m else name)
    return frozenset(groups)

def _looks_like_mcxn_gtsrb(groups: Dict[str, List[Task]]) -> bool:
    required = {"CONV_2D", "DEPTHWISE_CONV_2D", "MAX_POOL_2D", "MEAN", "FULLY_CONNECTED", "SOFTMAX"}
    return required.issubset(set(groups.keys()))


def _apply_mcxn_real_profile_costs(configs: List[CNNConfig]) -> List[CNNConfig]:
    out: List[CNNConfig] = []
    for cfg in configs:
        sig = _mcxn_profile_signature(cfg)
        real_cost = MCXN_PROFILE_REAL_COST_CYCLES.get(sig)
        if real_cost is None:
            continue
        out.append(CNNConfig(cnn_id=cfg.cnn_id, name=cfg.name, options=cfg.options, cost=int(real_cost), accuracy_score=cfg.accuracy_score))
    return out

def build_cnn_configurations(tasks: Sequence[Task], arch: ArchConfig, max_configs_per_cnn: int = 512) -> Dict[str, List[CNNConfig]]:
    by_cnn: Dict[str, List[Task]] = defaultdict(list)
    for task in tasks:
        by_cnn[task.cnn_id or task.name].append(task)

    configs_by_cnn: Dict[str, List[CNNConfig]] = {}
    for cnn_id, cnn_tasks in by_cnn.items():
        # Group tasks by operator type so the mapper chooses real conversion
        # profiles such as CONV-only, CONV+FC, CONV+POOL+FC, etc.
        groups: Dict[str, List[Task]] = defaultdict(list)
        group_order: List[str] = []
        for task in cnn_tasks:
            g = _mapping_group_name(task)
            if g not in groups:
                group_order.append(g)
            groups[g].append(task)

        partials: List[Tuple[Tuple[LayerOption, ...], int, float]] = [(tuple(), 0, 0.0)]
        for group_name in group_order:
            opts_for_group = _group_options(groups[group_name], arch)
            next_partials: List[Tuple[Tuple[LayerOption, ...], int, float]] = []
            for prefix, prefix_cost, prefix_acc in partials:
                for group_opts, group_cost, group_acc in opts_for_group:
                    next_partials.append((prefix + group_opts, prefix_cost + group_cost, prefix_acc + group_acc))

            temp = [CNNConfig(cnn_id=cnn_id, name="", options=o, cost=c, accuracy_score=a) for o, c, a in next_partials]
            temp = _prune_cnn_configs(temp, max(max_configs_per_cnn * 2, 8))
            partials = [(cfg.options, cfg.cost, cfg.accuracy_score) for cfg in temp]

        configs = [CNNConfig(cnn_id=cnn_id, name="", options=o, cost=c, accuracy_score=a) for o, c, a in partials]
        if _looks_like_mcxn_gtsrb(groups):
            # The real eIQ/MCXN experiments were generated only for a finite set
            # of operator-class mappings. Do not let NPsim invent unmeasured
            # configurations such as POOL-only or DW-only. Then replace nominal
            # additive costs by measured whole-CNN MCXN costs.
            configs = [cfg for cfg in configs if _mcxn_profile_signature(cfg) in MCXN_ALLOWED_NPU_PROFILES]
            configs = _apply_mcxn_real_profile_costs(configs)
        configs_by_cnn[cnn_id] = _prune_cnn_configs(configs, max_configs_per_cnn)

    return configs_by_cnn


def _fastest_selection(configs_by_cnn: Dict[str, List[CNNConfig]]) -> Dict[str, CNNConfig]:
    return {cnn_id: min(configs, key=lambda c: (c.cost, -c.accuracy_score)) for cnn_id, configs in configs_by_cnn.items()}


def _all_resource_configs(configs: List[CNNConfig], resource: str) -> List[CNNConfig]:
    target = resource.upper()
    return [cfg for cfg in configs if all(opt.resource.upper() == target for opt in cfg.options)]


def _all_cpu_high_quality_selection(configs_by_cnn: Dict[str, List[CNNConfig]]) -> Dict[str, CNNConfig]:
    """QF start point: all layers on CPU, highest available quality.

    This is intentionally independent from task names. If a CNN has no strict
    all-CPU configuration because of malformed input, it falls back to the best
    quality configuration so the simulator can still run.
    """
    out: Dict[str, CNNConfig] = {}
    for cnn_id, configs in configs_by_cnn.items():
        cpu_only = _all_resource_configs(configs, "CPU")
        pool = cpu_only if cpu_only else configs
        out[cnn_id] = max(pool, key=lambda c: (c.accuracy_score, -c.cost, c.name))
    return out


def _all_npu_low_quality_selection(configs_by_cnn: Dict[str, List[CNNConfig]]) -> Dict[str, CNNConfig]:
    """DF start point: NPU-heavy, lowest/cheapest quality.

    CPU-only layers remain CPU because they have no NPU option. For NPU-capable
    generic layers this selects the all-NPU/low-precision corner.
    """
    return _npu_first_selection(configs_by_cnn)


def _max_accuracy_score(configs_by_cnn: Dict[str, List[CNNConfig]]) -> float:
    return sum(max(c.accuracy_score for c in configs) for configs in configs_by_cnn.values())


def _selection_accuracy(selection: Dict[str, CNNConfig]) -> float:
    return sum(cfg.accuracy_score for cfg in selection.values())


def _selection_cost(selection: Dict[str, CNNConfig]) -> int:
    return sum(cfg.cost for cfg in selection.values())


def _materialize_tasks(
    tasks: Sequence[Task],
    selection: Dict[str, CNNConfig],
    previous_selection: Optional[Dict[str, CNNConfig]],
    switch_cost: int,
    switch_cost_mode: str,
) -> List[Task]:
    selected_by_task: Dict[str, LayerOption] = {}
    selected_scale_by_task: Dict[str, float] = {}
    previous_by_task: Dict[str, LayerOption] = {}
    for cfg in selection.values():
        selected_by_task.update(cfg.by_task)
        nominal = max(1, sum(int(opt.cost) for opt in cfg.options))
        scale = float(cfg.cost) / float(nominal)
        for opt in cfg.options:
            selected_scale_by_task[opt.task_name] = scale
    if previous_selection:
        for cfg in previous_selection.values():
            previous_by_task.update(cfg.by_task)

    charged_cnn: set[str] = set()
    out: List[Task] = []
    for task in tasks:
        opt = selected_by_task.get(task.name)
        if opt is None:
            out.append(task)
            continue

        sw = 0
        prev = previous_by_task.get(task.name)
        changed = prev is not None and prev.key != opt.key
        if changed and switch_cost > 0:
            mode = switch_cost_mode.lower()
            if mode == "per_layer":
                sw = switch_cost
            else:
                cnn_id = task.cnn_id or task.name
                if cnn_id not in charged_cnn:
                    sw = switch_cost
                    charged_cnn.add(cnn_id)

        scale = selected_scale_by_task.get(task.name, 1.0)
        new_wcet_int8 = task.wcet_int8
        new_wcet_fp16 = task.wcet_fp16
        new_wcet_fp32 = task.wcet_fp32
        if opt.precision.upper() == "INT8":
            new_wcet_int8 = max(1, int(round(task.wcet_int8 * scale)))
        elif opt.precision.upper() == "FP16":
            new_wcet_fp16 = max(1, int(round(task.wcet_fp16 * scale)))
        else:
            new_wcet_fp32 = max(1, int(round(task.wcet_fp32 * scale)))
        out.append(replace(task, resource=opt.resource.upper(), precision=opt.precision.upper(), wcet_int8=new_wcet_int8, wcet_fp16=new_wcet_fp16, wcet_fp32=new_wcet_fp32, switch_cost=sw))
    return out


def evaluate_selection(
    tasks: Sequence[Task],
    selection: Dict[str, CNNConfig],
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    previous_selection: Optional[Dict[str, CNNConfig]] = None,
    switch_cost: int = 0,
    switch_cost_mode: str = "per_cnn",
) -> MappingEvaluation:
    eval_tasks = _materialize_tasks(tasks, selection, previous_selection, switch_cost, switch_cost_mode)
    jobs = generate_jobs(eval_tasks, sim_time)
    result = simulate(jobs, arch, policy=policy)
    issues = validate_schedule(result)

    deadline_misses: List[str] = []
    unfinished: List[str] = []
    worst_lateness = 0
    sum_lateness = 0
    for job in result.compute_jobs:
        if job.finish is None:
            unfinished.append(job.job_id)
            worst_lateness = max(worst_lateness, job.deadline - job.release + 1)
            sum_lateness += max(1, job.deadline - job.release + 1)
            continue
        lateness = max(0, job.finish - job.deadline)
        if lateness > 0:
            deadline_misses.append(job.job_id)
            worst_lateness = max(worst_lateness, lateness)
            sum_lateness += lateness

    return MappingEvaluation(
        selection=selection,
        result=result,
        issues=issues,
        deadline_misses=deadline_misses,
        unfinished=unfinished,
        total_accuracy_score=_selection_accuracy(selection),
        total_nominal_cost=_selection_cost(selection),
        worst_lateness=worst_lateness,
        sum_lateness=sum_lateness,
    )


def _candidate_ratio(current: CNNConfig, candidate: CNNConfig, switch_cost: int) -> Tuple[float, float, int]:
    delta_accuracy = candidate.accuracy_score - current.accuracy_score
    delta_cost = max(0, candidate.cost - current.cost) + (switch_cost if candidate.differs_from(current) else 0)
    if delta_cost <= 0:
        ratio = float("inf") if delta_accuracy > 0 else 0.0
    else:
        ratio = delta_accuracy / delta_cost
    return ratio, delta_accuracy, delta_cost


def _repair_selection(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    current_selection: Dict[str, CNNConfig],
    current_eval: MappingEvaluation,
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
    trace: List[CandidateTrace],
    max_iterations: int,
) -> Tuple[Dict[str, CNNConfig], MappingEvaluation, bool]:
    used_repair = False
    for _ in range(max_iterations):
        if current_eval.feasible:
            break

        best: Optional[Tuple[Tuple[int, int, int, float, int], str, CNNConfig, MappingEvaluation, float, float, int]] = None
        for cnn_id, cfgs in configs_by_cnn.items():
            old = current_selection[cnn_id]
            for candidate in cfgs:
                if candidate.name == old.name:
                    continue
                trial_selection = dict(current_selection)
                trial_selection[cnn_id] = candidate
                trial_eval = evaluate_selection(
                    tasks,
                    trial_selection,
                    arch,
                    sim_time,
                    policy,
                    previous_selection=current_selection,
                    switch_cost=switch_cost,
                    switch_cost_mode=switch_cost_mode,
                )
                ratio, delta_accuracy, delta_cost = _candidate_ratio(old, candidate, switch_cost)
                key = (
                    len(trial_eval.deadline_misses) + len(trial_eval.unfinished) + len(trial_eval.issues),
                    trial_eval.worst_lateness,
                    trial_eval.sum_lateness,
                    -trial_eval.total_accuracy_score,
                    trial_eval.total_nominal_cost,
                )
                if best is None or key < best[0]:
                    best = (key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost)

        if best is None:
            break

        old_eval_key = (
            len(current_eval.deadline_misses) + len(current_eval.unfinished) + len(current_eval.issues),
            current_eval.worst_lateness,
            current_eval.sum_lateness,
            -current_eval.total_accuracy_score,
            current_eval.total_nominal_cost,
        )
        key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost = best
        accepted = key < old_eval_key
        trace.append(
            CandidateTrace(
                phase="repair",
                cnn_id=cnn_id,
                from_config=current_selection[cnn_id].name,
                to_config=candidate.name,
                delta_accuracy=delta_accuracy,
                delta_cost=delta_cost,
                ratio=ratio,
                accepted=accepted,
                miss_count=len(trial_eval.deadline_misses) + len(trial_eval.unfinished) + len(trial_eval.issues),
                worst_lateness=trial_eval.worst_lateness,
                note="violation reduction" if accepted else "no strict improvement",
            )
        )
        if not accepted:
            break

        current_selection = dict(current_selection)
        current_selection[cnn_id] = candidate
        current_eval = trial_eval
        used_repair = True

    return current_selection, current_eval, used_repair


def _global_feasible_search(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    initial_selection: Dict[str, CNNConfig],
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
    max_global_evals: int,
) -> Optional[MappingEvaluation]:
    cnn_ids = list(configs_by_cnn)
    lists = [configs_by_cnn[c] for c in cnn_ids]
    total = 1
    for arr in lists:
        total *= max(1, len(arr))
        if total > max_global_evals:
            break
    if total > max_global_evals:
        return None

    best: Optional[MappingEvaluation] = None
    for combo in product(*lists):
        selection = {cnn_id: cfg for cnn_id, cfg in zip(cnn_ids, combo)}
        ev = evaluate_selection(
            tasks,
            selection,
            arch,
            sim_time,
            policy,
            previous_selection=initial_selection,
            switch_cost=switch_cost,
            switch_cost_mode=switch_cost_mode,
        )
        if not ev.feasible:
            continue
        if best is None:
            best = ev
            continue
        if (ev.total_accuracy_score, -ev.total_nominal_cost) > (best.total_accuracy_score, -best.total_nominal_cost):
            best = ev
    return best


def _improve_accuracy(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    current_selection: Dict[str, CNNConfig],
    current_eval: MappingEvaluation,
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
    trace: List[CandidateTrace],
    max_iterations: int,
) -> Tuple[Dict[str, CNNConfig], MappingEvaluation]:
    for _ in range(max_iterations):
        candidates: List[Tuple[float, float, int, str, CNNConfig]] = []
        for cnn_id, cfgs in configs_by_cnn.items():
            old = current_selection[cnn_id]
            for candidate in cfgs:
                if candidate.accuracy_score <= old.accuracy_score or candidate.name == old.name:
                    continue
                ratio, delta_accuracy, delta_cost = _candidate_ratio(old, candidate, switch_cost)
                if delta_accuracy > 0:
                    candidates.append((ratio, delta_accuracy, delta_cost, cnn_id, candidate))

        if not candidates:
            break

        candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3], x[4].name))
        accepted_any = False
        for ratio, delta_accuracy, delta_cost, cnn_id, candidate in candidates:
            old = current_selection[cnn_id]
            trial_selection = dict(current_selection)
            trial_selection[cnn_id] = candidate
            trial_eval = evaluate_selection(
                tasks,
                trial_selection,
                arch,
                sim_time,
                policy,
                previous_selection=current_selection,
                switch_cost=switch_cost,
                switch_cost_mode=switch_cost_mode,
            )
            accepted = trial_eval.feasible
            trace.append(
                CandidateTrace(
                    phase="improve",
                    cnn_id=cnn_id,
                    from_config=old.name,
                    to_config=candidate.name,
                    delta_accuracy=delta_accuracy,
                    delta_cost=delta_cost,
                    ratio=ratio,
                    accepted=accepted,
                    miss_count=len(trial_eval.deadline_misses) + len(trial_eval.unfinished) + len(trial_eval.issues),
                    worst_lateness=trial_eval.worst_lateness,
                    note="accepted" if accepted else "rollback: deadline violated",
                )
            )
            if accepted:
                current_selection = trial_selection
                current_eval = trial_eval
                accepted_any = True
                break

        if not accepted_any:
            break

    return current_selection, current_eval


def deadline_first_mapping(
    tasks: Sequence[Task],
    arch: ArchConfig,
    sim_time: int,
    policy: str = "np_fp",
    switch_cost: int = 0,
    switch_cost_mode: str = "per_cnn",
    max_configs_per_cnn: int = 512,
    max_iterations: int = 64,
    max_global_evals: int = 20000,
) -> DeadlineFirstMappingResult:
    """Offline Deadline-First mapping.

    The offline DF corner starts from an NPU-heavy, low-quality configuration
    (all NPU where legal, INT8/cheapest first). It repairs if that aggressive
    configuration misses deadlines, then raises quality only while the complete
    offline simulation remains deadline-feasible. The selected configuration is
    fixed for execution; there is no period-by-period remapping.
    """

    configs_by_cnn = build_cnn_configurations(tasks, arch, max_configs_per_cnn=max_configs_per_cnn)
    initial_selection = _all_npu_low_quality_selection(configs_by_cnn)
    trace: List[CandidateTrace] = []

    current_selection = dict(initial_selection)
    # Offline DF chooses one configuration before execution; repairs/improvements
    # are design-time search steps, not runtime remaps. Do not charge switch cost
    # while evaluating the fixed offline candidate.
    offline_switch_cost = 0
    current_eval = evaluate_selection(tasks, current_selection, arch, sim_time, policy)

    repair_used = False
    global_search_used = False
    if not current_eval.feasible:
        current_selection, current_eval, repair_used = _repair_selection(
            tasks,
            configs_by_cnn,
            current_selection,
            current_eval,
            arch,
            sim_time,
            policy,
            offline_switch_cost,
            switch_cost_mode,
            trace,
            max_iterations,
        )

    if not current_eval.feasible:
        global_eval = _global_feasible_search(
            tasks,
            configs_by_cnn,
            initial_selection,
            arch,
            sim_time,
            policy,
            offline_switch_cost,
            switch_cost_mode,
            max_global_evals,
        )
        if global_eval is not None:
            current_selection = global_eval.selection
            current_eval = global_eval
            global_search_used = True

    schedulable = current_eval.feasible
    if schedulable:
        current_selection, current_eval = _improve_accuracy(
            tasks,
            configs_by_cnn,
            current_selection,
            current_eval,
            arch,
            sim_time,
            policy,
            offline_switch_cost,
            switch_cost_mode,
            trace,
            max_iterations,
        )

    final_tasks = _materialize_tasks(
        tasks,
        current_selection,
        previous_selection=None,
        switch_cost=0,
        switch_cost_mode=switch_cost_mode,
    )

    # Re-run the final fixed offline selection. No runtime remap/switch cost is
    # charged because the chosen configuration is applied before execution.
    final_eval = evaluate_selection(
        tasks,
        current_selection,
        arch,
        sim_time,
        policy,
        previous_selection=None,
        switch_cost=0,
        switch_cost_mode=switch_cost_mode,
    )

    return DeadlineFirstMappingResult(
        tasks=final_tasks,
        selection=current_selection,
        initial_selection=initial_selection,
        configs_by_cnn=configs_by_cnn,
        evaluation=final_eval,
        trace=trace,
        repair_used=repair_used,
        global_search_used=global_search_used,
        schedulable=schedulable and final_eval.feasible,
        normalization_accuracy_score=_max_accuracy_score(configs_by_cnn),
    )


def deadline_first_offline_mapping(*args, **kwargs) -> DeadlineFirstMappingResult:
    return deadline_first_mapping(*args, **kwargs)


def _npu_first_selection(configs_by_cnn: Dict[str, List[CNNConfig]]) -> Dict[str, CNNConfig]:
    """Initial online DF state: push every NPU-compatible layer to NPU.

    CPU-only layers remain on CPU because the hardware model cannot execute
    unsupported operators on the accelerator. Among configurations with the
    maximum number of NPU layers, the cheapest/lowest-precision configuration is
    selected, so P0 intentionally stresses the NPU before any repair.
    """
    out: Dict[str, CNNConfig] = {}
    for cnn_id, configs in configs_by_cnn.items():
        def key(cfg: CNNConfig) -> Tuple[int, int, float, str]:
            npu_layers = sum(1 for opt in cfg.options if opt.resource.upper() == "NPU")
            return (-npu_layers, cfg.cost, cfg.accuracy_score, cfg.name)
        out[cnn_id] = min(configs, key=key)
    return out


def _dominant_instance_count(tasks: Sequence[Task], sim_time: int) -> int:
    count = 0
    for task in tasks:
        if task.period <= 0 or task.phase >= sim_time:
            continue
        # Number of releases phase, phase+period, ... strictly before sim_time.
        n = 1 + max(0, (sim_time - 1 - task.phase) // task.period)
        count = max(count, n)
    return max(1, count)


def _dominant_period_window(tasks: Sequence[Task]) -> int:
    periods = [int(t.period) for t in tasks if int(t.period) > 0]
    return min(periods) if periods else 1


def _release_instance_in_window(task: Task, start: int, end: int) -> Optional[Tuple[int, int]]:
    """Return (instance, release) if this task releases one job in [start, end).

    Deadline-First decisions are made on real time windows, not on the local
    instance number of each task. This matters for aperiodic tasks modelled as
    phase>0 and period>simulation_time: their first instance must be mapped
    only when its real release window is reached, not at global P0.
    """
    period = int(task.period)
    phase = int(task.phase)
    if period <= 0 or end <= phase:
        return None
    k = max(0, (start - phase + period - 1) // period)
    release = phase + k * period
    if start <= release < end:
        return k, release
    return None


def _cnn_ids_released_in_window(tasks: Sequence[Task], window: int, period_index: int) -> set[str]:
    start = max(0, int(period_index) * int(window))
    end = start + max(1, int(window))
    active: set[str] = set()
    for task in tasks:
        if _release_instance_in_window(task, start, end) is not None:
            active.add(task.cnn_id or task.name)
    return active


def _window_jobs_for_step(
    tasks: Sequence[Task],
    selection: Dict[str, CNNConfig],
    previous_selection: Optional[Dict[str, CNNConfig]],
    step: PeriodStep,
    window: int,
    switch_cost: int,
    switch_cost_mode: str,
) -> List[ComputeJob]:
    start = max(0, int(step.period_index) * int(window))
    end = start + max(1, int(window))
    charged_cnn: set[str] = set()
    jobs: List[ComputeJob] = []
    for task in tasks:
        hit = _release_instance_in_window(task, start, end)
        if hit is None:
            continue
        instance, release = hit
        mapped_task = _materialize_task_for_step(
            task,
            selection,
            previous_selection,
            step.period_index,
            switch_cost,
            switch_cost_mode,
            charged_cnn,
            step,
        )
        jobs.append(
            ComputeJob(
                task=mapped_task,
                instance=instance,
                release=release,
                deadline=release + int(mapped_task.deadline),
                wcet=mapped_task.wcet(),
            )
        )
    jobs.sort(key=lambda j: (j.release, j.task.priority, j.task.name, j.instance))
    return jobs


def _candidate_changed_layers(old: CNNConfig, candidate: CNNConfig) -> List[str]:
    return candidate.changed_layers_from(old)


def _single_step_candidates(configs: List[CNNConfig], old: CNNConfig) -> List[CNNConfig]:
    """Prefer candidates that modify one layer only to make the Gantt readable.

    If the Pareto frontier does not contain a one-layer neighbor, fall back to
    the complete frontier so the mapper can still progress.
    """
    one_layer = [cfg for cfg in configs if cfg.name != old.name and len(_candidate_changed_layers(old, cfg)) == 1]
    return one_layer if one_layer else [cfg for cfg in configs if cfg.name != old.name]


def _evaluation_from_result(selection: Dict[str, CNNConfig], result: SimulationResult) -> MappingEvaluation:
    issues = validate_schedule(result)
    deadline_misses: List[str] = []
    unfinished: List[str] = []
    worst_lateness = 0
    sum_lateness = 0
    for job in result.compute_jobs:
        if job.finish is None:
            unfinished.append(job.job_id)
            lateness = max(1, job.deadline - job.release + 1)
            worst_lateness = max(worst_lateness, lateness)
            sum_lateness += lateness
            continue
        lateness = max(0, job.finish - job.deadline)
        if lateness > 0:
            deadline_misses.append(job.job_id)
            worst_lateness = max(worst_lateness, lateness)
            sum_lateness += lateness
    return MappingEvaluation(
        selection=selection,
        result=result,
        issues=issues,
        deadline_misses=deadline_misses,
        unfinished=unfinished,
        total_accuracy_score=_selection_accuracy(selection),
        total_nominal_cost=_selection_cost(selection),
        worst_lateness=worst_lateness,
        sum_lateness=sum_lateness,
    )


def _choose_period_repair(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    current_selection: Dict[str, CNNConfig],
    current_eval: MappingEvaluation,
    arch: ArchConfig,
    window: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
) -> Tuple[Dict[str, CNNConfig], MappingEvaluation, PeriodStep, CandidateTrace]:
    old_key = (
        len(current_eval.deadline_misses) + len(current_eval.unfinished) + len(current_eval.issues),
        current_eval.worst_lateness,
        current_eval.sum_lateness,
        -current_eval.total_accuracy_score,
        current_eval.total_nominal_cost,
    )
    best: Optional[Tuple[Tuple[int, int, int, float, int], str, CNNConfig, MappingEvaluation, float, float, int]] = None
    for cnn_id, cfgs in configs_by_cnn.items():
        old = current_selection[cnn_id]
        for candidate in _single_step_candidates(cfgs, old):
            trial_selection = dict(current_selection)
            trial_selection[cnn_id] = candidate
            trial_eval = evaluate_selection(
                tasks,
                trial_selection,
                arch,
                window,
                policy,
                previous_selection=current_selection,
                switch_cost=switch_cost,
                switch_cost_mode=switch_cost_mode,
            )
            ratio, delta_accuracy, delta_cost = _candidate_ratio(old, candidate, switch_cost)
            key = (
                len(trial_eval.deadline_misses) + len(trial_eval.unfinished) + len(trial_eval.issues),
                trial_eval.worst_lateness,
                trial_eval.sum_lateness,
                -trial_eval.total_accuracy_score,
                trial_eval.total_nominal_cost,
            )
            if best is None or key < best[0]:
                best = (key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost)

    if best is None:
        step = PeriodStep(0, dict(current_selection), "hold", accepted=False, miss_count=old_key[0], worst_lateness=current_eval.worst_lateness, note="aucun candidat")
        tr = CandidateTrace("repair", "-", "-", "-", 0.0, 0, 0.0, False, old_key[0], current_eval.worst_lateness, "aucun candidat")
        return current_selection, current_eval, step, tr

    key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost = best
    old = current_selection[cnn_id]
    accepted = key < old_key
    if accepted:
        new_selection = dict(current_selection)
        new_selection[cnn_id] = candidate
        new_eval = trial_eval
        action = "repair"
        note = "repair accepted: reduces violations/lateness"
    else:
        new_selection = current_selection
        new_eval = current_eval
        action = "hold"
        note = "no strict repair"

    step = PeriodStep(
        0,
        dict(new_selection),
        action,
        changed_cnn=cnn_id if accepted else "",
        from_config=old.name if accepted else "",
        to_config=candidate.name if accepted else "",
        accepted=accepted,
        miss_count=(len(new_eval.deadline_misses) + len(new_eval.unfinished) + len(new_eval.issues)),
        worst_lateness=new_eval.worst_lateness,
        note=note,
    )
    tr = CandidateTrace(
        phase="repair",
        cnn_id=cnn_id,
        from_config=old.name,
        to_config=candidate.name,
        delta_accuracy=delta_accuracy,
        delta_cost=delta_cost,
        ratio=ratio,
        accepted=accepted,
        miss_count=len(trial_eval.deadline_misses) + len(trial_eval.unfinished) + len(trial_eval.issues),
        worst_lateness=trial_eval.worst_lateness,
        note=note,
    )
    return new_selection, new_eval, step, tr


def _choose_period_improvement(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    current_selection: Dict[str, CNNConfig],
    current_eval: MappingEvaluation,
    arch: ArchConfig,
    window: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
) -> Tuple[Dict[str, CNNConfig], MappingEvaluation, PeriodStep, Optional[CandidateTrace]]:
    candidates: List[Tuple[float, float, int, str, CNNConfig]] = []
    for cnn_id, cfgs in configs_by_cnn.items():
        old = current_selection[cnn_id]
        for candidate in _single_step_candidates(cfgs, old):
            if candidate.accuracy_score <= old.accuracy_score:
                continue
            ratio, delta_accuracy, delta_cost = _candidate_ratio(old, candidate, switch_cost)
            if delta_accuracy > 0:
                candidates.append((ratio, delta_accuracy, delta_cost, cnn_id, candidate))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3], x[4].name))
    last_trace: Optional[CandidateTrace] = None
    for ratio, delta_accuracy, delta_cost, cnn_id, candidate in candidates:
        old = current_selection[cnn_id]
        trial_selection = dict(current_selection)
        trial_selection[cnn_id] = candidate
        trial_eval = evaluate_selection(
            tasks,
            trial_selection,
            arch,
            window,
            policy,
            previous_selection=current_selection,
            switch_cost=switch_cost,
            switch_cost_mode=switch_cost_mode,
        )
        accepted = trial_eval.feasible
        last_trace = CandidateTrace(
            phase="improve",
            cnn_id=cnn_id,
            from_config=old.name,
            to_config=candidate.name,
            delta_accuracy=delta_accuracy,
            delta_cost=delta_cost,
            ratio=ratio,
            accepted=accepted,
            miss_count=len(trial_eval.deadline_misses) + len(trial_eval.unfinished) + len(trial_eval.issues),
            worst_lateness=trial_eval.worst_lateness,
            note="improvement acceptede" if accepted else "rollback: deadline violated",
        )
        if accepted:
            step = PeriodStep(
                0,
                dict(trial_selection),
                "improve",
                changed_cnn=cnn_id,
                from_config=old.name,
                to_config=candidate.name,
                accepted=True,
                miss_count=0,
                worst_lateness=0,
                note="precision improvement accepted",
            )
            return trial_selection, trial_eval, step, last_trace

    miss_count = len(current_eval.deadline_misses) + len(current_eval.unfinished) + len(current_eval.issues)
    step = PeriodStep(0, dict(current_selection), "hold", accepted=False, miss_count=miss_count, worst_lateness=current_eval.worst_lateness, note="no feasible improvement")
    return current_selection, current_eval, step, last_trace



def _miss_count(evaluation: MappingEvaluation) -> int:
    return len(evaluation.deadline_misses) + len(evaluation.unfinished) + len(evaluation.issues)


def _one_period_observation(
    tasks: Sequence[Task],
    selection: Dict[str, CNNConfig],
    previous_selection: Optional[Dict[str, CNNConfig]],
    step: PeriodStep,
    arch: ArchConfig,
    window: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
) -> MappingEvaluation:
    """Simulate the real time window that has just been executed."""
    jobs = _window_jobs_for_step(
        tasks,
        selection,
        previous_selection,
        step,
        window,
        switch_cost,
        switch_cost_mode,
    )
    result = simulate(jobs, arch, policy=policy)
    return _evaluation_from_result(selection, result)


def _selection_edge_key(cnn_id: str, old: CNNConfig, candidate: CNNConfig) -> Tuple[str, str, str]:
    return (cnn_id, old.name, candidate.name)


def _task_to_cnn(tasks: Sequence[Task]) -> Dict[str, str]:
    return {task.name: (task.cnn_id or task.name) for task in tasks}


def _observed_problem_cnn_order(tasks: Sequence[Task], observed: MappingEvaluation, configs_by_cnn: Dict[str, List[CNNConfig]]) -> List[str]:
    by_task = _task_to_cnn(tasks)
    score: Dict[str, int] = defaultdict(int)
    for jid in observed.deadline_misses + observed.unfinished:
        task_name = jid.split("[", 1)[0]
        score[by_task.get(task_name, task_name)] += 1
    for issue in observed.issues:
        head = issue.split(":", 1)[0]
        task_name = head.split("[", 1)[0]
        if task_name in by_task:
            score[by_task[task_name]] += 1
    ordered = sorted(configs_by_cnn, key=lambda cid: (-score.get(cid, 0), cid))
    return ordered


def _resource_count(cfg: CNNConfig, resource: str) -> int:
    r = resource.upper()
    return sum(1 for opt in cfg.options if opt.resource.upper() == r)


def _changed_layer_pair(old: CNNConfig, candidate: CNNConfig) -> Tuple[Optional[LayerOption], Optional[LayerOption]]:
    old_by = old.by_task
    new_by = candidate.by_task
    for name in sorted(set(old_by) | set(new_by)):
        o = old_by.get(name)
        n = new_by.get(name)
        if o is None or n is None or o.key != n.key:
            return o, n
    return None, None



def _predict_one_period(
    tasks: Sequence[Task],
    selection: Dict[str, CNNConfig],
    previous_selection: Optional[Dict[str, CNNConfig]],
    arch: ArchConfig,
    window: int,
    period_index: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
) -> MappingEvaluation:
    """Virtual simulation for one real time window before applying a DF decision."""
    step = PeriodStep(
        period_index=period_index,
        selection=dict(selection),
        action="precheck",
        accepted=True,
    )
    jobs = _window_jobs_for_step(
        tasks,
        selection,
        previous_selection,
        step,
        window,
        switch_cost,
        switch_cost_mode,
    )
    result = simulate(jobs, arch, policy=policy)
    return _evaluation_from_result(selection, result)


def _propose_safe_repair(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    current_selection: Dict[str, CNNConfig],
    observed: MappingEvaluation,
    arch: ArchConfig,
    window: int,
    period_index: int,
    policy: str,
    blocked: set[Tuple[str, str, str]],
    switch_cost: int,
    switch_cost_mode: str,
) -> Tuple[Dict[str, CNNConfig], PeriodStep, List[CandidateTrace]]:
    """Choose one repair for the next period using virtual validation.

    A repair is allowed to keep the same accuracy score, because its objective
    is timing feasibility. It is applied only if a one-period simulation predicts
    a strict improvement on the deadline-first key:
        miss count -> max lateness -> total lateness -> precision -> cost.
    """
    base_miss = _miss_count(observed)
    old_key = (
        base_miss,
        observed.worst_lateness,
        observed.sum_lateness,
        -observed.total_accuracy_score,
        observed.total_nominal_cost,
    )
    traces: List[CandidateTrace] = []
    best: Optional[Tuple[Tuple[int, int, int, float, int], str, CNNConfig, MappingEvaluation, float, float, int]] = None
    active_cnn = _cnn_ids_released_in_window(tasks, window, period_index)
    problem_order = [cid for cid in _observed_problem_cnn_order(tasks, observed, configs_by_cnn) if cid in active_cnn]
    if not problem_order:
        problem_order = [cid for cid in configs_by_cnn if cid in active_cnn]
    problem_rank = {cid: idx for idx, cid in enumerate(problem_order)}

    for cnn_id in problem_order:
        old = current_selection[cnn_id]
        for candidate in _single_step_candidates(configs_by_cnn[cnn_id], old):
            edge = _selection_edge_key(cnn_id, old, candidate)
            if edge in blocked:
                continue
            old_opt, new_opt = _changed_layer_pair(old, candidate)
            if new_opt is None:
                continue
            old_npu = _resource_count(old, "NPU")
            new_npu = _resource_count(candidate, "NPU")
            npu_reduction = old_npu - new_npu
            delta_accuracy = candidate.accuracy_score - old.accuracy_score
            delta_cost_raw = candidate.cost - old.cost
            ratio, _, delta_cost = _candidate_ratio(old, candidate, switch_cost)
            npu_to_cpu = 1 if old_opt is not None and old_opt.resource.upper() == "NPU" and new_opt.resource.upper() == "CPU" else 0
            lower_precision = 1 if new_opt.accuracy_score < (old_opt.accuracy_score if old_opt is not None else new_opt.accuracy_score) else 0
            cheaper = 1 if delta_cost_raw < 0 else 0

            # Ignore moves that have no plausible repair effect.
            if not (npu_reduction > 0 or cheaper or lower_precision or delta_accuracy <= 0):
                continue

            trial_selection = dict(current_selection)
            trial_selection[cnn_id] = candidate
            trial_eval = _predict_one_period(
                tasks,
                trial_selection,
                current_selection,
                arch,
                window,
                period_index,
                policy,
                switch_cost,
                switch_cost_mode,
            )
            key = (
                _miss_count(trial_eval),
                trial_eval.worst_lateness,
                trial_eval.sum_lateness,
                -trial_eval.total_accuracy_score,
                trial_eval.total_nominal_cost,
            )
            accepted_by_key = key < old_key
            traces.append(CandidateTrace(
                phase="repair_precheck",
                cnn_id=cnn_id,
                from_config=old.name,
                to_config=candidate.name,
                delta_accuracy=delta_accuracy,
                delta_cost=delta_cost,
                ratio=ratio,
                accepted=accepted_by_key,
                miss_count=_miss_count(trial_eval),
                worst_lateness=trial_eval.worst_lateness,
                note="prevalidation: improves lateness" if accepted_by_key else "prevalidation: rejected, does not improve real-time score",
            ))
            if accepted_by_key:
                # Tie-breaker: prefer affected CNNs, then fewer predicted misses,
                # then less lateness, then less accuracy loss, then deterministic.
                tie_key = (
                    key[0],
                    key[1],
                    key[2],
                    problem_rank.get(cnn_id, 999),
                    -npu_to_cpu,
                    -npu_reduction,
                    -cheaper,
                    delta_accuracy,
                    max(0, delta_cost_raw),
                    cnn_id,
                    candidate.name,
                )
                # Store only the deadline-first part as best[0]; tie_key is encoded
                # by expanding with stable extra terms through comparison below.
                full_best_key = (tie_key[0], tie_key[1], tie_key[2], key[3], key[4], tie_key[3:])
                if best is None:
                    best = (key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost)
                    best_full_key = full_best_key
                else:
                    try:
                        old_full = best_full_key  # type: ignore[name-defined]
                    except NameError:
                        old_full = (best[0][0], best[0][1], best[0][2], best[0][3], best[0][4], tuple())
                    if full_best_key < old_full:
                        best = (key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost)
                        best_full_key = full_best_key
            else:
                blocked.add(edge)
                blocked.add((cnn_id, candidate.name, old.name))

    if best is None:
        step = PeriodStep(
            period_index=0,
            selection=dict(current_selection),
            action="hold",
            accepted=False,
            miss_count=base_miss,
            worst_lateness=observed.worst_lateness,
            note="no prevalidated repair; configuration held",
        )
        return current_selection, step, traces

    _, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost = best
    old = current_selection[cnn_id]
    next_selection = dict(current_selection)
    next_selection[cnn_id] = candidate
    step = PeriodStep(
        period_index=0,
        selection=dict(next_selection),
        action="repair",
        changed_cnn=cnn_id,
        from_config=old.name,
        to_config=candidate.name,
        accepted=True,
        miss_count=_miss_count(trial_eval),
        worst_lateness=trial_eval.worst_lateness,
        note="repair prevalidated; applied to the next period",
    )
    traces.append(CandidateTrace(
        phase="repair_selected",
        cnn_id=cnn_id,
        from_config=old.name,
        to_config=candidate.name,
        delta_accuracy=delta_accuracy,
        delta_cost=delta_cost,
        ratio=ratio,
        accepted=True,
        miss_count=_miss_count(trial_eval),
        worst_lateness=trial_eval.worst_lateness,
        note="selected after prevalidation",
    ))
    return next_selection, step, traces


def _propose_safe_improvement(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    current_selection: Dict[str, CNNConfig],
    arch: ArchConfig,
    window: int,
    period_index: int,
    policy: str,
    blocked: set[Tuple[str, str, str]],
    switch_cost: int,
    switch_cost_mode: str,
) -> Tuple[Dict[str, CNNConfig], PeriodStep, List[CandidateTrace]]:
    """Choose one precision improvement using virtual validation.

    Unlike the previous blind implementation, rejected improvements are never
    executed in the real schedule. They are logged as virtual rejections and
    blocked, which prevents the loop improvement -> deadline miss -> rollback.
    """
    traces: List[CandidateTrace] = []
    candidates: List[Tuple[float, float, int, str, CNNConfig]] = []
    active_cnn = _cnn_ids_released_in_window(tasks, window, period_index)
    for cnn_id, cfgs in configs_by_cnn.items():
        if cnn_id not in active_cnn:
            continue
        old = current_selection[cnn_id]
        for candidate in _single_step_candidates(cfgs, old):
            if candidate.accuracy_score <= old.accuracy_score:
                continue
            edge = _selection_edge_key(cnn_id, old, candidate)
            if edge in blocked:
                continue
            ratio, delta_accuracy, delta_cost = _candidate_ratio(old, candidate, switch_cost)
            if delta_accuracy > 0:
                candidates.append((ratio, delta_accuracy, delta_cost, cnn_id, candidate))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3], x[4].name))
    for ratio, delta_accuracy, delta_cost, cnn_id, candidate in candidates:
        old = current_selection[cnn_id]
        trial_selection = dict(current_selection)
        trial_selection[cnn_id] = candidate
        trial_eval = _predict_one_period(
            tasks,
            trial_selection,
            current_selection,
            arch,
            window,
            period_index,
            policy,
            switch_cost,
            switch_cost_mode,
        )
        accepted = trial_eval.feasible
        edge = _selection_edge_key(cnn_id, old, candidate)
        traces.append(CandidateTrace(
            phase="improve_precheck",
            cnn_id=cnn_id,
            from_config=old.name,
            to_config=candidate.name,
            delta_accuracy=delta_accuracy,
            delta_cost=delta_cost,
            ratio=ratio,
            accepted=accepted,
            miss_count=_miss_count(trial_eval),
            worst_lateness=trial_eval.worst_lateness,
            note="prevalidation: safe improvement" if accepted else "prevalidation: rejected, predicted deadline violation; not executed",
        ))
        if accepted:
            step = PeriodStep(
                period_index=0,
                selection=dict(trial_selection),
                action="improve",
                changed_cnn=cnn_id,
                from_config=old.name,
                to_config=candidate.name,
                accepted=True,
                miss_count=0,
                worst_lateness=0,
                note="improvement prevalidated; applied to the next period",
            )
            return trial_selection, step, traces
        blocked.add(edge)

    step = PeriodStep(
        period_index=0,
        selection=dict(current_selection),
        action="hold",
        accepted=True,
        note="no safe improvement left; configuration stabilized",
    )
    return current_selection, step, traces


def _build_online_period_steps(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
) -> Tuple[List[PeriodStep], List[CandidateTrace]]:
    """Build the safe online Deadline-First trace.

    P0 remains NPU-first so the simulator can observe the actual contention of
    the aggressive initial mapping. After that first observation, every repair
    or precision improvement is selected by a virtual one-period simulation
    before it is executed. This keeps the original Gantt period-by-period while
    avoiding an infinite improvement/rollback loop and avoiding real deadline
    misses caused only by unsafe precision exploration.
    """
    period_count = _dominant_instance_count(tasks, sim_time)
    window = _dominant_period_window(tasks)
    trace: List[CandidateTrace] = []
    blocked_improvements: set[Tuple[str, str, str]] = set()
    blocked_repairs: set[Tuple[str, str, str]] = set()

    current_selection = _npu_first_selection(configs_by_cnn)
    previous_selection: Optional[Dict[str, CNNConfig]] = None
    current_step = PeriodStep(
        period_index=0,
        selection=dict(current_selection),
        action="init_npu_first",
        note="P0: all compatible layers are placed on NPU; result known only after simulation",
    )

    steps: List[PeriodStep] = []

    for period_index in range(period_count):
        current_step.period_index = period_index
        current_step.selection = dict(current_selection)

        observed = _one_period_observation(
            tasks,
            current_selection,
            previous_selection,
            current_step,
            arch,
            window,
            policy,
            switch_cost,
            switch_cost_mode,
        )
        observed_misses = _miss_count(observed)
        current_step.miss_count = observed_misses
        current_step.worst_lateness = observed.worst_lateness
        current_step.accepted = observed_misses == 0 or current_step.action in ("init_npu_first", "repair")

        if current_step.action == "init_npu_first":
            current_step.note = (
                "observed after execution: "
                + ("deadline missed; repair prevalidated for the next period" if observed_misses else "feasible; safe improvement searched for the next period")
            )
        elif current_step.action == "improve":
            # This should remain feasible because it was prevalidated. If the
            # runtime still misses due to a modelling issue, the transition is
            # blocked and DF repairs next; it is not retried.
            edge = (current_step.changed_cnn, current_step.from_config, current_step.to_config)
            if observed_misses:
                blocked_improvements.add(edge)
                current_step.accepted = False
                current_step.note = "improvement prevalidated but period missed; transition blocked and next repair selected"
            else:
                current_step.accepted = True
                current_step.note = "improvement validated during execution"
        elif current_step.action == "repair":
            if observed_misses:
                current_step.note = "repair prevalidated but misses remain; selecting another repair"
                if current_step.changed_cnn and current_step.from_config and current_step.to_config:
                    blocked_repairs.add((current_step.changed_cnn, current_step.from_config, current_step.to_config))
                    blocked_repairs.add((current_step.changed_cnn, current_step.to_config, current_step.from_config))
            else:
                current_step.note = "repair validated during execution; system feasible"
        elif current_step.action == "hold":
            current_step.note = current_step.note or "configuration maintenue"

        steps.append(current_step)

        # Decide the NEXT period using the observation above plus virtual
        # prevalidation. Rejected virtual candidates go to the log, not to Gantt.
        if period_index == period_count - 1:
            break

        if observed_misses:
            next_selection, next_step, decision_traces = _propose_safe_repair(
                tasks,
                configs_by_cnn,
                current_selection,
                observed,
                arch,
                window,
                period_index + 1,
                policy,
                blocked_repairs,
                switch_cost,
                switch_cost_mode,
            )
            trace.extend(decision_traces)
        else:
            hold_eval = _predict_one_period(
                tasks,
                current_selection,
                current_selection,
                arch,
                window,
                period_index + 1,
                policy,
                switch_cost,
                switch_cost_mode,
            )
            if _miss_count(hold_eval):
                next_selection, next_step, decision_traces = _propose_safe_repair(
                    tasks,
                    configs_by_cnn,
                    current_selection,
                    hold_eval,
                    arch,
                    window,
                    period_index + 1,
                    policy,
                    blocked_repairs,
                    switch_cost,
                    switch_cost_mode,
                )
                if next_step.action == "repair":
                    next_step.note = "predicted hold is unsafe in the next window; adaptation applied before execution"
                trace.extend(decision_traces)
            else:
                next_selection, next_step, decision_traces = _propose_safe_improvement(
                    tasks,
                    configs_by_cnn,
                    current_selection,
                    arch,
                    window,
                    period_index + 1,
                    policy,
                    blocked_improvements,
                    switch_cost,
                    switch_cost_mode,
                )
                trace.extend(decision_traces)

        previous_selection = dict(current_selection)
        current_selection = dict(next_selection)
        current_step = next_step

    return steps, trace


def _materialize_task_for_step(
    task: Task,
    selection: Dict[str, CNNConfig],
    previous_selection: Optional[Dict[str, CNNConfig]],
    period_index: int,
    switch_cost: int,
    switch_cost_mode: str,
    charged_cnn: set[str],
    step: PeriodStep,
) -> Task:
    cfg = selection.get(task.cnn_id or task.name)
    if cfg is None:
        return replace(task, df_period=period_index, df_action=step.action, switch_cost=0, switch_instance=period_index)
    opt = cfg.by_task.get(task.name)
    if opt is None:
        return replace(task, df_period=period_index, df_config=cfg.name, df_action=step.action, switch_cost=0, switch_instance=period_index)

    prev_opt: Optional[LayerOption] = None
    if previous_selection:
        prev_cfg = previous_selection.get(task.cnn_id or task.name)
        if prev_cfg is not None:
            prev_opt = prev_cfg.by_task.get(task.name)
    changed = prev_opt is not None and prev_opt.key != opt.key
    sw = 0
    if changed and switch_cost > 0:
        if switch_cost_mode.lower() == "per_layer":
            sw = switch_cost
        else:
            cnn_id = task.cnn_id or task.name
            if cnn_id not in charged_cnn:
                sw = switch_cost
                charged_cnn.add(cnn_id)

    return replace(
        task,
        resource=opt.resource.upper(),
        precision=opt.precision.upper(),
        switch_cost=sw,
        switch_instance=period_index,
        df_period=period_index,
        df_config=cfg.name,
        df_action=step.action,
        df_changed=changed,
    )


def generate_jobs_online_periods(
    tasks: Sequence[Task],
    period_steps: Sequence[PeriodStep],
    sim_time: int,
    switch_cost: int,
    switch_cost_mode: str,
) -> List[ComputeJob]:
    jobs: List[ComputeJob] = []
    if not period_steps:
        return generate_jobs(list(tasks), sim_time)

    window = _dominant_period_window(tasks)
    # Build with a period-level cache for per_cnn switch charging.
    charged_by_period: Dict[int, set[str]] = defaultdict(set)
    for task in tasks:
        release = task.phase
        instance = 0
        while release < sim_time:
            step_index = min(max(0, int(release) // max(1, int(window))), len(period_steps) - 1)
            step = period_steps[step_index]
            previous_selection = period_steps[step_index - 1].selection if step_index > 0 else None
            mapped_task = _materialize_task_for_step(
                task,
                step.selection,
                previous_selection,
                step_index,
                switch_cost,
                switch_cost_mode,
                charged_by_period[step_index],
                step,
            )
            job = ComputeJob(
                task=mapped_task,
                instance=instance,
                release=release,
                deadline=release + mapped_task.deadline,
                wcet=mapped_task.wcet(),
            )
            job.notes.append(f"DF_PERIOD=P{step_index}")
            job.notes.append(f"DF_ACTION={step.action}")
            if mapped_task.df_config:
                job.notes.append(f"DF_CONFIG={mapped_task.df_config}")
            if mapped_task.df_changed:
                job.notes.append("DF_CHANGED=1")
            jobs.append(job)
            release += task.period
            instance += 1

    jobs.sort(key=lambda j: (j.release, j.task.priority, j.task.name, j.instance))
    return jobs


def deadline_first_online_period_mapping(
    tasks: Sequence[Task],
    arch: ArchConfig,
    sim_time: int,
    policy: str = "np_fp",
    switch_cost: int = 0,
    switch_cost_mode: str = "per_cnn",
    max_configs_per_cnn: int = 512,
) -> DeadlineFirstMappingResult:
    """Online Deadline-First dedicated to the simulator periods.

    P0 is always NPU-first and is executed even if it violates deadlines. For
    every following task instance/period, the mapper accepts at most one repair
    or one improvement. The generated ComputeJob list contains different Task
    objects per period, so the original simulator and original Gantt show the
    progression directly on CPU/NPU/communication lanes.
    """
    configs_by_cnn = build_cnn_configurations(tasks, arch, max_configs_per_cnn=max_configs_per_cnn)
    period_steps, trace = _build_online_period_steps(tasks, configs_by_cnn, arch, sim_time, policy, switch_cost, switch_cost_mode)
    jobs = generate_jobs_online_periods(tasks, period_steps, sim_time, switch_cost, switch_cost_mode)
    result = simulate(jobs, arch, policy=policy)
    final_selection = dict(period_steps[-1].selection) if period_steps else _npu_first_selection(configs_by_cnn)
    evaluation = _evaluation_from_result(final_selection, result)
    schedulable = evaluation.feasible
    final_tasks = [j.task for j in jobs]
    return DeadlineFirstMappingResult(
        tasks=final_tasks,
        selection=final_selection,
        initial_selection=period_steps[0].selection if period_steps else final_selection,
        configs_by_cnn=configs_by_cnn,
        evaluation=evaluation,
        trace=trace,
        repair_used=any(s.action == "repair" for s in period_steps),
        global_search_used=False,
        schedulable=schedulable,
        normalization_accuracy_score=_max_accuracy_score(configs_by_cnn),
        period_steps=list(period_steps),
    )

def _config_by_name(configs_by_cnn: Dict[str, List[CNNConfig]], cnn_id: str, name: str) -> Optional[CNNConfig]:
    for cfg in configs_by_cnn.get(cnn_id, []):
        if cfg.name == name:
            return cfg
    return None


def _format_config_delta(configs_by_cnn: Dict[str, List[CNNConfig]], cnn_id: str, from_name: str, to_name: str) -> str:
    old = _config_by_name(configs_by_cnn, cnn_id, from_name)
    new = _config_by_name(configs_by_cnn, cnn_id, to_name)
    if old is None or new is None:
        return ""
    old_by = old.by_task
    new_by = new.by_task
    chunks: List[str] = []
    for task_name in sorted(set(old_by) | set(new_by)):
        o = old_by.get(task_name)
        n = new_by.get(task_name)
        if o is None or n is None or o.key != n.key:
            left = "-" if o is None else f"{o.resource}/{o.precision}"
            right = "-" if n is None else f"{n.resource}/{n.precision}"
            chunks.append(f"{task_name}: {left}→{right}")
    return "; ".join(chunks)


def build_mapping_report(mapping: DeadlineFirstMappingResult) -> str:
    lines: List[str] = []
    lines.append("=== Mapping Online Deadline-First ===" if mapping.period_steps else "=== Mapping Offline Deadline-First ===")
    lines.append(f"Schedulable: {'yes' if mapping.schedulable else 'no'}")
    lines.append(f"Repair phase used: {'yes' if mapping.repair_used else 'no'}")
    lines.append(f"Global search used: {'yes' if mapping.global_search_used else 'no'}")
    lines.append(f"Global precision score: {mapping.evaluation.total_accuracy_score:.2f} / {mapping.normalization_accuracy_score:.2f} ({mapping.normalized_accuracy_percent:.2f}%)")
    lines.append(f"Global nominal cost: {mapping.evaluation.total_nominal_cost} cycles")
    if mapping.period_steps:
        lines.append("")
        lines.append("Deadline-First evolution executed period by period:")
        for step in mapping.period_steps:
            cfg_txt = ", ".join(f"{cnn}:{cfg.name}" for cnn, cfg in sorted(step.selection.items()))
            detail = ""
            if step.changed_cnn and step.from_config and step.to_config:
                delta = _format_config_delta(mapping.configs_by_cnn, step.changed_cnn, step.from_config, step.to_config)
                if delta:
                    detail = f"; changement: {delta}"
            if step.action == "init_npu_first":
                action = "online DF NPU-first initialization"
            elif step.action == "repair":
                action = f"repair {step.changed_cnn}: {step.from_config} -> {step.to_config}{detail}"
            elif step.action == "improve":
                action = f"improvement {step.changed_cnn}: {step.from_config} -> {step.to_config}{detail}"
            elif step.action == "rollback":
                action = f"rollback {step.changed_cnn}: {step.from_config} -> {step.to_config}{detail}"
            else:
                action = "hold"
            lines.append(
                f"- P{step.period_index}: {action}; miss={step.miss_count}; "
                f"max_lateness={step.worst_lateness}; configs=[{cfg_txt}] ({step.note})"
            )
    lines.append("")
    lines.append("Selected configurations:")
    for cnn_id in sorted(mapping.selection):
        cfg = mapping.selection[cnn_id]
        init = mapping.initial_selection[cnn_id]
        switch_note = " (unchanged)" if not cfg.differs_from(init) else f" (switch from {init.name})"
        lines.append(f"- {cnn_id}: {cfg.name}{switch_note}, cost={cfg.cost}, precision-score={cfg.accuracy_score:.2f}")
        for opt in cfg.options:
            lines.append(f"    {opt.task_name}: {opt.resource}/{opt.precision} cost={opt.cost} score={opt.accuracy_score:.2f}")

    lines.append("")
    lines.append("Generated configuration frontiers:")
    for cnn_id in sorted(mapping.configs_by_cnn):
        configs = mapping.configs_by_cnn[cnn_id]
        preview = ", ".join(f"{c.name}(C={c.cost},P={c.accuracy_score:.1f})" for c in configs[:12])
        suffix = "" if len(configs) <= 12 else f" ... +{len(configs) - 12}"
        lines.append(f"- {cnn_id}: {preview}{suffix}")

    if mapping.trace:
        lines.append("")
        lines.append("Journal des candidats:")
        for tr in mapping.trace[:80]:
            if tr.phase == "repair_precheck" and tr.accepted:
                status = "PREVALIDATED"
            elif tr.accepted:
                status = "ACCEPTED"
            elif "precheck" in tr.phase:
                status = "VIRTUALLY REJECTED"
            elif tr.phase.startswith("repair"):
                status = "rejected: no real-time gain"
            else:
                status = "REJECTED"
            ratio = "inf" if tr.ratio == float("inf") else f"{tr.ratio:.4f}"
            delta = _format_config_delta(mapping.configs_by_cnn, tr.cnn_id, tr.from_config, tr.to_config)
            delta_txt = f", changement={delta}" if delta else ""
            lines.append(
                f"- [{tr.phase}] {tr.cnn_id}: {tr.from_config} -> {tr.to_config}{delta_txt}, "
                f"ΔP={tr.delta_accuracy:.2f}, ΔC={tr.delta_cost}, ρ={ratio}, "
                f"miss={tr.miss_count}, max_lateness={tr.worst_lateness} => {status} ({tr.note})"
            )
        if len(mapping.trace) > 80:
            lines.append(f"... {len(mapping.trace) - 80} other trials not displayed")

    if mapping.evaluation.deadline_misses or mapping.evaluation.unfinished or mapping.evaluation.issues:
        lines.append("")
        lines.append("Violations restantes:")
        for jid in mapping.evaluation.deadline_misses:
            lines.append(f"- Deadline missed: {jid}")
        for jid in mapping.evaluation.unfinished:
            lines.append(f"- Unfinished job: {jid}")
        for issue in mapping.evaluation.issues:
            lines.append(f"- Consistency: {issue}")

    return "\n".join(lines)
