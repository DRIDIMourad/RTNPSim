from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .models import ComputeJob, SimulationResult
from .npu_model import BYTES_PER_ELEMENT, FLIT_BYTES, INPUT_FACTOR, KIND_WCET_SCALE, KERNEL_OPS, PRECISION_THROUGHPUT, WEIGHT_FACTOR


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _safe_ratio(num: Optional[int], den: Optional[int]) -> Optional[float]:
    if num is None or den is None or den <= 0:
        return None
    return float(num) / float(den)


def _job_duration(job: ComputeJob) -> Optional[int]:
    if job.start is None or job.finish is None:
        return None
    return max(0, job.finish - job.start)


def _stage_sum(job: ComputeJob, labels: Iterable[str]) -> int:
    wanted = set(labels)
    return sum(int(s.duration) for s in job.stages if s.label in wanted)


def _build_equations_section() -> List[str]:
    bpe = ", ".join(f"{k}={v}" for k, v in BYTES_PER_ELEMENT.items())
    eff = ", ".join(f"{k}={v:g}" for k, v in PRECISION_THROUGHPUT.items())
    kinds = ", ".join(f"{k}:{v}" for k, v in sorted(KERNEL_OPS.items()))
    alpha = ", ".join(f"{k}:{v:g}" for k, v in sorted(INPUT_FACTOR.items()))
    beta = ", ".join(f"{k}:{v:g}" for k, v in sorted(WEIGHT_FACTOR.items()))
    cpu_scale = ", ".join(f"{k}:{v:g}" for k, v in sorted(KIND_WCET_SCALE["CPU"].items()))
    npu_scale = ", ".join(f"{k}:{v:g}" for k, v in sorted(KIND_WCET_SCALE["NPU"].items()))

    return [
        "=== Hardware model equations ===",
        "Preserved convention: the tasks.txt format does not change.",
        "Important rule: kind=generic is WCET-only. For a generic task, size, memory, SA, SIMD and vector stages do not modify the duration.",
        "Precision support rule: CPU executes FP16 or FP32; NPU executes INT8 only.",
        "Generic:",
        "  CPU_WCET = selected CPU WCET = wcet_fp16 or wcet_fp32",
        "  NPU_WCET = selected NPU WCET = wcet_int8",
        f"  payload_flits = minimum 1 flit of {FLIT_BYTES} bytes to materialize DAG dependencies.",
        "  duration_job = SWITCH_CONFIG + CPU_WCET or NPU_WCET",
        "",
        "Non-generic layers:",
        "  the size field is interpreted as an output size in Ki-elements.",
        "  Notations: S=size, Eout=output elements, Ein=input elements, W=weights, bpe=bytes/element, B=octets, C=cycles.",
        f"  bpe per precision: {bpe}; SA efficiency per precision: {eff}.",
        "  Legal execution points: CPU/FP16, CPU/FP32, NPU/INT8.",
        f"  ops per element by layer kind: {kinds}.",
        f"  input factor alpha(kind): {alpha}.",
        f"  weight factor beta(kind): {beta}.",
        "",
        "Non-generic tensor and memory:",
        "  Eout = max(size, minimum_kind) × 1024",
        "  Ein = ceil(Eout × alpha(kind))",
        "  W = ceil(Eout × beta(kind))",
        "  read_bytes = max(64, (Ein + W) × bpe)",
        "  write_bytes = max(64, Eout × bpe)",
        "  output_bytes = write_bytes",
        "  working_set = read_bytes + write_bytes",
        "  mem_cycles(B) = latency + ceil(B / bandwidth_bytes_per_cycle)",
        "  spill_cycles = 0 if working_set <= capacity; otherwise latency + ceil(2 × (working_set - capacity) / bandwidth)",
        "",
        "Non-generic systolic / convolution / GEMM compute:",
        "  MACs = Eout × ops_per_output(kind)",
        "  peak_MAC_per_cycle = rows × cols × arrays × precision_efficiency",
        "  ideal_compute_cycles = ceil(MACs / peak_MAC_per_cycle)",
        "  tile_cycles = setup_SA + tile_M + tile_N + tile_K - 2",
        "  tiling_overhead = ceil(tiles_total × tile_cycles / arrays)",
        "  SA_COMPUTE = ideal_compute_cycles + tiling_overhead",
        "",
        "Non-generic vector / CPU:",
        "  vector_ops = Eout × ops_per_output(kind) for pool/bn/activation/fuse/nms",
        "  VECTOR_NPU = setup_vector + ceil(vector_ops / vector_lanes)",
        "  CPU_SIMD = setup_CPU + ceil(MACs / simd_mac_per_cycle)",
        "  CPU_VECTOR = setup_CPU + ceil(vector_ops / vector_elements_per_cycle)",
        "",
        "Non-generic communication:",
        f"  payload_flits = ceil(output_bytes / {FLIT_BYTES})",
        "  CPU↔NPU = setup + per_flit × payload_flits",
        "  NoC = noc_setup + noc_per_flit × payload_flits + hops × router_latency",
        "",
        "Non-generic kind-aware calibration:",
        "  generic is exact WCET-only, but non-generic layers are kind-sensitive.",
        "  input_WCET is a nominal calibration point, not a hard floor that erases the kind effect.",
        f"  CPU kind scale: {cpu_scale}.",
        f"  NPU kind scale: {npu_scale}.",
        "  analytic = sum of the stages computed by the equations above",
        "  kind_target = ceil(input_WCET × scale(kind, resource))",
        "  KIND_CALIBRATION = max(0, kind_target - analytic)",
        "  duration_job = SWITCH_CONFIG + analytic + KIND_CALIBRATION",
    ]

def _build_ratio_section(compute_jobs: List[ComputeJob]) -> List[str]:
    lines: List[str] = ["=== Duration / deadline / period ratio ==="]
    if not compute_jobs:
        lines.append("No compute job.")
        return lines

    rows = []
    for j in compute_jobs:
        dur = _job_duration(j)
        if dur is None:
            continue
        deadline_span = max(0, int(j.deadline) - int(j.release))
        period = int(j.task.period) if j.task.period else 0
        deadline_ratio = _safe_ratio(dur, deadline_span)
        period_ratio = _safe_ratio(dur, period)
        slack = int(j.deadline) - int(j.finish) if j.finish is not None else None
        compute_part = _stage_sum(j, ["SA_COMPUTE", "VECTOR", "CPU_SIMD", "CPU_VECTOR", "CPU_CONTROL", "NPU_CONTROL", "CPU_WCET", "NPU_WCET"])
        memory_part = _stage_sum(j, ["DMA_IN", "LOCAL_MEM_RD", "CAPACITY_SPILL", "LOCAL_MEM_WR", "DMA_OUT", "CPU_CACHE_RD", "CPU_CACHE_SPILL", "CPU_RMW", "CPU_CACHE_WR"])
        floor_part = _stage_sum(j, ["MEASURED_WCET_FLOOR", "CPU_MEASURED_WCET_FLOOR", "KIND_CALIBRATION", "CPU_KIND_CALIBRATION"])
        rows.append((deadline_ratio or 0.0, j, dur, deadline_span, period, deadline_ratio, period_ratio, slack, compute_part, memory_part, floor_part))

    if not rows:
        lines.append("No finished compute job.")
        return lines

    rows.sort(key=lambda x: (-x[0], x[1].job_id))
    lines.append("The most constrained rows are shown first.")
    lines.append("job | resource | duration | deadline_span | duration/deadline | period | duration/period | slack | compute | memory | calibration")
    for _ratio, j, dur, deadline_span, period, dr, pr, slack, compute_part, memory_part, floor_part in rows[:80]:
        dr_txt = _pct(dr) if dr is not None else "-"
        pr_txt = _pct(pr) if pr is not None else "-"
        slack_txt = str(slack) if slack is not None else "-"
        lines.append(
            f"{j.job_id} | {j.mapped_resource} | {dur} | {deadline_span} | {dr_txt} | "
            f"{period} | {pr_txt} | {slack_txt} | {compute_part} | {memory_part} | {floor_part}"
        )
    if len(rows) > 80:
        lines.append(f"... {len(rows) - 80} additional jobs hidden in this summary.")
    return lines


def build_report(result: SimulationResult) -> str:
    compute_jobs = result.compute_jobs
    message_jobs = result.message_jobs

    lines: List[str] = []
    lines.extend(_build_equations_section())
    lines.append("")
    lines.extend(_build_ratio_section(compute_jobs))
    lines.append("")
    lines.append("=== Compute schedule ===")
    for j in compute_jobs:
        miss = "  MISS" if j.finish is not None and j.finish > j.deadline else ""
        stage_txt = ", ".join([f"{s.label}=[{s.start},{s.finish}]" for s in j.stages])
        lines.append(f"{j.job_id} on {j.mapped_resource}: release={j.release}, deadline={j.deadline}, finish={j.finish}{miss}")
        lines.append(f"    {stage_txt}")
        if j.metrics:
            meta = []
            for k in (
                "kind", "output_elements", "read_bytes", "write_bytes", "output_bytes", "payload_flits_64B",
                "working_set_bytes", "macs", "vector_ops", "M", "N", "K", "tile_M", "tile_N", "tile_K",
                "tiles_total", "shape_util", "peak_macs_per_cycle", "ideal_compute_cycles", "tiling_overhead_cycles",
                "sa_cycles", "vector_cycles", "spill_cycles", "input_wcet_cycles", "kind_wcet_scale",
                "kind_calibrated_target_cycles", "kind_calibration_residual_cycles",
                "measured_wcet_cycles", "wcet_floor_residual_cycles",
            ):
                if k in j.metrics:
                    v = j.metrics[k]
                    meta.append(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
            if meta:
                lines.append("    Hardware model: " + ", ".join(meta))

    lines.append("")
    lines.append("=== Communication schedule ===")
    for m in message_jobs:
        lines.append(
            f"{m.msg_id}: {m.src_resource} -> {m.dst_resource}, medium={m.medium}, release={m.release}, start={m.start}, finish={m.finish}, cost={m.cost}, payload={m.payload_units} flits"
        )
        if m.route_nodes:
            lines.append(f"    Route: {' -> '.join(m.route_nodes)}")
        if m.notes:
            lines.append("    " + "; ".join(m.notes))

    hi_miss = sum(1 for j in compute_jobs if j.finish is not None and j.finish > j.deadline and j.task.criticity == "HI")
    lo_miss = sum(1 for j in compute_jobs if j.finish is not None and j.finish > j.deadline and j.task.criticity == "LO")
    unfinished = sum(1 for j in compute_jobs if j.finish is None)

    worst_task: Dict[str, int] = {}
    for j in compute_jobs:
        if j.response_time is not None:
            worst_task[j.task.name] = max(worst_task.get(j.task.name, 0), j.response_time)

    worst_medium: Dict[str, int] = {}
    for m in message_jobs:
        if m.response_time is not None:
            worst_medium[m.medium] = max(worst_medium.get(m.medium, 0), m.response_time)

    worst_chain: Dict[str, int] = defaultdict(int)
    for j in compute_jobs:
        if j.finish is not None and j.task.cnn_id and j.task.cnn_id != "-":
            root_release = j.chain_root_release if j.chain_root_release is not None else j.release
            worst_chain[j.task.cnn_id] = max(worst_chain[j.task.cnn_id], j.finish - root_release)

    lines.append("")
    lines.append("=== NPU task assignment to internal units ===")
    matrix_tasks = sorted({j.task.name for j in compute_jobs if any(s.label == "SA_COMPUTE" for s in j.stages)})
    vector_tasks = sorted({j.task.name for j in compute_jobs if any(s.label == "VECTOR" for s in j.stages)})
    lines.append("Tasks executed in MATRIX_UNIT: " + (", ".join(matrix_tasks) if matrix_tasks else "(none)"))
    lines.append("Tasks executed in VECTOR_UNIT: " + (", ".join(vector_tasks) if vector_tasks else "(none)"))

    lines.append("")
    lines.append("=== Summary ===")
    lines.append(f"HI deadline misses: {hi_miss}")
    lines.append(f"LO deadline misses: {lo_miss}")
    lines.append(f"Unfinished jobs: {unfinished}")

    lines.append("")
    lines.append("=== Worst observed response time by task ===")
    for k in sorted(worst_task):
        lines.append(f"{k}: {worst_task[k]} cycles")

    lines.append("")
    lines.append("=== Worst observed response time by medium ===")
    for k in sorted(worst_medium):
        lines.append(f"{k}: {worst_medium[k]} cycles")

    lines.append("")
    lines.append("=== Worst end-to-end response time by chain ===")
    for k in sorted(worst_chain):
        lines.append(f"{k}: {worst_chain[k]} cycles")

    lines.append("")
    lines.append("=== Systolic model statistics ===")
    sa_jobs = [j for j in compute_jobs if any(s.label == "SA_COMPUTE" for s in j.stages)]
    if sa_jobs:
        avg_util = sum(float(j.metrics.get("shape_util", 0.0)) for j in sa_jobs) / len(sa_jobs)
        total_tiles = sum(int(j.metrics.get("tiles_total", 0)) for j in sa_jobs)
        total_sa = sum(int(j.metrics.get("sa_cycles", 0)) for j in sa_jobs)
        total_vec = sum(int(j.metrics.get("vector_cycles", 0)) for j in compute_jobs)
        lines.append(f"Systolic jobs: {len(sa_jobs)}")
        lines.append(f"Average shape utilization: {avg_util:.3f}")
        lines.append(f"Total tiles: {total_tiles}")
        lines.append(f"Cumulative SA cycles: {total_sa}")
        lines.append(f"Cumulative vector cycles: {total_vec}")
    else:
        lines.append("No systolic job in this example.")

    return "\n".join(lines)


def write_report(path: str, text: str) -> None:
    Path(path).write_text(text, encoding="utf-8")
