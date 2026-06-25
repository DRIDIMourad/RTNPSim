from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

from .models import CPUConfig, ComputeJob, StageSegment, SystolicArrayConfig, Task


# The simulator keeps the historical tasks.txt schema. Explicit `generic` tasks
# are WCET-only. For non-generic tasks, the last integer field (`size`) is
# interpreted as an approximate output-tensor size in Ki-elements for
# architectural estimation.
BYTES_PER_ELEMENT = {"INT8": 1, "FP16": 2, "FP32": 4}
PRECISION_THROUGHPUT = {"INT8": 1.0, "FP16": 0.5, "FP32": 0.25}
FLIT_BYTES = 64


@dataclass(frozen=True)
class LayerProfile:
    kind: str
    output_elements: int
    input_elements: int
    weight_elements: int
    macs: int
    vector_ops: int
    read_bytes: int
    write_bytes: int
    output_bytes: int
    working_set_bytes: int
    M: int
    N: int
    K: int
    kernel_ops_per_output: int


LAYER_ALIASES = {
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
    "bn": "bn",
    "batchnorm": "bn",
    "activation": "activation",
    "relu": "activation",
    "fuse": "fuse",
    "concat": "fuse",
    "merge": "fuse",
    "add": "fuse",
    "identity": "fuse",
    "softmax": "fuse",
    "postprocess": "fuse",
    "post": "fuse",
    "nms": "nms",
    "control": "control",
    "ctrl": "control",
}


# Per-kind architectural defaults.  These are deliberately expressed as real
# tensor/operation concepts instead of WCET-derived proxies.
KERNEL_OPS = {
    "systolic": 576,      # e.g. 3x3 conv with ~64 input channels per output element
    "depthwise": 9,
    "pool": 4,
    "bn": 4,
    "activation": 1,
    "fuse": 2,
    "nms": 16,
    "control": 1,
    "generic": 0,
}

INPUT_FACTOR = {
    "systolic": 1.50,
    "depthwise": 1.10,
    "pool": 1.25,
    "bn": 1.00,
    "activation": 1.00,
    "fuse": 2.00,
    "nms": 1.00,
    "control": 1.00,
    "generic": 0.00,
}

WEIGHT_FACTOR = {
    "systolic": 0.25,
    "depthwise": 0.02,
    "pool": 0.00,
    "bn": 0.02,
    "activation": 0.00,
    "fuse": 0.00,
    "nms": 0.00,
    "control": 0.00,
    "generic": 0.00,
}


# Non-generic tasks are not WCET-only.  The WCET columns are treated as
# nominal calibration points, then the layer kind/resource adjusts the target
# duration so a convolution, depthwise layer, pooling layer and activation no
# longer collapse to the same timing as `generic`.  Generic tasks remain exact
# WCET-only by design.
KIND_WCET_SCALE = {
    "CPU": {
        "systolic": 1.35,
        "depthwise": 0.85,
        "pool": 0.45,
        "bn": 0.65,
        "activation": 0.35,
        "fuse": 0.55,
        "nms": 1.40,
        "control": 1.00,
    },
    "NPU": {
        "systolic": 1.10,
        "depthwise": 0.70,
        "pool": 0.40,
        "bn": 0.55,
        "activation": 0.30,
        "fuse": 0.50,
        "nms": 1.10,
        "control": 1.00,
    },
}


def kind_wcet_scale(kind: str, resource: str) -> float:
    resource_key = "NPU" if str(resource).upper().startswith("NPU") else "CPU"
    return float(KIND_WCET_SCALE.get(resource_key, {}).get(kind, 1.0))


def infer_layer_kind(task: Task) -> str:
    """Return the hardware-model layer family used by timing equations.

    The parser already performs name-based fallback for legacy tasks.txt files
    that do not contain an explicit kind column. Therefore an explicit GUI kind
    of ``generic`` stays generic here. Generic tasks are intentionally WCET-only.
    """
    token = (task.layer_kind or "generic").strip().lower().replace("-", "_")
    kind = LAYER_ALIASES.get(token, token if token else "generic")
    return kind if kind in KERNEL_OPS else "generic"

def _precision_bytes(task: Task) -> int:
    return BYTES_PER_ELEMENT.get(task.precision.upper(), 1)


def _precision_efficiency(task: Task) -> float:
    return PRECISION_THROUGHPUT.get(task.precision.upper(), 1.0)


def _size_to_output_elements(task: Task, kind: str) -> int:
    """Convert tasks.txt `size` to approximate output tensor elements.

    Convention for non-generic tasks: size = output Ki-elements. This keeps the
    text file unchanged while giving memory, NoC and vector equations a physical
    unit. Generic tasks skip this path and use WCET only.
    """
    size = max(1, int(task.size))
    min_kie = {
        "systolic": 16,
        "depthwise": 16,
        "pool": 8,
        "bn": 8,
        "activation": 8,
        "fuse": 4,
        "nms": 1,
        "control": 1,
        "generic": 4,
    }.get(kind, 4)
    return max(min_kie, size) * 1024


def _name_adjusted_kernel_ops(task: Task, kind: str) -> int:
    name = task.name.upper()
    base = KERNEL_OPS.get(kind, KERNEL_OPS["generic"])
    if kind == "systolic":
        if "STEM" in name:
            return 3 * 3 * 32
        if "DEEP" in name or "BACKBONE" in name:
            return 3 * 3 * 128
        if "NECK" in name or "FPN" in name:
            return 1 * 1 * 256
        if "HEAD" in name or "DETECT" in name:
            return 1 * 1 * 128
        if "PW" in name or "POINTWISE" in name:
            return 1 * 1 * 128
        if "FC" in name:
            return 512
    if kind == "nms":
        return 32
    return base


def infer_layer_profile(task: Task) -> LayerProfile:
    """Infer operations and bytes from task kind/name/size, not from WCET.

    For non-generic architectural layer kinds, size is interpreted as output
    Ki-elements and the model estimates bytes/MACs/vector ops. For explicit
    ``generic`` tasks, the simulator uses WCET only; size does not affect compute
    time or memory time. Generic output payload is one 64-byte flit so DAG
    dependencies still have a minimal communication token.
    """
    kind = infer_layer_kind(task)
    if kind == "generic":
        return LayerProfile(
            kind="generic",
            output_elements=0,
            input_elements=0,
            weight_elements=0,
            macs=0,
            vector_ops=0,
            read_bytes=0,
            write_bytes=0,
            output_bytes=FLIT_BYTES,
            working_set_bytes=0,
            M=1,
            N=1,
            K=1,
            kernel_ops_per_output=0,
        )

    bpe = _precision_bytes(task)
    out_elems = _size_to_output_elements(task, kind)
    in_elems = max(1, int(math.ceil(out_elems * INPUT_FACTOR.get(kind, 1.0))))
    k_ops = max(1, _name_adjusted_kernel_ops(task, kind))

    if kind == "systolic":
        macs = out_elems * k_ops
        weight_elems = max(1024, int(math.ceil(out_elems * WEIGHT_FACTOR["systolic"])))
        vector_ops = 0
    elif kind == "depthwise":
        # Depthwise convolution is mostly memory/vector-like in this lightweight
        # model, but it still performs one small kernel per output element.
        macs = out_elems * k_ops
        weight_elems = max(128, int(math.ceil(out_elems * WEIGHT_FACTOR["depthwise"])))
        vector_ops = out_elems * 2
    elif kind in ("pool", "bn", "activation", "fuse", "nms"):
        macs = 0
        weight_elems = max(0, int(math.ceil(out_elems * WEIGHT_FACTOR.get(kind, 0.0))))
        if kind == "nms":
            # Approximate NMS as sub-quadratic post-processing over candidate
            # boxes; cap growth so a single GUI size typo does not explode.
            candidates = min(max(64, out_elems // 64), 8192)
            vector_ops = candidates * int(math.ceil(math.log2(max(2, candidates)))) * k_ops
        else:
            vector_ops = out_elems * k_ops
    elif kind == "control":
        macs = 0
        weight_elems = 0
        vector_ops = max(1, task.wcet())
    else:
        macs = 0
        weight_elems = 0
        vector_ops = 0

    input_bytes = in_elems * bpe
    weight_bytes = weight_elems * bpe
    output_bytes = out_elems * bpe

    # Read includes activations plus weights/parameters.  Fuse/add/concat style
    # layers read more than one input activation through INPUT_FACTOR.
    read_bytes = max(64, input_bytes + weight_bytes)
    write_bytes = max(64, output_bytes)
    working_set = read_bytes + write_bytes

    # Synthetic GEMM dimensions for SA stage details. Compute cost uses MACs,
    # but dimensions help the Gantt/report explain tiling and utilization.
    if kind == "systolic":
        K = max(1, k_ops)
        N = 64 if "HEAD" not in task.name.upper() else 128
        if "FC" in task.name.upper():
            N = 256
        M = max(1, int(math.ceil(out_elems / N)))
    elif kind == "depthwise":
        K = k_ops
        N = 1
        M = max(1, out_elems)
    else:
        K = max(1, k_ops)
        N = 1
        M = max(1, out_elems)

    return LayerProfile(
        kind=kind,
        output_elements=out_elems,
        input_elements=in_elems,
        weight_elements=weight_elems,
        macs=max(0, int(macs)),
        vector_ops=max(0, int(vector_ops)),
        read_bytes=max(64, int(read_bytes)),
        write_bytes=max(64, int(write_bytes)),
        output_bytes=max(64, int(output_bytes)),
        working_set_bytes=max(64, int(working_set)),
        M=max(1, int(M)),
        N=max(1, int(N)),
        K=max(1, int(K)),
        kernel_ops_per_output=max(1, int(k_ops)),
    )

def infer_bytes(task: Task) -> Tuple[int, int]:
    profile = infer_layer_profile(task)
    return profile.read_bytes, profile.write_bytes


def infer_output_bytes(task: Task) -> int:
    return infer_layer_profile(task).output_bytes


def infer_payload_flits(task: Task) -> int:
    return max(1, int(math.ceil(infer_output_bytes(task) / FLIT_BYTES)))


def infer_gemm_dims(task: Task, arch: SystolicArrayConfig) -> Dict[str, int]:
    profile = infer_layer_profile(task)
    return {"M": profile.M, "N": profile.N, "K": profile.K}


def _tile_shape(profile: LayerProfile, arch: SystolicArrayConfig) -> Tuple[int, int, int, int, int, int, int]:
    tm = min(profile.M, max(1, arch.rows))
    tn = min(profile.N, max(1, arch.cols))
    tk = min(profile.K, max(8, arch.cols))
    tiles_m = math.ceil(profile.M / tm)
    tiles_n = math.ceil(profile.N / tn)
    tiles_k = math.ceil(profile.K / tk)
    return tm, tn, tk, tiles_m, tiles_n, tiles_k, tiles_m * tiles_n * tiles_k


def _systolic_cycles(profile: LayerProfile, task: Task, arch: SystolicArrayConfig) -> Tuple[int, Dict[str, float]]:
    if profile.macs <= 0:
        return 0, {}
    precision_eff = max(0.05, _precision_efficiency(task))
    peak = max(1.0, arch.rows * arch.cols * max(1, arch.arrays) * precision_eff)
    ideal_cycles = math.ceil(profile.macs / peak)
    tm, tn, tk, tiles_m, tiles_n, tiles_k, tiles_total = _tile_shape(profile, arch)
    fill_drain_per_tile = arch.setup_cycles + tm + tn + tk - 2
    tiling_overhead = math.ceil(tiles_total * fill_drain_per_tile / max(1, arch.arrays))
    cycles = max(1, ideal_cycles + tiling_overhead)
    details = {
        "macs": profile.macs,
        "ideal_compute_cycles": ideal_cycles,
        "tiling_overhead_cycles": tiling_overhead,
        "precision_efficiency": precision_eff,
        "peak_macs_per_cycle": peak,
        "M": profile.M,
        "N": profile.N,
        "K": profile.K,
        "tile_M": tm,
        "tile_N": tn,
        "tile_K": tk,
        "tiles_m": tiles_m,
        "tiles_n": tiles_n,
        "tiles_k": tiles_k,
        "tiles_total": tiles_total,
        "shape_util": min(1.0, (tm * tn) / max(1, arch.rows * arch.cols)),
    }
    return cycles, details


def _memory_read_cycles(bytes_count: int, latency: int, bandwidth: int) -> int:
    return int(latency) + math.ceil(max(0, int(bytes_count)) / max(1, int(bandwidth)))


def _spill_cycles(profile: LayerProfile, capacity_kb: int, bandwidth: int, latency: int) -> int:
    capacity_bytes = max(1, int(capacity_kb)) * 1024
    if profile.working_set_bytes <= capacity_bytes:
        return 0
    spill_bytes = profile.working_set_bytes - capacity_bytes
    # Two touches: spill/fill between tiles. This is a simplified capacity term,
    # but it finally makes the capacity field operational.
    return int(latency) + math.ceil(2 * spill_bytes / max(1, int(bandwidth)))




def _apply_kind_calibration(
    stages: List[StageSegment],
    wcet_cycles: int,
    profile: LayerProfile,
    resource: str,
    label: str,
) -> Tuple[int, int, float]:
    """Append a residual stage to reach the kind-adjusted calibration target.

    For non-generic layers, the selected WCET is no longer a hard measured
    floor that makes every kind behave like `generic`.  Instead, WCET is a
    nominal calibration point and the layer kind/resource changes the final
    target: convolutions are heavier, pooling/activation are lighter, etc.
    The analytic hardware stages still win whenever they exceed that target.
    """
    analytic = sum(max(0, int(stage.duration)) for stage in stages)
    scale = kind_wcet_scale(profile.kind, resource)
    target = max(1, int(math.ceil(max(1, int(wcet_cycles)) * scale)))
    residual = max(0, target - analytic)
    if residual > 0:
        stages.append(StageSegment(label, residual, details={
            "input_wcet_cycles": int(wcet_cycles),
            "kind_wcet_scale": scale,
            "kind_calibrated_target_cycles": target,
            "analytic_cycles_before_calibration": analytic,
        }))
    return residual, target, scale

def build_npu_stages(job: ComputeJob, arch: SystolicArrayConfig) -> Tuple[List[StageSegment], Dict[str, float]]:
    task = job.task
    profile = infer_layer_profile(task)

    if profile.kind == "generic":
        cycles = max(1, int(job.wcet))
        return [StageSegment("NPU_WCET", cycles, details={"wcet": cycles, "generic_wcet_only": True})], {
            "model": 3,
            "kind": "generic",
            "generic_wcet_only": True,
            "measured_wcet_cycles": cycles,
            "input_wcet_cycles": cycles,
            "wcet_floor_residual_cycles": 0,
            "kind_calibration_residual_cycles": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "output_bytes": FLIT_BYTES,
            "payload_flits_64B": 1,
            "working_set_bytes": 0,
            "output_elements": 0,
            "input_elements": 0,
            "weight_elements": 0,
            "macs": 0,
            "vector_ops": 0,
        }

    dma_in = arch.dma_setup + math.ceil(profile.read_bytes / max(1, arch.dma_bandwidth_bytes_per_tick))
    local_rd = _memory_read_cycles(profile.read_bytes, arch.local_dram_latency, arch.local_dram_bandwidth_bytes_per_tick)
    sa_cycles, sa_details = _systolic_cycles(profile, task, arch) if profile.kind == "systolic" else (0, {})

    vector_cycles = 0
    if profile.kind in ("depthwise", "pool", "bn", "activation", "fuse", "nms", "generic") and profile.vector_ops > 0:
        vector_cycles = arch.vector_setup + math.ceil(profile.vector_ops / max(1, arch.vector_lanes))

    spill = _spill_cycles(profile, arch.local_dram_capacity_kb, arch.dma_bandwidth_bytes_per_tick, arch.local_dram_latency)
    local_wr = _memory_read_cycles(profile.write_bytes, arch.local_dram_latency, arch.local_dram_bandwidth_bytes_per_tick)
    dma_out = arch.dma_setup + math.ceil(profile.write_bytes / max(1, arch.dma_bandwidth_bytes_per_tick))

    stages: List[StageSegment] = [
        StageSegment("DMA_IN", dma_in, details={"bytes": profile.read_bytes}),
        StageSegment("LOCAL_MEM_RD", local_rd, details={"bytes": profile.read_bytes}),
    ]
    if spill > 0:
        stages.append(StageSegment("CAPACITY_SPILL", spill, details={
            "working_set_bytes": profile.working_set_bytes,
            "capacity_bytes": max(1, int(arch.local_dram_capacity_kb)) * 1024,
        }))
    if sa_cycles > 0:
        stages.append(StageSegment("SA_COMPUTE", sa_cycles, details=sa_details))
    if vector_cycles > 0:
        stages.append(StageSegment("VECTOR", vector_cycles, details={
            "ops": profile.vector_ops,
            "lanes": arch.vector_lanes,
        }))
    if profile.kind == "control":
        stages.append(StageSegment("NPU_CONTROL", max(1, job.wcet), details={"wcet_fallback": job.wcet}))
    stages.extend([
        StageSegment("LOCAL_MEM_WR", local_wr, details={"bytes": profile.write_bytes}),
        StageSegment("DMA_OUT", dma_out, details={"bytes": profile.write_bytes}),
    ])

    calibration_residual, calibration_target, calibration_scale = _apply_kind_calibration(
        stages, job.wcet, profile, "NPU", "KIND_CALIBRATION"
    )

    metrics = {
        "model": 2,
        "kind": profile.kind,
        "input_wcet_cycles": job.wcet,
        "kind_wcet_scale": calibration_scale,
        "kind_calibrated_target_cycles": calibration_target,
        "kind_calibration_residual_cycles": calibration_residual,
        "read_bytes": profile.read_bytes,
        "write_bytes": profile.write_bytes,
        "output_bytes": profile.output_bytes,
        "payload_flits_64B": max(1, math.ceil(profile.output_bytes / FLIT_BYTES)),
        "working_set_bytes": profile.working_set_bytes,
        "output_elements": profile.output_elements,
        "input_elements": profile.input_elements,
        "weight_elements": profile.weight_elements,
        "macs": profile.macs,
        "vector_ops": profile.vector_ops,
        "M": profile.M,
        "N": profile.N,
        "K": profile.K,
        "kernel_ops_per_output": profile.kernel_ops_per_output,
        "sa_cycles": sa_cycles,
        "vector_cycles": vector_cycles,
        "spill_cycles": spill,
        "peak_macs_per_cycle": sa_details.get("peak_macs_per_cycle", 0.0),
        "tile_M": sa_details.get("tile_M", 0),
        "tile_N": sa_details.get("tile_N", 0),
        "tile_K": sa_details.get("tile_K", 0),
        "tiles_total": sa_details.get("tiles_total", 0),
        "shape_util": sa_details.get("shape_util", 0.0),
    }
    return stages, metrics


def _cpu_mem_cycles(bytes_count: int, latency: int, bandwidth: int) -> int:
    return int(latency) + math.ceil(max(0, int(bytes_count)) / max(1, int(bandwidth)))


def _cpu_spill_cycles(profile: LayerProfile, capacity_kb: int, bandwidth: int, latency: int) -> int:
    capacity_bytes = max(1, int(capacity_kb)) * 1024
    if profile.working_set_bytes <= capacity_bytes:
        return 0
    spill_bytes = profile.working_set_bytes - capacity_bytes
    return int(latency) + math.ceil(spill_bytes / max(1, int(bandwidth)))


def build_cpu_stages(job: ComputeJob, cpu: CPUConfig, npu_ref: SystolicArrayConfig) -> Tuple[List[StageSegment], Dict[str, float]]:
    task = job.task
    profile = infer_layer_profile(task)

    if profile.kind == "generic":
        cycles = max(1, int(job.wcet))
        return [StageSegment("CPU_WCET", cycles, details={"wcet": cycles, "generic_wcet_only": True})], {
            "model": 3,
            "kind": "generic",
            "generic_wcet_only": True,
            "measured_wcet_cycles": cycles,
            "input_wcet_cycles": cycles,
            "wcet_floor_residual_cycles": 0,
            "kind_calibration_residual_cycles": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "output_bytes": FLIT_BYTES,
            "payload_flits_64B": 1,
            "working_set_bytes": 0,
            "output_elements": 0,
            "input_elements": 0,
            "weight_elements": 0,
            "macs": 0,
            "vector_ops": 0,
        }

    if profile.kind == "control":
        cycles = max(1, job.wcet)
        return [StageSegment("CPU_CONTROL", cycles, details={"wcet": job.wcet})], {
            "model": 2,
            "kind": profile.kind,
            "output_bytes": profile.output_bytes,
            "payload_flits_64B": max(1, math.ceil(profile.output_bytes / FLIT_BYTES)),
            "wcet": job.wcet,
        }

    cache_rd = _cpu_mem_cycles(profile.read_bytes, cpu.cache_latency, cpu.cache_bandwidth_bytes_per_tick)
    cache_wr = _cpu_mem_cycles(profile.write_bytes, cpu.cache_latency, cpu.cache_bandwidth_bytes_per_tick)
    spill = _cpu_spill_cycles(profile, cpu.cache_capacity_kb, cpu.rmw_bandwidth_bytes_per_tick, cpu.cache_latency)

    stages: List[StageSegment] = [StageSegment("CPU_CACHE_RD", cache_rd, details={"bytes": profile.read_bytes})]
    if spill > 0:
        stages.append(StageSegment("CPU_CACHE_SPILL", spill, details={
            "working_set_bytes": profile.working_set_bytes,
            "cache_capacity_bytes": max(1, int(cpu.cache_capacity_kb)) * 1024,
        }))

    metrics = {
        "model": 2,
        "kind": profile.kind,
        "read_bytes": profile.read_bytes,
        "write_bytes": profile.write_bytes,
        "output_bytes": profile.output_bytes,
        "payload_flits_64B": max(1, math.ceil(profile.output_bytes / FLIT_BYTES)),
        "working_set_bytes": profile.working_set_bytes,
        "output_elements": profile.output_elements,
        "input_elements": profile.input_elements,
        "weight_elements": profile.weight_elements,
        "macs": profile.macs,
        "vector_ops": profile.vector_ops,
        "M": profile.M,
        "N": profile.N,
        "K": profile.K,
        "kernel_ops_per_output": profile.kernel_ops_per_output,
        "cpu_spill_cycles": spill,
    }

    if profile.macs > 0 and profile.kind in ("systolic", "depthwise", "generic"):
        pack_work = profile.input_elements + profile.weight_elements
        pack_cycles = cpu.setup_cycles + math.ceil(pack_work / max(1, cpu.pack_elements_per_tick))
        simd_cycles = cpu.setup_cycles + math.ceil(profile.macs / max(1, cpu.simd_mac_per_tick))
        epilogue_cycles = cpu.setup_cycles + math.ceil(profile.output_elements / max(1, cpu.epilogue_elements_per_tick))
        stages.extend([
            StageSegment("CPU_LAYOUT_PACK", pack_cycles, details={"elements": pack_work}),
            StageSegment("CPU_SIMD", simd_cycles, details={"macs": profile.macs, "mac_per_cycle": cpu.simd_mac_per_tick}),
            StageSegment("CPU_EPILOGUE", epilogue_cycles, details={"elements": profile.output_elements}),
        ])
        metrics.update({
            "cpu_pack_cycles": pack_cycles,
            "cpu_simd_cycles": simd_cycles,
            "cpu_epilogue_cycles": epilogue_cycles,
        })
    else:
        ops = max(1, profile.vector_ops)
        vector_cycles = cpu.setup_cycles + math.ceil(ops / max(1, cpu.vector_elements_per_tick))
        rmw_cycles = _cpu_mem_cycles(profile.read_bytes + profile.write_bytes, cpu.cache_latency, cpu.rmw_bandwidth_bytes_per_tick)
        stages.extend([
            StageSegment("CPU_VECTOR", vector_cycles, details={"ops": ops, "lanes": cpu.vector_elements_per_tick}),
            StageSegment("CPU_RMW", rmw_cycles, details={"bytes": profile.read_bytes + profile.write_bytes}),
        ])
        metrics.update({
            "cpu_vector_cycles": vector_cycles,
            "cpu_rmw_cycles": rmw_cycles,
        })

    stages.append(StageSegment("CPU_CACHE_WR", cache_wr, details={"bytes": profile.write_bytes}))
    calibration_residual, calibration_target, calibration_scale = _apply_kind_calibration(
        stages, job.wcet, profile, "CPU", "CPU_KIND_CALIBRATION"
    )
    metrics.update({
        "input_wcet_cycles": job.wcet,
        "kind_wcet_scale": calibration_scale,
        "kind_calibrated_target_cycles": calibration_target,
        "kind_calibration_residual_cycles": calibration_residual,
    })
    return stages, metrics
