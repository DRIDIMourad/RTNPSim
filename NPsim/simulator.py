from __future__ import annotations
from collections import defaultdict
import math
from typing import Dict, List, Optional, Tuple

from .models import ArchConfig, ComputeJob, MessageJob, SimulationResult, StageSegment
from .npu_model import FLIT_BYTES, build_cpu_stages, build_npu_stages, infer_payload_flits
from .topology import path_to_links, shortest_path


def assign_resources(jobs: List[ComputeJob], arch: ArchConfig) -> None:
    """Assign each compute job to a concrete CPU/NPU resource.

    Earlier versions distributed *jobs* round-robin.  That meant the same
    layer could move from NPU0 to NPU1 between two releases even when the
    mapping policy had not changed.  For CNN scheduling this is misleading:
    a layer mapping should be stable across instances unless an algorithm
    explicitly changes it.

    The new behavior distributes unmapped NPU-capable *layers* round-robin
    once, then reuses that layer-to-NPU decision for every instance.  This
    still lets Deadline-First start P0 on multiple NPUs, but keeps the Gantt
    stable and easier to interpret.
    """
    npu_nodes = arch.mapping.npu_nodes or ["NPU0"]
    rr_index = 0
    stable_unmapped: Dict[str, str] = {}
    for j in sorted(jobs, key=lambda x: (x.release, x.task.cnn_id, x.task.priority, x.task.name, x.instance)):
        if j.task.resource == "CPU":
            j.mapped_resource = arch.mapping.cpu_name
            continue
        mapped = arch.mapping.task_to_npu.get(j.task.name)
        if mapped:
            j.mapped_resource = mapped
            continue
        if arch.mapping.round_robin_unmapped:
            if j.task.name not in stable_unmapped:
                stable_unmapped[j.task.name] = npu_nodes[rr_index % len(npu_nodes)]
                rr_index += 1
            j.mapped_resource = stable_unmapped[j.task.name]
        else:
            j.mapped_resource = npu_nodes[0]


def _scale_reference_cycles(cycles: int, reference_mhz: float, local_mhz: float) -> int:
    """Convert local-resource cycles into reference cycles."""
    cycles = int(cycles)
    if cycles <= 0:
        return 0
    if local_mhz <= 0 or reference_mhz <= 0:
        return cycles
    return max(1, int(math.ceil(cycles * reference_mhz / local_mhz)))


def _scale_stage_durations(job: ComputeJob, arch: ArchConfig) -> None:
    """Scale CPU/NPU stage durations according to architecture clocks.

    The simulator's global timeline is expressed in reference cycles.  Stage
    builders compute local CPU or NPU cycles.  This conversion makes the
    `clock` section of arch.yaml operational instead of only descriptive,
    without changing the tasks.txt format.
    """
    if job.mapped_resource == arch.mapping.cpu_name:
        freq = arch.clock.cpu_frequency_mhz
        source = "cpu"
    else:
        freq = arch.clock.npu_frequency_mhz
        source = "npu"
    ref = arch.clock.reference_frequency_mhz
    for st in job.stages:
        if st.label == "SWITCH_CONFIG":
            # Switch costs are user-provided global/reference-cycle overheads.
            continue
        local = int(st.duration)
        scaled = _scale_reference_cycles(local, ref, freq)
        if scaled != local:
            st.details["local_cycles"] = local
            st.details["scaled_from"] = source
            st.details["frequency_mhz"] = float(freq)
            st.duration = scaled


def _scale_message_costs(messages: List[MessageJob], arch: ArchConfig) -> None:
    ref = arch.clock.reference_frequency_mhz
    freq = arch.clock.interconnect_frequency_mhz
    for m in messages:
        local = int(m.cost)
        scaled = _scale_reference_cycles(local, ref, freq)
        if scaled != local:
            m.notes.append(f"Local interconnect cost={local} cycles, converted to {scaled} cycles reference")
            m.cost = scaled
            m.remaining_payload_units = m.payload_units


def _select_key(policy: str):
    p = (policy or "np_fp").lower()
    if p == "edf":
        return lambda obj: (getattr(obj, "deadline", 10**12), getattr(obj, "release", 10**12), getattr(obj, "msg_id", getattr(obj, "job_id", "")))
    if p == "rm":
        return lambda obj: (getattr(getattr(obj, "task", None), "period", 10**12), getattr(obj, "release", 10**12), getattr(obj, "msg_id", getattr(obj, "job_id", "")))
    return lambda obj: (getattr(getattr(obj, "task", None), "priority", 10**12), getattr(obj, "release", 10**12), getattr(obj, "msg_id", getattr(obj, "job_id", "")))


def _find_pred_job(dst: ComputeJob, pred_jobs: List[ComputeJob]) -> Optional[ComputeJob]:
    cands = [j for j in pred_jobs if j.release <= dst.release]
    if not cands:
        return None
    return max(cands, key=lambda j: (j.release, j.instance))


def _payload_units(src: ComputeJob, dst: ComputeJob) -> int:
    """Return communication payload in 64-byte flits.

    Older versions multiplied task size by WCET, which made NoC cost depend on
    compute time rather than produced tensor data.  The new model uses the
    producer output tensor inferred by npu_model.py; if a job was profiled
    already, reuse its metrics, otherwise infer from the source task.
    """
    payload = int(src.metrics.get("payload_flits_64B", 0) or 0)
    if payload <= 0:
        payload = infer_payload_flits(src.task)
    return max(1, payload)


def _message_kind_and_cost(src: ComputeJob, dst: ComputeJob, arch: ArchConfig) -> Tuple[str, int, int, List[str], List[Tuple[str, str]]]:
    ic = arch.interconnect
    payload = _payload_units(src, dst)
    src_res = src.mapped_resource
    dst_res = dst.mapped_resource
    if src_res == dst_res:
        return "LOCAL", payload, 0, [src_res], []
    if src_res == arch.mapping.cpu_name and dst_res.startswith("NPU"):
        medium = "CPU_NPU_SHARED" if ic.cpu_npu_mode == "shared_half_duplex" else "CPU_TO_NPU"
        cost = ic.shared_setup + ic.shared_per_unit * payload if medium == "CPU_NPU_SHARED" else ic.cpu_to_npu_setup + ic.cpu_to_npu_per_unit * payload
        return medium, payload, cost, [src_res, dst_res], []
    if src_res.startswith("NPU") and dst_res == arch.mapping.cpu_name:
        medium = "CPU_NPU_SHARED" if ic.cpu_npu_mode == "shared_half_duplex" else "NPU_TO_CPU"
        cost = ic.shared_setup + ic.shared_per_unit * payload if medium == "CPU_NPU_SHARED" else ic.npu_to_cpu_setup + ic.npu_to_cpu_per_unit * payload
        return medium, payload, cost, [src_res, dst_res], []
    route = shortest_path(ic.noc_topology, arch.mapping.npu_nodes, src_res, dst_res)
    route_links = path_to_links(route)
    hops = max(0, len(route) - 1)
    cost = ic.noc_setup + ic.noc_per_unit * payload + hops * ic.router_latency
    return "NOC", payload, cost, route, route_links


def build_message_jobs(compute_jobs: List[ComputeJob], arch: ArchConfig, policy: str) -> List[MessageJob]:
    by_task: Dict[str, List[ComputeJob]] = defaultdict(list)
    for j in compute_jobs:
        by_task[j.task.name].append(j)
    messages: List[MessageJob] = []
    select_key = _select_key(policy)
    for dst in compute_jobs:
        dst.chain_root_release = dst.release
        for pred_name in dst.task.preds:
            pred = _find_pred_job(dst, by_task.get(pred_name, []))
            if pred is None:
                continue
            dst.chain_root_release = min(dst.chain_root_release, pred.release) if dst.chain_root_release is not None else pred.release
            if pred.mapped_resource == dst.mapped_resource:
                dst.local_pred_ids.append(pred.job_id)
                dst.direct_predecessors.append(pred.job_id)
                continue
            medium, payload, cost, route_nodes, route_links = _message_kind_and_cost(pred, dst, arch)
            msg = MessageJob(
                msg_id=f"MSG_{pred.job_id}_TO_{dst.job_id}",
                pred_job_id=pred.job_id,
                dst_job_id=dst.job_id,
                src_resource=pred.mapped_resource,
                dst_resource=dst.mapped_resource,
                medium=medium,
                payload_units=payload,
                cost=cost,
                priority=dst.task.priority,
                deadline=dst.deadline,
                period=dst.task.period,
                route_nodes=route_nodes,
                route_links=route_links,
                priority_key=select_key(dst),
            )
            msg.notes.append(f"Inferred payload={payload} flits of {FLIT_BYTES} bytes")
            if medium == "NOC":
                msg.notes.append(f"Route: {' -> '.join(route_nodes)}")
            messages.append(msg)
            dst.incoming_msg_ids.append(msg.msg_id)
    return messages


def _all_incoming_messages_done(job: ComputeJob, msg_by_id: Dict[str, MessageJob], t: int) -> bool:
    for mid in job.incoming_msg_ids:
        m = msg_by_id[mid]
        if m.finish is None or m.finish > t:
            return False
    return True


def _all_local_preds_done(job: ComputeJob, jobs_by_id: Dict[str, ComputeJob], t: int) -> bool:
    for pid in job.local_pred_ids:
        p = jobs_by_id[pid]
        if p.finish is None or p.finish > t:
            return False
    return True


def _pred_finished(msg: MessageJob, jobs_by_id: Dict[str, ComputeJob]) -> Optional[int]:
    pred = jobs_by_id[msg.pred_job_id]
    return pred.finish


def _apply_stage_times(job: ComputeJob, start: int) -> int:
    t = start
    for st in job.stages:
        st.start = t
        st.finish = t + st.duration
        t = st.finish
    job.start = start
    job.finish = t
    return t


def simulate(compute_jobs: List[ComputeJob], arch: ArchConfig, policy: str = "np_fp") -> SimulationResult:
    assign_resources(compute_jobs, arch)
    for j in compute_jobs:
        if j.mapped_resource == arch.mapping.cpu_name:
            j.stages, j.metrics = build_cpu_stages(j, arch.cpu, arch.npu)
        else:
            j.stages, j.metrics = build_npu_stages(j, arch.npu)
        _scale_stage_durations(j, arch)
        switch_cost = int(getattr(j.task, "switch_cost", 0) or 0)
        switch_instance = getattr(j.task, "switch_instance", 0)
        switch_applies = switch_instance is None or int(switch_instance) == int(j.instance)
        if switch_cost > 0 and switch_applies:
            j.stages = [StageSegment("SWITCH_CONFIG", switch_cost, details={"cost": switch_cost})] + j.stages
            j.metrics["switch_cost"] = switch_cost

    msg_jobs = build_message_jobs(compute_jobs, arch, policy)
    _scale_message_costs(msg_jobs, arch)
    jobs_by_id = {j.job_id: j for j in compute_jobs}
    msg_by_id = {m.msg_id: m for m in msg_jobs}

    compute_busy_until: Dict[str, int] = {arch.mapping.cpu_name: 0}
    for n in arch.mapping.npu_nodes:
        compute_busy_until[n] = 0

    medium_busy_until: Dict[str, int] = {"CPU_NPU_SHARED": 0, "CPU_TO_NPU": 0, "NPU_TO_CPU": 0}
    noc_link_busy_until: Dict[Tuple[str, str], int] = defaultdict(int)

    select_key = _select_key(policy)
    current = 0
    pending_compute = set(j.job_id for j in compute_jobs)
    pending_messages = set(m.msg_id for m in msg_jobs)

    while pending_compute or pending_messages:
        progress = False

        # Start messages first when predecessor compute has completed.
        ready_msgs = []
        for mid in list(pending_messages):
            m = msg_by_id[mid]
            rel = _pred_finished(m, jobs_by_id)
            if rel is None or rel > current:
                continue
            m.release = rel
            ready_msgs.append(m)
        # shared/separate bus messages
        bus_msgs = [m for m in ready_msgs if m.medium in ("CPU_NPU_SHARED", "CPU_TO_NPU", "NPU_TO_CPU")]
        bus_msgs.sort(key=lambda m: m.priority_key)
        for m in bus_msgs:
            if m.msg_id not in pending_messages:
                continue
            medium = m.medium
            if medium_busy_until[medium] > current:
                continue
            start = max(current, m.release or 0)
            m.start = start
            m.finish = start + m.cost
            medium_busy_until[medium] = m.finish
            pending_messages.remove(m.msg_id)
            progress = True

        # NoC messages with greedy link reservation
        noc_msgs = [m for m in ready_msgs if m.medium == "NOC"]
        arb = arch.interconnect.arb_policy.lower()
        if arb == "round_robin":
            noc_msgs.sort(key=lambda m: (m.release or 0, m.msg_id))
        else:
            noc_msgs.sort(key=lambda m: m.priority_key)
        for m in noc_msgs:
            if m.msg_id not in pending_messages:
                continue
            links = m.route_links
            if any(noc_link_busy_until[lk] > current for lk in links):
                continue
            start = max(current, m.release or 0)
            m.start = start
            m.finish = start + m.cost
            for lk in links:
                noc_link_busy_until[lk] = m.finish
            pending_messages.remove(m.msg_id)
            progress = True

        # Compute jobs
        ready_compute = []
        for jid in list(pending_compute):
            j = jobs_by_id[jid]
            if j.release > current:
                continue
            if not _all_local_preds_done(j, jobs_by_id, current):
                continue
            if not _all_incoming_messages_done(j, msg_by_id, current):
                continue
            ready_compute.append(j)
        ready_compute.sort(key=select_key)
        used_resources = set()
        for j in ready_compute:
            if j.job_id not in pending_compute:
                continue
            res = j.mapped_resource
            if res in used_resources:
                continue
            if compute_busy_until[res] > current:
                continue
            finish = _apply_stage_times(j, current)
            compute_busy_until[res] = finish
            pending_compute.remove(j.job_id)
            used_resources.add(res)
            progress = True

        if progress:
            continue

        # Advance to next interesting event
        candidates: List[int] = []
        for j in compute_jobs:
            if j.job_id in pending_compute and j.release > current:
                candidates.append(j.release)
        for val in compute_busy_until.values():
            if val > current:
                candidates.append(val)
        for mid in pending_messages:
            m = msg_by_id[mid]
            rel = _pred_finished(m, jobs_by_id)
            if rel is not None and rel > current:
                candidates.append(rel)
        for val in medium_busy_until.values():
            if val > current:
                candidates.append(val)
        for val in noc_link_busy_until.values():
            if val > current:
                candidates.append(val)
        if not candidates:
            break
        current = min(candidates)

    stats = {
        "compute_jobs": len(compute_jobs),
        "message_jobs": len(msg_jobs),
        "cpu_jobs": sum(1 for j in compute_jobs if j.mapped_resource == arch.mapping.cpu_name),
        "npu_jobs": sum(1 for j in compute_jobs if j.mapped_resource.startswith("NPU")),
        "noc_messages": sum(1 for m in msg_jobs if m.medium == "NOC"),
        "cpu_npu_messages": sum(1 for m in msg_jobs if m.medium != "NOC"),
    }
    return SimulationResult(compute_jobs=compute_jobs, message_jobs=msg_jobs, stats=stats)


def validate_schedule(result: SimulationResult) -> List[str]:
    """Run structural checks that make the Gantt trustworthy.

    The simulator is intentionally lightweight, so the Gantt is the main way to
    inspect CNN behavior.  These checks catch the most damaging visualization
    errors: compute overlap on a CPU/NPU lane, stages outside their job box,
    broken dependency ordering, and overlapping communications.
    """
    issues: List[str] = []
    jobs_by_id = {j.job_id: j for j in result.compute_jobs}
    msg_by_id = {m.msg_id: m for m in result.message_jobs}

    by_resource: Dict[str, List[ComputeJob]] = defaultdict(list)

    for j in result.compute_jobs:
        if j.start is None or j.finish is None:
            issues.append(f"{j.job_id}: compute job not scheduled")
            continue

        by_resource[j.mapped_resource].append(j)

        if j.start < j.release:
            issues.append(f"{j.job_id}: start={j.start} avant release={j.release}")
        if j.finish < j.start:
            issues.append(f"{j.job_id}: finish={j.finish} avant start={j.start}")
        if j.finish > j.deadline:
            j.notes.append(f"Deadline miss: finish={j.finish} > deadline={j.deadline}")

        # Stage consistency: a job bar must be exactly the concatenation of its
        # internal stages.  This protects the detailed CPU/NPU stage reports.
        expected_start = j.start
        for st in j.stages:
            if st.start is None or st.finish is None:
                issues.append(f"{j.job_id}: stage {st.label} not scheduled")
                continue
            if st.start < j.start or st.finish > j.finish:
                issues.append(
                    f"{j.job_id}: stage {st.label} [{st.start},{st.finish}] hors job [{j.start},{j.finish}]"
                )
            if st.start != expected_start:
                issues.append(
                    f"{j.job_id}: stage {st.label} starts at {st.start}, expected {expected_start}"
                )
            if st.finish < st.start:
                issues.append(f"{j.job_id}: stage {st.label} finish={st.finish} avant start={st.start}")
            expected_start = st.finish
        if j.stages and expected_start != j.finish:
            issues.append(f"{j.job_id}: finish job={j.finish}, finish dernier stage={expected_start}")

        for pid in j.local_pred_ids:
            p = jobs_by_id[pid]
            if p.finish is None:
                issues.append(f"{j.job_id}: local predecessor {pid} not finished")
            elif j.start < p.finish:
                issues.append(f"{j.job_id}: starts before local predecessor finishes {pid} ({p.finish})")
        for mid in j.incoming_msg_ids:
            m = msg_by_id[mid]
            if m.finish is None:
                issues.append(f"{j.job_id}: incoming message {mid} not finished")
            elif j.start < m.finish:
                issues.append(f"{j.job_id}: starts before message finishes {mid} ({m.finish})")

    for resource, arr in by_resource.items():
        arr.sort(key=lambda x: (x.start or 0, x.finish or 0, x.job_id))
        for a, b in zip(arr, arr[1:]):
            if a.finish is not None and b.start is not None and a.finish > b.start:
                issues.append(
                    f"Chevauchement compute {resource}: {a.job_id} [{a.start},{a.finish}] avec {b.job_id} [{b.start},{b.finish}]"
                )

    by_medium: Dict[str, List[MessageJob]] = defaultdict(list)
    for m in result.message_jobs:
        if m.start is None or m.finish is None:
            issues.append(f"{m.msg_id}: message not scheduled")
            continue
        if m.finish < m.start:
            issues.append(f"{m.msg_id}: finish={m.finish} avant start={m.start}")
        if m.start < (m.release or 0):
            issues.append(f"{m.msg_id}: start={m.start} avant release={m.release}")
        pred = jobs_by_id.get(m.pred_job_id)
        dst = jobs_by_id.get(m.dst_job_id)
        if pred is not None and pred.finish is not None and m.start < pred.finish:
            issues.append(f"{m.msg_id}: starts before producer finishes {m.pred_job_id} ({pred.finish})")
        if dst is not None and dst.start is not None and dst.start < m.finish:
            issues.append(f"{m.dst_job_id}: starts before transfer finishes {m.msg_id} ({m.finish})")
        by_medium[m.medium].append(m)

    for medium, arr in by_medium.items():
        if medium not in ("CPU_NPU_SHARED", "CPU_TO_NPU", "NPU_TO_CPU"):
            continue
        arr.sort(key=lambda x: (x.start or 0, x.finish or 0, x.msg_id))
        for a, b in zip(arr, arr[1:]):
            if a.finish is not None and b.start is not None and a.finish > b.start:
                issues.append(f"Chevauchement medium {medium}: {a.msg_id} [{a.start},{a.finish}] avec {b.msg_id} [{b.start},{b.finish}]")

    link_use: Dict[Tuple[str, str], List[MessageJob]] = defaultdict(list)
    for m in result.message_jobs:
        if m.start is None or m.finish is None:
            continue
        for lk in m.route_links:
            link_use[lk].append(m)
    for lk, arr in link_use.items():
        arr.sort(key=lambda x: (x.start or 0, x.finish or 0, x.msg_id))
        for a, b in zip(arr, arr[1:]):
            if a.finish is not None and b.start is not None and a.finish > b.start:
                issues.append(f"Chevauchement lien NoC {lk}: {a.msg_id} [{a.start},{a.finish}] avec {b.msg_id} [{b.start},{b.finish}]")

    return issues
