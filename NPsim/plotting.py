from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

from .models import ComputeJob, MessageJob, SimulationResult


def _time_axis_transform(unit: str, frequency_mhz: float, tick_cycles: int = 1) -> Tuple[float, str]:
    """Return multiplier from reference cycles to the requested display unit.

    The simulator timeline is stored in reference cycles. If the architecture
    defines reference_frequency_mhz=1000, then 1000 cycles = 1 microsecond.
    """
    u = (unit or "cycles").strip().lower()
    f = max(float(frequency_mhz or 1000.0), 1e-9)
    if u in {"tick", "ticks", "t"}:
        return 1.0 / max(1, int(tick_cycles or 1)), "Time (ticks)"
    if u in {"cycle", "cycles", "cy"}:
        return 1.0, "Time (cycles)"
    if u in {"ns", "nanosecond", "nanoseconds"}:
        return 1000.0 / f, "Time (ns)"
    if u in {"us", "µs", "microsecond", "microseconds"}:
        return 1.0 / f, "Time (µs)"
    if u in {"ms", "millisecond", "milliseconds"}:
        return 1.0 / (f * 1000.0), "Time (ms)"
    if u in {"s", "sec", "second", "seconds"}:
        return 1.0 / (f * 1_000_000.0), "Time (s)"
    return 1.0, "Time (cycles)"


def _short_label(text: str, max_len: int = 28) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "..."


def _job_label(j: ComputeJob) -> str:
    changed = bool(getattr(j.task, "df_changed", False))
    df_period = int(getattr(j.task, "df_period", -1))
    precision = getattr(j.task, "precision", "")
    star = "★" if changed else ""
    if df_period >= 0:
        prefix = f"P{df_period}{star} "
    else:
        prefix = ""
    return f"{prefix}{j.task.name}\n{j.mapped_resource}/{precision}"


def _chain_markers(compute_jobs: Iterable[ComputeJob]) -> Dict[Tuple[str, int], Tuple[int, int]]:
    """Return end-to-end release/deadline markers per CNN instance.

    This is independent from Deadline-First periods.  It makes static runs and
    adaptive runs comparable: each CNN frame/block gets a release line and an
    end-to-end deadline line on the Gantt.
    """
    markers: Dict[Tuple[str, int], List[int]] = defaultdict(list)
    releases: Dict[Tuple[str, int], int] = {}
    for j in compute_jobs:
        cnn = getattr(j.task, "cnn_id", "-") or "-"
        if cnn == "-":
            continue
        key = (cnn, int(j.instance))
        releases[key] = min(releases.get(key, j.release), j.release)
        markers[key].append(j.deadline)
    return {key: (releases[key], min(deadlines)) for key, deadlines in markers.items() if deadlines}


def _dependency_pairs(compute_jobs: List[ComputeJob], message_jobs: List[MessageJob]) -> List[Tuple[str, str, bool]]:
    pairs: List[Tuple[str, str, bool]] = []
    seen = set()
    for j in compute_jobs:
        for pid in j.local_pred_ids:
            key = (pid, j.job_id, False)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    for m in message_jobs:
        key = (m.pred_job_id, m.dst_job_id, True)
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def _action_label(action: str) -> str:
    return {
        "init_npu_first": "init NPU-first",
        "init_cpu_quality": "init CPU QF",
        "qf_repair": "QF repair",
        "repair": "repair",
        "improve": "improve",
        "rollback": "rollback",
        "hold": "hold",
        "quality_first": "Quality-First",
    }.get(action, action)


def _rounded_time(value: float) -> float | int:
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return round(value, 6)


def schedule_to_gantt_metadata(result: SimulationResult, x_unit: str = "cycles", reference_frequency_mhz: float = 1000.0, tick_cycles: int = 1) -> Dict[str, Any]:
    """Return serialisable schedule data for the internal Tkinter Gantt viewer.

    The PNG/SVG backends are static images.  This sidecar keeps the scheduling
    semantics available to the GUI so it can provide zooming and hover tooltips
    without opening the Plotly HTML file in a browser.
    """
    compute_jobs = result.compute_jobs
    message_jobs = result.message_jobs

    lane_order: List[str] = []
    seen = set()
    for j in compute_jobs:
        if j.mapped_resource not in seen:
            lane_order.append(j.mapped_resource)
            seen.add(j.mapped_resource)
    for m in message_jobs:
        if m.medium not in seen:
            lane_order.append(m.medium)
            seen.add(m.medium)

    max_finish = 0
    for j in compute_jobs:
        if j.finish is not None:
            max_finish = max(max_finish, j.finish)
    for m in message_jobs:
        if m.finish is not None:
            max_finish = max(max_finish, m.finish)

    x_scale, x_label = _time_axis_transform(x_unit, reference_frequency_mhz, tick_cycles)

    def tx(value: float | int | None) -> float | None:
        if value is None:
            return None
        return float(value) * x_scale

    compute_items: List[Dict[str, Any]] = []
    for j in compute_jobs:
        if j.start is None or j.finish is None:
            continue
        start = int(j.start)
        finish = int(j.finish)
        deadline = int(j.deadline)
        release = int(j.release)
        duration = finish - start
        miss = finish > deadline
        changed = bool(getattr(j.task, "df_changed", False))
        compute_items.append(
            {
                "type": "compute",
                "id": j.job_id,
                "label": _job_label(j).replace("\n", "  "),
                "task": j.task.name,
                "cnn_id": getattr(j.task, "cnn_id", ""),
                "instance": int(j.instance),
                "resource": j.mapped_resource,
                "support": getattr(j.task, "resource", ""),
                "precision": getattr(j.task, "precision", ""),
                "criticity": getattr(j.task, "criticity", ""),
                "layer_kind": getattr(j.task, "layer_kind", ""),
                "priority": int(getattr(j.task, "priority", 0)),
                "period_cycles": int(getattr(j.task, "period", 0)),
                "wcet_cycles": int(getattr(j, "wcet", 0)),
                "release_cycles": release,
                "start_cycles": start,
                "finish_cycles": finish,
                "duration_cycles": duration,
                "deadline_cycles": deadline,
                "response_cycles": j.response_time,
                "release": _rounded_time(tx(release) or 0),
                "start": _rounded_time(tx(start) or 0),
                "finish": _rounded_time(tx(finish) or 0),
                "duration": _rounded_time(tx(duration) or 0),
                "deadline": _rounded_time(tx(deadline) or 0),
                "response": None if j.response_time is None else _rounded_time(tx(j.response_time) or 0),
                "miss": miss,
                "changed": changed,
                "df_period": int(getattr(j.task, "df_period", -1)),
                "df_action": str(getattr(j.task, "df_action", "")),
                "df_action_label": _action_label(str(getattr(j.task, "df_action", ""))),
                "preds": list(getattr(j.task, "preds", [])),
                "local_pred_ids": list(j.local_pred_ids),
                "incoming_msg_ids": list(j.incoming_msg_ids),
                "direct_predecessors": list(j.direct_predecessors),
                "notes": list(j.notes),
                "stages": [
                    {
                        "label": st.label,
                        "start_cycles": st.start,
                        "finish_cycles": st.finish,
                        "duration_cycles": st.duration,
                        "duration": _rounded_time(tx(st.duration) or 0),
                        "details": st.details,
                    }
                    for st in j.stages
                ],
            }
        )

    message_items: List[Dict[str, Any]] = []
    for m in message_jobs:
        if m.start is None or m.finish is None:
            continue
        start = int(m.start)
        finish = int(m.finish)
        duration = finish - start
        message_items.append(
            {
                "type": "message",
                "id": m.msg_id,
                "label": _short_label(m.msg_id, 42),
                "resource": m.medium,
                "medium": m.medium,
                "pred_job_id": m.pred_job_id,
                "dst_job_id": m.dst_job_id,
                "src_resource": m.src_resource,
                "dst_resource": m.dst_resource,
                "payload_units": int(m.payload_units),
                "cost_cycles": int(m.cost),
                "priority": int(m.priority),
                "period_cycles": int(m.period),
                "release_cycles": m.release,
                "start_cycles": start,
                "finish_cycles": finish,
                "duration_cycles": duration,
                "deadline_cycles": int(m.deadline),
                "release": None if m.release is None else _rounded_time(tx(m.release) or 0),
                "start": _rounded_time(tx(start) or 0),
                "finish": _rounded_time(tx(finish) or 0),
                "duration": _rounded_time(tx(duration) or 0),
                "deadline": _rounded_time(tx(m.deadline) or 0),
                "route_nodes": list(m.route_nodes),
                "route_links": list(m.route_links),
                "notes": list(m.notes),
            }
        )

    df_jobs = [j for j in compute_jobs if getattr(j.task, "df_period", -1) >= 0]
    period_releases: Dict[int, int] = {}
    period_actions: Dict[int, str] = {}
    for j in df_jobs:
        p = int(getattr(j.task, "df_period", -1))
        period_releases[p] = min(period_releases.get(p, j.release), j.release)
        period_actions[p] = str(getattr(j.task, "df_action", ""))

    periods = [
        {
            "period": p,
            "release_cycles": release,
            "release": _rounded_time(tx(release) or 0),
            "action": period_actions.get(p, ""),
            "action_label": _action_label(period_actions.get(p, "")),
        }
        for p, release in sorted(period_releases.items())
    ]

    markers = []
    for (cnn_id, inst), (release, deadline) in sorted(_chain_markers(compute_jobs).items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1])):
        markers.append(
            {
                "cnn_id": cnn_id,
                "instance": int(inst),
                "release_cycles": release,
                "deadline_cycles": deadline,
                "release": _rounded_time(tx(release) or 0),
                "deadline": _rounded_time(tx(deadline) or 0),
            }
        )

    deps = [
        {"pred_job_id": pred_id, "dst_job_id": dst_id, "crossed_medium": crossed}
        for pred_id, dst_id, crossed in _dependency_pairs(compute_jobs, message_jobs)
    ]

    return {
        "schema": "npsim.gantt.v1",
        "x_unit": x_unit,
        "x_label": x_label,
        "x_scale": x_scale,
        "reference_frequency_mhz": reference_frequency_mhz,
        "tick_cycles": max(1, int(tick_cycles or 1)),
        "lanes": lane_order,
        "max_finish_cycles": max_finish,
        "max_finish": _rounded_time(tx(max_finish) or 0),
        "compute": compute_items,
        "messages": message_items,
        "periods": periods,
        "markers": markers,
        "dependencies": deps,
    }


def write_schedule_metadata(result: SimulationResult, outpath: str, x_unit: str = "cycles", reference_frequency_mhz: float = 1000.0, tick_cycles: int = 1) -> str:
    metadata_path = Path(outpath).with_suffix(".json")
    metadata = schedule_to_gantt_metadata(result, x_unit=x_unit, reference_frequency_mhz=reference_frequency_mhz, tick_cycles=tick_cycles)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(metadata_path)


def plot_schedule(result: SimulationResult, outpath: str, x_unit: str = "cycles", reference_frequency_mhz: float = 1000.0, tick_cycles: int = 1) -> None:
    compute_jobs = result.compute_jobs
    message_jobs = result.message_jobs

    lane_order: List[str] = []
    seen = set()
    for j in compute_jobs:
        if j.mapped_resource not in seen:
            lane_order.append(j.mapped_resource)
            seen.add(j.mapped_resource)
    for m in message_jobs:
        if m.medium not in seen:
            lane_order.append(m.medium)
            seen.add(m.medium)

    lanes = {name: idx for idx, name in enumerate(lane_order)}
    max_finish = 0
    for j in compute_jobs:
        if j.finish is not None:
            max_finish = max(max_finish, j.finish)
    for m in message_jobs:
        if m.finish is not None:
            max_finish = max(max_finish, m.finish)

    x_scale, x_label = _time_axis_transform(x_unit, reference_frequency_mhz, tick_cycles)

    def tx(value: float | int) -> float:
        return float(value) * x_scale

    df_jobs = [j for j in compute_jobs if getattr(j.task, "df_period", -1) >= 0]
    period_releases: Dict[int, int] = {}
    period_actions: Dict[int, str] = {}
    for j in df_jobs:
        p = int(getattr(j.task, "df_period", -1))
        period_releases[p] = min(period_releases.get(p, j.release), j.release)
        period_actions[p] = str(getattr(j.task, "df_action", ""))

    fig_h = max(5.2, 0.9 * len(lane_order) + 2.2)
    fig, ax = plt.subplots(figsize=(18, fig_h))
    colors: Dict[str, str] = {}

    def color_for(name: str) -> str:
        if name not in colors:
            colors[name] = f"C{len(colors) % 10}"
        return colors[name]

    def action_label(action: str) -> str:
        return {
            "init_npu_first": "init NPU-first",
            "repair": "repair",
            "improve": "improve",
            "rollback": "rollback",
            "hold": "hold",
            "quality_first": "Quality-First",
        }.get(action, action)

    # Keep period/adaptation labels in a dedicated band above the resource lanes.
    # Otherwise labels such as "P0 / NPU first" can be hidden by CPU/NPU bars.
    top_y = len(lane_order) + 0.35 if lane_order else 0.35
    bottom_y = -0.55

    # CNN frame/block release and deadline markers.  These are the primary
    # visual guide for realistic compact examples: simulate only a few frames,
    # then read the x-axis in ms.
    chain_markers = _chain_markers(compute_jobs)
    marker_items = sorted(chain_markers.items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1]))
    for marker_idx, ((cnn_id, inst), (release, deadline)) in enumerate(marker_items):
        ax.axvline(tx(release), color="0.30", linestyle="-", linewidth=0.9, alpha=0.35, zorder=0)
        ax.axvline(tx(deadline), color="red", linestyle="--", linewidth=1.0, alpha=0.45, zorder=0)
        # Label only the first few and then every distinct release to avoid
        # unreadable text on larger horizons.
        if marker_idx < 12:
            ax.text(
                tx(release),
                bottom_y,
                f"R {cnn_id}[{inst}]",
                fontsize=7,
                rotation=90,
                va="bottom",
                ha="right",
                color="0.25",
                alpha=0.85,
            )
            ax.text(
                tx(deadline),
                bottom_y,
                f"D {cnn_id}[{inst}]",
                fontsize=7,
                rotation=90,
                va="bottom",
                ha="left",
                color="red",
                alpha=0.75,
            )

    # Online DF/QF period labels remain useful, but they are now separate from
    # the CNN release/deadline markers above.
    for p, x in sorted(period_releases.items()):
        ax.axvline(tx(x), color="0.20", linestyle=":" if p else "-", linewidth=1.0, alpha=0.8, zorder=0)
        ax.text(
            tx(x + max(1, int(max_finish * 0.003) if max_finish else 1)),
            top_y,
            f"P{p}\n{action_label(period_actions.get(p, ''))}",
            fontsize=8,
            va="top",
            ha="left",
            color="0.20",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.85},
        )

    job_by_id = {j.job_id: j for j in compute_jobs}

    for j in compute_jobs:
        if j.start is None or j.finish is None:
            continue
        dur = j.finish - j.start
        miss = j.finish > j.deadline
        changed = bool(getattr(j.task, "df_changed", False))
        lane = lanes[j.mapped_resource]
        edge = "red" if miss else ("orange" if changed else "black")
        lw = 2.6 if miss else (2.0 if changed else 0.9)
        ax.barh(lane, dur * x_scale, left=tx(j.start), height=0.50, color=color_for(j.task.name), edgecolor=edge, linewidth=lw, zorder=3)

        # Very small bars cannot hold text; leave them clean rather than making
        # the Gantt unreadable.
        if tx(dur) >= max(tx(max_finish) * 0.012, 0.02):
            ax.text(tx(j.start + dur / 2), lane, _job_label(j), ha="center", va="center", fontsize=7, color="white", zorder=4)

    for m in message_jobs:
        if m.start is None or m.finish is None:
            continue
        dur = m.finish - m.start
        lane = lanes[m.medium]
        ax.barh(lane, dur * x_scale, left=tx(m.start), height=0.36, color="lightgray", edgecolor="black", hatch="//", linewidth=0.8, zorder=2)
        if tx(dur) >= max(tx(max_finish) * 0.015, 0.02):
            ax.text(tx(m.start + dur / 2), lane, _short_label(m.msg_id, 32), ha="center", va="center", fontsize=6.5, color="black", zorder=4)

    # Dependency arrows: local dependencies are solid, dependencies crossing a
    # communication message are dashed.  The arrow endpoints deliberately use
    # producer finish and consumer start so the CNN DAG behavior is visible.
    pairs = _dependency_pairs(compute_jobs, message_jobs)
    if len(pairs) <= 120:
        for pred_id, dst_id, crossed_medium in pairs:
            pred = job_by_id.get(pred_id)
            dst = job_by_id.get(dst_id)
            if pred is None or dst is None or pred.finish is None or dst.start is None:
                continue
            if pred.mapped_resource not in lanes or dst.mapped_resource not in lanes:
                continue
            ax.annotate(
                "",
                xy=(tx(dst.start), lanes[dst.mapped_resource]),
                xytext=(tx(pred.finish), lanes[pred.mapped_resource]),
                arrowprops={
                    "arrowstyle": "->",
                    "color": "0.15" if not crossed_medium else "0.40",
                    "lw": 0.7,
                    "alpha": 0.55 if not crossed_medium else 0.45,
                    "linestyle": "--" if crossed_medium else "-",
                    "shrinkA": 4,
                    "shrinkB": 4,
                },
                zorder=1,
            )

    ax.set_yticks(range(len(lane_order)))
    ax.set_yticklabels(lane_order)
    ax.set_xlabel(x_label)
    title = "CNN schedule: compute, communication, releases, deadlines"
    if df_jobs:
        actions_seen = {str(getattr(j.task, "df_action", "")) for j in df_jobs}
        if any(a.startswith("quality") for a in actions_seen):
            title += " — Quality-First"
        else:
            title += " — Deadline-First"
    ax.set_title(title)
    right_pad = max(5, int(max_finish * 0.04) if max_finish else 5)
    ax.set_xlim(0, tx(max_finish + right_pad))
    ax.set_ylim(bottom_y, len(lane_order) + 0.75 if lane_order else 0.75)
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    plt.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
