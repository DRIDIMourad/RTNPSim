from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

try:  # Plotly is optional at import time so PNG/SVG users are not blocked.
    import plotly.graph_objects as go
except Exception as exc:  # pragma: no cover - exercised only when dependency is missing
    go = None  # type: ignore[assignment]
    _PLOTLY_IMPORT_ERROR = exc
else:
    _PLOTLY_IMPORT_ERROR = None

from .models import ComputeJob, MessageJob, SimulationResult
from .plotting import _time_axis_transform


_COMPUTE_COLORS = [
    "#2563eb", "#7c3aed", "#059669", "#dc2626", "#ea580c", "#0891b2",
    "#9333ea", "#16a34a", "#be123c", "#4f46e5", "#0f766e", "#a16207",
]


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
    return f"{prefix}{j.task.name}<br>{j.mapped_resource}/{precision}"


def _plain_job_label(j: ComputeJob) -> str:
    changed = bool(getattr(j.task, "df_changed", False))
    df_period = int(getattr(j.task, "df_period", -1))
    precision = getattr(j.task, "precision", "")
    star = "★" if changed else ""
    if df_period >= 0:
        prefix = f"P{df_period}{star} "
    else:
        prefix = ""
    return f"{prefix}{j.task.name} {j.mapped_resource}/{precision}"


def _chain_markers(compute_jobs: Iterable[ComputeJob]) -> Dict[Tuple[str, int], Tuple[int, int]]:
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


def _format_duration(value: float, unit: str) -> str:
    if abs(value) >= 100:
        return f"{value:,.1f} {unit}"
    if abs(value) >= 10:
        return f"{value:,.2f} {unit}"
    return f"{value:,.3f} {unit}"


def _display_unit_label(x_label: str) -> str:
    start = x_label.find("(")
    end = x_label.find(")", start + 1)
    if start >= 0 and end > start:
        return x_label[start + 1 : end]
    return x_label.replace("Time", "").strip() or "cycles"


def plot_schedule_html(
    result: SimulationResult,
    outpath: str,
    x_unit: str = "ms",
    reference_frequency_mhz: float = 1000.0,
    tick_cycles: int = 1,
    *,
    show_dependencies: bool = True,
    show_markers: bool = True,
) -> None:
    """Write a polished interactive HTML Gantt chart using Plotly.

    The simulator timeline remains in reference cycles.  x_unit controls only
    display, exactly like the Matplotlib backend.
    """
    if go is None:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "Plotly is required for HTML Gantt output. Install it with `pip install plotly`."
        ) from _PLOTLY_IMPORT_ERROR

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

    lane_index = {name: idx for idx, name in enumerate(lane_order)}
    max_finish = 0
    for j in compute_jobs:
        if j.finish is not None:
            max_finish = max(max_finish, j.finish)
    for m in message_jobs:
        if m.finish is not None:
            max_finish = max(max_finish, m.finish)

    x_scale, x_label = _time_axis_transform(x_unit, reference_frequency_mhz, tick_cycles)
    unit_label = _display_unit_label(x_label)

    def tx(value: float | int) -> float:
        return float(value) * x_scale

    def color_for(task_name: str) -> str:
        return _COMPUTE_COLORS[abs(hash(task_name)) % len(_COMPUTE_COLORS)]

    fig = go.Figure()
    short_job_points = []  # jobs too narrow to read on the global axis; shown as explicit markers later

    # Compute bars: one trace per job so hover, outline, and text can be exact.
    for j in compute_jobs:
        if j.start is None or j.finish is None:
            continue
        duration = j.finish - j.start
        miss = j.finish > j.deadline
        changed = bool(getattr(j.task, "df_changed", False))
        edge_color = "#ef4444" if miss else ("#f59e0b" if changed else "rgba(15,23,42,0.45)")
        edge_width = 3 if miss else (2.5 if changed else 1)
        stage_summary = "<br>".join(
            f"{s.label}: {_format_duration(tx(s.duration), unit_label)}" for s in getattr(j, "stages", [])[:8]
        )
        if len(getattr(j, "stages", [])) > 8:
            stage_summary += "<br>..."
        custom = [
            j.job_id,
            getattr(j.task, "cnn_id", "-"),
            j.instance,
            j.mapped_resource,
            getattr(j.task, "precision", ""),
            _format_duration(tx(j.start), unit_label),
            _format_duration(tx(j.finish), unit_label),
            _format_duration(tx(duration), unit_label),
            _format_duration(tx(j.release), unit_label),
            _format_duration(tx(j.deadline), unit_label),
            "MISS" if miss else "met",
            "yes" if changed else "no",
            getattr(j.task, "layer_kind", "generic"),
            stage_summary or "-",
        ]
        # Very short jobs are real tasks but can be visually narrower than a pixel on a 50–100 ms axis.
        # Keep the true bar duration, and add a non-distorting marker/label so the task never disappears.
        short_job_points.append((j, custom, tx(j.start + duration / 2), tx(duration)))

        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[j.mapped_resource],
                x=[tx(duration)],
                base=[tx(j.start)],
                width=[0.58],
                marker=dict(color=color_for(j.task.name), line=dict(color=edge_color, width=edge_width)),
                text=[_plain_job_label(j)],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="white", size=11),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "CNN: %{customdata[1]} | instance: %{customdata[2]}<br>"
                    "Resource: %{customdata[3]} | precision: %{customdata[4]}<br>"
                    "Start: %{customdata[5]}<br>Finish: %{customdata[6]}<br>Duration: %{customdata[7]}<br>"
                    "Release: %{customdata[8]}<br>Deadline: %{customdata[9]}<br>Deadline: %{customdata[10]}<br>"
                    "Adaptive mapping changed: %{customdata[11]}<br>Layer kind: %{customdata[12]}<br>"
                    "<br><b>Stages</b><br>%{customdata[13]}"
                    "<extra></extra>"
                ),
                customdata=[custom],
                showlegend=False,
            )
        )

    # Explicit markers for short jobs. They do not change the x-axis or bar duration;
    # they only make narrow but scheduled tasks discoverable without zooming.
    if short_job_points:
        visible_range = max(tx(max_finish), 1e-9)
        short_threshold = max(0.35, visible_range * 0.004)  # ms-scale default, adaptive to horizon
        xs, ys, texts, customs = [], [], [], []
        for j, custom, mid_x, display_duration in short_job_points:
            if display_duration <= short_threshold:
                xs.append(mid_x)
                ys.append(j.mapped_resource)
                texts.append(_short_label(j.task.name, 18))
                customs.append(custom)
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    text=texts,
                    textposition="top center",
                    textfont=dict(size=10, color="#0f172a"),
                    marker=dict(symbol="diamond", size=8, color="#0f172a", line=dict(color="#ffffff", width=1)),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "CNN: %{customdata[1]} | instance: %{customdata[2]}<br>"
                        "Resource: %{customdata[3]} | precision: %{customdata[4]}<br>"
                        "Start: %{customdata[5]}<br>Finish: %{customdata[6]}<br>Duration: %{customdata[7]}<br>"
                        "Release: %{customdata[8]}<br>Deadline: %{customdata[9]}<br>Deadline: %{customdata[10]}<br>"
                        "Layer kind: %{customdata[12]}<br>"
                        "<extra>short task marker</extra>"
                    ),
                    customdata=customs,
                    showlegend=False,
                    cliponaxis=False,
                )
            )

    # Communication bars.
    for m in message_jobs:
        if m.start is None or m.finish is None:
            continue
        duration = m.finish - m.start
        custom = [
            m.msg_id,
            m.pred_job_id,
            m.dst_job_id,
            m.src_resource,
            m.dst_resource,
            m.medium,
            m.payload_units,
            _format_duration(tx(m.start), unit_label),
            _format_duration(tx(m.finish), unit_label),
            _format_duration(tx(duration), unit_label),
        ]
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=[m.medium],
                x=[tx(duration)],
                base=[tx(m.start)],
                width=[0.36],
                marker=dict(color="rgba(148,163,184,0.65)", line=dict(color="#334155", width=1), pattern=dict(shape="/")),
                text=[_short_label(m.msg_id, 36)],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#0f172a", size=10),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Producer: %{customdata[1]}<br>Consumer: %{customdata[2]}<br>"
                    "Route: %{customdata[3]} → %{customdata[4]}<br>Medium: %{customdata[5]}<br>"
                    "Payload units: %{customdata[6]}<br>"
                    "Start: %{customdata[7]}<br>Finish: %{customdata[8]}<br>Duration: %{customdata[9]}"
                    "<extra></extra>"
                ),
                customdata=[custom],
                showlegend=False,
            )
        )

    shapes = []
    annotations = []

    if show_markers:
        marker_items = sorted(_chain_markers(compute_jobs).items(), key=lambda kv: (kv[1][0], kv[0][0], kv[0][1]))
        for marker_idx, ((cnn_id, inst), (release, deadline)) in enumerate(marker_items):
            shapes.append(
                dict(type="line", xref="x", yref="paper", x0=tx(release), x1=tx(release), y0=0, y1=1,
                     line=dict(color="rgba(30,41,59,0.28)", width=1))
            )
            shapes.append(
                dict(type="line", xref="x", yref="paper", x0=tx(deadline), x1=tx(deadline), y0=0, y1=1,
                     line=dict(color="rgba(220,38,38,0.45)", width=1.4, dash="dash"))
            )
            if marker_idx < 24:
                annotations.append(
                    dict(x=tx(release), y=1.02, xref="x", yref="paper", showarrow=False, text=f"R {cnn_id}[{inst}]",
                         textangle=-90, font=dict(size=9, color="#475569"), xanchor="right", yanchor="bottom")
                )
                annotations.append(
                    dict(x=tx(deadline), y=1.02, xref="x", yref="paper", showarrow=False, text=f"D {cnn_id}[{inst}]",
                         textangle=-90, font=dict(size=9, color="#dc2626"), xanchor="left", yanchor="bottom")
                )

    # Online DF/QF period annotations.
    period_releases: Dict[int, int] = {}
    period_actions: Dict[int, str] = {}
    for j in compute_jobs:
        p = int(getattr(j.task, "df_period", -1))
        if p >= 0:
            period_releases[p] = min(period_releases.get(p, j.release), j.release)
            period_actions[p] = str(getattr(j.task, "df_action", ""))
    action_names = {
        "init_npu_first": "init NPU-first",
        "init_cpu_quality": "init CPU QF",
        "qf_repair": "QF repair",
        "repair": "repair",
        "improve": "improve",
        "rollback": "rollback",
        "hold": "hold",
        "quality_first": "Quality-First",
    }
    for p, x in sorted(period_releases.items()):
        shapes.append(
            dict(type="line", xref="x", yref="paper", x0=tx(x), x1=tx(x), y0=0, y1=1,
                 line=dict(color="rgba(15,23,42,0.50)", width=1, dash="dot" if p else "solid"))
        )
        annotations.append(
            dict(
                x=tx(x), y=0.98, xref="x", yref="paper", showarrow=False,
                text=f"P{p}<br>{action_names.get(period_actions.get(p, ''), period_actions.get(p, ''))}",
                font=dict(size=10, color="#0f172a"), align="left", xanchor="left", yanchor="top",
                bgcolor="rgba(255,255,255,0.82)", bordercolor="rgba(148,163,184,0.8)", borderwidth=1,
            )
        )

    if show_dependencies:
        job_by_id = {j.job_id: j for j in compute_jobs}
        pairs = _dependency_pairs(compute_jobs, message_jobs)
        if len(pairs) <= 160:
            for pred_id, dst_id, crossed_medium in pairs:
                pred = job_by_id.get(pred_id)
                dst = job_by_id.get(dst_id)
                if pred is None or dst is None or pred.finish is None or dst.start is None:
                    continue
                if pred.mapped_resource not in lane_index or dst.mapped_resource not in lane_index:
                    continue
                dash = "dot" if crossed_medium else "solid"
                color = "rgba(71,85,105,0.45)" if crossed_medium else "rgba(15,23,42,0.42)"
                shapes.append(
                    dict(
                        type="line", xref="x", yref="y",
                        x0=tx(pred.finish), x1=tx(dst.start),
                        y0=pred.mapped_resource, y1=dst.mapped_resource,
                        line=dict(color=color, width=1, dash=dash),
                        layer="below",
                    )
                )

    right_pad = max(5, int(max_finish * 0.04) if max_finish else 5)
    title = "CNN CPU/NPU Schedule — interactive Gantt"
    if any(int(getattr(j.task, "df_period", -1)) >= 0 for j in compute_jobs):
        actions_seen = {str(getattr(j.task, "df_action", "")) for j in compute_jobs}
        title += " — Quality-First" if any(a.startswith("quality") for a in actions_seen) else " — Deadline-First"

    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=22, color="#0f172a")),
        template="plotly_white",
        barmode="overlay",
        bargap=0.34,
        height=max(560, 86 * max(1, len(lane_order)) + 220),
        margin=dict(l=120, r=40, t=110, b=95),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            title=x_label,
            range=[0, tx(max_finish + right_pad)],
            gridcolor="rgba(148,163,184,0.30)",
            zeroline=False,
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikedash="dot",
        ),
        yaxis=dict(
            title="Resource / medium",
            categoryorder="array",
            categoryarray=list(reversed(lane_order)),
            gridcolor="rgba(226,232,240,0.9)",
        ),
        shapes=shapes,
        annotations=annotations,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, Arial, sans-serif"),
        font=dict(family="Inter, Arial, sans-serif", color="#0f172a"),
    )

    fig.update_traces(cliponaxis=False)

    out = Path(outpath)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        full_html=True,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "toImageButtonOptions": {"format": "svg", "filename": out.stem, "height": 900, "width": 1600, "scale": 1},
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
        },
    )
    # Add a small CSS polish wrapper. Plotly remains fully standalone.
    html = html.replace(
        "<head>",
        "<head>\n<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<style>body{margin:0;background:#f1f5f9;} .plotly-graph-div{box-shadow:0 24px 70px rgba(15,23,42,.12);}\n"
        "@media print{body{background:white}.modebar{display:none!important}}</style>",
        1,
    )
    out.write_text(html, encoding="utf-8")
