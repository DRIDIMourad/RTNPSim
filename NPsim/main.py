from __future__ import annotations
import argparse
import webbrowser
from pathlib import Path
from typing import Tuple

from .analysis import build_report, write_report
from .deadline_first import build_mapping_report, deadline_first_mapping, deadline_first_online_period_mapping
from .quality_first import build_quality_mapping_report, quality_first_mapping, quality_first_online_mapping
from .parser import compute_hyperperiod, generate_jobs, load_arch_config, parse_tasks_file
from .plotting import plot_schedule, write_schedule_metadata
from .plotly_gantt import plot_schedule_html
from .simulator import simulate, validate_schedule


def run(
    tasks_path: str,
    arch_config: str | None = None,
    policy: str = "np_fp",
    mapping_policy: str = "file",
    switch_cost: int = 0,
    switch_cost_mode: str = "per_cnn",
    max_configs_per_cnn: int = 512,
    gantt_time_unit: str = "tick",
    gantt_format: str = "png",
    open_gantt: bool = False,
    show_gantt_dependencies: bool = True,
) -> Tuple[str, str]:
    tasks, sim_time, npu_size = parse_tasks_file(tasks_path)
    arch = load_arch_config(arch_config)
    if npu_size is not None and not arch_config:
        arch.mapping.npu_nodes = [f"NPU{i}" for i in range(max(1, int(npu_size)))]
    task_tick_cycles = max((int(getattr(t, "tick_cycles", 1)) for t in tasks), default=0)
    tick_cycles = task_tick_cycles if task_tick_cycles > 1 else max(1, int(getattr(arch.clock, "tick_cycles", 1)))

    if sim_time is None:
        sim_time = compute_hyperperiod(tasks)

    mapping_policy_norm = (mapping_policy or "file").lower()
    mapping_report = ""

    if mapping_policy_norm in ("online_df", "deadline_first", "deadline-first", "df", "ia_df"):
        mapping = deadline_first_online_period_mapping(
            tasks,
            arch,
            sim_time,
            policy=policy,
            switch_cost=switch_cost,
            switch_cost_mode=switch_cost_mode,
            max_configs_per_cnn=max_configs_per_cnn,
        )
        result = mapping.evaluation.result
        mapping_report = build_mapping_report(mapping)
    elif mapping_policy_norm in ("offline_df", "df_offline", "deadline_first_offline"):
        mapping = deadline_first_mapping(
            tasks,
            arch,
            sim_time,
            policy=policy,
            switch_cost=switch_cost,
            switch_cost_mode=switch_cost_mode,
            max_configs_per_cnn=max_configs_per_cnn,
        )
        result = mapping.evaluation.result
        mapping_report = build_mapping_report(mapping)
    elif mapping_policy_norm in ("online_qf", "qf_online", "quality_first_online"):
        mapping = quality_first_online_mapping(
            tasks,
            arch,
            sim_time,
            policy=policy,
            switch_cost=switch_cost,
            switch_cost_mode=switch_cost_mode,
            max_configs_per_cnn=max_configs_per_cnn,
        )
        result = mapping.evaluation.result
        mapping_report = build_quality_mapping_report(mapping)
    elif mapping_policy_norm in ("offline_qf", "quality_first", "quality-first", "qf", "qf_offline"):
        mapping = quality_first_mapping(
            tasks,
            arch,
            sim_time,
            policy=policy,
            switch_cost=switch_cost,
            switch_cost_mode=switch_cost_mode,
            max_configs_per_cnn=max_configs_per_cnn,
        )
        result = mapping.evaluation.result
        mapping_report = build_quality_mapping_report(mapping)
    else:
        jobs = generate_jobs(tasks, sim_time)
        result = simulate(jobs, arch, policy=policy)

    issues = validate_schedule(result)

    report_text = ""
    input_unit = (getattr(tasks[0], "input_time_unit", "cycles") if tasks else "cycles")
    report_text += (
        "=== Clocks and units ===\n"
        f"File time base: {input_unit}\n"
        f"Internal time base: cycles\n"
        f"GUI/Gantt conversion: 1 tick = {tick_cycles} internal cycle(s)\n"
        f"Reference frequency: {arch.clock.reference_frequency_mhz:g} MHz\n"
        f"CPU frequency: {arch.clock.cpu_frequency_mhz:g} MHz\n"
        f"NPU frequency: {arch.clock.npu_frequency_mhz:g} MHz\n"
        f"Interconnect frequency: {arch.clock.interconnect_frequency_mhz:g} MHz\n"
        f"Gantt X-axis unit: {gantt_time_unit}\n"
        f"Format Gantt: {gantt_format}\n"
        "Gantt markers: R=CNN/frame release, D=end-to-end deadline; "
        "solid arrows=local dependencies, dashed arrows=dependencies with transfer.\n\n"
    )
    if mapping_report:
        report_text += mapping_report + "\n\n"
    report_text += build_report(result)
    report_text += "\n\n=== Consistency checks ===\n"
    if issues:
        report_text += "Detected errors :\n" + "\n".join(f"- {x}" for x in issues)
    else:
        report_text += "No inconsistency detected in the main internal checks."

    tasks_file = Path(tasks_path)
    stem = tasks_file.stem
    policy_tag = f"{policy.upper()}_{mapping_policy_norm.upper()}"
    log_path = tasks_file.with_name(f"{stem}_report_{policy_tag}.log")
    gantt_format_norm = (gantt_format or "html").lower().strip()
    if gantt_format_norm not in {"html", "png", "svg"}:
        raise ValueError(f"Unknown Gantt format: {gantt_format}. Use html, png or svg.")
    gantt_path = tasks_file.with_name(f"{stem}_schedule_{policy_tag}.{gantt_format_norm}")

    write_report(str(log_path), report_text)
    if gantt_format_norm == "html":
        plot_schedule_html(
            result,
            str(gantt_path),
            x_unit=gantt_time_unit,
            reference_frequency_mhz=arch.clock.reference_frequency_mhz,
            tick_cycles=tick_cycles,
            show_dependencies=show_gantt_dependencies,
            show_markers=True,
        )
    else:
        plot_schedule(result, str(gantt_path), x_unit=gantt_time_unit, reference_frequency_mhz=arch.clock.reference_frequency_mhz, tick_cycles=tick_cycles)
    metadata_path = write_schedule_metadata(result, str(gantt_path), x_unit=gantt_time_unit, reference_frequency_mhz=arch.clock.reference_frequency_mhz, tick_cycles=tick_cycles)

    if open_gantt and gantt_format_norm == "html":
        try:
            opened = webbrowser.open(Path(gantt_path).resolve().as_uri())
            if not opened:
                print("[WARN] Could not automatically open the browser. Open the HTML file manually.")
        except Exception as exc:
            print(f"[WARN] Could not automatically open the Gantt HTML: {exc}")

    horizon_ticks = (int(sim_time) + tick_cycles - 1) // tick_cycles
    print(f"[INFO] Horizon = {horizon_ticks} ticks ({sim_time} internal cycles)")
    print(f"[INFO] Scheduling policy = {policy}")
    print(f"[INFO] Mapping policy = {mapping_policy_norm}")
    if mapping_policy_norm != "file":
        print(f"[INFO] Switch cost = {switch_cost} ({switch_cost_mode})")
        print(f"[INFO] Max configs/CNN = {max_configs_per_cnn}")
    print(f"[INFO] Report : {log_path}")
    print(f"[INFO] Gantt X-axis unit = {gantt_time_unit} (tick = {tick_cycles} cycle(s), reference frequency = {arch.clock.reference_frequency_mhz:g} MHz)")
    print(f"[INFO] Format Gantt = {gantt_format_norm}")
    print(f"[INFO] Graph  : {gantt_path}")
    print(f"[INFO] Interactive Gantt data : {metadata_path}")
    print()
    print(report_text)
    return str(log_path), str(gantt_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="CPU/NPU simulator with online/offline Deadline-First and Quality-First mappings")
    ap.add_argument("tasks_file", help="tasks .txt file: old 14-column format or new 15-column format with kind")
    ap.add_argument("--arch-config", default=None, help="Optional architecture YAML")
    ap.add_argument("--policy", default="np_fp", choices=["np_fp", "edf", "rm"], help="Scheduling policy runtime")
    ap.add_argument(
        "--mapping-policy",
        default="file",
        choices=["file", "online_df", "online_qf", "offline_df", "offline_qf", "deadline_first", "quality_first"],
        help="online_df/online_qf evolve period by period; offline_df/offline_qf choose a fixed configuration before execution",
    )
    ap.add_argument("--switch-cost", type=int, default=0, help="Switch cost de configuration, en internal cycles")
    ap.add_argument(
        "--switch-cost-mode",
        default="per_cnn",
        choices=["per_cnn", "per_layer"],
        help="Switch-cost charging: once per modified CNN or per modified layer",
    )
    ap.add_argument("--max-configs-per-cnn", type=int, default=512, help="Maximum number of Pareto configurations kept per CNN")
    ap.add_argument("--gantt-time-unit", default="tick", choices=["tick", "ticks", "cycles", "ns", "us", "µs", "ms", "s"], help="Gantt X-axis display unit")
    ap.add_argument("--gantt-format", default="png", choices=["html", "png", "svg"], help="Gantt output format. png = local static display; html = interactive Plotly")
    ap.add_argument("--open-gantt", dest="open_gantt", action="store_true", default=False, help="Automatically open the HTML Gantt after execution")
    ap.add_argument("--no-open-gantt", dest="open_gantt", action="store_false", help="Do not automatically open the HTML Gantt")
    ap.add_argument("--no-gantt-dependencies", dest="show_gantt_dependencies", action="store_false", default=True, help="Hide dependency lines in the HTML Gantt")
    args = ap.parse_args()
    run(
        args.tasks_file,
        arch_config=args.arch_config,
        policy=args.policy,
        mapping_policy=args.mapping_policy,
        switch_cost=args.switch_cost,
        switch_cost_mode=args.switch_cost_mode,
        max_configs_per_cnn=args.max_configs_per_cnn,
        gantt_time_unit=args.gantt_time_unit,
        gantt_format=args.gantt_format,
        open_gantt=args.open_gantt,
        show_gantt_dependencies=args.show_gantt_dependencies,
    )


if __name__ == "__main__":
    main()
