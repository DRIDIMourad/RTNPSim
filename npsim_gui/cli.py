from __future__ import annotations

import argparse
from pathlib import Path

from .exporters import export_project_files, run_generated_script
from .scenario import default_arch_settings, scenario_examples


def _resolve_template(value: str, examples: list[dict]) -> int:
    aliases = {
        "single": 0,
        "generic": 0,
        "quality": 1,
        "two": 2,
        "contention": 2,
        "dag": 3,
        "noc": 4,
        "camera": 5,
        "multicam": 5,
    }
    v = (value or "ex02").lower()
    if v in aliases:
        return min(aliases[v], len(examples) - 1)
    if v.startswith("ex"):
        try:
            idx = int(v[2:]) - 1
            if 0 <= idx < len(examples):
                return idx
        except ValueError:
            pass
    try:
        idx = int(v) - 1
        if 0 <= idx < len(examples):
            return idx
    except ValueError:
        pass
    raise SystemExit(f"Unknown example: {value}. Use --list-examples.")


def main() -> None:
    examples = scenario_examples()
    parser = argparse.ArgumentParser(description="User-friendly NPsim scenario generator")
    parser.add_argument("--out", default="workspace", help="Output folder for simulation files")
    parser.add_argument("--template", default="ex02", help="Example to load: ex01, ex02, ...")
    parser.add_argument("--list-examples", action="store_true", help="Show available examples")
    parser.add_argument("--horizon-ticks", type=int, default=None, help="GUI horizon to simulate, in ticks")
    parser.add_argument("--simulation-time", type=int, default=None, help="Alias of --horizon-ticks, kept for compatibility")
    parser.add_argument("--periods", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--policy", choices=["np_fp", "edf", "rm"], default=None)
    parser.add_argument("--mapping-policy", choices=["file", "online_df", "online_qf", "offline_df", "offline_qf", "deadline_first", "quality_first"], default=None)
    parser.add_argument("--switch-cost", type=int, default=None)
    parser.add_argument("--switch-cost-mode", choices=["per_cnn", "per_layer"], default=None)
    parser.add_argument("--max-configs-per-cnn", type=int, default=None)
    parser.add_argument("--gantt-time-unit", choices=["tick", "ticks", "cycles", "ns", "us", "ms", "s"], default=None, help="Gantt X-axis display unit")
    parser.add_argument("--execute", action="store_true", help="Run the simulation after generation")
    args = parser.parse_args()

    if args.list_examples:
        for i, ex in enumerate(examples, start=1):
            print(f"ex{i:02d}: {ex['label']} — {ex['description']}")
        return

    root = Path(__file__).resolve().parents[1]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    example = examples[_resolve_template(args.template, examples)]
    tasks = example["tasks"]
    arch = default_arch_settings()
    arch.update(example.get("arch_updates", {}))
    selected_horizon_ticks = args.horizon_ticks if args.horizon_ticks is not None else args.simulation_time
    if selected_horizon_ticks is None:
        selected_horizon_ticks = int(example.get("horizon_ticks", example.get("simulation_time", 1000)))
    if args.periods is not None:
        # Old hidden mode: converts a number of periods into a tick horizon.
        base_period = min((int(t.get("period", 0)) for t in tasks if int(t.get("period", 0)) > 0), default=1)
        selected_horizon_ticks = max(1, int(args.periods)) * base_period

    paths = export_project_files(
        out,
        tasks=tasks,
        simulation_time=selected_horizon_ticks,
        npu_size=int(arch.get("npu_count", 1)),
        arch=arch,
        runtime_policy=args.policy if args.policy is not None else example.get("runtime_policy", "np_fp"),
        mapping_policy=args.mapping_policy if args.mapping_policy is not None else example.get("mapping_policy", "online_df"),
        switch_cost=args.switch_cost if args.switch_cost is not None else int(example.get("switch_cost", 1)),
        switch_cost_mode=args.switch_cost_mode if args.switch_cost_mode is not None else example.get("switch_cost_mode", "per_layer"),
        max_configs_per_cnn=args.max_configs_per_cnn if args.max_configs_per_cnn is not None else int(example.get("max_configs_per_cnn", 96)),
        gantt_time_unit=args.gantt_time_unit if args.gantt_time_unit is not None else example.get("gantt_time_unit", "tick"),
    )
    print("Generated files:")
    for key in ("tasks", "arch", "scenario", "run"):
        print(f"- {key}: {paths[key]}")

    if args.execute:
        print("\nRunning simulation...")
        result = run_generated_script(out)
        print(result["stdout"])
        if result["stderr"]:
            print(result["stderr"])
        print(f"Return code: {result['returncode']}")
        if result["log"]:
            print(f"Report: {result['log']}")
        if result["gantt"]:
            print(f"Gantt: {result['gantt']}")


if __name__ == "__main__":
    main()
