from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .scenario import normalize_task_table, periods_to_simulation_time, scenario_to_yaml_dict, tick_cycles_from_arch, ticks_to_cycles, tasks_ticks_to_cycles


TASKS_HEADER = "# name kind resource period_cycles wcet_int8_cycles wcet_fp16_cycles wcet_fp32_cycles deadline_cycles priority phase_cycles precision preds cnn_id criticity size"


def _resource_from_support(support: str) -> str:
    # The simulator input has only CPU or NPU. In the Deadline-First mapper,
    # resource=CPU means CPU-only. resource=NPU means NPU-supported with CPU
    # fallback. The GUI therefore stores richer support text and converts it
    # here for backward compatibility.
    return "CPU" if str(support).lower() in {"cpu only", "cpu seulement"} else "NPU"


def tasks_to_txt(
    tasks: List[Dict[str, Any]],
    simulation_time: int | None = None,
    npu_size: int | None = None,
    periods: int | None = None,
    tick_cycles: int = 100000,
) -> str:
    """Export a simulator tasks.txt file.

    The GUI table is expressed in ticks. This function writes the simulator file
    in internal cycles again, while keeping tick metadata so Gantt display can
    default to ticks. The optional `periods` argument is accepted only for
    backward compatibility with older callers.
    """
    gui_rows = normalize_task_table(tasks)
    tick_cycles = max(1, int(tick_cycles))
    if simulation_time is None:
        if periods is None:
            horizon_ticks = periods_to_simulation_time(gui_rows, 1)
        else:
            horizon_ticks = periods_to_simulation_time(gui_rows, max(1, int(periods)))
    else:
        horizon_ticks = max(1, int(simulation_time))
    simulation_time_cycles = ticks_to_cycles(horizon_ticks, tick_cycles, minimum=1)
    rows = normalize_task_table(tasks_ticks_to_cycles(gui_rows, tick_cycles))

    lines = [
        "#TIME_UNIT cycles",
        f"#TICK_CYCLES {tick_cycles}",
        f"#HORIZON_TICKS {int(horizon_ticks)}",
        f"#SIMULATION_TIME {simulation_time_cycles}",
        TASKS_HEADER,
    ]
    for t in rows:
        resource = _resource_from_support(t["support"])
        preds = t["preds"] if t["preds"] else "-"
        # Export the user-selected layer kind explicitly. The simulator parser
        # treats this kind as authoritative and falls back to name inference only
        # for legacy tasks.txt files that do not contain this column.
        line = (
            f"{t['name']} {t['kind']} {resource} {t['period']} {t['wcet_int8']} {t['wcet_fp16']} {t['wcet_fp32']} "
            f"{t['deadline']} {t['priority']} {t['phase']} {t['precision']} {preds} "
            f"{t['cnn_id']} {t['criticity']} {t['size']}"
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def arch_to_yaml(arch: Dict[str, Any]) -> str:
    npu_count = max(1, int(arch.get("npu_count", 1)))
    data = {
        "clock": {
            "time_base": "cycles",
            "user_time_unit": "tick",
            "tick_cycles": tick_cycles_from_arch(arch),
            "reference_frequency_mhz": float(arch.get("reference_frequency_mhz", 1000)),
            "cpu_frequency_mhz": float(arch.get("cpu_frequency_mhz", 1000)),
            "npu_frequency_mhz": float(arch.get("npu_frequency_mhz", 1000)),
            "interconnect_frequency_mhz": float(arch.get("interconnect_frequency_mhz", 1000)),
        },
        "resources": {
            "cpu_name": arch.get("cpu_name", "CPU"),
            "npu_nodes": [f"NPU{i}" for i in range(npu_count)],
        },
        "communications": {
            "cpu_npu": {
                "mode": arch.get("cpu_npu_mode", "separate_full_duplex"),
                "cpu_to_npu_setup": int(arch.get("cpu_to_npu_setup", 2)),
                "cpu_to_npu_per_unit": int(arch.get("cpu_to_npu_per_unit", 1)),
                "npu_to_cpu_setup": int(arch.get("npu_to_cpu_setup", 2)),
                "npu_to_cpu_per_unit": int(arch.get("npu_to_cpu_per_unit", 1)),
                "shared_setup": int(arch.get("shared_setup", 2)),
                "shared_per_unit": int(arch.get("shared_per_unit", 1)),
            },
            "noc": {
                "topology": arch.get("noc_topology", "mesh"),
                "setup": int(arch.get("noc_setup", 1)),
                "per_unit": int(arch.get("noc_per_unit", 1)),
                "router_latency": int(arch.get("router_latency", 1)),
                "arb_policy": arch.get("arb_policy", "fixed_priority"),
                "rr_quantum_units": int(arch.get("noc_rr_quantum_units", 32)),
            },
        },
        "npu": {
            "sa_rows": int(arch.get("sa_rows", 16)),
            "sa_cols": int(arch.get("sa_cols", 16)),
            "systolic_arrays": int(arch.get("systolic_arrays", 1)),
            "setup_cycles": int(arch.get("npu_setup_cycles", 2)),
            "precision_modes": ["INT8"],
            "local_dram_capacity_kb": int(arch.get("local_dram_capacity_kb", 2048)),
            "local_dram_latency": int(arch.get("local_dram_latency", 6)),
            "local_dram_bandwidth_bytes_per_cycle": int(arch.get("local_dram_bandwidth_bytes_per_cycle", arch.get("local_dram_bandwidth_bytes_per_tick", 256))),
            "dma_setup": int(arch.get("dma_setup", 2)),
            "dma_bandwidth_bytes_per_cycle": int(arch.get("dma_bandwidth_bytes_per_cycle", arch.get("dma_bandwidth_bytes_per_tick", 512))),
            "vector_lanes": int(arch.get("vector_lanes", 128)),
            "vector_setup": int(arch.get("vector_setup", 1)),
        },
        "cpu": {
            "setup_cycles": int(arch.get("cpu_setup_cycles", 1)),
            "cache_capacity_kb": int(arch.get("cache_capacity_kb", 512)),
            "cache_latency": int(arch.get("cache_latency", 6)),
            "cache_bandwidth_bytes_per_cycle": int(arch.get("cache_bandwidth_bytes_per_cycle", arch.get("cache_bandwidth_bytes_per_tick", 512))),
            "simd_mac_per_cycle": int(arch.get("simd_mac_per_cycle", arch.get("simd_mac_per_tick", 64))),
            "pack_elements_per_cycle": int(arch.get("pack_elements_per_cycle", arch.get("pack_elements_per_tick", 128))),
            "epilogue_elements_per_cycle": int(arch.get("epilogue_elements_per_cycle", arch.get("epilogue_elements_per_tick", 64))),
            "vector_elements_per_cycle": int(arch.get("vector_elements_per_cycle", arch.get("vector_elements_per_tick", 32))),
            "rmw_bandwidth_bytes_per_cycle": int(arch.get("rmw_bandwidth_bytes_per_cycle", arch.get("rmw_bandwidth_bytes_per_tick", 256))),
        },
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def run_sh_text(
    runtime_policy: str,
    mapping_policy: str,
    switch_cost: int,
    switch_cost_mode: str,
    max_configs_per_cnn: int,
    gantt_time_unit: str = "tick",
) -> str:
    args = [
        "python",
        "-m",
        "NPsim.main",
        "${SCRIPT_DIR}/tasks.txt",
        "--arch-config",
        "${SCRIPT_DIR}/arch.yaml",
        "--policy",
        runtime_policy,
        "--mapping-policy",
        mapping_policy,
        "--switch-cost",
        str(int(switch_cost)),
        "--switch-cost-mode",
        switch_cost_mode,
        "--max-configs-per-cnn",
        str(int(max_configs_per_cnn)),
        "--gantt-time-unit",
        gantt_time_unit,
        "--gantt-format",
        "png",
        "--no-open-gantt",
    ]
    # Keep shell variables unquoted in the list above, then quote the fixed
    # literals safely. This script remains readable and portable within the
    # generated workspace.
    command = " ".join(a if a.startswith("${") else shlex.quote(a) for a in args)
    package_root = Path(__file__).resolve().parents[1]
    return f"""#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
# Case 1: run.sh is at the package root.
if [ -f "${{SCRIPT_DIR}}/NPsim/main.py" ]; then
  PROJECT_ROOT="${{SCRIPT_DIR}}"
# Cas 2: run.sh est dans <package>/workspace, donc le paquet NPsim est dans le parent.
elif [ -f "${{SCRIPT_DIR}}/../NPsim/main.py" ]; then
  PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/.." && pwd)"
# Cas 3: run.sh est dans <package>/examples/<exemple>.
elif [ -f "${{SCRIPT_DIR}}/../../NPsim/main.py" ]; then
  PROJECT_ROOT="$(cd "${{SCRIPT_DIR}}/../.." && pwd)"
else
  # Fallback if the output folder is elsewhere; this value is written at generation time.
  PROJECT_ROOT={shlex.quote(str(package_root))}
fi
export PYTHONPATH="${{PROJECT_ROOT}}:${{PYTHONPATH:-}}"
cd "${{SCRIPT_DIR}}"
{command}
"""


def export_project_files(
    output_dir: str | Path,
    tasks: List[Dict[str, Any]],
    simulation_time: int | None = None,
    npu_size: int = 1,
    periods: int | None = None,
    arch: Dict[str, Any] | None = None,
    runtime_policy: str = "np_fp",
    mapping_policy: str = "online_df",
    switch_cost: int = 1,
    switch_cost_mode: str = "per_cnn",
    max_configs_per_cnn: int = 512,
    gantt_time_unit: str = "tick",
) -> Dict[str, Path]:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks_norm = normalize_task_table(tasks)
    arch = arch or {}
    tick_cycles = tick_cycles_from_arch(arch)
    if simulation_time is None:
        if periods is None:
            simulation_time = periods_to_simulation_time(tasks_norm, 1)
        else:
            simulation_time = periods_to_simulation_time(tasks_norm, max(1, int(periods)))
    simulation_time = max(1, int(simulation_time))
    switch_cost_cycles = ticks_to_cycles(int(switch_cost), tick_cycles, minimum=0)

    tasks_path = out / "tasks.txt"
    arch_path = out / "arch.yaml"
    scenario_path = out / "scenario.yaml"
    run_path = out / "run.sh"

    tasks_path.write_text(tasks_to_txt(tasks_norm, simulation_time, None, tick_cycles=tick_cycles), encoding="utf-8")
    arch_path.write_text(arch_to_yaml(arch), encoding="utf-8")
    scenario_yaml = scenario_to_yaml_dict(
        tasks_norm,
        int(simulation_time),
        int(npu_size),
        mapping_policy,
        runtime_policy,
        switch_cost_cycles,
        switch_cost_mode,
        int(max_configs_per_cnn),
        arch,
        gantt_time_unit,
    )
    scenario_path.write_text(yaml.safe_dump(scenario_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    run_path.write_text(
        run_sh_text(runtime_policy, mapping_policy, switch_cost_cycles, switch_cost_mode, max_configs_per_cnn, gantt_time_unit),
        encoding="utf-8",
    )
    os.chmod(run_path, 0o755)
    return {
        "output_dir": out,
        "tasks": tasks_path,
        "arch": arch_path,
        "scenario": scenario_path,
        "run": run_path,
    }


def find_latest_outputs(output_dir: str | Path) -> Tuple[Path | None, Path | None]:
    out = Path(output_dir)
    logs = sorted(list(out.glob("*_report_*.log")) + list(out.glob("*_rapport_*.log")), key=lambda p: p.stat().st_mtime, reverse=True)
    gantts = []
    for pattern in ("*_schedule_*.png", "*_schedule_*.svg", "*_schedule_*.html", "*_ordonnancement_*.png", "*_ordonnancement_*.svg", "*_ordonnancement_*.html"):
        gantts.extend(out.glob(pattern))
    gantts = sorted(gantts, key=lambda p: p.stat().st_mtime, reverse=True)
    return (logs[0] if logs else None, gantts[0] if gantts else None)


def run_generated_script(output_dir: str | Path, timeout: int = 120) -> Dict[str, Any]:
    out = Path(output_dir).resolve()
    run_path = out / "run.sh"
    if not run_path.exists():
        raise FileNotFoundError(f"run.sh not found in {out}")

    proc = subprocess.run(
        ["bash", str(run_path)],
        cwd=str(out),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    log_path, png_path = find_latest_outputs(out)
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "log": log_path,
        "gantt": png_path,
    }
