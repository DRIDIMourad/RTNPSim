# NPsim — CPU/NPU simulator

NPsim simulates real-time workloads on heterogeneous CPU/NPU architectures: periodic tasks, DAG dependencies, precisions, CPU↔NPU communication, NoC transfers, deadlines, mapping policies, and an internal Gantt viewer with zoom, tooltips, and DAG edges.

## Launch the GUI

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start_gui.sh
```

On Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
start_gui.bat
```

## Time units

The GUI uses **ticks** for user-facing fields: period, WCET, deadline, phase, switch cost, and horizon.

Default scale:

```text
1 tick = 10000 internal cycles
```

This conversion is configurable in **Architecture → Clocks / frequencies → Cycles per tick**. The simulator files remain written in **internal cycles**; the GUI converts ticks ⇄ cycles automatically.

The Gantt uses **tick** as the default X-axis unit. Other available units are `cycles`, `ns`, `us`, `ms`, and `s`.

## Timing model

The `tasks.txt` format is :

```text
# name kind resource period_cycles wcet_int8_cycles wcet_fp16_cycles wcet_fp32_cycles deadline_cycles priority phase_cycles precision preds cnn_id criticity size
```

### Generic tasks

For `kind=generic`, compute duration is strictly WCET-only:

```text
CPU_WCET = selected_WCET
NPU_WCET = selected_WCET
duration_job = optional_SWITCH_CONFIG + CPU_WCET or NPU_WCET
```

### Non-generic tasks

Non-generic kinds are (`conv`, `pool`, `bn`, `activation`, `pointwise_conv`, etc.). For those kinds, the architectural model estimates memory, DMA, cache, NoC, SIMD, vector unit, and systolic-array timing.

## Run an example from the CLI

List examples:

```bash
python -m npsim_gui.cli --list-examples
```

Generate and run an example with a tick horizon:

```bash
python -m npsim_gui.cli --template ex01 --out workspace/ex01 --execute
```

Each example automatically applies:

- the runtime scheduling policy (`edf`, `np_fp`, `rm`);
- the mapping policy (`file`, `online_df`, `online_qf`, `offline_df`, `offline_qf`);
- the switch cost;
- the architecture;
- the Gantt X-axis unit.

Generated `tasks.txt` files contain:

```text
#TIME_UNIT cycles
#TICK_CYCLES 10000
#HORIZON_TICKS ...
#SIMULATION_TIME ...
```

The `period`, `wcet_*`, `deadline`, and `phase` columns are expressed in **internal cycles** to keep the engine simple and compatible. `#HORIZON_TICKS` documents the horizon entered in the GUI.

A generated folder contains:

```text
tasks.txt
arch.yaml
scenario.yaml
run.sh
*_rapport_*.log
*_ordonnancement_*.png by default, or *.html/*.svg with --gantt-format
```

## Linux Tkinter installation

If Tkinter is not available:

```bash
sudo apt-get install python3-tk
```
