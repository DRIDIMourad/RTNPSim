from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

from .deadline_first import (
    CNNConfig,
    CandidateTrace,
    DeadlineFirstMappingResult,
    MappingEvaluation,
    PeriodStep,
    build_cnn_configurations,
    evaluate_selection,
    generate_jobs_online_periods,
    _all_cpu_high_quality_selection,
    _all_resource_configs,
    _candidate_ratio,
    _dominant_instance_count,
    _dominant_period_window,
    _evaluation_from_result,
    _max_accuracy_score,
    _materialize_tasks,
    _one_period_observation,
    _predict_one_period,
    _selection_cost,
    _selection_accuracy,
    _selection_edge_key,
    _single_step_candidates,
)
from .models import ArchConfig, Task
from .simulator import simulate


def _miss_count(ev: MappingEvaluation) -> int:
    return len(ev.deadline_misses) + len(ev.unfinished) + len(ev.issues)


def _npu_count(cfg: CNNConfig) -> int:
    return sum(1 for opt in cfg.options if opt.resource.upper().startswith("NPU"))


def _highest_quality_selection(configs_by_cnn: Dict[str, List[CNNConfig]]) -> Dict[str, CNNConfig]:
    """Initial Quality-First selection.

    The highest numerical quality is selected first. For equal quality, the
    mapper prefers NPU-heavy configurations, then the cheapest one. This makes
    the initial point the top-quality mirror of Deadline-First's low-cost start.
    """
    out: Dict[str, CNNConfig] = {}
    for cnn_id, configs in configs_by_cnn.items():
        out[cnn_id] = max(configs, key=lambda c: (c.accuracy_score, _npu_count(c), -c.cost, c.name))
    return out


def _quality_loss(old: CNNConfig, candidate: CNNConfig) -> float:
    return max(0.0, old.accuracy_score - candidate.accuracy_score)


def _repair_timing_gain(old_eval: MappingEvaluation, trial_eval: MappingEvaluation) -> float:
    """Weighted timing gain used only for ranking repair candidates.

    Deadline feasibility remains dominant in the acceptance key; this score
    tells which repair buys the largest timing improvement per lost quality.
    """
    old_miss = _miss_count(old_eval)
    new_miss = _miss_count(trial_eval)
    miss_gain = max(0, old_miss - new_miss)
    worst_gain = max(0, old_eval.worst_lateness - trial_eval.worst_lateness)
    sum_gain = max(0, old_eval.sum_lateness - trial_eval.sum_lateness)
    return 100_000.0 * miss_gain + 100.0 * worst_gain + float(sum_gain)


def _quality_first_repair(
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
    """Greedy offline repair phase for Quality-First.

    Unlike Deadline-First, this phase never tries to increase quality: Quality-
    First starts from the maximum-quality point and repeatedly applies the
    least damaging repair that improves the deadline situation.
    """
    repair_used = False
    blocked: set[Tuple[str, str, str]] = set()

    for _ in range(max_iterations):
        if current_eval.feasible:
            break

        old_key = (
            _miss_count(current_eval),
            current_eval.worst_lateness,
            current_eval.sum_lateness,
            -current_eval.total_accuracy_score,
            current_eval.total_nominal_cost,
        )

        best: Optional[Tuple[Tuple[int, int, int, float, float, int], str, CNNConfig, MappingEvaluation, float, float, int, float]] = None

        for cnn_id, cfgs in configs_by_cnn.items():
            old = current_selection[cnn_id]
            for candidate in cfgs:
                if candidate.name == old.name:
                    continue
                edge = (cnn_id, old.name, candidate.name)
                if edge in blocked:
                    continue

                # Quality-First repairs by keeping the same quality with a
                # better placement, or by downgrading precision. It never uses
                # a higher-quality candidate during repair.
                if candidate.accuracy_score > old.accuracy_score:
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

                trial_key = (
                    _miss_count(trial_eval),
                    trial_eval.worst_lateness,
                    trial_eval.sum_lateness,
                    -trial_eval.total_accuracy_score,
                    trial_eval.total_nominal_cost,
                )

                loss = _quality_loss(old, candidate)
                timing_gain = _repair_timing_gain(current_eval, trial_eval)
                denom = loss + (switch_cost if candidate.differs_from(old) else 0)
                ratio = float("inf") if timing_gain > 0 and denom <= 0 else (timing_gain / max(1.0, denom))
                delta_cost = candidate.cost - old.cost + (switch_cost if candidate.differs_from(old) else 0)
                delta_accuracy = candidate.accuracy_score - old.accuracy_score

                accepted_by_key = trial_key < old_key
                trace.append(
                    CandidateTrace(
                        phase="quality_repair_precheck",
                        cnn_id=cnn_id,
                        from_config=old.name,
                        to_config=candidate.name,
                        delta_accuracy=delta_accuracy,
                        delta_cost=delta_cost,
                        ratio=ratio,
                        accepted=accepted_by_key,
                        miss_count=_miss_count(trial_eval),
                        worst_lateness=trial_eval.worst_lateness,
                        note=(
                            "repair candidate: improves deadlines with minimal precision loss"
                            if accepted_by_key
                            else "rejected: no strict timing improvement"
                        ),
                    )
                )

                if not accepted_by_key:
                    blocked.add(edge)
                    continue

                # Feasibility first, then minimal quality sacrifice, then high
                # timing/quality ratio, then lower nominal cost.
                select_key = (
                    _miss_count(trial_eval),
                    trial_eval.worst_lateness,
                    trial_eval.sum_lateness,
                    loss,
                    -ratio,
                    trial_eval.total_nominal_cost,
                )
                if best is None or select_key < best[0]:
                    best = (select_key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost, timing_gain)

        if best is None:
            break

        _, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost, timing_gain = best
        old = current_selection[cnn_id]
        current_selection = dict(current_selection)
        current_selection[cnn_id] = candidate
        current_eval = trial_eval
        repair_used = True
        trace.append(
            CandidateTrace(
                phase="quality_repair_selected",
                cnn_id=cnn_id,
                from_config=old.name,
                to_config=candidate.name,
                delta_accuracy=delta_accuracy,
                delta_cost=delta_cost,
                ratio=ratio,
                accepted=True,
                miss_count=_miss_count(trial_eval),
                worst_lateness=trial_eval.worst_lateness,
                note=f"selected; timing_gain={timing_gain:.1f}, quality_loss={_quality_loss(old, candidate):.2f}",
            )
        )

    return current_selection, current_eval, repair_used


def _quality_first_global_search(
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
            previous_selection=None,
            switch_cost=0,
            switch_cost_mode=switch_cost_mode,
        )
        if not ev.feasible:
            continue
        if best is None or (ev.total_accuracy_score, -ev.total_nominal_cost) > (best.total_accuracy_score, -best.total_nominal_cost):
            best = ev
    return best


def _materialize_quality_tasks(
    tasks: Sequence[Task],
    selection: Dict[str, CNNConfig],
    initial_selection: Dict[str, CNNConfig],
    switch_cost: int,
    switch_cost_mode: str,
) -> List[Task]:
    # Offline QF chooses one configuration before execution; there is no runtime
    # remap from the all-CPU start point, therefore no switch cost is charged.
    mapped = _materialize_tasks(
        tasks,
        selection,
        previous_selection=None,
        switch_cost=0,
        switch_cost_mode=switch_cost_mode,
    )
    by_cnn = {cnn_id: cfg for cnn_id, cfg in selection.items()}
    init_by_cnn = {cnn_id: cfg for cnn_id, cfg in initial_selection.items()}
    out: List[Task] = []
    for t in mapped:
        cfg = by_cnn.get(t.cnn_id or t.name)
        init = init_by_cnn.get(t.cnn_id or t.name)
        changed = False
        if cfg is not None and init is not None:
            opt = cfg.by_task.get(t.name)
            old = init.by_task.get(t.name)
            changed = opt is not None and old is not None and opt.key != old.key
        out.append(
            replace(
                t,
                df_period=0,
                df_config=cfg.name if cfg is not None else "",
                df_action="quality_first",
                df_changed=False,
            )
        )
    return out


def _qf_feasible_search(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    initial_selection: Dict[str, CNNConfig],
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
    max_global_evals: int,
    trace: List[CandidateTrace],
) -> Optional[MappingEvaluation]:
    """Find the best fixed offline QF configuration.

    Offline QF starts all CPU/high quality, then searches offline over all
    legal resource/precision configurations. It may use NPU and may lower
    quality to meet deadlines, but the selected configuration is fixed before
    execution and is not remapped period-by-period.
    """
    cnn_ids = list(configs_by_cnn)
    lists: List[List[CNNConfig]] = []
    total = 1
    for cnn_id in cnn_ids:
        start_cfg = initial_selection[cnn_id]
        # QF may use NPU. The offline search starts conceptually from the
        # all-CPU/high-precision point, then considers every legal placement whose
        # quality is not higher than that start point. This permits equal-quality
        # CPU->NPU remapping as well as precision reduction when required.
        candidates = [cfg for cfg in configs_by_cnn[cnn_id] if cfg.accuracy_score <= start_cfg.accuracy_score]
        if not candidates:
            candidates = list(configs_by_cnn[cnn_id])
        candidates = sorted(candidates, key=lambda c: (-c.accuracy_score, c.cost, -_npu_count(c), c.name))
        lists.append(candidates)
        total *= max(1, len(candidates))
        if total > max_global_evals:
            # Keep the most QF-relevant candidates if the full product is too large.
            lists[-1] = candidates[: max(1, max_global_evals // max(1, total // max(1, len(candidates))))]

    best: Optional[MappingEvaluation] = None
    tested = 0
    for combo in product(*lists):
        tested += 1
        selection = {cnn_id: cfg for cnn_id, cfg in zip(cnn_ids, combo)}
        ev = evaluate_selection(
            tasks,
            selection,
            arch,
            sim_time,
            policy,
            previous_selection=None,
            switch_cost=0,
            switch_cost_mode=switch_cost_mode,
        )
        trace.append(
            CandidateTrace(
                phase="offline_qf_test",
                cnn_id="*",
                from_config=", ".join(f"{k}:{initial_selection[k].name}" for k in sorted(initial_selection)),
                to_config=", ".join(f"{k}:{selection[k].name}" for k in sorted(selection)),
                delta_accuracy=ev.total_accuracy_score - _selection_accuracy(initial_selection),
                delta_cost=ev.total_nominal_cost - _selection_cost(initial_selection),
                ratio=0.0,
                accepted=ev.feasible,
                miss_count=_miss_count(ev),
                worst_lateness=ev.worst_lateness,
                note="offline QF candidate; NPU allowed; accepted if deadline-feasible",
            )
        )
        if not ev.feasible:
            continue
        if best is None or (ev.total_accuracy_score, -ev.total_nominal_cost) > (best.total_accuracy_score, -best.total_nominal_cost):
            best = ev

    if best is not None:
        trace.append(
            CandidateTrace(
                phase="offline_qf_selected",
                cnn_id="*",
                from_config="CPU/FP32 start",
                to_config=", ".join(f"{k}:{v.name}" for k, v in sorted(best.selection.items())),
                delta_accuracy=best.total_accuracy_score - _selection_accuracy(initial_selection),
                delta_cost=best.total_nominal_cost - _selection_cost(initial_selection),
                ratio=0.0,
                accepted=True,
                miss_count=0,
                worst_lateness=0,
                note=f"best fixed feasible QF configuration after {tested} offline tests; NPU allowed",
            )
        )
    return best


def quality_first_mapping(
    tasks: Sequence[Task],
    arch: ArchConfig,
    sim_time: int,
    policy: str = "np_fp",
    switch_cost: int = 0,
    switch_cost_mode: str = "per_cnn",
    max_configs_per_cnn: int = 512,
    max_iterations: int = 8,
    max_global_evals: int = 4096,
) -> DeadlineFirstMappingResult:
    """Offline Quality-First mapping.

    Requested semantics: start with all tasks on CPU at maximum numerical precision, test
    offline configurations while decreasing quality and/or using NPU until
    deadlines are met, choose the best feasible fixed configuration, then
    execute it without period-by-period remapping.
    """
    configs_by_cnn = build_cnn_configurations(tasks, arch, max_configs_per_cnn=max_configs_per_cnn)
    initial_selection = _all_cpu_high_quality_selection(configs_by_cnn)
    trace: List[CandidateTrace] = []

    initial_eval = evaluate_selection(tasks, initial_selection, arch, sim_time, policy)
    trace.append(
        CandidateTrace(
            phase="offline_qf_init_cpu",
            cnn_id="*",
            from_config="-",
            to_config=", ".join(f"{k}:{v.name}" for k, v in sorted(initial_selection.items())),
            delta_accuracy=0.0,
            delta_cost=0,
            ratio=0.0,
            accepted=initial_eval.feasible,
            miss_count=_miss_count(initial_eval),
            worst_lateness=initial_eval.worst_lateness,
            note="all CPU / maximum numerical precision tested offline first",
        )
    )

    if initial_eval.feasible:
        current_selection = dict(initial_selection)
        current_eval = initial_eval
        repair_used = False
        global_search_used = False
    else:
        best = _qf_feasible_search(
            tasks,
            configs_by_cnn,
            initial_selection,
            arch,
            sim_time,
            policy,
            switch_cost,
            switch_cost_mode,
            max_global_evals,
            trace,
        )
        if best is not None:
            current_selection = dict(best.selection)
            current_eval = best
            repair_used = True
            global_search_used = True
        else:
            # No feasible fixed point exists. Pick the lowest-cost legal
            # configuration so the report exposes the remaining violation.
            current_selection = {}
            for cnn_id, cfgs in configs_by_cnn.items():
                start_cfg = initial_selection[cnn_id]
                pool = [cfg for cfg in cfgs if cfg.accuracy_score <= start_cfg.accuracy_score] or list(cfgs)
                current_selection[cnn_id] = min(pool, key=lambda c: (c.cost, -c.accuracy_score, -_npu_count(c), c.name))
            current_eval = evaluate_selection(tasks, current_selection, arch, sim_time, policy, previous_selection=None, switch_cost=0, switch_cost_mode=switch_cost_mode)
            repair_used = True
            global_search_used = False
            trace.append(
                CandidateTrace(
                    phase="offline_qf_no_feasible_fixed",
                    cnn_id="*",
                    from_config="CPU/FP32 start",
                    to_config=", ".join(f"{k}:{v.name}" for k, v in sorted(current_selection.items())),
                    delta_accuracy=current_eval.total_accuracy_score - _selection_accuracy(initial_selection),
                    delta_cost=current_eval.total_nominal_cost - _selection_cost(initial_selection),
                    ratio=0.0,
                    accepted=False,
                    miss_count=_miss_count(current_eval),
                    worst_lateness=current_eval.worst_lateness,
                    note="no fixed QF configuration met deadlines; lowest-cost legal point kept for diagnostics",
                )
            )

    final_tasks = _materialize_quality_tasks(tasks, current_selection, initial_selection, switch_cost, switch_cost_mode)
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
        schedulable=final_eval.feasible,
        normalization_accuracy_score=_max_accuracy_score(configs_by_cnn),
        period_steps=[],
    )


def quality_first_offline_mapping(*args, **kwargs) -> DeadlineFirstMappingResult:
    return quality_first_mapping(*args, **kwargs)


def _propose_online_qf_repair(
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
    """One online QF change: lower quality and/or remap, never raise quality."""
    old_key = (_miss_count(observed), observed.worst_lateness, observed.sum_lateness, -observed.total_accuracy_score, observed.total_nominal_cost)
    traces: List[CandidateTrace] = []
    best: Optional[Tuple[Tuple[int, int, int, float, int, str, str], str, CNNConfig, MappingEvaluation, float, float, int]] = None

    for cnn_id, cfgs in configs_by_cnn.items():
        old = current_selection[cnn_id]
        for candidate in _single_step_candidates(cfgs, old):
            edge = _selection_edge_key(cnn_id, old, candidate)
            if edge in blocked:
                continue
            # QF repair may remap at equal quality, or decrease quality. It may
            # not increase quality during online adaptation.
            if candidate.accuracy_score > old.accuracy_score:
                continue
            if candidate.name == old.name:
                continue
            trial_selection = dict(current_selection)
            trial_selection[cnn_id] = candidate
            trial_eval = _predict_one_period(tasks, trial_selection, current_selection, arch, window, period_index, policy, switch_cost, switch_cost_mode)
            key = (_miss_count(trial_eval), trial_eval.worst_lateness, trial_eval.sum_lateness, -trial_eval.total_accuracy_score, trial_eval.total_nominal_cost)
            ratio, delta_accuracy, delta_cost = _candidate_ratio(old, candidate, switch_cost)
            accepted_by_key = key < old_key
            traces.append(
                CandidateTrace(
                    phase="online_qf_repair_precheck",
                    cnn_id=cnn_id,
                    from_config=old.name,
                    to_config=candidate.name,
                    delta_accuracy=delta_accuracy,
                    delta_cost=delta_cost,
                    ratio=ratio,
                    accepted=accepted_by_key,
                    miss_count=_miss_count(trial_eval),
                    worst_lateness=trial_eval.worst_lateness,
                    note="prevalidated one-step QF repair" if accepted_by_key else "rejected: does not improve deadline key",
                )
            )
            if not accepted_by_key:
                blocked.add(edge)
                continue
            # Deadline-first key first, then smallest precision loss.
            full_key = (key[0], key[1], key[2], -candidate.accuracy_score, key[4], cnn_id, candidate.name)
            if best is None or full_key < best[0]:
                best = (full_key, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost)

    if best is None:
        return current_selection, PeriodStep(period_index, dict(current_selection), "hold", accepted=False, miss_count=old_key[0], worst_lateness=observed.worst_lateness, note="no safe QF repair found"), traces

    _, cnn_id, candidate, trial_eval, ratio, delta_accuracy, delta_cost = best
    old = current_selection[cnn_id]
    next_selection = dict(current_selection)
    next_selection[cnn_id] = candidate
    step = PeriodStep(
        period_index=period_index,
        selection=dict(next_selection),
        action="qf_repair",
        changed_cnn=cnn_id,
        from_config=old.name,
        to_config=candidate.name,
        accepted=True,
        miss_count=_miss_count(trial_eval),
        worst_lateness=trial_eval.worst_lateness,
        note="one QF adaptation prevalidated for next period",
    )
    traces.append(
        CandidateTrace(
            phase="online_qf_repair_selected",
            cnn_id=cnn_id,
            from_config=old.name,
            to_config=candidate.name,
            delta_accuracy=delta_accuracy,
            delta_cost=delta_cost,
            ratio=ratio,
            accepted=True,
            miss_count=_miss_count(trial_eval),
            worst_lateness=trial_eval.worst_lateness,
            note="selected for execution in the next period",
        )
    )
    return next_selection, step, traces


def _build_online_quality_period_steps(
    tasks: Sequence[Task],
    configs_by_cnn: Dict[str, List[CNNConfig]],
    arch: ArchConfig,
    sim_time: int,
    policy: str,
    switch_cost: int,
    switch_cost_mode: str,
) -> Tuple[List[PeriodStep], List[CandidateTrace]]:
    period_count = _dominant_instance_count(tasks, sim_time)
    window = _dominant_period_window(tasks)
    current_selection = _all_cpu_high_quality_selection(configs_by_cnn)
    previous_selection: Optional[Dict[str, CNNConfig]] = None
    current_step = PeriodStep(0, dict(current_selection), "init_cpu_quality", note="P0 forced all CPU / maximum numerical precision; executed even if deadlines pass")
    blocked: set[Tuple[str, str, str]] = set()
    steps: List[PeriodStep] = []
    trace: List[CandidateTrace] = []

    for period_index in range(period_count):
        current_step.period_index = period_index
        current_step.selection = dict(current_selection)
        observed = _one_period_observation(tasks, current_selection, previous_selection, current_step, arch, window, policy, switch_cost, switch_cost_mode)
        observed_misses = _miss_count(observed)
        current_step.miss_count = observed_misses
        current_step.worst_lateness = observed.worst_lateness
        current_step.accepted = observed_misses == 0 or current_step.action == "init_cpu_quality"
        if current_step.action == "init_cpu_quality":
            current_step.note = "P0 all CPU/high quality observed; " + ("deadline miss, QF adaptation searched" if observed_misses else "deadlines pass, configuration held")
        elif current_step.action == "qf_repair":
            current_step.note = "QF adaptation executed; " + ("misses remain" if observed_misses else "deadlines pass")
        elif current_step.action == "hold":
            current_step.note = current_step.note or "configuration held"
        steps.append(current_step)
        if period_index == period_count - 1:
            break

        # Online QF only changes if the current or next held period is unsafe.
        decision_eval = observed
        if observed_misses == 0:
            hold_eval = _predict_one_period(tasks, current_selection, current_selection, arch, window, period_index + 1, policy, switch_cost, switch_cost_mode)
            decision_eval = hold_eval
        if _miss_count(decision_eval):
            next_selection, next_step, traces = _propose_online_qf_repair(tasks, configs_by_cnn, current_selection, decision_eval, arch, window, period_index + 1, policy, blocked, switch_cost, switch_cost_mode)
            trace.extend(traces)
        else:
            next_selection = dict(current_selection)
            next_step = PeriodStep(period_index + 1, dict(next_selection), "hold", accepted=True, note="held: current all-CPU/QF configuration is deadline-safe")

        previous_selection = dict(current_selection)
        current_selection = dict(next_selection)
        current_step = next_step

    return steps, trace


def quality_first_online_mapping(
    tasks: Sequence[Task],
    arch: ArchConfig,
    sim_time: int,
    policy: str = "np_fp",
    switch_cost: int = 0,
    switch_cost_mode: str = "per_cnn",
    max_configs_per_cnn: int = 512,
) -> DeadlineFirstMappingResult:
    configs_by_cnn = build_cnn_configurations(tasks, arch, max_configs_per_cnn=max_configs_per_cnn)
    period_steps, trace = _build_online_quality_period_steps(tasks, configs_by_cnn, arch, sim_time, policy, switch_cost, switch_cost_mode)
    jobs = generate_jobs_online_periods(tasks, period_steps, sim_time, switch_cost, switch_cost_mode)
    result = simulate(jobs, arch, policy=policy)
    final_selection = dict(period_steps[-1].selection) if period_steps else _all_cpu_high_quality_selection(configs_by_cnn)
    evaluation = _evaluation_from_result(final_selection, result)
    return DeadlineFirstMappingResult(
        tasks=[j.task for j in jobs],
        selection=final_selection,
        initial_selection=period_steps[0].selection if period_steps else final_selection,
        configs_by_cnn=configs_by_cnn,
        evaluation=evaluation,
        trace=trace,
        repair_used=any(s.action == "qf_repair" for s in period_steps),
        global_search_used=False,
        schedulable=evaluation.feasible,
        normalization_accuracy_score=_max_accuracy_score(configs_by_cnn),
        period_steps=list(period_steps),
    )

def build_quality_mapping_report(mapping: DeadlineFirstMappingResult) -> str:
    lines: List[str] = []
    lines.append("=== Mapping Online Quality-First ===" if mapping.period_steps else "=== Mapping Offline Quality-First ===")
    if mapping.period_steps:
        lines.append("Principle: P0 all CPU / maximum numerical precision, then at most one validated adaptation per period; no online precision increase.")
    else:
        lines.append("Principle: starts all CPU / maximum numerical precision, then offline search with precision decrease and/or NPU usage; fixed execution configuration, no runtime remap.")
    lines.append(f"Schedulable: {'yes' if mapping.schedulable else 'no'}")
    lines.append(f"Repair phase used: {'yes' if mapping.repair_used else 'no'}")
    lines.append(f"Global search used: {'yes' if mapping.global_search_used else 'no'}")
    lines.append(f"Global precision score: {mapping.evaluation.total_accuracy_score:.2f} / {mapping.normalization_accuracy_score:.2f} ({mapping.normalized_accuracy_percent:.2f}%)")
    lines.append(f"Global nominal cost: {mapping.evaluation.total_nominal_cost} cycles")
    if mapping.period_steps:
        lines.append("")
        lines.append("Online Quality-First evolution executed period by period:")
        for step in mapping.period_steps:
            cfg_txt = ", ".join(f"{cnn}:{cfg.name}" for cnn, cfg in sorted(step.selection.items()))
            if step.action == "init_cpu_quality":
                action = "high-precision CPU initialization"
            elif step.action == "qf_repair":
                action = f"adaptation {step.changed_cnn}: {step.from_config} -> {step.to_config}"
            else:
                action = "hold"
            lines.append(
                f"- P{step.period_index}: {action}; miss={step.miss_count}; "
                f"max_lateness={step.worst_lateness}; configs=[{cfg_txt}] ({step.note})"
            )
    lines.append("")
    lines.append("Initial high-precision configurations:")
    for cnn_id in sorted(mapping.initial_selection):
        cfg = mapping.initial_selection[cnn_id]
        lines.append(f"- {cnn_id}: {cfg.name}, cost={cfg.cost}, precision-score={cfg.accuracy_score:.2f}")
    lines.append("")
    lines.append("Final selected configurations:")
    for cnn_id in sorted(mapping.selection):
        cfg = mapping.selection[cnn_id]
        init = mapping.initial_selection[cnn_id]
        switch_note = " (unchanged depuis precision max)" if not cfg.differs_from(init) else f" (repaired from {init.name})"
        loss = init.accuracy_score - cfg.accuracy_score
        lines.append(f"- {cnn_id}: {cfg.name}{switch_note}, cost={cfg.cost}, precision-score={cfg.accuracy_score:.2f}, precision loss={loss:.2f}")
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
        lines.append("Journal Quality-First:")
        for tr in mapping.trace[:120]:
            if tr.phase == "quality_init":
                status = "FAISABLE" if tr.accepted else "INFAISABLE"
            elif tr.accepted:
                status = "PREVALIDATED / ACCEPTED"
            else:
                status = "REJECTED"
            ratio = "inf" if tr.ratio == float("inf") else f"{tr.ratio:.4f}"
            lines.append(
                f"- [{tr.phase}] {tr.cnn_id}: {tr.from_config} -> {tr.to_config}, "
                f"ΔP={tr.delta_accuracy:.2f}, ΔC={tr.delta_cost}, score={ratio}, "
                f"miss={tr.miss_count}, max_lateness={tr.worst_lateness} => {status} ({tr.note})"
            )
        if len(mapping.trace) > 120:
            lines.append(f"... {len(mapping.trace) - 120} other trials not displayed")

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
