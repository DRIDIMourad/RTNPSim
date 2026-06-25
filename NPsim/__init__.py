"""NPsim: CPU/NPU scheduling and NoC-aware simulator."""

from .parser import parse_tasks_file, load_arch_config, generate_jobs, compute_hyperperiod
from .simulator import simulate, validate_schedule
from .deadline_first import deadline_first_mapping, deadline_first_online_period_mapping, build_cnn_configurations, build_mapping_report
from .quality_first import quality_first_mapping, quality_first_online_mapping, build_quality_mapping_report


def run(*args, **kwargs):
    from .main import run as _run
    return _run(*args, **kwargs)


__all__ = [
    "run",
    "parse_tasks_file",
    "load_arch_config",
    "generate_jobs",
    "compute_hyperperiod",
    "simulate",
    "validate_schedule",
    "deadline_first_mapping",
    "deadline_first_online_period_mapping",
    "build_cnn_configurations",
    "build_mapping_report",
    "quality_first_mapping",
    "quality_first_online_mapping",
    "build_quality_mapping_report",
]
