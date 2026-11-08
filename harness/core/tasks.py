"""Shared task sources and output locations for every UniWeaver entry point."""
import json
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SPECS = {
    'UniWeaver-Bench': {
        'version': 'uniweaver-bench-v1',
        'task_count': 120,
        'filename': 'bench.json',
    },
}
DEFAULT_BENCHMARK_FILE = (
    PROJECT_ROOT / 'datasets' / 'benchmarks' / 'uniweaver_bench' /
    BENCHMARK_SPECS['UniWeaver-Bench']['filename'])
UNIWEAVER_TRAJECTORY_ROOT = (
    PROJECT_ROOT / 'output_run' / 'runs' / 'uniweaver' / 'trajectories')


def load_benchmark_tasks(benchmark_file=None, filter_names=None):
    """Load UniWeaver-Bench tasks or an explicitly supplied task list."""
    path = Path(benchmark_file) if benchmark_file else DEFAULT_BENCHMARK_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"benchmark file not found: {path}. Pass --benchmark-file or "
            "place the benchmark matrix at this default path.")
    with path.open(encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        records = payload.get('tasks')
        if not isinstance(records, list):
            raise TypeError('benchmark payload must contain a task list')
        tasks = [
            dict(task, name=task.get('id', ''), prompt=task.get('user_prompt', ''))
            for task in records
        ]
    elif isinstance(payload, list):
        tasks = payload
    else:
        raise TypeError('benchmark payload must be a task list or object')
    if filter_names:
        lowered = [value.lower() for value in filter_names]
        tasks = [
            task for task in tasks
            if any(value in task.get('name', '').lower() for value in lowered)
        ]
    return tasks


def find_benchmark_task(task_key, benchmark_file=None):
    """Return the first task whose name contains ``task_key``, or ``None``."""
    matches = find_benchmark_tasks([task_key], benchmark_file)
    return matches[0][1] if matches else None


def find_benchmark_tasks(task_keys, benchmark_file=None):
    """Resolve several task keys while preserving their requested order."""
    tasks = load_benchmark_tasks(benchmark_file)
    resolved = []
    for task_key in task_keys:
        key = str(task_key).lower()
        match = next(
            (task for task in tasks if key in task.get('name', '').lower()),
            None,
        )
        if match is not None:
            resolved.append((task_key, match))
    return resolved


def validate_uniweaver_bench_tasks(tasks):
    """Enforce the published 20-family, 6-task, 40/40/40 composition."""
    records = list(tasks)
    if len(records) != 120:
        raise ValueError("UniWeaver-Bench must contain exactly 120 tasks")
    def field(task, *names):
        if isinstance(task, dict):
            return next((task.get(name) for name in names if task.get(name)), None)
        return next((getattr(task, name, None) for name in names
                     if getattr(task, name, None)), None)

    families = Counter(
        field(task, 'task_family', 'family') for task in records)
    if len(families) != 20 or any(count != 6 for count in families.values()):
        raise ValueError(
            "UniWeaver-Bench must contain 6 tasks in each of 20 families")
    difficulties = Counter(field(task, 'difficulty') for task in records)
    if difficulties != {'easy': 40, 'medium': 40, 'hard': 40}:
        raise ValueError(
            "UniWeaver-Bench must contain 40 easy, 40 medium, and 40 hard tasks")
