"""Shared job discovery and execution for UniWeaver trajectory transports."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.runner import run_trajectory_with_sender
from core.serialization import atomic_write_json, export_trajectory
from core.tasks import UNIWEAVER_TRAJECTORY_ROOT, find_benchmark_tasks


def construct_trajectory(
    task_key,
    task_prompt,
    output_path,
    send_backend,
    *,
    extra_fields=None,
    include_step_generation=True,
):
    trajectory = run_trajectory_with_sender(
        task_key,
        task_prompt,
        output_path,
        send_backend,
        extra_fields=extra_fields,
        include_step_generation=include_step_generation,
    )
    return export_trajectory(trajectory)


def benchmark_task_specs(task_keys, label, output_path=None, benchmark_file=None):
    requested = [key.strip() for key in task_keys.split(',') if key.strip()]
    found = [
        (key, task.get('prompt', ''), task.get('name', ''))
        for key, task in find_benchmark_tasks(requested, benchmark_file)
    ]
    found_keys = {key for key, _, _ in found}
    missing = [key for key in requested if key not in found_keys]
    if missing:
        raise ValueError(f'task-key(s) not found in benchmark: {missing}')

    run_tag = time.strftime('_%Y%m%d_%H%M%S')
    specs = []
    for key, prompt, name in found:
        default_path = UNIWEAVER_TRAJECTORY_ROOT / f'{key}_{label}{run_tag}.json'
        path = output_path if len(found) == 1 and output_path else default_path
        specs.append((key, prompt, str(path), name))
    return specs


def run_task_specs(specs, run_one, parallel=1):
    """Run independent trajectory jobs and preserve requested result order."""
    if not specs:
        return []
    worker_count = max(1, min(parallel, len(specs)))
    results = [None] * len(specs)
    errors = []

    def execute(position, spec):
        key, prompt, path = spec
        print(f'\n>>> {key} started', flush=True)
        return position, key, run_one(key, prompt, path)

    if worker_count == 1:
        completed = (execute(position, spec) for position, spec in enumerate(specs))
        for done, (position, key, _result) in enumerate(completed, 1):
            results[position] = _result
            print(f'Progress: {done}/{len(specs)} tasks done', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(execute, position, spec): (position, spec[0])
                for position, spec in enumerate(specs)
            }
            for done, future in enumerate(as_completed(futures), 1):
                position, key = futures[future]
                try:
                    _, _, results[position] = future.result()
                except Exception as error:  # noqa: BLE001 - task isolation boundary
                    errors.append((key, str(error)))
                    print(f'ERROR: {key} failed: {error}', flush=True)
                print(f'Progress: {done}/{len(specs)} tasks done', flush=True)

    if errors:
        details = '; '.join(
            f'{key}: {message}' for key, message in errors[:20]
        )
        if len(errors) > 20:
            details += f'; ... {len(errors) - 20} more'
        raise RuntimeError(
            f'trajectory batch incomplete ({len(errors)}/{len(specs)} failed): '
            f'{details}'
        )
    return results


def save_combined(results, label, task_count, output_path=None):
    if task_count <= 1:
        return None
    combined_path = Path(output_path) if output_path else (
        UNIWEAVER_TRAJECTORY_ROOT
        / f'{label}_combined_{task_count}tasks_{time.strftime("%Y%m%d_%H%M%S")}.json'
    )
    if combined_path.exists():
        raise FileExistsError(f'refusing to overwrite trajectory: {combined_path}')
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(combined_path, results)
    print(f'\nCombined ({len(results)} tasks) saved to {combined_path}', flush=True)
    return combined_path


def run_constructor_cli(args, parser, label, execute_specs):
    if args.parallel < 1:
        parser.error('--parallel must be at least 1')
    if args.output_dir and ',' in args.task_key:
        parser.error('--output-dir is only supported for one task')
    try:
        benchmark = benchmark_task_specs(
            args.task_key,
            label,
            args.output_dir,
            benchmark_file=getattr(args, "benchmark_file", None),
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    for key, _prompt, _path, name in benchmark:
        print(f'Task: {name}', flush=True)
    specs = [(key, prompt, path) for key, prompt, path, _name in benchmark]
    results = execute_specs(specs)
    save_combined(results, label, len(specs), args.output_dir)
    return results
