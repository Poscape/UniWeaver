"""Run one unified UMM backend over a sequential minimal task."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.umm_cli import (
    require_backend_repo,
    require_positive_image_steps,
    require_positive_runtime_timeouts,
    require_positive_text_limits,
    require_sampling_temperature,
)
from baselines.umm_framework import (
    UMMRunConfig,
    classify_task_error,
    run_minimal_task,
)
from baselines.umm_minimal_persistence import MinimalPersistenceSession
from baselines.umm_runtime_backend import load_supervised_backend
from baselines.umm_runtime_limits import (
    backend_task_scope,
    runtime_limits_from_args,
)
from core.tasks import find_benchmark_task


def run(args) -> int:
    task = find_benchmark_task(args.task_key, args.benchmark_file)
    if task is None:
        raise SystemExit(f"benchmark task not found: {args.task_key}")
    initial = {
        "framework": "umm-unified-minimal-v1",
        "backend": getattr(args, "backend", "unknown"),
        "task_key": args.task_key,
        "task_prompt": task["prompt"],
    }
    session = MinimalPersistenceSession(args.output_dir, initial, resume=args.resume)
    try:
        require_positive_text_limits(args)
        require_positive_image_steps(args)
        require_sampling_temperature(args)
        require_positive_runtime_timeouts(args)
        require_backend_repo(args)
        backend = load_supervised_backend(args)
        limits = runtime_limits_from_args(args)
        with backend_task_scope(backend, limits.task_timeout_seconds):
            result = run_minimal_task(
                backend,
                task["prompt"],
                args.task_key,
                UMMRunConfig(
                    max_steps=args.max_steps,
                    text_max_tokens=args.text_max_tokens,
                    text_temperature=args.text_temperature,
                    image_shape=(args.image_height, args.image_width),
                    image_steps=args.image_steps,
                    seed=args.seed,
                    do_sample=args.sample,
                ),
                on_step_start=session.step_started,
                on_step_done=session.step_done,
                on_step_terminal=session.step_terminal,
                on_step_error=session.step_failed,
                initial_steps=session.completed_steps() if args.resume else None,
            )
    except BaseException as error:  # persist terminal single-task state
        stage = (
            "minimal_step" if session.state.get("active_operation") is not None
            else "preflight"
        )
        status = classify_task_error(error, protocol="minimal")
        if stage == "preflight":
            # The task never reached the Step operation, so naming a Step
            # failure would repeat the same mislabelling this fixes.
            status = "preflight_error"
        try:
            session.fail(error, stage=stage, status=status)
        except Exception as persistence_error:  # noqa: BLE001
            session.persistence_failure(persistence_error)
        print({
            "status": status,
            "backend": session.state.get("backend", initial["backend"]),
            "error": str(error),
        })
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 1
    result["repo_source_identity"] = args.repo_source_identity
    try:
        persisted = session.finish(result)
    except Exception as error:
        session.persistence_failure(error)
        raise
    print(
        {
            "framework": persisted["framework"],
            "backend": persisted["backend"],
            "task_key": persisted["task_key"],
            "status": persisted["status"],
            "step_count": len(persisted["steps"]),
            "output_dir": str(Path(args.output_dir).resolve()),
        }
    )
    return 0
