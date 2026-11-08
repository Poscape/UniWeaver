"""Run any unified UMM backend over the UniWeaver-Bench suites."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.umm_benchmark_runner import (
    benchmark_status_counts,
    build_benchmark_provenance,
    run_benchmark_batch,
    task_dir_name_for_prompt,
)
from baselines.umm_cli import (
    require_backend_repo,
    require_positive_image_steps,
    require_positive_runtime_timeouts,
    require_positive_text_limits,
    require_sampling_temperature,
)
from baselines.umm_evaluation_prepare import (
    prepare_requests,
    prepare_review_variants,
)
from baselines.umm_framework import UMMRunConfig, classify_task_error, run_minimal_task
from baselines.umm_minimal_persistence import MinimalPersistenceSession
from baselines.umm_runtime_backend import load_supervised_backend
from baselines.umm_runtime_limits import backend_task_scope, runtime_limits_from_args
from baselines.umm_uniweaver_research import run_uniweaver_images, run_uniweaver_task
from core.hashing import sha256_file, sha256_text
from core.policy_registry import (
    EVALUATION_SPEC_ID,
    INTERRUPTION_RECOVERY_EVENT,
    ORCHESTRATION_FAILURE_EVENT,
    RUNTIME_MODEL_OPERATION_EVENT,
)
from core.production import IMAGE_GENERATION_CONFIG
from core.serialization import atomic_write_json
from core.tasks import BENCHMARK_SPECS, validate_uniweaver_bench_tasks

BATCH_SCHEMA_VERSION = "umm-bench-textonly-v1"
load_backend = load_supervised_backend
SUPPORTED_BENCHMARKS = {
    name: spec["task_count"] for name, spec in BENCHMARK_SPECS.items()
}
AGENT_VISIBLE_FIELDS = ("id", "user_prompt")
REQUIRED_METADATA_FIELDS = (
    "task_family",
    "difficulty",
)
EVALUATOR_ONLY_FIELDS = (
    "evaluation_reference",
)


def _load_tasks(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    benchmark_name = payload.get("benchmark_name")
    expected_task_count = SUPPORTED_BENCHMARKS.get(benchmark_name)
    if expected_task_count is None:
        raise ValueError(f"unexpected benchmark: {benchmark_name}")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected_task_count:
        raise ValueError(
            f"{benchmark_name} benchmark must contain {expected_task_count} tasks"
        )
    validate_uniweaver_bench_tasks(tasks)
    required = set(AGENT_VISIBLE_FIELDS) | set(REQUIRED_METADATA_FIELDS)
    seen = set()
    normalized = []
    for position, task in enumerate(tasks, 1):
        if not isinstance(task, dict):
            raise TypeError(f"task {position} is not an object")
        missing = sorted(required - set(task))
        if missing:
            raise ValueError(f"task {position} is missing required fields: {missing}")
        task_id = task["id"]
        prompt = task["user_prompt"]
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError(f"invalid or duplicate task id at position {position}")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id} has an empty user_prompt")
        seen.add(task_id)
        normalized.append(task)
    return payload, normalized


def _provenance(args, backend, benchmark_path: Path, benchmark: dict) -> dict:
    benchmark_name = benchmark["benchmark_name"]
    benchmark_task_count = len(benchmark["tasks"])
    return build_benchmark_provenance(
        backend,
        args,
        schema_version=BATCH_SCHEMA_VERSION,
        mode=args.protocol,
        seed_policy=(
            "image_seed_for(base_seed, kind, identity); "
            "kinds are condition and step"
        ),
        extra={
            "protocol": args.protocol,
            "benchmark_name": benchmark_name,
            "benchmark_file_sha256": sha256_file(benchmark_path),
            "benchmark_task_count": benchmark_task_count,
            "benchmark_input_policy": "prompt_only",
            "agent_visible_fields": list(AGENT_VISIBLE_FIELDS),
            "evaluator_only_fields": [
                *REQUIRED_METADATA_FIELDS, *EVALUATOR_ONLY_FIELDS,
            ],
            "sft_checkpoint": str(Path(args.sft_checkpoint).resolve())
            if args.sft_checkpoint
            else None,
            "step_image_shape": [args.image_height, args.image_width],
            "image_generation": {
                **IMAGE_GENERATION_CONFIG,
                "num_timesteps": args.image_steps,
            },
            "task_scheduling": "serial shared UMM model instance",
            "attachment_count": 0,
            "evidence_packet_count": 0,
        },
    )


def _failure_record(task: dict, task_dir: Path, status: str, error: Exception) -> dict:
    state_path = task_dir / "state.json"
    if not state_path.is_file() and (task_dir / "task_result.json").is_file():
        state_path = task_dir / "task_result.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    return {
        "task_id": task["id"],
        "task_family": task["task_family"],
        "difficulty": task["difficulty"],
        "prompt_sha256": sha256_text(task["user_prompt"]),
        "status": state.get("failure_kind", status),
        "error": state.get("error", f"{type(error).__name__}: {error}"),
        "output_dir": str(task_dir),
        "trajectory_path": None,
        "state_path": str(state_path) if state_path.is_file() else None,
        "step_count": len(state.get("steps", {})),
        "failure_provenance": state.get("failure_provenance", {
            "policy_id": RUNTIME_MODEL_OPERATION_EVENT["policy_id"],
            "classification": "model_or_runtime_failure",
            "resolved": True,
            "exception_type": type(error).__name__,
            "message": str(error),
        }),
    }


def _success_record(task: dict, task_dir: Path, trajectory_path: Path) -> dict:
    state = json.loads((task_dir / "state.json").read_text(encoding="utf-8"))
    steps = state.get("steps", {})
    return {
        "task_id": task["id"],
        "task_family": task["task_family"],
        "difficulty": task["difficulty"],
        "prompt_sha256": sha256_text(task["user_prompt"]),
        "status": "completed",
        "output_dir": str(task_dir),
        "trajectory_path": str(trajectory_path),
        "state_path": str(task_dir / "state.json"),
        "step_count": len(steps),
        "generated_step_count": sum(
            value.get("generated") is True for value in steps.values()
            if isinstance(value, dict)
        ),
        "condition_count": len(state.get("conditions", {})),
    }


def _minimal_success_record(task: dict, task_dir: Path) -> dict:
    state = json.loads((task_dir / "run.json").read_text(encoding="utf-8"))
    return {
        "task_id": task["id"],
        "task_family": task["task_family"],
        "difficulty": task["difficulty"],
        "prompt_sha256": sha256_text(task["user_prompt"]),
        "status": state.get("status"),
        "output_dir": str(task_dir),
        "trajectory_path": None,
        "state_path": str(task_dir / "task_result.json"),
        "step_count": len(state.get("steps", [])),
        "generated_step_count": len(state.get("steps", [])),
        "condition_count": 0,
    }


def _summary(records: list[dict], benchmark_name: str, task_count: int) -> dict:
    return {
        "schema_version": BATCH_SCHEMA_VERSION,
        "benchmark_name": benchmark_name,
        "task_count": task_count,
        "processed_count": len(records),
        **benchmark_status_counts(records),
        "records": records,
    }


def _resume_records(run_dir: Path, tasks: list[dict], provenance: dict) -> list[dict]:
    """Load a validated completed prefix without changing prior evidence."""
    snapshot = run_dir / "benchmark_task_matrix.json"
    provenance_path = run_dir / "provenance.json"
    manifest_path = run_dir / "manifest.json"
    if not snapshot.is_file() or not provenance_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            "resume requires benchmark_task_matrix.json, provenance.json, and manifest.json"
        )
    if sha256_file(snapshot) != provenance["benchmark_file_sha256"]:
        raise ValueError("resume benchmark snapshot does not match requested benchmark")
    persisted_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for field in ("benchmark_file_sha256", "model_identity", "image_execution_identity"):
        if persisted_provenance.get(field) != provenance.get(field):
            raise ValueError(f"resume provenance mismatch: {field}")
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = persisted_manifest.get("records")
    if not isinstance(records, list) or len(records) > len(tasks):
        raise ValueError("resume manifest has an invalid records prefix")
    expected_prefix = [task["id"] for task in tasks[:len(records)]]
    actual_prefix = [record.get("task_id") for record in records if isinstance(record, dict)]
    if actual_prefix != expected_prefix:
        raise ValueError("resume manifest records are not the benchmark task prefix")
    return records


@contextmanager
def _batch_resume_lease(run_dir: Path, timeout_seconds: float):
    """Hold one process-level lock while a resumed batch can mutate its manifest."""
    path = run_dir / ".resume_owner.json"
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            owner = handle.read().strip() or "unknown owner"
            raise RuntimeError(
                f"refusing concurrent batch resume; active owner: {owner}"
            ) from error
        now = time.time()
        owner = {
            "policy_id": INTERRUPTION_RECOVERY_EVENT["policy_id"],
            "status": "running",
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "acquired_at": now,
            "heartbeat_at": now,
            "lease_expires_at": now + max(300.0, float(timeout_seconds)),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        yield
        owner.update({"status": "released", "released_at": time.time()})
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def run(args) -> int:
    if not args.resume:
        return _run(args)
    run_dir = args.output_dir.resolve()
    if not run_dir.is_dir():
        raise ValueError("resume output root must be an existing directory")
    lease_seconds = (
        float(getattr(args, "task_timeout_seconds", 21_600.0))
        + float(getattr(args, "backend_startup_timeout_seconds", 3_600.0))
        + 600.0
    )
    with _batch_resume_lease(run_dir, lease_seconds):
        return _run(args)


def _run(args) -> int:
    run_dir = args.output_dir.resolve()
    if args.resume:
        if not run_dir.is_dir():
            raise ValueError("resume output root must be an existing directory")
    elif run_dir.exists():
        raise FileExistsError(f"refusing to reuse output root: {run_dir}")
    try:
        require_positive_image_steps(args)
        require_positive_runtime_timeouts(args)
        if args.protocol == "minimal":
            require_positive_text_limits(args)
            require_sampling_temperature(args)
        if args.image_height != 1024 or args.image_width != 1024:
            raise SystemExit("UniWeaver-Bench requires 1024x1024 Step images")
        require_backend_repo(args)
        benchmark_path = args.benchmark_file.resolve()
        benchmark, tasks = _load_tasks(benchmark_path)
        benchmark_name = benchmark["benchmark_name"]
        task_count = len(tasks)
        backend = load_backend(args)
        provenance = _provenance(args, backend, benchmark_path, benchmark)
    except BaseException as error:
        failure = {
            "policy_id": ORCHESTRATION_FAILURE_EVENT["policy_id"],
            "status": "error",
            "stage": "preflight",
            "exception_type": type(error).__name__,
            "message": str(error),
            "classification": "orchestration_failure",
            "resolved": False,
        }
        if not run_dir.exists():
            try:
                run_dir.mkdir(parents=True)
                atomic_write_json(run_dir / "terminal_error.json", failure)
            except Exception as persistence_error:  # noqa: BLE001
                if hasattr(error, "add_note"):
                    error.add_note(
                        "preflight terminal persistence failed: "
                        f"{type(persistence_error).__name__}: {persistence_error}"
                    )
        raise
    initial_records = _resume_records(run_dir, tasks, provenance) if args.resume else []
    if args.resume:
        provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))

    text_dir = run_dir / "text"
    image_dir = run_dir / "images"
    minimal_dir = run_dir / "tasks"

    def prepare_output(output):
        if args.protocol == "uniweaver":
            text_dir.mkdir()
            image_dir.mkdir()
        else:
            minimal_dir.mkdir()
        shutil.copy2(benchmark_path, output / "benchmark_task_matrix.json")
        atomic_write_json(output / "provenance.json", provenance)

    def process_task(_index, task):
        task_id = task["id"]
        prompt = task["user_prompt"]
        task_root = image_dir if args.protocol == "uniweaver" else minimal_dir
        task_dir = task_root / task_dir_name_for_prompt(task_id, prompt)
        trajectory_path = text_dir / f"{task_id}.json"
        try:
            if args.protocol == "minimal":
                initial = {
                    "framework": "umm-unified-minimal-v1",
                    "backend": getattr(args, "backend", "unknown"),
                    "task_key": task_id,
                    "task_prompt": prompt,
                    "runtime_configuration": provenance,
                }
                session = MinimalPersistenceSession(
                    task_dir, initial, resume=args.resume,
                )
                try:
                    limits = runtime_limits_from_args(args)
                    with backend_task_scope(backend, limits.task_timeout_seconds):
                        result = run_minimal_task(
                            backend, prompt, task_id,
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
                            initial_steps=(
                                session.completed_steps() if args.resume else None
                            ),
                        )
                except BaseException as error:
                    # Record the same stage name the summary record will carry,
                    # so the per-task file and summary.json agree.
                    stage = (
                        "minimal_step"
                        if session.state.get("active_operation") is not None
                        else "preflight"
                    )
                    session.fail(
                        error,
                        stage=stage,
                        status=(
                            classify_task_error(error, protocol="minimal")
                            if stage != "preflight"
                            else "preflight_error"
                        ),
                    )
                    raise
                result["repo_source_identity"] = args.repo_source_identity
                session.finish(result)
                return _minimal_success_record(task, task_dir)
            state_path = task_dir / "state.json"
            state = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if args.resume and state_path.is_file() else {}
            )
            if args.resume and state.get("status") == "done" and trajectory_path.is_file():
                return _success_record(task, task_dir, trajectory_path)
            if args.resume and trajectory_path.is_file():
                trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
                plan = trajectory.get("plan")
                planner_complete = (
                    isinstance(plan, dict)
                    and all(stage in plan for stage in ("plan1", "plan2", "plan3", "plan4"))
                    and isinstance(trajectory.get("_text_validation"), dict)
                )
                if planner_complete and state_path.is_file():
                    run_uniweaver_images(
                        backend, trajectory, task_dir, provenance,
                        condition_shape=(512, 512),
                        step_shape=(args.image_height, args.image_width),
                        image_steps=args.image_steps,
                        resume=True,
                        review_retry=args.review_retry,
                    )
                    atomic_write_json(trajectory_path, trajectory)
                    return _success_record(task, task_dir, trajectory_path)
                if planner_complete and (
                    not task_dir.exists() or not any(task_dir.iterdir())
                ):
                    run_uniweaver_images(
                        backend, trajectory, task_dir, provenance,
                        condition_shape=(512, 512),
                        step_shape=(args.image_height, args.image_width),
                        image_steps=args.image_steps,
                        review_retry=args.review_retry,
                    )
                    atomic_write_json(trajectory_path, trajectory)
                    return _success_record(task, task_dir, trajectory_path)
                interrupted_path = trajectory_path.with_name(
                    f"{task_id}.interrupted-planner."
                    f"{sha256_file(trajectory_path)[:12]}.json"
                )
                if not interrupted_path.exists():
                    shutil.copy2(trajectory_path, interrupted_path)
                if task_dir.exists() and any(task_dir.iterdir()):
                    raise RuntimeError(
                        "incomplete Planner evidence conflicts with an existing task state"
                    )
            run_uniweaver_task(
                backend, task_id, prompt, task_dir, provenance,
                condition_shape=(512, 512),
                step_shape=(args.image_height, args.image_width),
                image_steps=args.image_steps,
                trajectory_path=trajectory_path,
                review_retry=args.review_retry,
            )
            return _success_record(task, task_dir, trajectory_path)
        except Exception as error:  # noqa: BLE001 - preserve every benchmark task in denominator
            return _failure_record(
                task, task_dir,
                classify_task_error(error, protocol=args.protocol), error)

    manifest = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "benchmark": {
            "name": benchmark_name,
            "task_count": task_count,
            "source_sha256": provenance["benchmark_file_sha256"],
        },
        "provenance": provenance,
        "duration_seconds": None,
        "records": [],
    }
    summary = run_benchmark_batch(
        tasks=tasks,
        output=run_dir,
        manifest=manifest,
        process_task=process_task,
        summary_builder=lambda records: _summary(
            records, benchmark_name, task_count),
        manifest_record_builder=lambda _task, record, _output: record,
        prepare_output=prepare_output,
        progress_printer=lambda index, total, task, _summary: print(
            f"[{index}/{total}] {task['id']}: "
            f"{_summary['records'][-1]['status']}", flush=True),
        resume=args.resume,
        initial_records=initial_records,
    )
    try:
        if args.protocol == "uniweaver":
            variants = prepare_review_variants(run_dir, benchmark_name)
            preparation = {
                "status": "prepared",
                "evaluation_version": EVALUATION_SPEC_ID,
                "manifest_path": "review_variants.json",
                "judge_called": False,
                "report_variants": variants["variants"],
            }
        else:
            evaluation = prepare_requests(
                run_dir, benchmark_name, protocol=args.protocol,
            )
            preparation = {
                "status": "prepared",
                "evaluation_version": evaluation["evaluation_version"],
                "manifest_path": "evaluation/scoring/manifest.json",
                "judge_called": False,
                "request_counts": {
                    metric: record["request_count"]
                    for metric, record in evaluation["outputs"].items()
                },
            }
    except Exception as error:
        preparation = {
            "policy_id": ORCHESTRATION_FAILURE_EVENT["policy_id"],
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
            "judge_called": False,
            "classification": "judge_packet_missing",
            "resolved": False,
        }
        error_path = run_dir / "evaluation" / "scoring" / "prepare_error.json"
        atomic_write_json(error_path, preparation)
        summary["evaluation_preparation"] = preparation
        manifest["evaluation_preparation"] = preparation
        atomic_write_json(run_dir / "summary.json", summary)
        atomic_write_json(run_dir / "manifest.json", manifest)
        raise
    summary["evaluation_preparation"] = preparation
    manifest["evaluation_preparation"] = preparation
    atomic_write_json(run_dir / "summary.json", summary)
    atomic_write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({
        "run_dir": str(run_dir),
        **{key: summary[key] for key in summary if key != "records"},
    }))
    return 0
