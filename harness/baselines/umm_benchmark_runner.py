"""Shared batch lifecycle for UMM benchmark policies."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path

from baselines.umm_backend_factory import backend_spec
from baselines.umm_framework import (
    build_umm_provenance,
    capabilities_for,
    run_serial_batch,
)
from baselines.umm_runtime_limits import runtime_limits_from_args
from core.hashing import sha256_text
from core.policy_registry import ORCHESTRATION_FAILURE_EVENT
from core.production import (
    REVIEW_MAX_RETRIES,
    image_execution_identity,
    image_runtime_identity,
)
from core.serialization import atomic_write_json

UNIWEAVER_MAX_WORKERS = 4


def backend_execution_workers(backend, requested: int = UNIWEAVER_MAX_WORKERS) -> int:
    """Honor the backend's declared safe in-process concurrency."""
    declared = capabilities_for(backend).max_parallel_calls
    return min(max(1, int(declared)), max(1, requested))


def task_dir_name(task_key: str, prompt_sha256: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ".-_" else "_" for char in task_key
    )
    return f"{safe}__{prompt_sha256[:12]}"


def task_dir_name_for_prompt(task_key: str, prompt: str) -> str:
    return task_dir_name(task_key, sha256_text(prompt))


def benchmark_status_counts(records: list[dict]) -> dict[str, int]:
    """Count terminal outcomes per protocol."""
    return {
        "completed": sum(item.get("status") == "completed" for item in records),
        "incomplete": sum(item.get("status") == "incomplete" for item in records),
        "planner_error": sum(item.get("status") == "planner_error" for item in records),
        "image_error": sum(item.get("status") == "image_error" for item in records),
        "step_error": sum(item.get("status") == "step_error" for item in records),
        "preflight_error": sum(
            item.get("status") == "preflight_error" for item in records
        ),
        "timed_out": sum(item.get("status") == "timed_out" for item in records),
    }


def build_benchmark_provenance(
    backend,
    args,
    *,
    schema_version: str | None,
    mode: str,
    seed_policy: str,
    extra: dict | None = None,
) -> dict:
    capabilities = capabilities_for(backend)
    execution_workers = backend_execution_workers(backend)
    image_config = dict(capabilities.image_config or {})
    image_config["num_timesteps"] = args.image_steps
    common = {
        "mode": mode,
        "step_image_shape": [args.image_height, args.image_width],
        "image_steps_requested": args.image_steps,
        "backend_image_steps": {
            "requested_parameter_consumed": (
                capabilities.requested_image_steps_consumed),
            "note": (
                "backend consumes image_steps"
                if capabilities.requested_image_steps_consumed
                else "image_steps handled by the backend generation protocol"
            ),
        },
        "image_generation": image_config,
        "review_retry": bool(getattr(args, "review_retry", False)),
        # Recorded next to the boolean so a reader can tell "review disabled"
        # from "review enabled with N rounds" without the launcher environment
        # that produced the run.
        "review_max_retries": (
            REVIEW_MAX_RETRIES if getattr(args, "review_retry", False) else 0),
        "review": bool(getattr(args, "review_retry", False)),
        "repair": bool(getattr(args, "review_retry", False)),
        "retry": bool(getattr(args, "review_retry", False)),
        "condition_scheduling": (
            "parallel" if mode == "uniweaver" and execution_workers > 1
            else "sequential by backend capability"
            if mode == "uniweaver" else "sequential by design"
        ),
        "step_scheduling": (
            "parallel after all Conditions complete"
            if mode == "uniweaver" and execution_workers > 1
            else "sequential after all Conditions complete"
            if mode == "uniweaver" else "sequential by design"
        ),
        "parallel_workers": (
            execution_workers if mode == "uniweaver" else None
        ),
        "condition_image_shape": [512, 512] if mode == "uniweaver" else None,
        "runtime_control": {
            "isolation": "persistent spawned backend worker process",
            "model_call_timeout_seconds": getattr(
                args, "model_call_timeout_seconds", None),
            "backend_startup_timeout_seconds": getattr(
                args, "backend_startup_timeout_seconds", None),
            "limits": runtime_limits_from_args(args).as_dict(),
            "timeout_policy": (
                "terminate worker and persist timed_out; batch callers continue "
                "with the next task"
            ),
            "one_hour_single_step_policy": False,
        },
        "image_execution_identity": image_execution_identity(
            extra_sources=(
                backend_spec(getattr(args, "backend", capabilities.name))
                .adapter_source,
            )
        ),
        "image_runtime_identity": image_runtime_identity(),
    }
    for name in ("repo_path", "model_path", "vq_path", "device"):
        value = getattr(args, name, None)
        if value:
            common[name] = str(Path(value).resolve()) if name != "device" else value
    if getattr(args, "repo_source_identity", None) is not None:
        common["repo_source_identity"] = args.repo_source_identity
    if getattr(args, "sft_checkpoint", None):
        common["sft_checkpoint"] = str(Path(args.sft_checkpoint).resolve())
    common.update(extra or {})
    return build_umm_provenance(
        backend,
        mode,
        schema_version=schema_version,
        base_seed=args.seed,
        seed_policy=seed_policy,
        extra=common,
    )


def run_benchmark_batch(
    *,
    tasks: Sequence[dict],
    output: Path,
    manifest: dict,
    process_task: Callable[[int, dict], dict],
    summary_builder: Callable[[list[dict]], dict],
    manifest_record_builder: Callable[[dict, dict, Path], dict],
    prepare_output: Callable[[Path], None] | None = None,
    initial_manifest_path: Path | None = None,
    progress_printer: Callable[[int, int, dict, dict], None] | None = None,
    resume: bool = False,
    initial_records: list[dict] | None = None,
) -> dict:
    """Run serial benchmark tasks with one manifest/snapshot lifecycle."""
    if output.exists() and not resume:
        raise SystemExit(
            f"output root already exists; refusing to overwrite: {output}")
    if resume:
        if not output.is_dir():
            raise SystemExit(f"resume output root not found: {output}")
    else:
        output.mkdir(parents=True)

    manifest_records: list[dict] = list(initial_records or [])

    def process(index: int, task: dict) -> dict:
        record = process_task(index, task)
        manifest_records.append(manifest_record_builder(task, record, output))
        return record

    def write_progress(
        index: int,
        task: dict,
        _record: dict,
        records: list[dict],
        started: float,
    ) -> None:
        manifest["records"] = manifest_records[:index]
        manifest["duration_seconds"] = round(time.time() - started, 3)
        summary = summary_builder(records)
        atomic_write_json(output / "manifest.json", manifest)
        atomic_write_json(output / "summary.json", summary)
        if progress_printer is not None:
            progress_printer(index, len(tasks), task, summary)

    try:
        if prepare_output is not None and not resume:
            prepare_output(output)
        if initial_manifest_path is not None and not resume:
            atomic_write_json(initial_manifest_path, manifest)
        records = run_serial_batch(
            list(tasks), process, write_progress,
            start_index=len(manifest_records),
            initial_results=manifest_records,
        )
        summary = summary_builder(records)
        atomic_write_json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        failure = {
            "policy_id": ORCHESTRATION_FAILURE_EVENT["policy_id"],
            "status": "error",
            "stage": "batch",
            "exception_type": type(error).__name__,
            "message": str(error),
            "records_completed": len(manifest_records),
            "recorded_at": time.time(),
            "classification": "orchestration_failure",
            "resolved": False,
        }
        try:
            atomic_write_json(output / "terminal_error.json", failure)
            manifest["terminal_failure"] = failure
            atomic_write_json(output / "manifest.json", manifest)
        except Exception as persistence_error:  # noqa: BLE001
            if hasattr(error, "add_note"):
                error.add_note(
                    "batch terminal persistence failed: "
                    f"{type(persistence_error).__name__}: {persistence_error}"
                )
        raise
