"""Prepare frozen UniWeaver-Bench judge packets as a post-run UMM step."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from core.hashing import sha256_file
from core.policy_registry import (
    EVALUATION_SPEC_ID,
    REPORT_FAILURE_PROVENANCE_GATE,
    REPORT_REQUEST_LAYERS,
    REVIEW_VARIANT_DERIVATION_EVENT,
)
from core.serialization import atomic_write_json, atomic_write_text
from core.tasks import BENCHMARK_SPECS
from evaluation.uniweaver_eval.data import load_matrix_tasks
from evaluation.uniweaver_eval.prompts import build_report_requests
from evaluation.uniweaver_eval.reports import (
    input_echo_units,
    load_report_artifacts,
    prose_completion_tasks,
    report_contract,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256_json(value: dict) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _run_relative_path(path_value: str, run_dir: Path) -> str:
    path = Path(path_value).resolve()
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"evaluation artifact path escapes run directory: {path}"
        ) from exc


def _make_request_portable(request: dict, run_dir: Path) -> dict:
    """Replace host-specific artifact paths with hashed run-relative references."""
    portable = dict(request)
    replacements: dict[str, str] = {}

    source_value = portable.get("source_artifact")
    if not isinstance(source_value, str):
        raise TypeError("prepared request is missing source_artifact")
    source_relative = _run_relative_path(source_value, run_dir)
    replacements[source_value] = source_relative
    portable["source_artifact"] = source_relative
    portable["source_artifact_sha256"] = sha256_file(run_dir / source_relative)

    image_artifacts = []
    relative_images = []
    for path_value in portable.get("image_paths", []):
        if not isinstance(path_value, str):
            raise TypeError("prepared request image path must be a string")
        relative = _run_relative_path(path_value, run_dir)
        replacements[path_value] = relative
        relative_images.append(relative)
        image_artifacts.append({
            "path": relative,
            "sha256": sha256_file(run_dir / relative),
        })
    portable["image_paths"] = relative_images
    portable["image_artifacts"] = image_artifacts
    portable["artifact_path_base"] = "run_dir"
    portable["artifact_path_schema"] = "umm-run-relative-v1"

    user_prompt = portable.get("user_prompt")
    if isinstance(user_prompt, str):
        for absolute, relative in replacements.items():
            user_prompt = user_prompt.replace(absolute, relative)
        portable["user_prompt"] = user_prompt

    portable.pop("request_content_sha256", None)
    portable["request_content_sha256"] = _sha256_json(portable)
    return portable

def _write_jsonl(path: Path, records: list[dict]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, payload)


def _link_or_copy_immutable(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(
                f"review variant artifact conflicts with existing file: {destination}"
            )
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _selected_attempt(state: dict, step_index: str, variant: str) -> dict:
    if variant == "no_review":
        attempts = state.get("step_attempts", {}).get(step_index, [])
        matches = [item for item in attempts if item.get("attempt_index") == 0]
        if len(matches) != 1:
            raise ValueError(
                f"Step {step_index} lacks one immutable attempt_0 record"
            )
        return matches[0]
    step = state.get("steps", {}).get(step_index, {})
    selected = step.get("selected_attempt")
    attempts = state.get("step_attempts", {}).get(step_index, [])
    matches = [item for item in attempts if item.get("attempt_index") == selected]
    if len(matches) != 1:
        raise ValueError(
            f"Step {step_index} lacks its selected reviewed attempt record"
        )
    return matches[0]


def _task_review_provenance(state: dict, variant: str) -> dict:
    reviews = [
        item for item in state.get("reviews", []) if isinstance(item, dict)
    ]
    executed = any(
        item.get("status") in {"reviewed", "partial_review"}
        for item in reviews
    )
    unavailable = next(
        (
            item.get("reason") for item in reviews
            if item.get("status") == "review_unavailable"
        ),
        None,
    )
    return {
        "selection_policy": (
            "attempt_0" if variant == "no_review" else "final_selected_attempt"
        ),
        "review_applied": executed if variant == "reviewed" else False,
        "source_review_executed": executed,
        "source_review_unavailable_reason": unavailable,
        "source_review_round_count": len(reviews),
    }


def _prepare_review_variant(run_dir: Path, variant: str) -> Path:
    """Materialize one self-contained report view from immutable attempts."""
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    for name in ("benchmark_task_matrix.json",):
        _link_or_copy_immutable(run_dir / name, variant_dir / name)

    root_provenance_path = run_dir / "provenance.json"
    root_provenance = json.loads(root_provenance_path.read_text(encoding="utf-8"))
    root_summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    source_identity = {
        "run_dir": run_dir.name,
        "provenance_sha256": sha256_file(root_provenance_path),
        "benchmark_snapshot_sha256": sha256_file(
            run_dir / "benchmark_task_matrix.json"
        ),
    }
    variant_records = copy.deepcopy(root_summary.get("records", []))
    records_by_task = {
        record.get("task_id"): record
        for record in variant_records if isinstance(record, dict)
    }
    reports_dir = variant_dir / "reports"
    task_review_records = {}
    for state_path in sorted((run_dir / "images").glob("*/state.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        task_id = state.get("task_key")
        if state.get("status") != "done" or not isinstance(task_id, str):
            continue
        source_trajectory = run_dir / "text" / f"{task_id}.json"
        if not source_trajectory.is_file():
            raise FileNotFoundError(
                f"completed task lacks trajectory for review variants: {task_id}"
            )
        trajectory = json.loads(source_trajectory.read_text(encoding="utf-8"))
        task_source_identity = {
            **source_identity,
            "state_path": state_path.relative_to(run_dir).as_posix(),
            "state_sha256": sha256_file(state_path),
            "trajectory_path": source_trajectory.relative_to(run_dir).as_posix(),
            "trajectory_sha256": sha256_file(source_trajectory),
        }
        review_provenance = _task_review_provenance(state, variant)
        task_review_records[task_id] = review_provenance
        derived_state = copy.deepcopy(state)
        derived_state["report_variant"] = variant
        derived_state["derived_from"] = task_source_identity
        derived_state["review_derivation"] = review_provenance
        derived_state["reviews"] = (
            copy.deepcopy(state.get("reviews", [])) if variant == "reviewed" else []
        )
        derived_state["step_attempts"] = {}
        units = []
        task_name = state_path.parent.name
        for step_index, step in sorted(
            state.get("steps", {}).items(), key=lambda item: int(item[0])
        ):
            attempt = _selected_attempt(state, step_index, variant)
            source_image = state_path.parent / attempt["image_path"]
            if sha256_file(source_image) != attempt["image_sha256"]:
                raise ValueError(
                    f"immutable attempt hash mismatch for {task_id} Step {step_index}"
                )
            relative_image = Path("steps") / f"step_{step_index}.png"
            destination = variant_dir / "images" / task_name / relative_image
            _link_or_copy_immutable(source_image, destination)
            derived_step = copy.deepcopy(step)
            derived_step.update({
                "image_path": relative_image.as_posix(),
                "image_sha256": attempt["image_sha256"],
                "retry_round": attempt.get("retry_round"),
                "selected_attempt": attempt["attempt_index"],
            })
            derived_state["steps"][step_index] = derived_step
            derived_attempt = copy.deepcopy(attempt)
            derived_attempt["source_image_path"] = attempt["image_path"]
            derived_attempt["image_path"] = relative_image.as_posix()
            derived_state["step_attempts"][step_index] = [derived_attempt]
            generation = trajectory.get("step_generations", {}).get(
                f"step_{step_index}", {}
            )
            generation["image_path"] = relative_image.as_posix()
            generation["image_sha256"] = attempt["image_sha256"]
            generation["retry_round"] = attempt.get("retry_round")
            generation["selected_attempt"] = attempt["attempt_index"]
            units.append({
                "unit_index": int(step_index),
                "content": step.get("content", ""),
                "image_path": (
                    Path("images") / task_name / relative_image
                ).as_posix(),
                "image_sha256": attempt["image_sha256"],
                "attempt_index": attempt["attempt_index"],
            })
        trajectory["report_variant"] = variant
        trajectory["derived_from"] = task_source_identity
        trajectory["review_derivation"] = review_provenance
        variant_state_path = variant_dir / "images" / task_name / "state.json"
        variant_trajectory_path = variant_dir / "text" / f"{task_id}.json"
        atomic_write_json(variant_state_path, derived_state)
        atomic_write_json(variant_trajectory_path, trajectory)
        atomic_write_json(reports_dir / f"{task_id}.json", {
            "schema_version": "umm-uniweaver-report-variant-v1",
            "task_key": task_id,
            "task_prompt": state.get("task_prompt", ""),
            "status": "completed",
            "variant": variant,
            "derived_from": task_source_identity,
            "review_derivation": review_provenance,
            "units": units,
        })
        record = records_by_task.get(task_id)
        if record is not None:
            record["output_dir"] = str(variant_dir / "images" / task_name)
            record["trajectory_path"] = str(variant_trajectory_path)
            record["state_path"] = str(variant_state_path)

    variant_provenance = {
        **root_provenance,
        "derivation_policy_id": REVIEW_VARIANT_DERIVATION_EVENT["policy_id"],
        "report_variant": variant,
        "selection_policy": (
            "attempt_0" if variant == "no_review" else "final_selected_attempt"
        ),
        "evaluation_run_id": f"{run_dir.name}-{variant}",
        "derived_from": source_identity,
        "task_review_provenance": task_review_records,
        "review_application_summary": {
            "completed_task_count": len(task_review_records),
            "review_applied_task_count": sum(
                bool(item["review_applied"])
                for item in task_review_records.values()
            ),
            "source_review_executed_task_count": sum(
                bool(item["source_review_executed"])
                for item in task_review_records.values()
            ),
        },
    }
    atomic_write_json(variant_dir / "provenance.json", variant_provenance)

    variant_summary = {
        **root_summary,
        "run_id": variant_dir.name,
        "report_variant": variant,
        "derived_from": source_identity,
        "records": variant_records,
    }
    variant_summary.pop("evaluation_preparation", None)
    atomic_write_json(variant_dir / "summary.json", variant_summary)
    return variant_dir


def prepare_review_variants(run_dir: Path, benchmark_name: str) -> dict:
    """Build independently evaluable attempt-0 and reviewed report views."""
    run_dir = Path(run_dir).resolve()
    manifests = {}
    for variant in ("no_review", "reviewed"):
        variant_dir = _prepare_review_variant(run_dir, variant)
        manifest = prepare_requests(
            variant_dir, benchmark_name, protocol="uniweaver"
        )
        manifests[variant] = manifest
        summary_path = variant_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["evaluation_preparation"] = {
            "status": "prepared",
            "evaluation_version": manifest["evaluation_version"],
            "manifest_path": "evaluation/scoring/manifest.json",
            "judge_called": False,
            "request_counts": {
                metric: record["request_count"]
                for metric, record in manifest["outputs"].items()
            },
        }
        atomic_write_json(summary_path, summary)
    index = {
        "schema_version": "umm-uniweaver-review-variants-v1",
        "policy_id": REVIEW_VARIANT_DERIVATION_EVENT["policy_id"],
        "source_run": run_dir.name,
        "variants": {
            name: {
                "run_dir": name,
                "evaluation_manifest": f"{name}/evaluation/scoring/manifest.json",
                "evaluation_manifest_sha256": sha256_file(
                    run_dir / name / "evaluation" / "scoring" / "manifest.json"
                ),
            }
            for name in manifests
        },
    }
    atomic_write_json(run_dir / "review_variants.json", index)
    return index


def prepare_requests(
    run_dir: Path, benchmark_name: str, protocol: str | None = None,
) -> dict:
    """Prepare every UniWeaver-Bench request layer without calling a judge endpoint."""
    benchmark_spec = BENCHMARK_SPECS.get(benchmark_name)
    if benchmark_spec is None:
        raise ValueError(
            f"no UniWeaver-Bench specification for benchmark {benchmark_name!r}"
        )
    run_dir = Path(run_dir).resolve()
    benchmark_snapshot = run_dir / "benchmark_task_matrix.json"
    if not benchmark_snapshot.is_file():
        raise FileNotFoundError(
            "automatic UniWeaver-Bench preparation requires the exact run benchmark "
            f"snapshot: {benchmark_snapshot}"
        )
    benchmark_payload = json.loads(benchmark_snapshot.read_text(encoding="utf-8"))
    if benchmark_payload.get("benchmark_name") != benchmark_name:
        raise ValueError("run benchmark snapshot name does not match run provenance")
    tasks = load_matrix_tasks(benchmark_snapshot)
    expected_task_count = benchmark_spec["task_count"]
    if len(tasks) != expected_task_count:
        raise ValueError(
            f"{benchmark_name} snapshot must contain {expected_task_count} tasks"
        )
    provenance_path = run_dir / "provenance.json"
    provenance = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        recorded_hash = provenance.get("benchmark_file_sha256")
        snapshot_hash = sha256_file(benchmark_snapshot)
        if recorded_hash and recorded_hash != snapshot_hash:
            raise ValueError("run benchmark snapshot hash does not match provenance")
    protocol = protocol or provenance.get("protocol") or provenance.get("mode") or "uniweaver"
    if protocol not in {"uniweaver", "minimal"}:
        raise ValueError(f"unsupported evaluation preparation protocol: {protocol!r}")
    prepared_run_id = provenance.get("evaluation_run_id") or run_dir.name
    if not isinstance(prepared_run_id, str) or not prepared_run_id:
        raise ValueError("evaluation_run_id must be a non-empty string")
    report_artifacts = load_report_artifacts(
        PROJECT_ROOT,
        prepared_run_id,
        "bagel" if protocol == "uniweaver" else "sequential",
        report_root=(run_dir / "tasks") if protocol == "minimal" else None,
        run_dir=run_dir,
    )
    report_contracts = {
        task_id: report_contract(artifact)
        for task_id, artifact in report_artifacts.items()
    }
    # Gates that passed but with something worth flagging: a report below the
    # expected unit count, or a completion inferred from the last step because
    # the run never announced it. Neither changes a score.
    report_contract_warnings = {
        task_id: contract["warnings"]
        for task_id, contract in sorted(report_contracts.items())
        if contract.get("warnings")
    }
    # Diagnostics only: a unit that merely restates an input prompt is recorded
    # for a reader to judge. It never changes a gate, a count, or a score.
    report_unit_diagnostics = {
        task_id: {"input_echo_units": echoes}
        for task_id, artifact in sorted(report_artifacts.items())
        if (echoes := input_echo_units(artifact))
    }
    eligible_reports = {
        task_id: artifact
        for task_id, artifact in report_artifacts.items()
        if report_contracts[task_id]["status"] == "PASS"
    }
    expected_ids = set(tasks)

    output_dir = run_dir / "evaluation" / "scoring"
    request_dir = output_dir / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    for metric, layer in REPORT_REQUEST_LAYERS.items():
        path = request_dir / f"report_{metric.lower()}.requests.jsonl"
        requests = [
            _make_request_portable(request, run_dir)
            for request in build_report_requests(tasks, eligible_reports, layer)
        ]
        _write_jsonl(path, requests)
        outputs[metric] = {
            "layer": layer,
            "path": str(path.relative_to(run_dir)),
            "sha256": sha256_file(path),
            "request_count": len(requests),
        }

    summary_path = run_dir / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {}
    run_records = {
        record.get("task_id"): record
        for record in summary.get("records", [])
        if isinstance(record, dict) and isinstance(record.get("task_id"), str)
    }
    failure_tasks = {}
    for task_id in sorted(expected_ids):
        record = run_records.get(task_id, {})
        status = record.get("status")
        if status in {
            "planner_error", "image_error",  # UniWeaver stages
            "step_error", "preflight_error",  # minimal protocol has no Planner
            "timed_out",
        }:
            failure_tasks[task_id] = {
                "gate_id": REPORT_FAILURE_PROVENANCE_GATE["gate_id"],
                "kind": "run_failure",
                "resolved": True,
                "source_status": status,
                "error": record.get("error"),
                "runtime_failure_provenance": record.get("failure_provenance"),
            }
        elif task_id not in report_artifacts:
            failure_tasks[task_id] = {
                "gate_id": REPORT_FAILURE_PROVENANCE_GATE["gate_id"],
                "kind": (
                    "file_omission" if record else "artifact_collection"
                ),
                "resolved": False,
                "source_status": status,
            }
    failure_provenance = {
        "schema_version": "umm-report-failure-provenance-v1",
        "evaluation_version": EVALUATION_SPEC_ID,
        "tasks": failure_tasks,
    }
    failure_path = output_dir / "failure_provenance.json"
    atomic_write_json(failure_path, failure_provenance)
    manifest = {
        "schema_version": "umm-auto-evaluation-prepare-v1",
        "evaluation_version": EVALUATION_SPEC_ID,
        "created_at_unix": time.time(),
        "automatic_after_run_umm": True,
        "judge_called": False,
        "artifact_path_schema": "umm-run-relative-v1",
        "artifact_path_base": "run_dir",
        "run_id": prepared_run_id,
        "protocol": protocol,
        "benchmark_name": benchmark_name,
        "benchmark_snapshot_path": str(benchmark_snapshot.relative_to(run_dir)),
        "benchmark_snapshot_sha256": sha256_file(benchmark_snapshot),
        "expected_task_count": len(tasks),
        "report_artifact_count": len(report_artifacts),
        "report_contract_pass_count": len(eligible_reports),
        "missing_report_task_ids": sorted(expected_ids - set(report_artifacts)),
        "report_unit_diagnostics": report_unit_diagnostics,
        "report_contract_warnings": report_contract_warnings,
        # Tasks whose per-task file says "incomplete" but whose completion was
        # declared in prose, which the terminal gate accepts. Diagnostic only.
        "prose_completion_task_ids": prose_completion_tasks(report_artifacts),
        "report_contract_failures": {
            task_id: contract
            for task_id, contract in sorted(report_contracts.items())
            if contract["status"] != "PASS"
        },
        "failure_provenance_path": str(failure_path.relative_to(run_dir)),
        "failure_provenance_sha256": sha256_file(failure_path),
        "outputs": outputs,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest
