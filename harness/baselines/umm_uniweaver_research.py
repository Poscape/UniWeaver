"""Production UniWeaver planner, renderer, reviewer, and persistence runner."""

from __future__ import annotations

import json
import sys
import time
import uuid
from functools import partial
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.umm_benchmark_runner import (
    UNIWEAVER_MAX_WORKERS,
    backend_execution_workers,
    build_benchmark_provenance,
)
from baselines.umm_cli import (
    require_backend_repo,
    require_positive_image_steps,
    require_positive_runtime_timeouts,
)
from baselines.umm_framework import (
    capabilities_for,
    classify_task_error,
    send_planner_text,
)
from baselines.umm_runtime_backend import (
    is_timeout_error,
    load_supervised_backend,
)
from baselines.umm_runtime_limits import (
    backend_task_scope,
    project_uniweaver_runtime_budget,
    runtime_limits_from_provenance,
    uniweaver_runtime_usage,
)
from baselines.umm_uniweaver_persistence import UniWeaverPersistenceTransition
from core.hashing import sha256_file
from core.policy_registry import ORCHESTRATION_FAILURE_EVENT
from core.production import (
    REVIEW_MAX_RETRIES,
    image_seed_for,
    require_trajectory_identity,
    trajectory_contract_view,
)
from core.render_inputs import (
    compile_condition_execution_input,
    compile_condition_reference_instruction,
    compile_condition_render_input,
    compile_step_execution_input,
    compile_step_render_input,
)
from core.review import (
    checks_for_step,
    retry_feedback,
    review_prompt,
    review_step_image,
)
from core.runner import TrajectoryError, run_trajectory_with_sender
from core.serialization import atomic_write_json, project_runtime_plan
from core.tasks import find_benchmark_task
from core.uniweaver_execution import run_uniweaver_stages
from core.validation import check_text_validation, task_text


def _model_operation(persistence, backend, operation, detail):
    if persistence is None:
        from contextlib import nullcontext
        return nullcontext()
    return persistence.manager.operation(
        operation,
        detail=detail,
        deadline_seconds=getattr(backend, "call_timeout_seconds", None),
    )


def _run_planner(
    backend,
    task_key,
    task_prompt,
    trajectory_path,
):
    model_identity = getattr(backend, "model_identity", {})
    model_artifact_sha256 = (
        model_identity.get("sha256")
        if isinstance(model_identity, dict)
        else None
    )

    def send_backend(
        *, user_text, label, max_tokens, system_text, do_sample, temperature
    ):
        del label
        started = time.time()
        try:
            raw_output = send_planner_text(
                backend,
                user_text=user_text,
                system_text=system_text,
                max_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )
        except BaseException as error:
            send_backend.last_metadata = {
                "provider": backend.name,
                "model_name": backend.name,
                "model_artifact_sha256": model_artifact_sha256,
                "duration_seconds": time.time() - started,
                "status": "failed",
                "exception_type": type(error).__name__,
            }
            raise
        metadata = {
            "provider": backend.name,
            "model_name": backend.name,
            "model_artifact_sha256": model_artifact_sha256,
            "duration_seconds": time.time() - started,
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": 0,
            "token_count_scope": "adapter_not_reported",
            # Explicitly recorded as unknown rather than omitted: no adapter
            # surfaces the sampler stop reason yet, and a missing key would hide
            # that truncation was never verified.
            "finish_reason": None,
            "finish_reason_scope": "adapter_not_reported",
        }
        token_counter = getattr(backend, "planner_token_counts", None)
        if token_counter is not None:
            input_tokens, output_tokens = token_counter(
                system_text, user_text, raw_output)
            metadata.update({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "token_count_scope": (
                    "adapter_reported: backend-rendered prompt (including the "
                    "backend's own chat template where applicable) + returned "
                    "content"
                ),
            })
            # No adapter exposes the sampler stop reason, so a generation that
            # used its whole allowance is recorded as an inference.
            if (isinstance(output_tokens, int)
                    and isinstance(max_tokens, int) and max_tokens > 0):
                metadata["finish_reason"] = (
                    "length" if output_tokens >= max_tokens else "stop")
                metadata["finish_reason_scope"] = "inferred_from_token_count"
        send_backend.last_metadata = metadata
        return raw_output

    extra_fields = {
        "model_provider": backend.name,
        "model_name": backend.name,
    }
    if model_artifact_sha256 is not None:
        extra_fields["model_artifact_sha256"] = model_artifact_sha256

    from core.execution_recovery import repair_planner_stage

    return run_trajectory_with_sender(
        task_key,
        task_prompt,
        trajectory_path,
        send_backend,
        extra_fields=extra_fields,
        include_step_generation=False,
        mechanical_repairer=repair_planner_stage,
    )


def _require_execution_eligible(trajectory):
    """Allow recovered plans to execute while retaining their provenance."""
    text_check = check_text_validation(
        trajectory["_text_validation"], criterion="execution")
    if not text_check["passed"]:
        raise TrajectoryError(
            "text trajectory is not executable/renderable: "
            + "; ".join(text_check["reasons"])
        )


def _condition_images(
    backend,
    runtime_plan,
    output,
    seed,
    image_shape,
    image_steps,
    persistence=None,
    resume=False,
    max_workers=UNIWEAVER_MAX_WORKERS,
):
    declared_items = runtime_plan.get("condition_bank", {}).get("items", [])
    images = {}
    records = {}
    if persistence is not None and resume:
        completed_conditions, _ = persistence.manager.completed_items()
        for item in declared_items:
            key = item["key"]
            if key not in completed_conditions:
                continue
            path = output / "conditions" / f"{Path(key).name}.png"
            with Image.open(path) as persisted_image:
                images[key] = persisted_image.copy()
            records[key] = dict(
                persistence.manager.state["conditions"].get(key, {}))
    items = [
        item for item in declared_items if item["key"] not in images
    ]
    conditions_dir = output / "conditions"
    if items and persistence is None:
        conditions_dir.mkdir()

    def generate_one(item):
        render_input = compile_condition_render_input(runtime_plan, item)
        execution_input = compile_condition_execution_input(render_input)
        condition_seed = image_seed_for(seed, "condition", item["key"])
        started = time.time()
        with _model_operation(
            persistence, backend, "condition_generation", {
                "condition_key": item["key"],
                "raw_input": execution_input.prompt,
                "seed": condition_seed,
            }
        ):
            image = backend.generate_image(
                execution_input.prompt,
                None,
                condition_seed,
                image_shape,
                image_steps,
            )
        if not isinstance(image, Image.Image):
            raise TypeError(f"condition {item['key']} returned no PIL image")
        return item, render_input, execution_input, condition_seed, image, time.time() - started

    def persist_generated(generated_item):
        (
            item,
            render_input,
            execution_input,
            condition_seed,
            image,
            duration,
        ) = generated_item
        key = item["key"]
        record = {
            "type": item.get("type", "reference"),
            "image_seed": condition_seed,
            "image_size": list(image.size),
            "duration": duration,
            "semantic_image_prompt": render_input.semantic_image_prompt,
            "model_image_prompt": render_input.model_image_prompt,
            "raw_input": execution_input.prompt,
            "raw_output": "",
        }
        path = conditions_dir / f"{Path(key).name}.png"
        if persistence is None:
            image.save(path)
            image_sha256 = sha256_file(path)
        else:
            image_sha256 = persistence.persist_condition(key, record, image)
        record["image_path"] = str(path.relative_to(output))
        record["image_sha256"] = image_sha256
        images[key] = image
        records[key] = record

    def on_condition(generated_item):
        persist_generated(generated_item)
        torch.cuda.empty_cache()

    run_uniweaver_stages(
        condition_items=items,
        condition_worker=generate_one,
        max_workers=max_workers,
        on_condition=on_condition,
    )

    if items:
        torch.cuda.empty_cache()
    missing = sorted(
        item["key"] for item in items if item["key"] not in images)
    if missing:
        raise RuntimeError(
            "condition generation remained incomplete: "
            + ", ".join(missing))
    return images, records


def _run_images(
    backend,
    trajectory,
    runtime_plan,
    output,
    condition_shape,
    step_shape,
    image_steps,
    seed,
    max_workers=None,
    persistence=None,
    resume=False,
    review_retry=False,
    max_checks_per_step=3,
):
    if max_workers is None:
        max_workers = backend_execution_workers(backend)
    condition_images, condition_records = _condition_images(
        backend,
        runtime_plan,
        output,
        seed,
        condition_shape,
        image_steps,
        persistence=persistence,
        resume=resume,
        max_workers=max_workers,
    )
    steps_dir = output / "steps"
    if persistence is None:
        steps_dir.mkdir()
    step_items = list(runtime_plan["steps"])
    runtime_steps = {
        step["step_index"]: step for step in step_items
    }
    generated = {}
    trajectory["step_generations"] = {}
    if persistence is not None and resume:
        _, completed_steps = persistence.manager.completed_items()
        for step in step_items:
            step_index = step["step_index"]
            if str(step_index) not in completed_steps:
                continue
            image_path = output / "steps" / f"step_{step_index}.png"
            with Image.open(image_path) as persisted_image:
                image = persisted_image.copy()
            state_record = persistence.manager.state["steps"][str(step_index)]
            render_input = compile_step_render_input(runtime_plan, step)
            image_trace = state_record.get("img_trace", {})
            stored_refs = state_record.get("condition_refs", [])
            result = {
                "step_index": step_index,
                "render_input": render_input,
                "image": image,
                "image_seed": state_record.get("image_seed"),
                "condition_refs": stored_refs,
                # Older state files only stored the declared refs; fall back to
                # that assumption instead of inventing a consumption split.
                "condition_refs_passed": state_record.get(
                    "condition_refs_passed", stored_refs
                ),
                "condition_refs_used": state_record.get(
                    "condition_refs_used", stored_refs
                ),
                "condition_refs_ignored": state_record.get(
                    "condition_refs_ignored", []
                ),
                "raw_input": image_trace.get("raw_input", ""),
                "duration": image_trace.get("duration", 0),
                "input_structured": image_trace.get("input_structured", {}),
            }
            generated[step_index] = result
            trajectory["step_generations"][f"step_{step_index}"] = {
                "parsed": render_input.as_dict(),
                "image_path": str(image_path.relative_to(output)),
                "image_seed": result["image_seed"],
                "image_sha256": state_record["image_sha256"],
                "image_size": list(image.size),
                "duration": result["duration"],
                "condition_refs": result["condition_refs"],
                "condition_refs_passed": result["condition_refs_passed"],
                "condition_refs_used": result["condition_refs_used"],
                "condition_refs_ignored": result["condition_refs_ignored"],
            }
        step_items = [
            step for step in step_items
            if str(step["step_index"]) not in completed_steps
        ]

    def generate_one(step, feedback="", *, retry_round=None):
        step_index = step["step_index"]
        render_input = compile_step_render_input(runtime_plan, step)
        refs = list(dict.fromkeys(step.get("condition_refs", [])))
        usable_refs = [key for key in refs if key in condition_images]
        capabilities = capabilities_for(backend)
        if not capabilities.multiple_condition_images_consumed:
            # Single-source recipes (ILLUME+ and SEED-X editing, TokenFlow and
            # Janus-Pro which consume nothing) take at most one reference; the
            # remaining ones stay recorded as ignored rather than claimed.
            usable_refs = usable_refs[:1]
        passed_images = [condition_images[key] for key in usable_refs]
        execution_input = compile_step_execution_input(
            render_input,
            condition_images,
            usable_refs,
            feedback=feedback,
        )
        step_seed = image_seed_for(seed, "step", step_index)
        native_generator = getattr(backend, "generate_uniweaver_image", None)
        reference_instructions = [
            compile_condition_reference_instruction(key)
            for key in usable_refs
        ]
        started = time.time()
        with _model_operation(
            persistence, backend, "step_generation", {
                "step_index": step_index,
                "raw_input": execution_input.prompt,
                "seed": step_seed,
                "retry_round": retry_round,
            }
        ):
            if native_generator is not None:
                image = native_generator(
                    input_lists=execution_input.model_inputs,
                    seed=step_seed,
                    image_shape=step_shape,
                    image_steps=image_steps,
                )
            else:
                reference_text = "\n".join(reference_instructions)
                prompt = (
                    f"{reference_text}\n\n{execution_input.prompt}"
                    if reference_text
                    else execution_input.prompt
                )
                image = backend.generate_image(
                    prompt,
                    passed_images[0] if passed_images else None,
                    step_seed,
                    step_shape,
                    image_steps,
                    previous_images=passed_images,
                )
        if not isinstance(image, Image.Image):
            raise TypeError(f"step {step_index} returned no PIL image")
        consumed_refs = (
            usable_refs if capabilities.condition_images_consumed else []
        )
        return {
            "step_index": step_index,
            "render_input": render_input,
            "image": image,
            "image_seed": step_seed,
            "condition_refs": refs,
            "condition_refs_passed": usable_refs,
            "condition_refs_used": consumed_refs,
            "condition_refs_ignored": [
                key for key in refs if key not in consumed_refs
            ],
            "raw_input": execution_input.prompt,
            "feedback": feedback,
            "duration": time.time() - started,
            "retry_round": retry_round,
            "input_structured": {
                "condition_refs": usable_refs,
                "reference_instructions": reference_instructions,
                "condition_image_sizes": [
                    list(image.size) for image in passed_images
                ],
                "image_seed": step_seed,
            },
        }

    def persist_step_result(result):
        step_index = result["step_index"]
        image_path = steps_dir / f"step_{step_index}.png"
        if persistence is None:
            result["image"].save(image_path)
            image_sha256 = sha256_file(image_path)
        else:
            image_sha256 = persistence.persist_step(
                step_index, result, runtime_steps[step_index])
        trajectory["step_generations"][f"step_{step_index}"] = {
            "parsed": result["render_input"].as_dict(),
            "image_path": str(image_path.relative_to(output)),
            "image_seed": result["image_seed"],
            "image_sha256": image_sha256,
            "image_size": list(result["image"].size),
            "duration": result["duration"],
            "condition_refs": result["condition_refs"],
            "condition_refs_passed": result["condition_refs_passed"],
            "condition_refs_used": result["condition_refs_used"],
            "condition_refs_ignored": result["condition_refs_ignored"],
        }

    def on_step(result):
        generated[result["step_index"]] = result
        persist_step_result(result)

    def regenerate_one(item, *, retry_round):
        step, feedback = item
        return generate_one(step, feedback, retry_round=retry_round)

    run_uniweaver_stages(
        step_items=step_items,
        step_worker=generate_one,
        max_workers=max_workers,
        on_step=on_step,
    )
    review_history = (
        list(persistence.manager.state.get("reviews", []))
        if persistence is not None and resume else []
    )
    if review_retry:
        checklist = runtime_plan.get("reviewer_checklist", [])
        capabilities = capabilities_for(backend)
        unavailable_reason = None
        if not checklist:
            unavailable_reason = "NO_SAFE_PLAN4_CHECKLIST"
        elif not capabilities.reviewer_supported:
            unavailable_reason = "BACKEND_REVIEW_UNSUPPORTED"
        terminal_review = bool(review_history and (
            review_history[-1].get("status") == "review_unavailable"
            or not review_history[-1].get("retry_step_indexes")
            or review_history[-1].get("round", 0) >= REVIEW_MAX_RETRIES
        ))
        if unavailable_reason is not None and not review_history:
            round_record = {
                "round": 0,
                "status": "review_unavailable",
                "reason": unavailable_reason,
                "approved": None,
                "steps": {},
                "unreviewed_step_indexes": sorted(generated),
                "retry_step_indexes": [],
            }
            review_history.append(round_record)
            if persistence is not None:
                persistence.persist_review(round_record)
        start_round = 0
        if review_history:
            start_round = int(review_history[-1].get("round", 0)) + 1
        if review_history and not terminal_review:
            previous_round = review_history[-1]
            retry_items = []
            for step_index in previous_round.get("retry_step_indexes", []):
                state_step = persistence.manager.state.get(
                    "steps", {}).get(str(step_index), {})
                if state_step.get("retry_round") == start_round:
                    continue
                review = previous_round.get("steps", {}).get(str(step_index))
                if review is None:
                    raise RuntimeError(
                        f"review round {start_round - 1} lacks Step "
                        f"{step_index} evidence needed for resume"
                    )
                retry_items.append((
                    runtime_steps[step_index], retry_feedback(review)
                ))
            if retry_items:
                run_uniweaver_stages(
                    step_items=retry_items,
                    step_worker=partial(
                        regenerate_one, retry_round=start_round),
                    max_workers=max_workers,
                    on_step=on_step,
                )
        for review_round in (
                range(start_round, REVIEW_MAX_RETRIES + 1)
                if unavailable_reason is None and not terminal_review else ()):
            step_reviews = {}
            failed = {}
            unreviewed = []
            for step_index in sorted(generated):
                checks, _dropped_checks = checks_for_step(
                    checklist, step_index, max_checks_per_step,
                )
                if not checks:
                    unreviewed.append(step_index)
                    continue
                result = generated[step_index]
                with _model_operation(
                    persistence, backend, "step_review", {
                        "step_index": step_index,
                        "raw_input": review_prompt(checks),
                        "condition_images_excluded": True,
                    }
                ):
                    review = review_step_image(
                        backend,
                        result["image"],
                        checks,
                    )
                review["condition_images_excluded"] = True
                review["review_mode"] = "plan4_checklist"
                step_reviews[str(step_index)] = review
                if not review["approved"]:
                    failed[step_index] = retry_feedback(review)

            round_record = {
                "round": review_round,
                "status": "reviewed" if not unreviewed else "partial_review",
                "approved": not failed and not unreviewed,
                "steps": step_reviews,
                "unreviewed_step_indexes": unreviewed,
                "retry_step_indexes": sorted(failed),
            }
            review_history.append(round_record)
            if persistence is not None:
                persistence.persist_review(round_record)
            if not failed or review_round == REVIEW_MAX_RETRIES:
                break

            retry_items = [
                (runtime_steps[step_index], feedback)
                for step_index, feedback in failed.items()
            ]

            run_uniweaver_stages(
                step_items=retry_items,
                step_worker=partial(
                    regenerate_one, retry_round=review_round + 1),
                max_workers=max_workers,
                on_step=on_step,
            )

    trajectory["step_generations"] = dict(sorted(
        trajectory["step_generations"].items(),
        key=lambda item: int(item[0].split("_", 1)[1]),
    ))
    return {
        "condition_images": condition_images,
        "condition_records": condition_records,
        "steps": generated,
        "review": {
            "enabled": review_retry,
            "max_retries": REVIEW_MAX_RETRIES if review_retry else 0,
            "approved": review_history[-1]["approved"] if review_history else None,
            "status": review_history[-1].get("status") if review_history else "disabled",
            "rounds": review_history,
        },
    }


def _execute_uniweaver_images(
    backend,
    trajectory,
    runtime_plan,
    output,
    provenance,
    condition_shape,
    step_shape,
    image_steps,
    trajectory_path=None,
    resume=False,
    review_retry=False,
    task_started_at=None,
):
    limits = runtime_limits_from_provenance(provenance)
    runtime_plan, runtime_projection = project_uniweaver_runtime_budget(
        runtime_plan, limits,
    )
    transition = UniWeaverPersistenceTransition.start(
        output,
        trajectory["task_key"],
        task_text(trajectory),
        provenance,
        runtime_plan=runtime_plan,
        trajectory=trajectory,
        resume=resume,
        started_at=task_started_at,
    )
    try:
        transition.prepare(trajectory, runtime_plan)
        runtime_usage = uniweaver_runtime_usage(
            runtime_plan,
            review_retry=review_retry,
            max_checks_per_step=limits.max_checks_per_step,
        )
        transition.manager.update_state(
            runtime_projection=runtime_projection,
            runtime_budget={
                "status": (
                    "projected"
                    if runtime_projection["changed"] else "within_limits"
                ),
                "usage": runtime_usage,
                "limits": limits.as_dict(),
            },
        )
        execution = _run_images(
            backend,
            trajectory,
            runtime_plan,
            output,
            condition_shape,
            step_shape,
            image_steps,
            provenance["base_seed"],
            persistence=transition,
            resume=resume,
            review_retry=review_retry,
            max_checks_per_step=limits.max_checks_per_step,
        )
        trajectory["status"] = "done"
        if trajectory_path is not None:
            atomic_write_json(trajectory_path, trajectory)
        transition.finish(trajectory, runtime_plan, execution)
    except TrajectoryError as error:
        transition.fail(error, "planner_error")
        raise
    except Exception as error:
        failure_kind = (
            "timed_out" if is_timeout_error(error) else "image_error"
        )
        transition.fail(error, failure_kind)
        trajectory["status"] = "error"
        trajectory["failure_kind"] = failure_kind
        if trajectory_path is not None:
            atomic_write_json(trajectory_path, trajectory)
        raise
    return trajectory, runtime_plan, execution


def _run_uniweaver_task(
    backend,
    task_key,
    task_prompt,
    output,
    provenance,
    condition_shape=(512, 512),
    step_shape=(1024, 1024),
    image_steps=50,
    trajectory_path=None,
    resume=False,
    review_retry=False,
):
    implicit_trace_path = trajectory_path is None
    # Planning runs before any OutputManager exists. Record the task's real
    # start here so a planning failure still reports the time it consumed.
    task_started_at = time.time()
    planner_trace_path = Path(trajectory_path) if trajectory_path is not None else (
        Path(output).parent
        / f".{Path(output).name}.planner-{uuid.uuid4().hex}.json"
    )
    trajectory = None
    try:
        trajectory = _run_planner(
            backend,
            task_key,
            task_prompt,
            planner_trace_path,
        )
        _require_execution_eligible(trajectory)
        if trajectory_path is not None:
            atomic_write_json(trajectory_path, trajectory)
        runtime_plan = project_runtime_plan(
            trajectory["plan"],
            task_prompt,
            trajectory_design=trajectory["trajectory_design"],
            contract_version=trajectory["contract_version"],
        )
        if implicit_trace_path:
            planner_trace_path.unlink(missing_ok=True)
    except Exception as error:
        transition = UniWeaverPersistenceTransition.start(
            output, task_key, task_prompt, provenance,
            started_at=task_started_at)
        if trajectory is None and planner_trace_path.is_file():
            with planner_trace_path.open(encoding="utf-8") as handle:
                trajectory = json.load(handle)
        if trajectory is not None:
            transition.persist_planner_failure(trajectory)
            if implicit_trace_path:
                planner_trace_path.unlink(missing_ok=True)
        failure_kind = (
            "timed_out" if is_timeout_error(error) else "planner_error"
        )
        transition.fail(error, failure_kind)
        if trajectory is not None and not implicit_trace_path:
            trajectory["status"] = "error"
            trajectory["failure_kind"] = failure_kind
            atomic_write_json(planner_trace_path, trajectory)
        raise

    return _execute_uniweaver_images(
        backend,
        trajectory,
        runtime_plan,
        output,
        provenance,
        condition_shape,
        step_shape,
        image_steps,
        trajectory_path=trajectory_path,
        resume=resume,
        review_retry=review_retry,
        task_started_at=task_started_at,
    )


def run_uniweaver_task(
    backend,
    task_key,
    task_prompt,
    output,
    provenance,
    condition_shape=(512, 512),
    step_shape=(1024, 1024),
    image_steps=50,
    trajectory_path=None,
    resume=False,
    review_retry=False,
):
    limits = runtime_limits_from_provenance(provenance)
    with backend_task_scope(backend, limits.task_timeout_seconds):
        return _run_uniweaver_task(
            backend,
            task_key,
            task_prompt,
            output,
            provenance,
            condition_shape=condition_shape,
            step_shape=step_shape,
            image_steps=image_steps,
            trajectory_path=trajectory_path,
            resume=resume,
            review_retry=review_retry,
        )


def _run_uniweaver_images(
    backend,
    trajectory,
    output,
    provenance,
    condition_shape=(512, 512),
    step_shape=(1024, 1024),
    image_steps=50,
    resume=False,
    review_retry=False,
):
    # Accept a trajectory that was persisted by this pipeline (it carries view
    # metadata such as ``status``) as well as a freshly constructed one.
    trajectory_design, contract_version = require_trajectory_identity(
        trajectory_contract_view(trajectory)
    )
    _require_execution_eligible(trajectory)
    runtime_plan = project_runtime_plan(
        trajectory["plan"],
        task_text(trajectory),
        trajectory_design=trajectory_design,
        contract_version=contract_version,
    )
    return _execute_uniweaver_images(
        backend,
        trajectory,
        runtime_plan,
        output,
        provenance,
        condition_shape,
        step_shape,
        image_steps,
        resume=resume,
        review_retry=review_retry,
    )


def run_uniweaver_images(
    backend,
    trajectory,
    output,
    provenance,
    condition_shape=(512, 512),
    step_shape=(1024, 1024),
    image_steps=50,
    resume=False,
    review_retry=False,
):
    limits = runtime_limits_from_provenance(provenance)
    with backend_task_scope(backend, limits.task_timeout_seconds):
        return _run_uniweaver_images(
            backend,
            trajectory,
            output,
            provenance,
            condition_shape=condition_shape,
            step_shape=step_shape,
            image_steps=image_steps,
            resume=resume,
            review_retry=review_retry,
        )


def run(args) -> int:
    require_positive_image_steps(args)
    require_positive_runtime_timeouts(args)
    if args.image_height < 1 or args.image_width < 1:
        raise SystemExit("image settings must be positive")
    task = find_benchmark_task(args.task_key, args.benchmark_file)
    if task is None:
        raise SystemExit(f"benchmark task not found: {args.task_key}")
    output = Path(args.output_dir)
    if args.resume:
        if not output.is_dir():
            raise SystemExit(f"resume output directory not found: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    try:
        require_backend_repo(args)
        backend = load_supervised_backend(args)
        provenance = build_benchmark_provenance(
            backend,
            args,
            schema_version=None,
            mode="uniweaver",
            seed_policy="image_seed_for(base_seed, kind, identity); kinds are condition and step",
            extra={
                "note": "common UMM UniWeaver protocol with backend-specific capability provenance",
                "repo_source_identity": getattr(args, "repo_source_identity", None),
                "condition_policy": "all declared usable Condition images are passed in declared order; backend consumption is recorded",
            },
        )
    except BaseException as error:
        atomic_write_json(output / "terminal_error.json", {
            "policy_id": ORCHESTRATION_FAILURE_EVENT["policy_id"],
            "status": "error",
            "stage": "preflight",
            "exception_type": type(error).__name__,
            "message": str(error),
            "classification": "orchestration_failure",
            "resolved": False,
        })
        raise
    try:
        run_uniweaver_task(
            backend,
            args.task_key,
            task["prompt"],
            output,
            provenance,
            condition_shape=(512, 512),
            step_shape=(args.image_height, args.image_width),
            image_steps=args.image_steps,
            resume=args.resume,
            review_retry=args.review_retry,
        )
        print({"status": "completed", "backend": backend.name, "task_key": args.task_key})
        return 0
    except TrajectoryError as error:
        print({"status": "planner_error", "backend": backend.name, "error": str(error)})
        return 1
    except Exception as error:  # noqa: BLE001 - preserve adapter failure provenance
        print({
            "status": classify_task_error(error),
            "backend": backend.name,
            "error": str(error),
        })
        return 1
