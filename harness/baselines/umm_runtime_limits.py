"""Bounded runtime projection and task deadlines for UMM tasks."""

from __future__ import annotations

import copy
import json
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass

from core.hashing import sha256_text
from core.policy_registry import (
    RUNTIME_CHECKLIST_PREFIX_POLICY,
    RUNTIME_CHECKLIST_STEP_REFERENCE_POLICY,
    RUNTIME_CONDITION_PREFIX_POLICY,
    RUNTIME_CONDITION_REFERENCE_POLICY,
    RUNTIME_GENERAL_CHECK_POLICY,
    RUNTIME_STEP_PREFIX_POLICY,
    RUNTIME_UNREFERENCED_CONDITION_POLICY,
    validate_operation_provenance,
)
from core.review import checks_for_step, split_checklist

DEFAULT_MAX_RUNTIME_CONDITIONS = RUNTIME_CONDITION_PREFIX_POLICY["default_limit"]
DEFAULT_MAX_RUNTIME_STEPS = RUNTIME_STEP_PREFIX_POLICY["default_limit"]
DEFAULT_MAX_CHECKS_PER_STEP = RUNTIME_CHECKLIST_PREFIX_POLICY["default_limit"]
DEFAULT_TASK_TIMEOUT_SECONDS = 21_600.0


@dataclass(frozen=True)
class RuntimeLimits:
    max_conditions: int = DEFAULT_MAX_RUNTIME_CONDITIONS
    max_steps: int = DEFAULT_MAX_RUNTIME_STEPS
    max_checks_per_step: int = DEFAULT_MAX_CHECKS_PER_STEP
    task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS

    def as_dict(self) -> dict:
        return asdict(self)


def runtime_limits_from_args(args) -> RuntimeLimits:
    defaults = RuntimeLimits()
    return RuntimeLimits(
        max_conditions=getattr(
            args, "max_runtime_conditions", defaults.max_conditions),
        max_steps=getattr(args, "max_runtime_steps", defaults.max_steps),
        max_checks_per_step=getattr(
            args, "max_checks_per_step", defaults.max_checks_per_step),
        task_timeout_seconds=getattr(
            args, "task_timeout_seconds", defaults.task_timeout_seconds),
    )


def runtime_limits_from_provenance(provenance: dict) -> RuntimeLimits:
    stored = provenance.get("runtime_control", {}).get("limits", {})
    defaults = RuntimeLimits().as_dict()
    return RuntimeLimits(**{
        key: stored.get(key, value) for key, value in defaults.items()
    })


def _json_sha256(value) -> str:
    canonical = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return sha256_text(canonical)


def project_uniweaver_runtime_budget(
    runtime_plan: dict,
    limits: RuntimeLimits,
) -> tuple[dict, dict]:
    """Project an oversized Plan onto bounded execution without altering source."""
    projected = copy.deepcopy(runtime_plan)
    operations = []

    def record(policy_id, operation, path, before, after, dropped):
        if before == after:
            return
        operations.append({
            "sequence": len(operations) + 1,
            "policy_id": policy_id,
            "operation": operation,
            "path": path,
            "input_sha256": _json_sha256(before),
            "output_sha256": _json_sha256(after),
            "input_count": len(before),
            "output_count": len(after),
            "dropped": copy.deepcopy(dropped),
        })

    condition_bank = projected.get("condition_bank")
    condition_items = (
        condition_bank.get("items", [])
        if isinstance(condition_bank, dict) else []
    )
    if isinstance(condition_items, list):
        retained_conditions = condition_items[:limits.max_conditions]
        record(
            RUNTIME_CONDITION_PREFIX_POLICY["policy_id"],
            RUNTIME_CONDITION_PREFIX_POLICY["operation"],
            "$.condition_bank.items",
            condition_items,
            retained_conditions,
            [
                {"position": index, "value": item, "reason": "condition_limit"}
                for index, item in enumerate(
                    condition_items[limits.max_conditions:],
                    limits.max_conditions,
                )
            ],
        )
        condition_bank["items"] = retained_conditions
    else:
        retained_conditions = []

    retained_condition_keys = {
        item.get("key")
        for item in retained_conditions
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }

    steps = projected.get("steps", [])
    if isinstance(steps, list):
        retained_steps = steps[:limits.max_steps]
        record(
            RUNTIME_STEP_PREFIX_POLICY["policy_id"],
            RUNTIME_STEP_PREFIX_POLICY["operation"],
            "$.steps",
            steps,
            retained_steps,
            [
                {"position": index, "value": item, "reason": "step_limit"}
                for index, item in enumerate(
                    steps[limits.max_steps:], limits.max_steps,
                )
            ],
        )
        projected["steps"] = retained_steps
    else:
        retained_steps = []

    retained_step_indexes = set()
    for position, step in enumerate(retained_steps):
        if not isinstance(step, dict):
            continue
        step_index = step.get("step_index")
        if isinstance(step_index, int) and not isinstance(step_index, bool):
            retained_step_indexes.add(step_index)
        refs = step.get("condition_refs", [])
        if not isinstance(refs, list):
            continue
        retained_refs = [ref for ref in refs if ref in retained_condition_keys]
        record(
            RUNTIME_CONDITION_REFERENCE_POLICY["policy_id"],
            RUNTIME_CONDITION_REFERENCE_POLICY["operation"],
            f"$.steps[{position}].condition_refs",
            refs,
            retained_refs,
            [
                {
                    "position": index,
                    "value": ref,
                    "reason": "condition_not_retained",
                }
                for index, ref in enumerate(refs)
                if ref not in retained_condition_keys
            ],
        )
        step["condition_refs"] = retained_refs

    referenced_condition_keys = {
        ref
        for step in retained_steps if isinstance(step, dict)
        for ref in step.get("condition_refs", [])
        if isinstance(step.get("condition_refs"), list)
    }
    used_conditions = [
        item for item in retained_conditions
        if isinstance(item, dict) and item.get("key") in referenced_condition_keys
    ]
    record(
        RUNTIME_UNREFERENCED_CONDITION_POLICY["policy_id"],
        RUNTIME_UNREFERENCED_CONDITION_POLICY["operation"],
        "$.condition_bank.items",
        retained_conditions,
        used_conditions,
        [
            {"position": index, "value": item, "reason": "not_referenced_by_retained_steps"}
            for index, item in enumerate(retained_conditions)
            if item not in used_conditions
        ],
    )
    if isinstance(condition_bank, dict):
        condition_bank["items"] = used_conditions

    checklist = projected.get("reviewer_checklist", [])
    if isinstance(checklist, list):
        filtered = []
        removed_targets = []
        for position, item in enumerate(checklist):
            match = (
                re.match(r"^step_(\d+):", item)
                if isinstance(item, str) else None
            )
            if match and int(match.group(1)) not in retained_step_indexes:
                removed_targets.append({
                    "position": position,
                    "value": item,
                    "reason": "step_not_retained",
                })
            else:
                filtered.append(item)
        record(
            RUNTIME_CHECKLIST_STEP_REFERENCE_POLICY["policy_id"],
            RUNTIME_CHECKLIST_STEP_REFERENCE_POLICY["operation"],
            "$.reviewer_checklist",
            checklist,
            filtered,
            removed_targets,
        )

        retained_checklist = []
        counts = {}
        overflow = []
        for position, item in enumerate(filtered):
            match = (
                re.match(r"^step_(\d+):", item)
                if isinstance(item, str) else None
            )
            if not match:
                retained_checklist.append(item)
                continue
            step_index = int(match.group(1))
            count = counts.get(step_index, 0)
            counts[step_index] = count + 1
            if count < limits.max_checks_per_step:
                retained_checklist.append(item)
            else:
                overflow.append({
                    "position": position,
                    "value": item,
                    "reason": "per_step_check_limit",
                    "step_index": step_index,
                })
        record(
            RUNTIME_CHECKLIST_PREFIX_POLICY["policy_id"],
            RUNTIME_CHECKLIST_PREFIX_POLICY["operation"],
            "$.reviewer_checklist",
            filtered,
            retained_checklist,
            overflow,
        )
        projected["reviewer_checklist"] = retained_checklist
        _per_step, general = split_checklist(retained_checklist)
        if general:
            applications = {}
            dropped_applications = []
            for step_index in sorted(retained_step_indexes):
                effective, dropped = checks_for_step(
                    retained_checklist,
                    step_index,
                    limits.max_checks_per_step,
                )
                applications[str(step_index)] = effective
                dropped_applications.extend({
                    "step_index": step_index,
                    "value": item,
                    "reason": "per_step_check_limit",
                } for item in dropped)
            operations.append({
                "sequence": len(operations) + 1,
                "policy_id": RUNTIME_GENERAL_CHECK_POLICY["policy_id"],
                "operation": RUNTIME_GENERAL_CHECK_POLICY["operation"],
                "path": "$.reviewer_checklist",
                "input_sha256": _json_sha256(retained_checklist),
                "output_sha256": _json_sha256(applications),
                "input_count": len(retained_checklist),
                "output_count": sum(len(items) for items in applications.values()),
                "applications": applications,
                "dropped": dropped_applications,
            })

    provenance = {
        "policy_version": "umm_runtime_projection_v1",
        "input_sha256": _json_sha256(runtime_plan),
        "output_sha256": _json_sha256(projected),
        "changed": bool(operations),
        "operations": operations,
    }
    validate_operation_provenance(operations)
    return projected, provenance


def uniweaver_runtime_usage(
    runtime_plan: dict,
    *,
    review_retry: bool,
    max_checks_per_step: int = DEFAULT_MAX_CHECKS_PER_STEP,
) -> dict:
    conditions = runtime_plan.get("condition_bank", {}).get("items", [])
    steps = runtime_plan.get("steps", [])
    checklist = runtime_plan.get("reviewer_checklist", [])
    per_step, general = split_checklist(checklist)
    step_indexes = [
        step.get("step_index") for step in steps
        if isinstance(step, dict) and isinstance(step.get("step_index"), int)
    ]
    check_counts = {
        str(index): len(checks_for_step(checklist, index, max_checks_per_step)[0])
        for index in step_indexes
    }
    check_chars = {
        str(index): sum(
            len(item)
            for item in checks_for_step(
                checklist, index, max_checks_per_step,
            )[0]
        )
        for index in step_indexes
    }
    reviewed_steps = sum(
        bool(per_step.get(step.get("step_index")) or general) for step in steps
    )
    maximum_internal_model_operations = (
        4 + len(conditions) + len(steps)
        + (7 * reviewed_steps if review_retry else 0)
    )
    return {
        "conditions": len(conditions),
        "steps": len(steps),
        "check_counts_by_step": check_counts,
        "check_chars_by_step": check_chars,
        "maximum_internal_model_operations": maximum_internal_model_operations,
        "reviewed_steps": reviewed_steps,
    }


def minimal_runtime_usage(max_steps: int) -> dict:
    return {
        "steps": max_steps,
        "maximum_internal_model_operations": 2 * max_steps,
    }


def backend_task_scope(backend, timeout_seconds: float):
    scope = getattr(backend, "task_scope", None)
    return scope(timeout_seconds) if scope is not None else nullcontext()
