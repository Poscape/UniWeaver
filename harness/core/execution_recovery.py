"""Closed registry and provenance helpers for Planner execution recovery.

Recovery never edits ``raw_attempts`` and never upgrades strict schema or
training acceptance. It may only produce a separately audited render input.
"""

from __future__ import annotations

import copy
import hashlib
import json

from .contract import find_planner_placeholders
from .policy_registry import (
    PLAN1_STEP_ALIAS_POLICY,
    PLAN2_STEP_ALIAS_POLICY,
    PLAN2_STEP_INDEX_REALIGNMENT_POLICY,
    PLAN2_VISUAL_STYLE_DEFAULT_POLICY,
    PLAN3_FRAGMENT_MERGE_POLICY,
    PLAN4_CHECKLIST_WRAPPER_POLICY,
    RECOVERY_POLICIES,
    recovery_policy_for_method,
    validate_operation_provenance,
)

EXECUTION_RECOVERY_CATALOG = tuple(
    policy for policy in RECOVERY_POLICIES
    if set(policy.get("stages", ())) & {"plan1", "plan2", "plan3", "plan4"}
)
PARSER_RECOVERY_METHODS = frozenset(
    method
    for policy in EXECUTION_RECOVERY_CATALOG
    if policy["category"] == "parser_recovery"
    for method in policy["methods"]
)


def _json_sha256(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _wrapper_normalizations(raw_output):
    text = str(raw_output or "").strip()
    actions = []
    think_end = text.rfind("</think>")
    if think_end >= 0:
        actions.append("select_after_last_think_close")
        text = text[think_end + len("</think>"):].strip()
    if text.startswith("###Assistant:"):
        actions.append("strip_assistant_prefix")
        text = text[len("###Assistant:"):].lstrip()
    elif text.startswith("###Assistant"):
        actions.append("strip_assistant_prefix")
        text = text[len("###Assistant"):].lstrip(":").lstrip()
    if text.lstrip("\ufeff\u200b\u200c\u200d\u200e\u200f\ufffe> ").strip() != text:
        actions.append("strip_transport_marker_prefix")
    return actions


def parser_recovery_provenance(stage: str, result: dict):
    method = result.get("parse_method") if isinstance(result, dict) else None
    decoded = result.get("decoded") if isinstance(result, dict) else None
    if method not in PARSER_RECOVERY_METHODS and isinstance(decoded, dict):
        method = decoded.get("parse_method")
    decoded_operations = (decoded.get('recovery_operations', [])
                          if isinstance(decoded, dict) else [])
    if method not in PARSER_RECOVERY_METHODS and not decoded_operations:
        return None
    parsed = result.get("parsed")
    if not isinstance(parsed, dict) and isinstance(decoded, dict):
        parsed = decoded.get("candidate")
    policy = recovery_policy_for_method(method)
    operations = [item for item in decoded_operations if isinstance(item, dict)]
    if not isinstance(parsed, (dict, list)) and not operations:
        return None
    if not operations and policy is not None:
        operations = [{
            'sequence': 1,
            'policy_id': policy['policy_id'], 'operation': method,
            'input_sha256': _text_sha256(result.get('raw_output')),
            'output_sha256': _json_sha256(parsed),
        }]
    for sequence, operation in enumerate(operations, 1):
        operation['sequence'] = sequence
    validate_operation_provenance(operations)
    return {
        "policy_id": (policy["policy_id"] if policy else operations[-1]['policy_id']),
        "stage": stage,
        "parse_method": method,
        "source_raw_output_sha256": _text_sha256(result.get("raw_output")),
        "execution_plan_sha256": _json_sha256(parsed),
        "recovery_produced_candidate": isinstance(parsed, (dict, list)),
        "wrapper_normalizations": _wrapper_normalizations(
            result.get("raw_output")),
        "semantic_values_added": False,
        "operations": operations,
    }


def _repair_plan2_visual_style(decoded, result, candidate):
    """Fill Plan2's rendering metadata when only ``visual_style`` is missing."""
    if not isinstance(candidate, dict):
        return None
    steps = candidate.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    default_style = {
        "medium": "photo",
        "color_treatment": "natural",
        "lighting_character": "daylight",
        "rendering_finish": "realistic",
    }
    current = candidate.get("visual_style")
    repaired = copy.deepcopy(candidate)
    changes = []
    if not isinstance(current, dict):
        if current is not None:
            return None
        repaired["visual_style"] = dict(default_style)
        for field, value in default_style.items():
            changes.append({
                "json_path": "$.visual_style",
                "operation": "add_default_visual_style_field",
                "destination": f"$.visual_style.{field}",
                "value_before": None,
                "value_after": value,
            })
    else:
        for field, value in default_style.items():
            existing = current.get(field)
            if isinstance(existing, str) and existing.strip():
                continue
            if field in current and existing is not None and not isinstance(existing, str):
                return None
            repaired["visual_style"][field] = value
            changes.append({
                "json_path": f"$.visual_style.{field}",
                "operation": "fill_empty_visual_style_field",
                "destination": f"$.visual_style.{field}",
                "value_before": existing,
                "value_after": value,
            })
    if not changes:
        return None
    for step in repaired["steps"]:
        if not isinstance(step, dict):
            return None
        for field in ("image_prompt", "content"):
            value = step.get(field)
            if not isinstance(value, str) or not value.strip():
                return None
            if find_planner_placeholders(value):
                return None
    return _repair_result(
        PLAN2_VISUAL_STYLE_DEFAULT_POLICY["policy_id"],
        "plan2", decoded, result, candidate, repaired, changes)


def _repair_plan2_step_aliases(plan, decoded, result, candidate):
    """Rename Plan2 field synonyms when nothing else is wrong."""
    if not isinstance(candidate, dict):
        return None
    plan1 = plan.get("plan1") if isinstance(plan, dict) else None
    plan1_steps = plan1.get("steps") if isinstance(plan1, dict) else None
    steps = candidate.get("steps")
    if not isinstance(plan1_steps, list) or not isinstance(steps, list):
        return None
    if not steps or len(steps) != len(plan1_steps):
        return None
    expected = [
        step.get("step_index") if isinstance(step, dict) else None
        for step in plan1_steps
    ]
    if expected != list(range(1, len(expected) + 1)):
        return None

    field_aliases = (
        ("image_prompt", "image_subject",
         lambda value: isinstance(value, str) and bool(value.strip())),
        ("step_index", "index",
         lambda value: isinstance(value, int) and not isinstance(value, bool)),
    )
    repaired = copy.deepcopy(candidate)
    changes = []
    for position, step in enumerate(repaired["steps"]):
        if not isinstance(step, dict):
            return None
        content = step.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        if find_planner_placeholders(content):
            return None
        for canonical, synonym, is_valid in field_aliases:
            has_canonical = canonical in step
            has_synonym = synonym in step
            if has_canonical and has_synonym and is_valid(step.get(canonical)):
                # Both spellings carry a usable value: an extra field, not an
                # alias. Validation already tolerates it; never guess here.
                return None
            if has_synonym:
                step[canonical] = step.pop(synonym)
                changes.append({
                    "json_path": f"$.steps[{position}].{synonym}",
                    "operation": "rename_key",
                    "destination": f"$.steps[{position}].{canonical}",
                    "value_before": step[canonical],
                    "value_after": step[canonical],
                })
            if not is_valid(step.get(canonical)):
                return None
        if find_planner_placeholders(step["image_prompt"]):
            return None
    if not changes:
        return None
    if [step["step_index"] for step in repaired["steps"]] != expected:
        return None
    return _repair_result(
        PLAN2_STEP_ALIAS_POLICY["policy_id"],
        "plan2", decoded, result, candidate, repaired, changes)


def _repair_plan2_step_indexes(plan, decoded, result, candidate):
    """Re-number Plan2 steps when only the copied index drifted."""
    if not isinstance(candidate, dict):
        return None
    plan1 = plan.get("plan1") if isinstance(plan, dict) else None
    plan1_steps = plan1.get("steps") if isinstance(plan1, dict) else None
    steps = candidate.get("steps")
    if not isinstance(plan1_steps, list) or not isinstance(steps, list):
        return None
    if not steps or len(steps) != len(plan1_steps):
        return None
    expected = [
        step.get("step_index") if isinstance(step, dict) else None
        for step in plan1_steps
    ]
    if expected != list(range(1, len(expected) + 1)):
        return None
    current = []
    for step in steps:
        if (not isinstance(step, dict)
                or not isinstance(step.get("image_prompt"), str)
                or not step["image_prompt"].strip()
                or not isinstance(step.get("content"), str)
                or not step["content"].strip()):
            return None
        index = step.get("step_index")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        current.append(index)
    if current == expected:
        return None

    repaired = copy.deepcopy(candidate)
    changes = []
    for position, (step, index) in enumerate(zip(repaired["steps"], expected)):
        changes.append({
            "json_path": f"$.steps[{position}].step_index",
            "operation": "realign_step_index_by_position",
            "destination": f"$.steps[{position}].step_index",
            "value_before": step["step_index"],
            "value_after": index,
        })
        step["step_index"] = index
    return _repair_result(
        PLAN2_STEP_INDEX_REALIGNMENT_POLICY["policy_id"],
        "plan2", decoded, result, candidate, repaired, changes)


def repair_planner_stage(label: str, result: dict, context=None):
    """Return one unambiguous mechanical schema repair, or ``None``."""
    if not isinstance(result, dict):
        return None
    decoded = result.get("decoded")
    candidate = decoded.get("candidate") if isinstance(decoded, dict) else None
    if label == "plan4":
        if not isinstance(candidate, list) or not all(
                isinstance(item, str) for item in candidate):
            return None
        repaired = {"reviewer_checklist": copy.deepcopy(candidate)}
        return _repair_result(
            PLAN4_CHECKLIST_WRAPPER_POLICY["policy_id"],
            label, decoded, result, candidate,
            repaired, [{
                "operation": "wrap_top_level_list",
                "json_path": "$",
                "destination": "$.reviewer_checklist",
            }])
    if label == "plan3":
        return _repair_plan3_step_conditions(decoded, result, candidate)
    if label == "plan2":
        plan = context.get("plan") if isinstance(context, dict) else None
        # Field synonyms first (the model used a registered name for the same
        # value), then pure index drift; a plan that needs both is left alone.
        aliased = _repair_plan2_step_aliases(plan, decoded, result, candidate)
        if aliased is not None:
            return aliased
        styled = _repair_plan2_visual_style(decoded, result, candidate)
        if styled is not None:
            return styled
        return _repair_plan2_step_indexes(plan, decoded, result, candidate)
    if label != "plan1":
        return None
    if not isinstance(candidate, dict) or not isinstance(candidate.get("steps"), list):
        return None
    if not candidate["steps"]:
        return None

    repaired = copy.deepcopy(candidate)
    changes = []
    # Per-field synonym acceptance: canonical names and registered synonyms may
    # mix, as long as no field is ambiguous and none is missing.
    field_aliases = (
        ("step_index", "index", lambda value: isinstance(value, int)
         and not isinstance(value, bool)),
        ("image_subject", "subject",
         lambda value: isinstance(value, str) and bool(value.strip())),
        ("content_goal", "goal",
         lambda value: isinstance(value, str) and bool(value.strip())),
    )
    for position, step in enumerate(repaired["steps"]):
        if not isinstance(step, dict):
            return None
        for canonical, synonym, is_valid in field_aliases:
            has_canonical = canonical in step
            has_synonym = synonym in step
            if has_canonical and has_synonym:
                # Ambiguous spelling: never guess which one the model meant.
                return None
            if not has_canonical and not has_synonym:
                return None
            if has_synonym:
                step[canonical] = step.pop(synonym)
                changes.append({
                    "json_path": f"$.steps[{position}].{synonym}",
                    "operation": "rename_key",
                    "destination": f"$.steps[{position}].{canonical}",
                    "value_before": step[canonical],
                    "value_after": step[canonical],
                })
            if not is_valid(step[canonical]):
                return None
    if not changes:
        return None

    return _repair_result(
        PLAN1_STEP_ALIAS_POLICY["policy_id"],
        label, decoded, result, candidate, repaired,
        changes)


def _repair_result(policy_id, label, decoded, result, candidate, repaired, changes):
    operations = []
    for sequence, change in enumerate(changes, 1):
        operation = dict(change)
        value_before = operation.pop("value_before", None)
        value_after = operation.pop("value_after", None)
        if value_before is None and value_after is None:
            before, after = candidate, repaired
        else:
            before = {
                "path": operation.get("json_path"), "value": value_before,
            }
            after = {
                "path": operation.get("destination"), "value": value_after,
            }
        operation.update({
            "sequence": sequence,
            "policy_id": policy_id,
            "input_sha256": _json_sha256(before),
            "output_sha256": _json_sha256(after),
        })
        operations.append(operation)
    validate_operation_provenance(operations)
    return {
        "parsed": repaired,
        "provenance": {
            "policy_id": policy_id,
            "stage": label,
            "source_parse_method": decoded.get("parse_method"),
            "source_raw_output_sha256": _text_sha256(result.get("raw_output")),
            "source_candidate_sha256": _json_sha256(candidate),
            "execution_plan_sha256": _json_sha256(repaired),
            "changes": changes,
            "operations": operations,
            "wrapper_normalizations": _wrapper_normalizations(
                result.get("raw_output")),
            "semantic_values_added": False,
        },
    }


def _repair_plan3_step_conditions(decoded, result, candidate):
    """Merge only the explicitly allowed Plan3 fragment representation."""
    if not isinstance(candidate, dict):
        return None
    fragments = candidate.get("step_conditions")
    bank = candidate.get("condition_bank")
    if (set(candidate) != {"condition_bank", "step_conditions"}
            or not isinstance(bank, dict) or not isinstance(fragments, list)
            or not fragments):
        return None
    merged = {}
    for fragment in fragments:
        if not isinstance(fragment, dict) or len(fragment) != 1:
            return None
        key, refs = next(iter(fragment.items()))
        if (not isinstance(key, str) or not key.startswith("step_")
                or key in merged or not isinstance(refs, list)
                or not all(isinstance(ref, str) and ref.strip() for ref in refs)):
            return None
        merged[key] = copy.deepcopy(refs)
    repaired = {"condition_bank": copy.deepcopy(bank), "step_conditions": merged}
    # Import lazily: validation imports decoding but not this recovery module.
    from .validation import plan3_output_issues
    plan1 = result.get("plan1")
    if not isinstance(plan1, dict):
        return None
    issues = plan3_output_issues(plan1, repaired)
    allowed = [issue for issue in issues if issue.startswith("condition bank")]
    if len(allowed) != len(issues):
        return None
    return _repair_result(
        PLAN3_FRAGMENT_MERGE_POLICY["policy_id"],
        "plan3", decoded, result,
        candidate, repaired, [{
            "operation": "merge_step_conditions_fragments",
            "json_path": "$.step_conditions",
            "destination": "$.step_conditions",
            "condition_count_noncompliant": bool(issues),
        }])
