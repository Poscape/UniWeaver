"""Variant prompt texts derived from the production ones, with drift guards."""
from __future__ import annotations

import json

from core import contract as c


def _require(haystack: str, needle: str, variant: str) -> str:
    if needle not in haystack:
        raise RuntimeError(
            f"{variant}: anchor text not found in the production prompt; "
            f"core/contract.py changed under the ablation layer: {needle!r}")
    return haystack


def _steps_summary(steps, fields) -> str:
    return c._steps_summary(steps, fields)


# --------------------------------------------------------------------------- #
# plan1 variants
# --------------------------------------------------------------------------- #
_P1_SCHEMA = (
    '{"title":"...","endpoint":"earliest completed outcome or last explicit '
    'event",\n"steps":[{"step_index":1,"image_subject":"one concrete visible '
    'subject or state",\n"content_goal":"short reader-facing question"}]}')


def plan1_no_endpoint() -> str:
    prompt = _require(c.PLAN1_PROMPT, _P1_SCHEMA, "F1")
    replaced = prompt.replace(
        _P1_SCHEMA,
        '{"title":"...",\n"steps":[{"step_index":1,"image_subject":"one '
        'concrete visible subject or state",\n"content_goal":"short '
        'reader-facing question"}]}')
    replaced = replaced.replace(
        "The top-level object has exactly these three keys.",
        "The top-level object has exactly these two keys: title and steps.")
    replaced = replaced.replace(
        "The endpoint is the earliest completed requested outcome",
        "Do not record an endpoint field; judge scope from the task itself. "
        "The earliest completed requested outcome")
    return replaced


def plan1_no_content_goal() -> str:
    prompt = _require(c.PLAN1_PROMPT, _P1_SCHEMA, "F2")
    replaced = prompt.replace(
        _P1_SCHEMA,
        '{"title":"...","endpoint":"earliest completed outcome or last '
        'explicit event",\n"steps":[{"step_index":1,"image_subject":"one '
        'concrete visible subject or state"}]}')
    replaced = replaced.replace(
        "For each unit separate:\n- image_subject",
        "For each unit record only the part that a single image can show:\n"
        "- image_subject")
    replaced = replaced.replace(
        "Every object inside steps has exactly three keys:\n"
        "step_index, image_subject, and content_goal.",
        "Every object inside steps has exactly two keys: step_index and "
        "image_subject. Do not emit a content_goal field.")
    replaced = replaced.replace(
        "\n- content_goal: one short reader-facing question asking what useful "
        "explanation, interpretation,\n  comparison, instruction, chronology, "
        "or decision the adjacent prose must answer. Ask about the\n"
        "  subject itself.",
        "")
    replaced = replaced.replace(
        "put the properties and comparison criteria together in\n"
        "that unit's content_goal.",
        "keep the properties and comparison criteria within\n"
        "that unit's single view.")
    replaced = replaced.replace(
        "its content_goal carries all criteria, trade-offs, and the decision.",
        "that single view must expose the criteria, trade-offs, and the decision.")
    replaced = replaced.replace(
        " Put synthesis and\nthe decision in the content_goal of the last "
        "alternative's existing unit.",
        "")
    return replaced


# --------------------------------------------------------------------------- #
# plan2 variants
# --------------------------------------------------------------------------- #
def plan2_single_text_field() -> str:
    prompt = _require(c.PLAN2_PROMPT, "image_prompt: one complete", "F3")
    replaced = prompt.replace(
        "- image_prompt: one complete, self-contained rendering instruction for "
        "the observable subject,\n  state, relations, and context required by "
        "this contribution;\n- content: finished report prose carrying useful "
        "explanation that need not be visible in pixels.",
        "- text: one self-contained instruction and explanation used both for "
        "the image and for the reader; write it once.")
    return replaced.replace(
        '"steps":{step_semantic_entries}', '"steps":{step_semantic_entries}')


def _single_text_entries(steps) -> str:
    entries = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        entries.append({
            "step_index": step.get("step_index"),
            "text": "one self-contained sentence used for image and reader",
        })
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":"))


def plan2_request_single_text_field(task_prompt, plan1) -> str:
    """F3: same frozen steps, but the response carries a single text field."""
    plan1 = plan1 if isinstance(plan1, dict) else {}
    steps = plan1.get("steps") if isinstance(plan1.get("steps"), list) else []
    return (c.prompt_template("plan2")
            .replace("{task}", c._stringify(task_prompt))
            .replace("{steps_summary}",
                     _steps_summary(steps, ("image_subject", "content_goal")))
            .replace("{endpoint}", c._stringify(plan1.get("endpoint")))
            .replace("{step_indexes}", json.dumps(
                [step.get("step_index") for step in steps
                 if isinstance(step, dict)],
                ensure_ascii=False, separators=(",", ":")))
            .replace("{step_semantic_entries}", _single_text_entries(steps)))


def plan2_no_visual_style() -> str:
    prompt = _require(c.PLAN2_PROMPT, "Choose one shared visual_style", "F4")
    replaced = prompt.replace(
        "Choose one shared visual_style for rendering treatment only: medium, "
        "color treatment, lighting\ncharacter, and rendering finish. Keep "
        "subjects, actions, locations, viewpoint, composition, and\nprose voice "
        "in the individual steps.",
        "Do not emit a visual_style block; choose the rendering treatment "
        "inside each step's own image_prompt.")
    replaced = replaced.replace(
        'Output schema:\n{"visual_style":{"medium":"...",'
        '"color_treatment":"...",\n"lighting_character":"...",'
        '"rendering_finish":"..."},\n"steps":{step_semantic_entries}}',
        'Output schema:\n{"steps":{step_semantic_entries}}')
    return replaced


# --------------------------------------------------------------------------- #
# request builders for the ordering variants
# --------------------------------------------------------------------------- #
def plan3_request_from_plan1(task_prompt, plan1, plan2) -> str:
    """O32: references are chosen from the frozen subjects, not the prose."""
    plan1 = plan1 if isinstance(plan1, dict) else {}
    plan2 = plan2 if isinstance(plan2, dict) else {}
    steps = plan1.get("steps") if isinstance(plan1.get("steps"), list) else []
    plan2_steps = plan2.get("steps")
    if not isinstance(plan2_steps, list):
        raise TypeError("O32 requires plan2 steps for the index skeleton")
    return (c.prompt_template("plan3")
            .replace("{task}", c._stringify(task_prompt))
            .replace("{semantic_subjects}",
                     _steps_summary(steps, ("image_subject",)))
            .replace("{steps_summary}",
                     _steps_summary(steps, ("image_subject",)))
            .replace("{step_condition_entries}",
                     c.build_step_condition_entries(len(plan2_steps))))


def plan2_request_without_endpoint(task_prompt, plan1) -> str:
    """F1: plan2 keeps the frozen steps but no endpoint line."""
    plan1 = plan1 if isinstance(plan1, dict) else {}
    steps = plan1.get("steps") if isinstance(plan1.get("steps"), list) else []
    return (c.prompt_template("plan2")
            .replace("{task}", c._stringify(task_prompt))
            .replace("{steps_summary}",
                     _steps_summary(steps, ("image_subject", "content_goal")))
            .replace("{endpoint}", "(endpoint not recorded)")
            .replace("{step_indexes}", json.dumps(
                [step.get("step_index") for step in steps
                 if isinstance(step, dict)],
                ensure_ascii=False, separators=(",", ":")))
            .replace("{step_semantic_entries}",
                     c.build_step_semantic_entries(steps)))


def plan2_request_subject_only(task_prompt, plan1) -> str:
    """F2: plan2 sees subjects only, because content_goal no longer exists."""
    plan1 = plan1 if isinstance(plan1, dict) else {}
    steps = plan1.get("steps") if isinstance(plan1.get("steps"), list) else []
    return (c.prompt_template("plan2")
            .replace("{task}", c._stringify(task_prompt))
            .replace(
                "Directly answer each content_goal using task details,",
                "Decide what reader-facing content each unit needs, using task "
                "details,")
            .replace("{steps_summary}",
                     _steps_summary(steps, ("image_subject",)))
            .replace("{endpoint}", c._stringify(plan1.get("endpoint")))
            .replace("{step_indexes}", json.dumps(
                [step.get("step_index") for step in steps
                 if isinstance(step, dict)],
                ensure_ascii=False, separators=(",", ":")))
            .replace("{step_semantic_entries}",
                     c.build_step_semantic_entries(steps)))
