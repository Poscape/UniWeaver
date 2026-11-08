"""Image Reviewer parsing and retry decisions for optional UMM quality control."""

from __future__ import annotations

import re
import time

from core.decoding import (
    detect_repetition_loop,
    detect_think_loop,
    split_think,
)
from core.policy_registry import RUNTIME_CHECKLIST_PREFIX_POLICY

REVIEW_MAX_TOKENS = 1536
REVIEWER_PROMPT = """Look at the generated image and check every numbered item below.
Use only directly visible evidence in this image. Do not infer hidden facts or read text.

{check_items}

Output exactly one result per line in this form:
1: PASS
2: FAIL: concise visible reason
Do not omit, combine, or reorder results."""


def check_result(passed, reasons=None, retryable=False):
    return {
        "passed": bool(passed),
        "reasons": list(reasons or []),
        "retryable": bool(retryable and not passed),
    }


def split_checklist(checklist):
    """Split the canonical flat checklist into per-step and general items."""
    per_step = {}
    general = []
    for item in checklist or []:
        if not isinstance(item, str):
            continue
        if item.startswith("general:"):
            general.append(item)
            continue
        match = re.match(r"step_(\d+):", item)
        if match:
            per_step.setdefault(int(match.group(1)), []).append(item)
    return per_step, general


def checks_for_step(checklist, step_index, limit=None):
    """Apply general checks after Step-specific checks within one Step budget."""
    per_step, general = split_checklist(checklist)
    checks = list(per_step.get(step_index, [])) + list(general)
    effective_limit = (
        RUNTIME_CHECKLIST_PREFIX_POLICY["default_limit"]
        if limit is None else limit
    )
    return checks[:effective_limit], checks[effective_limit:]


def parse_numbered_checks(raw, item_count):
    """Parse one numbered PASS/FAIL response and fail closed on missing rows."""
    think_text, after_think = split_think(raw)
    has_reasoning = "</think" in str(raw or "")
    answer = after_think if has_reasoning else str(raw or "")
    # Check the reasoning block only when one exists, then scan the answer text
    # that the numbered-check parser actually reads.
    loop_warning = (
        (detect_think_loop(think_text) if has_reasoning else None)
        or detect_repetition_loop(answer, label='Reviewer output')
    )
    parsed = {}
    for line in answer.splitlines():
        match = re.match(
            r"\s*(\d+)[:.)]\s*(PASS|FAIL)\b[:\s-]*(.*)",
            line,
            re.IGNORECASE,
        )
        if match:
            parsed[int(match.group(1)) - 1] = (
                match.group(2).upper(),
                match.group(3).strip(),
            )

    missing_indexes = [
        index + 1 for index in range(item_count) if index not in parsed
    ]
    results = []
    for index in range(item_count):
        if index in parsed:
            status, reason = parsed[index]
        else:
            status = "FAIL"
            reason = (
                "reviewer response omitted this result; "
                f"parsed={len(parsed)}/{item_count}; missing={missing_indexes}"
            )
        results.append({
            "status": status,
            "reason": reason,
            "check": check_result(
                status == "PASS", [reason] if reason else [], True
            ),
        })
    return think_text if answer != think_text else "", results, loop_warning


def review_prompt(checks):
    """Build the exact single-image Reviewer prompt for persistence."""
    numbered = "\n".join(
        f"{index}. {check}" for index, check in enumerate(checks, 1)
    )
    return REVIEWER_PROMPT.replace("{check_items}", numbered)


def review_step_image(backend, image, checks):
    """Review one Step image without Condition/reference-image inputs."""
    prompt = review_prompt(checks)
    started = time.time()
    raw_output = backend.review_image(
        image=image,
        prompt=prompt,
        condition_images={},
        max_tokens=REVIEW_MAX_TOKENS,
    )
    think_text, parsed, loop_warning = parse_numbered_checks(
        raw_output, len(checks)
    )
    results = [
        {
            "item": check,
            "status": result["status"],
            "reason": result["reason"],
        }
        for check, result in zip(checks, parsed)
    ]
    reasons = [
        result["reason"] for result in results if result["status"] == "FAIL"
    ]
    return {
        "approved": not reasons,
        "check": check_result(not reasons, reasons, True),
        "results": results,
        "raw_input": prompt,
        "raw_output": raw_output,
        "think_text": think_text,
        "loop_warning": loop_warning,
        "duration_seconds": time.time() - started,
    }


def retry_feedback(review):
    """Compile failed checklist rows into one retry-only image instruction."""
    parts = []
    for result in review.get("results", []):
        if result.get("status") != "FAIL":
            continue
        required = str(result.get("item", "")).split(":", 1)[-1].strip()
        reason = str(result.get("reason", "")).strip()
        parts.append(
            f"Required visible result: {required}. Observed issue: {reason}"
        )
    return "\n".join(parts)
