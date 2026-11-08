from __future__ import annotations

import hashlib
import json
from typing import Any

from core.text_normalization import normalize_prompt

from .data import TaskSpec
from .reports import ReportArtifact

TEXT_SYSTEM_PROMPT = """You are a strict evaluator for an interleaved text-and-image report agent.
You receive one isolated task packet. Evaluate only the materials explicitly supplied in that packet.
Do not browse, use external knowledge, or reward plausible details not present in the packet.
This is not a 1-10 quality rating. For every predefined criterion return exactly one tier: MET, PARTIAL, or NOT_MET.
Give one short, concrete observation from the actual artifact for every tier. Do not compensate one failed criterion with another.
Do not output chain-of-thought, narrative rationale, markdown, or any text outside the required JSON object.
Return one JSON object only.
"""


VISION_SYSTEM_PROMPT = """You are a strict visual evaluator for an interleaved text-and-image report agent.
You receive one isolated task packet and its attached generated images. Evaluate only those materials.
Do not browse, use external knowledge, or demand a real-world reference that was not supplied.
This is not a 1-10 quality rating. For every predefined criterion return exactly one tier: MET, PARTIAL, or NOT_MET.
Give one short, concrete visible observation from the actual image or image-text pair for every tier. Do not treat readable image text as evidence unless rendered text is itself the task subject.
Do not output chain-of-thought, narrative rationale, markdown, or any text outside the required JSON object.
Return one JSON object only.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _visual_atomic_checks(task: TaskSpec) -> list[dict[str, Any]]:
    reference = task.evaluation_reference
    checks: list[dict[str, Any]] = []
    for index, evidence in enumerate(reference.get("visible_evidence", []), start=1):
        if isinstance(evidence, str) and evidence.strip():
            checks.append(
                {
                    "id": f"visible_evidence_{index:02d}",
                    "type": "visible_required_evidence",
                    "weight": 3,
                    "assertion": f"At least one attached final image visibly shows: {evidence}",
                }
            )
    if not checks:
        raise ValueError(f"Task {task.task_id} has no visual atomic checks.")
    return checks


def _report_units(artifact: ReportArtifact, include_images: bool) -> list[dict[str, Any]]:
    return [
        {
            "unit_index": unit.get("unit_index"),
            "content": unit.get("content"),
            **({"image_path": unit.get("image_path")} if include_images else {}),
        }
        for unit in artifact.units
    ]


def _report_checks(
    layer: str,
    task: TaskSpec,
    artifact: ReportArtifact | None = None,
) -> list[dict[str, Any]]:
    reference = task.evaluation_reference
    if layer == "report_text_atomic":
        checks = [
            {
                "id": f"text_coverage_{index:02d}",
                "type": "required_coverage",
                "weight": 3,
                "assertion": assertion,
            }
            for index, assertion in enumerate(
                reference.get("essential_coverage", []), start=1
            )
            if isinstance(assertion, str) and assertion.strip()
        ]
        endpoint = reference.get("endpoint")
        if isinstance(endpoint, str) and endpoint.strip():
            checks.append(
                {
                    "id": "text_endpoint",
                    "type": "required_endpoint",
                    "weight": 3,
                    "assertion": endpoint,
                }
            )
        return checks
    if layer == "content_quality_atomic":
        return [
            {
                "id": "reader_ready_clarity",
                "type": "content_quality",
                "weight": 1,
                "assertion": "Every final prose unit is complete, grammatical, reader-facing, and free of material dangling references or process meta-commentary.",
            },
            {
                "id": "concrete_task_value",
                "type": "content_quality",
                "weight": 1,
                "assertion": "The final prose provides concrete task-relevant explanation, instruction, comparison, limitation, or interpretation beyond merely captioning an image.",
            },
        ]
    if layer == "visual_atomic":
        return _visual_atomic_checks(task)
    if layer == "report_interleaved_atomic":
        units = list(getattr(artifact, "units", None) or [])
        if not units:
            raise ValueError(
                f"Task {task.task_id} has no text-image unit to pair.")
        return [
            {
                "id": f"paired_evidence_{index:02d}",
                "type": "paired_evidence",
                "weight": 1,
                "image_index": index,
                "assertion": (
                    "The prose of this unit explains or makes usable a "
                    "task-relevant fact that is visibly present in the "
                    "unit's paired image."
                ),
            }
            for index in range(1, len(units) + 1)
        ]
    raise ValueError(f"Unknown report layer: {layer}")


def _report_response_example(metric: str, checks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "verdicts": {
            metric: {
                check["id"]: {
                    "tier": "MET",
                    "evidence": "short concrete observation from the supplied report",
                }
                for check in checks
            }
        },
    }


def build_report_request(
    task: TaskSpec,
    artifact: ReportArtifact,
    layer: str,
) -> dict[str, Any]:
    recorded_prompt = (artifact.prompt or "").strip()
    if recorded_prompt and normalize_prompt(recorded_prompt) != normalize_prompt(
            task.prompt or ""):
        raise ValueError(
            f"Run record {artifact.task_id} carries a task prompt that does not "
            "match the benchmark task; refusing to score it against the "
            "benchmark reference."
        )
    checks = _report_checks(layer, task, artifact)
    metric = {
        "report_text_atomic": "TC",
        "content_quality_atomic": "WQ",
        "visual_atomic": "VC",
        "report_interleaved_atomic": "ITA",
    }[layer]
    if layer == "report_text_atomic":
        reference = {
            "essential_coverage": task.evaluation_reference.get("essential_coverage", []),
            "endpoint": task.evaluation_reference.get("endpoint"),
        }
        user_prompt = f"""Evaluate only the ordered final reader-facing prose for one report task.

Task ID: {task.task_id}
Primary family: {task.family}
User task:
{artifact.prompt or task.prompt or "MISSING"}

Hidden text reference:
{_json(reference)}

Ordered final prose, with no image prompts or trajectory metadata:
{_json(_report_units(artifact, include_images=False))}

Apply only these frozen checks:
{_json(checks)}
For every check return MET, PARTIAL, or NOT_MET with one short concrete observation. Do not use images, image prompts, Plan fields, style, or general quality. Return exactly:
{_json(_report_response_example(metric, checks))}
"""
    elif layer == "content_quality_atomic":
        user_prompt = f"""Evaluate only whether the final prose is usable by a reader.

Task ID: {task.task_id}
User task:
{artifact.prompt or task.prompt or "MISSING"}

Final prose only:
{_json(_report_units(artifact, include_images=False))}

Frozen checks:
{_json(checks)}
Do not judge task coverage, factual truth, images, image-text alignment, literary style, or length. Return exactly:
{_json(_report_response_example(metric, checks))}
"""
    elif layer == "visual_atomic":
        user_prompt = f"""Evaluate only visible evidence in the attached final report images.

Task ID: {task.task_id}
User task:
{artifact.prompt or task.prompt or "MISSING"}

Images are attached in this order:
{_json([str(path) for path in artifact.image_paths])}

Frozen checks:
{_json(checks)}
Judge only final pixels. Do not use Plan fields, image prompts, prose claims, readable image text unless rendered text is itself the task subject, taste, or photorealism. Return exactly:
{_json(_report_response_example(metric, checks))}
"""
    else:
        user_prompt = f"""Evaluate rendered image-text interleaving for one report task.

Task ID: {task.task_id}
User task:
{artifact.prompt or task.prompt or "MISSING"}

Exact adjacent prose and attached image order:
{_json(_report_units(artifact, include_images=True))}
{_json([str(path) for path in artifact.image_paths])}

Frozen hidden paired-evidence checks:
{_json(checks)}
Inspect each named image together with only its adjacent prose. Do not use Plan fields, image prompts, prose from another unit, or image aesthetics. Return exactly:
{_json(_report_response_example(metric, checks))}
"""
    request = {
        "request_id": f"{artifact.run_id}:{task.task_id}:{layer}",
        "layer": layer,
        "metric": metric,
        "task_id": task.task_id,
        "family": task.family,
        "run_id": artifact.run_id,
        "source_artifact": str(artifact.source_path),
        "image_paths": [str(path) for path in artifact.image_paths]
        if layer in {"visual_atomic", "report_interleaved_atomic"}
        else [],
        "criteria": checks,
        "report_track": True,
        "report_input": {
            "user_task": task.prompt,
            "artifact_prompt": artifact.prompt,
        },
        "system_prompt": VISION_SYSTEM_PROMPT
        if layer in {"visual_atomic", "report_interleaved_atomic"}
        else TEXT_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "task_prompt_sha256": hashlib.sha256(
            (task.prompt or "").encode("utf-8")
        ).hexdigest(),
    }
    request["request_content_sha256"] = _sha256_json(request)
    return request


def build_report_requests(
    tasks: dict[str, TaskSpec],
    artifacts: dict[str, ReportArtifact],
    layer: str,
) -> list[dict[str, Any]]:
    if layer not in {
        "report_text_atomic",
        "content_quality_atomic",
        "visual_atomic",
        "report_interleaved_atomic",
    }:
        raise ValueError(f"Unknown report layer: {layer}")
    return [
        build_report_request(tasks[task_id], artifacts[task_id], layer)
        for task_id in sorted(artifacts)
        if task_id in tasks
        and (layer not in {"visual_atomic", "report_interleaved_atomic"} or artifacts[task_id].image_paths)
    ]
