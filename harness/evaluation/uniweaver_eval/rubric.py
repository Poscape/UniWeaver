from __future__ import annotations

from typing import Any

TIER_TO_ANCHOR = {"NOT_MET": 0, "PARTIAL": 1, "MET": 2}
ANCHOR_TO_TIER = {value: key for key, value in TIER_TO_ANCHOR.items()}

VISION_DIMENSIONS = {
    "RQS": (
        "indispensable_visual_evidence",
        "state_relation_and_mechanism_fidelity",
        "cross_unit_identity_and_continuity",
        "local_prompt_to_image_fidelity",
        "visual_legibility_and_no_prohibited_substitute",
    ),
    "IRS": (
        "image_text_correspondence",
        "image_text_complementarity",
        "sequence_pacing_and_navigation",
    ),
}


def dimensions_for(
    layer: str,
    criteria: list[dict[str, Any]] | None = None,
) -> dict[str, tuple[str, ...]]:
    if layer in {
        "visual_atomic",
        "report_text_atomic",
        "content_quality_atomic",
        "report_interleaved_atomic",
    }:
        if not criteria:
            raise ValueError(f"{layer} requires nonempty task-specific criteria.")
        identifiers = tuple(
            criterion["id"]
            for criterion in criteria
            if isinstance(criterion, dict) and isinstance(criterion.get("id"), str)
        )
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{layer} criteria must have unique string ids.")
        metric = {
            "visual_atomic": "VC",
            "report_text_atomic": "TC",
            "content_quality_atomic": "WQ",
            "report_interleaved_atomic": "ITA",
        }[layer]
        return {metric: identifiers}
    if layer == "vision":
        return VISION_DIMENSIONS
    raise ValueError(f"Unknown layer: {layer}")


def anchor_from_judgment(
    judgment: dict[str, Any], metric: str, dimension: str
) -> int | None:
    """Read the tier a judge assigned to one criterion."""
    verdict = (
        judgment.get("verdicts", {})
        .get(metric, {})
        .get(dimension)
    )
    if isinstance(verdict, dict):
        tier = verdict.get("tier")
        if tier in TIER_TO_ANCHOR:
            return TIER_TO_ANCHOR[tier]
    return None
