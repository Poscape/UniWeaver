"""The seven ablation profiles and the D3 identity self-check."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Profile:
    id: str
    title: str
    claim: str
    factor: str = ""
    prompt: dict[str, str] = field(default_factory=dict)
    request: dict[str, str] = field(default_factory=dict)
    schema: tuple[str, ...] = ()
    runtime: dict[str, object] = field(default_factory=dict)
    notes: str = ""


IDENTITY = Profile(
    id="D3",
    title="identity (production UniWeaver)",
    claim="patch-layer identity",
    factor="none",
    notes=(
        "D3 is a self-check: run_variant.py --variant D3 must produce a "
        "trajectory byte-identical to baselines/run_umm.py --protocol uniweaver."
    ),
)


REFERENCE_FIXED = (
    Profile(
        id="O32",
        title="References Fixed before the Text",
        claim="reference selection",
        factor="references_before_prose",
        request={"plan3": "plan1_only"},
        notes=(
            "Plan 3 receives Plan 1 scope and subjects without Plan 2's "
            "rendered step prompts."
        ),
    ),
)


FIELDS = (
    Profile(
        id="F1",
        title="w/o Task Goal",
        claim="task goal coverage",
        factor="plan1_endpoint_removed",
        prompt={"plan1": "no_endpoint"},
        schema=("plan1:endpoint",),
        notes="Plan 2 receives no task-goal line.",
    ),
    Profile(
        id="F2",
        title="w/o Per-step Content Goal",
        claim="per-step content guidance",
        factor="plan1_content_goal_removed",
        prompt={"plan1": "no_content_goal"},
        schema=("plan1:steps[].content_goal",),
        notes=(
            "Plan 1 steps retain image subjects without content goals; Plan 2 "
            "receives the reduced step descriptions."
        ),
    ),
    Profile(
        id="F4",
        title="w/o Shared Visual Style",
        claim="shared visual style",
        factor="plan2_visual_style_removed",
        prompt={"plan2": "no_visual_style"},
        schema=("plan2:visual_style",),
        notes="Rendering treatment is specified independently in each step.",
    ),
    Profile(
        id="F3",
        title="w/ Merged Image Prompt and Content",
        claim="image and text differentiation",
        factor="plan2_fields_merged",
        prompt={"plan2": "single_text_field"},
        schema=("plan2:steps[].image_prompt",),
        runtime={"merged_step_text_field": "text"},
        notes="Each step uses one text field for both image and prose inputs.",
    ),
)


REFERENCES = (
    Profile(
        id="F5",
        title="w/o Per-step Reference Routing",
        claim="per-step reference routing",
        factor="plan3_routing_removed",
        runtime={"routing": "all_steps"},
        notes=(
            "Plan 3 builds the condition bank; every step receives every "
            "reference instead of the routed subset."
        ),
    ),
    Profile(
        id="F5b",
        title="w/o All Reference Images",
        claim="reference-image conditioning",
        factor="references_removed",
        runtime={"conditions": "none"},
        notes="The condition bank is removed before runtime projection.",
    ),
)


ALL = (IDENTITY, *FIELDS, *REFERENCE_FIXED, *REFERENCES)
EXPERIMENTS = tuple(p for p in ALL if p.id != "D3")
BY_ID = {p.id: p for p in ALL}


def profile(variant_id: str) -> Profile:
    try:
        return BY_ID[variant_id]
    except KeyError:
        raise SystemExit(
            f"unknown variant {variant_id!r}; known: {', '.join(BY_ID)}")
