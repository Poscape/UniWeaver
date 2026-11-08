from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ablation import profiles as P
from ablation import prompts as VP
from core import contract as c

EXPERIMENT_IDS = ("F1", "F2", "F4", "F3", "O32", "F5", "F5b")


def test_the_experiment_set_is_exactly_the_seven_variants():
    assert tuple(p.id for p in P.EXPERIMENTS) == EXPERIMENT_IDS
    assert len(P.EXPERIMENTS) == 7
    assert P.IDENTITY.id == "D3"
    assert "D3" not in EXPERIMENT_IDS


def test_every_variant_declares_one_factor_and_only_that_factor():
    factors = [profile.factor for profile in P.EXPERIMENTS]
    assert len(set(factors)) == len(factors)
    for profile in P.EXPERIMENTS:
        assert profile.factor and profile.factor != "none"
        assert any((profile.prompt, profile.request, profile.schema,
                    profile.runtime)), profile.id


def _plan1_sample():
    return {
        "title": "T",
        "endpoint": "Serve dinner.",
        "steps": [
            {"step_index": 1, "image_subject": "Bowl of dough",
             "content_goal": "How is it kneaded?"},
            {"step_index": 2, "image_subject": "Baked loaf",
             "content_goal": "How is it served?"},
        ],
    }


def _plan2_sample():
    return {
        "visual_style": {"medium": "photo", "color_treatment": "warm",
                         "lighting_character": "soft",
                         "rendering_finish": "clean"},
        "steps": [
            {"step_index": 1, "image_prompt": "hands kneading dough",
             "content": "Knead for ten minutes."},
            {"step_index": 2, "image_prompt": "sliced loaf on a board",
             "content": "Cool before slicing."},
        ],
    }


def test_plan1_variants_drop_the_field_and_keep_the_rest():
    no_endpoint = VP.plan1_no_endpoint()
    assert '"endpoint"' not in no_endpoint.split("Output exactly:")[1]
    assert "image_subject" in no_endpoint and "content_goal" in no_endpoint
    assert no_endpoint != c.PLAN1_PROMPT

    no_goal = VP.plan1_no_content_goal()
    goal_schema = no_goal.split("Output exactly:")[1]
    assert '"content_goal"' not in goal_schema
    assert '"step_index"' in goal_schema and '"image_subject"' in goal_schema
    assert no_goal != c.PLAN1_PROMPT


def test_plan2_variants_drop_the_field_and_keep_the_rest():
    single = VP.plan2_single_text_field()
    assert "- text:" in single
    assert "- image_prompt: one complete" not in single

    no_style = VP.plan2_no_visual_style()
    assert '"visual_style"' not in no_style
    assert "- image_prompt: one complete" in no_style


def _filled(prompt: str) -> str:
    return prompt.replace("{task}", "TASK").replace(
        "{steps_summary}", "STEPS").replace(
        "{endpoint}", "ENDPOINT").replace(
        "{step_indexes}", "[1,2]").replace(
        "{step_semantic_entries}", "[]").replace(
        "{step_condition_entries}", "{}").replace(
        "{step_checklist_entries}", "c").replace(
        "{semantic_subjects}", "SUBJECTS")


PATCH_PROBE = r"""
import json, sys
sys.path.insert(0, {root!r})
from ablation import patches
variant = {variant!r}
profile = patches.apply(variant)
import core.contract as contract
import core.serialization as serialization
import core.render_inputs as render_inputs
import baselines.umm_uniweaver_research as research

plan1 = {plan1!r}
plan2 = {plan2!r}
result = {{"applied": patches.applied()}}

if variant == "D3":
    result["no_patches"] = patches.applied() == []
    result["prompt_is_production"] = contract.PLAN1_PROMPT == {plan1_prompt!r}
elif variant in ("O32", "F1", "F2", "F3"):
    stage = {{"O32": "plan3",
              "F1": "plan2", "F2": "plan2", "F3": "plan2"}}[variant]
    request = contract.build_plan_request(
        stage, "TASK", {{"plan1": plan1, "plan2": plan2}})
    result["request"] = request
    result["request_head"] = request[:400]
elif variant in ("F5", "F5b"):
    projected = serialization.project_runtime_plan(
        {{"plan1": plan1, "plan2": plan2,
          "plan3": {{"condition_bank": {{"items": [
              {{"key": "core", "type": "reference", "image_prompt": "x"}}]}},
                    "step_conditions": {{"step_1": ["core"], "step_2": []}}}},
          "plan4": {{"reviewer_checklist": ["step_1: shows core",
                                            "step_2: shows loaf"]}}}},
        "TASK", trajectory_design="uniweaver", contract_version="UniWeaver")
    result["refs"] = [s.get("condition_refs") for s in projected["steps"]]
    result["checklist"] = projected["reviewer_checklist"]
    result["note"] = projected.get("_ablation_note")
elif variant == "F4":
    result["prompt_changed"] = contract.PLAN2_PROMPT != {plan2_prompt!r}
print(json.dumps(result))
"""


@functools.cache
def _probe(variant: str) -> dict:
    code = PATCH_PROBE.format(
        root=str(PROJECT_ROOT), variant=variant,
        plan1=_plan1_sample(), plan2=_plan2_sample(),
        plan1_prompt=c.PLAN1_PROMPT, plan2_prompt=c.PLAN2_PROMPT)
    completed = subprocess.run([sys.executable, "-c", code],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("variant", EXPERIMENT_IDS)
def test_every_variant_patches_something(variant):
    """Every experiment must rebind at least one production name."""
    assert _probe(variant)["applied"], variant


# ``run_variant.py`` forwards all production CLI arguments after consuming its
# own flags. This probe records the forwarded argument vector.
_ARGV_PROBE = r"""
import json, runpy, sys
sys.path.insert(0, {root!r})
def fake_run_module(name, run_name=None):
    # ``SystemExit`` unwinds past the caller, so report from inside the stub.
    print("ARGV " + json.dumps(list(sys.argv[1:])))
    raise SystemExit(0)
runpy.run_module = fake_run_module
from ablation import run_variant
run_variant.main(["--variant", {variant!r}, "--backend", "bagel",
                  "--protocol", "uniweaver", "--dataset", "single",
                  "--task-key", "task_001",
                  "--output-dir", {output_dir!r}])
"""


def _forwarded_argv(tmp_path, variant):
    completed = subprocess.run(
        [sys.executable, "-c", _ARGV_PROBE.format(
            root=str(PROJECT_ROOT), variant=variant,
            output_dir=str(tmp_path / variant))],
        capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT))
    assert completed.returncode == 0, completed.stderr[-2000:]
    lines = [line for line in completed.stdout.splitlines()
             if line.startswith("ARGV ")]
    assert lines, completed.stdout[-2000:]
    return json.loads(lines[-1][len("ARGV "):])


def test_o32_gives_plan3_subjects_instead_of_written_prompts():
    request = _probe("O32")["request"]
    assert "Bowl of dough" in request
    assert "hands kneading dough" not in request


def test_f5_routes_every_reference_to_every_step():
    result = _probe("F5")
    assert result["refs"] == [["core"], ["core"]]


def test_f5b_removes_references_entirely():
    result = _probe("F5b")
    assert result["refs"] == [[], []]


# field relaxations on the production messages
PLAN1_MISSING_ENDPOINT = {
    "title": "T", "steps": [
        {"step_index": 1, "image_subject": "A", "content_goal": "q"},
        {"step_index": 2, "image_subject": "B", "content_goal": "q"}]}
PLAN1_MISSING_CONTENT_GOAL = {
    "title": "T", "endpoint": "E", "steps": [
        {"step_index": 1, "image_subject": "A"},
        {"step_index": 2, "image_subject": "B"}]}
PLAN2_MISSING_VISUAL_STYLE = {"steps": [
    {"step_index": 1, "image_prompt": "p", "content": "c"},
    {"step_index": 2, "image_prompt": "p", "content": "c"}]}
PLAN2_MERGED_TEXT = {"steps": [
    {"step_index": 1, "text": "one sentence"},
    {"step_index": 2, "text": "one sentence"}]}

_RELAXATION_PROBE = r"""
import json, sys
sys.path.insert(0, {root!r})
from ablation import patches
patches.apply({variant!r})
from core import validation
plan1 = {plan1!r}
plan2 = {plan2!r}
print(json.dumps({{
    "no_endpoint": validation.plan1_output_issues("task", {p1_no_endpoint!r}),
    "no_content_goal": validation.plan1_output_issues(
        "task", {p1_no_content_goal!r}),
    "no_visual_style": validation.plan2_output_issues(
        plan1, {p2_no_visual_style!r}),
    "merged_text": validation.plan2_output_issues(plan1, {p2_merged_text!r}),
}}))
"""


def _relaxation_probe(variant):
    completed = subprocess.run(
        [sys.executable, "-c", _RELAXATION_PROBE.format(
            root=str(PROJECT_ROOT), variant=variant,
            plan1=_plan1_sample(), plan2=_plan2_sample(),
            p1_no_endpoint=PLAN1_MISSING_ENDPOINT,
            p1_no_content_goal=PLAN1_MISSING_CONTENT_GOAL,
            p2_no_visual_style=PLAN2_MISSING_VISUAL_STYLE,
            p2_merged_text=PLAN2_MERGED_TEXT,
        )],
        capture_output=True, text=True, check=True, cwd=str(PROJECT_ROOT))
    return json.loads(completed.stdout)


def _mentions(issues, needle):
    return [issue for issue in issues if needle in issue]


# the runner's own stage predicate must see the variant's schema
_STAGE_PREDICATE_PROBE = r"""
import json, sys
sys.path.insert(0, {root!r})
from ablation import patches
patches.apply({variant!r})
from core import runner
response = {response!r}
def send(**kwargs):
    return response
def decode(raw):
    return {{"parsed": json.loads(raw), "parse_method": "exact"}}
def strict_predicate(parsed):
    steps = parsed.get("steps") if isinstance(parsed, dict) else None
    return bool(steps) and all(
        isinstance(step.get("content_goal"), str) for step in steps)
out = runner.run_json_text_stage(
    "req", "plan1", 32, send, decode, ("steps",),
    parsed_is_valid=strict_predicate)
print("RESULT " + json.dumps(
    {{"parsed": out["parsed"] is not None, "method": out["parse_method"]}}))
"""

_F2_PLAN1 = ('{"title":"T","endpoint":"E","steps":['
             '{"step_index":1,"image_subject":"A"},'
             '{"step_index":2,"image_subject":"B"}]}')


def _stage_predicate_probe(variant, response):
    completed = subprocess.run(
        [sys.executable, "-c", _STAGE_PREDICATE_PROBE.format(
            root=str(PROJECT_ROOT), variant=variant, response=response)],
        capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT))
    lines = [line for line in completed.stdout.splitlines()
             if line.startswith("RESULT ")]
    assert lines, completed.stdout[-2000:] + completed.stderr[-2000:]
    return json.loads(lines[-1][len("RESULT "):])


# run_variant must leave output-root creation to the runner
_RUNNER_HARNESS = r"""
import json, runpy, sys
from pathlib import Path
sys.path.insert(0, {root!r})
output_dir = Path({output_dir!r})
observed = {{}}
def fake_run_module(name, run_name=None):
    # run_umm.py: refuse if the root already exists, then create it.
    observed["existed_before_runner"] = output_dir.exists()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output root: {{output_dir}}")
    output_dir.mkdir(parents=True)
    if {fail!r}:
        raise RuntimeError("simulated task failure")
runpy.run_module = fake_run_module
from ablation import run_variant
try:
    run_variant.main(["--variant", {variant!r}, "--backend", "bagel",
                      "--protocol", "uniweaver", "--dataset", "bench",
                      "--output-dir", str(output_dir)])
except BaseException as error:
    observed["error"] = type(error).__name__
observed["existed_after_runner"] = output_dir.exists()
print("OBSERVED " + json.dumps(observed))
"""


def _run_harness(tmp_path, variant, fail=False):
    output_dir = tmp_path / f"abl_{variant}"
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER_HARNESS.format(
            root=str(PROJECT_ROOT), output_dir=str(output_dir),
            variant=variant, fail=fail)],
        capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT))
    lines = [line for line in completed.stdout.splitlines()
             if line.startswith("OBSERVED ")]
    assert lines, completed.stdout[-3000:] + completed.stderr[-2000:]
    return output_dir, json.loads(lines[-1][len("OBSERVED "):])


# every variant's runner start, in a single interpreter
_ALL_VARIANTS_HARNESS = r"""
import json, runpy, sys, tempfile
from pathlib import Path
sys.path.insert(0, {root!r})
from ablation import run_variant

results = {{}}
for variant in {variants!r}:
    output_dir = Path(tempfile.mkdtemp()) / ("abl_" + variant)
    observed = {{}}
    def fake_run_module(name, run_name=None, output_dir=output_dir,
                        observed=observed):
        # run_umm.py: refuse if the root already exists, then create it.
        observed["existed_before_runner"] = output_dir.exists()
        if output_dir.exists():
            raise FileExistsError(str(output_dir))
        output_dir.mkdir(parents=True)
    runpy.run_module = fake_run_module
    try:
        run_variant.main(["--variant", variant, "--backend", "bagel",
                          "--protocol", "uniweaver", "--dataset", "bench",
                          "--output-dir", str(output_dir)])
    except BaseException as error:
        observed["error"] = type(error).__name__
    observed["existed_after_runner"] = output_dir.exists()
    results[variant] = observed
print("ALLVARIANTS " + json.dumps(results))
"""


def _all_variant_harness() -> dict:
    completed = subprocess.run(
        [sys.executable, "-c", _ALL_VARIANTS_HARNESS.format(
            root=str(PROJECT_ROOT), variants=list(EXPERIMENT_IDS))],
        capture_output=True, text=True, check=False, cwd=str(PROJECT_ROOT))
    lines = [line for line in completed.stdout.splitlines()
             if line.startswith("ALLVARIANTS ")]
    assert lines, completed.stdout[-3000:] + completed.stderr[-2000:]
    return json.loads(lines[-1][len("ALLVARIANTS "):])


# interpreter diagnostics


# patch state isolation
def test_restore_puts_the_production_prompts_back():
    from ablation import patches
    from core import contract

    production = contract.PLAN1_PROMPT
    patches.apply("F2")
    assert contract.PLAN1_PROMPT != production
    patches.restore()
    assert contract.PLAN1_PROMPT == production
    assert patches.applied() == []


def test_applying_a_second_variant_does_not_stack_on_the_first():
    from ablation import patches
    from core import contract

    plan1 = contract.PLAN1_PROMPT
    plan2 = contract.PLAN2_PROMPT
    patches.apply("F2")          # rebinds PLAN1_PROMPT
    patches.apply("F4")          # rebinds PLAN2_PROMPT only
    # F2's rebind must be gone: otherwise this "F4 run" carries F2 as well.
    assert contract.PLAN1_PROMPT == plan1
    assert contract.PLAN2_PROMPT != plan2


# production must not depend on the ablation layer
def test_production_sources_never_import_or_launch_the_ablation_layer():
    from ablation import isolation as ISO

    offenders = []
    for directory in ISO.PRODUCTION_DIRS:
        root = PROJECT_ROOT / directory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix not in (".py", ".sh") or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ISO.mentions_ablation(text):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []
