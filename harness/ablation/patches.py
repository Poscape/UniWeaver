"""Variant patches applied to the production modules at runtime.

``apply(profile)`` rebinds module attributes; no file under ``core/`` or
``baselines/`` is edited, and profile ``D3`` rebinds nothing.
"""
from __future__ import annotations

import copy
import importlib
import sys

from ablation import profiles as P
from ablation import prompts as VP

_APPLIED: list[str] = []
# module name -> why importing it failed, so a missing dependency is reported
# as itself instead of as a failed rebind.
_IMPORT_ERRORS: dict[str, str] = {}
# (module, attribute, original value) for every rebind still in effect.
_ORIGINALS: list[tuple[object, str, object]] = []


def applied() -> list[str]:
    return list(_APPLIED)


def restore() -> None:
    """Undo every rebind ``apply()`` made, newest first."""
    while _ORIGINALS:
        module, attribute, original = _ORIGINALS.pop()
        setattr(module, attribute, original)
    _APPLIED.clear()
    _IMPORT_ERRORS.clear()


# --------------------------------------------------------------------------- #
# patch plumbing
# --------------------------------------------------------------------------- #
def _target_module(module_name: str):
    """Import the module if the patch has to land before anything else did."""
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        _IMPORT_ERRORS[module_name] = f"{type(error).__name__}: {error}"
        return None


def _rebind(module_name: str, attribute: str, value, why: str) -> None:
    module = _target_module(module_name)
    if module is None or not hasattr(module, attribute):
        return
    # Remember the production value so ``restore()`` can put it back. Tests run
    # several variants in one process, so restoring each rebind keeps the suite
    # independent of execution order.
    _ORIGINALS.append((module, attribute, getattr(module, attribute)))
    setattr(module, attribute, value)
    _APPLIED.append(f"{module_name}.{attribute} <- {why}")


def _rebind_all(attribute: str, value, why: str, *module_names: str) -> None:
    """Rebind a name in its definition module and in every consumer."""
    for module_name in module_names:
        _rebind(module_name, attribute, value, why)


# Stage/field contract relaxations run the production validator on a corrected
# copy of the stage output, never by matching validator message text.
_RELAXED_VALUE = "ablation-relaxed"


def _visual_style_stub() -> dict:
    """A complete visual_style block the production validator accepts."""
    from core.schema import VISUAL_STYLE_FIELDS

    return {name: _RELAXED_VALUE for name in VISUAL_STYLE_FIELDS}


_RELAXATIONS: dict[str, dict[str, object]] = {
    "plan1:endpoint": {
        "validator": "plan1_output_issues",
        "top_level": {"endpoint": _RELAXED_VALUE},
    },
    "plan1:steps[].content_goal": {
        "validator": "plan1_output_issues",
        "step": {"content_goal": _RELAXED_VALUE},
    },
    "plan2:visual_style": {
        "validator": "plan2_output_issues",
        "top_level_keys": ("visual_style",),
    },
    "plan2:steps[].image_prompt": {
        "validator": "plan2_output_issues",
        "step": {"image_prompt": _RELAXED_VALUE,
                 "content": _RELAXED_VALUE},
        # F3 replaces both fields with one ``text`` field; the production
        # closed-field check would otherwise report it as unexpected.
        "drop_step": ("text",),
    },
}


# ``parsed_is_valid`` closes over the full production schema, so a variant that
# drops a field must wrap it.
_STAGE_PREDICATES: dict[str, dict[str, object]] = {
    "plan1:steps[].content_goal": {
        "stage": "plan1",
        # ``step_index`` is an integer in the production schema, so it is
        # checked for presence while these must be non-empty strings.
        "required_step_text_fields": ("image_subject",),
    },
}


def _relaxed_stage_predicate(original, spec):
    """Accept the variant's own step shape, then fall back to production."""
    required = tuple(str(f) for f in spec["required_step_text_fields"])

    def predicate(parsed):
        steps = parsed.get("steps") if isinstance(parsed, dict) else None
        if isinstance(steps, list) and steps and all(
                isinstance(step, dict)
                and step.get("step_index") is not None
                and all(isinstance(step.get(field), str)
                        and step[field].strip() for field in required)
                for step in steps):
            return True
        return original(parsed)

    predicate.__ablation__ = True
    return predicate


def _apply_stage_predicate_relaxations(profile: P.Profile) -> None:
    specs = {
        str(_STAGE_PREDICATES[key]["stage"]): _STAGE_PREDICATES[key]
        for key in profile.schema if key in _STAGE_PREDICATES}
    if not specs:
        return
    from core import runner

    original = runner.run_json_text_stage

    def run_json_text_stage(user_text, label, max_tokens, send_backend, decoder,
                            required_keys, parsed_is_valid=None, **kwargs):
        spec = specs.get(label)
        if spec is not None and parsed_is_valid is not None:
            parsed_is_valid = _relaxed_stage_predicate(parsed_is_valid, spec)
        return original(
            user_text, label, max_tokens, send_backend, decoder,
            required_keys, parsed_is_valid=parsed_is_valid, **kwargs)

    run_json_text_stage.__ablation__ = True
    _rebind("core.runner", "run_json_text_stage", run_json_text_stage,
            "stage_predicate:" + ",".join(sorted(specs)))


def _relaxed_stage_output(stage_output, spec):
    """Copy ``stage_output`` with ``spec`` applied, leaving the original alone."""
    if not isinstance(stage_output, dict):
        return stage_output
    relaxed = copy.deepcopy(stage_output)
    for field, injected in dict(spec.get("top_level") or {}).items():
        relaxed[field] = copy.deepcopy(injected)
    for field in tuple(spec.get("top_level_keys") or ()):
        relaxed[field] = _visual_style_stub()
    injected = dict(spec.get("step") or {})
    dropped = tuple(spec.get("drop_step") or ())
    if (injected or dropped) and isinstance(relaxed.get("steps"), list):
        steps = []
        for step in relaxed["steps"]:
            if not isinstance(step, dict):
                steps.append(step)
                continue
            step = dict(step)
            for field in dropped:
                step.pop(field, None)
            for field, value in injected.items():
                step[field] = copy.deepcopy(value)
            steps.append(step)
        relaxed["steps"] = steps
    return relaxed


def _relax_validator(validator, spec):
    """Wrap a stage validator so it sees the variant's own schema."""
    kwarg = spec["validator"].replace("_output_issues", "")

    def wrapped(*args, **kwargs):
        args = list(args)
        if len(args) > 1:
            args[1] = _relaxed_stage_output(args[1], spec)
        elif kwarg in kwargs:
            kwargs = dict(kwargs)
            kwargs[kwarg] = _relaxed_stage_output(kwargs[kwarg], spec)
        return validator(*args, **kwargs)

    wrapped.__ablation__ = True
    return wrapped


def _apply_schema_relaxations(profile: P.Profile) -> None:
    from core import validation

    for key in profile.schema:
        spec = _RELAXATIONS.get(key)
        if spec is None:
            continue
        name = str(spec["validator"])
        base = getattr(validation, name, None)
        if base is None:
            continue
        _rebind_all(name, _relax_validator(base, spec), f"relax:{key}",
                    "core.validation", "core.runner")


# --------------------------------------------------------------------------- #
# plan1 / plan2 prompt+request layer
# --------------------------------------------------------------------------- #
def _apply_prompts(profile: P.Profile) -> None:
    from core import contract

    overrides = {
        "no_endpoint": ("plan1", VP.plan1_no_endpoint),
        "no_content_goal": ("plan1", VP.plan1_no_content_goal),
        "single_text_field": ("plan2", VP.plan2_single_text_field),
        "no_visual_style": ("plan2", VP.plan2_no_visual_style),
    }
    for key, (stage, builder) in overrides.items():
        if profile.prompt.get(stage) != key:
            continue
        attr = f"{stage.upper()}_PROMPT"
        _rebind("core.contract", attr, builder(), f"prompt:{key}")

    plan2_override = None
    plan3_override = None
    if profile.id == "F1":
        plan2_override = VP.plan2_request_without_endpoint
    elif profile.id == "F2":
        plan2_override = VP.plan2_request_subject_only
    elif profile.id == "F3":
        plan2_override = VP.plan2_request_single_text_field
    if profile.request.get("plan3") == "plan1_only":
        plan3_override = VP.plan3_request_from_plan1

    if not any((plan2_override, plan3_override)):
        return

    original = contract.build_plan_request

    def build_plan_request(stage, task_prompt, plan):
        blocks = plan if isinstance(plan, dict) else {}
        plan1 = blocks.get("plan1", {})
        plan2 = blocks.get("plan2", {})
        if stage == "plan2" and plan2_override is not None:
            return plan2_override(task_prompt, plan1)
        if stage == "plan3" and plan3_override is not None:
            return plan3_override(task_prompt, plan1, plan2)
        return original(stage, task_prompt, plan)

    build_plan_request.__ablation__ = True
    _rebind_all("build_plan_request", build_plan_request,
                "request:" + ",".join(sorted(profile.request)),
                "core.contract", "core.runner")


# --------------------------------------------------------------------------- #
# runtime projection: F5 / F5b
# --------------------------------------------------------------------------- #
def _apply_runtime_overrides(profile: P.Profile) -> None:
    runtime = profile.runtime
    if not any(key in runtime for key in ("conditions", "routing")):
        return
    from core import serialization

    original = serialization.project_runtime_plan

    def project_runtime_plan(plan_blocks, user_prompt="", **kwargs):
        projected = original(plan_blocks, user_prompt, **kwargs)
        if runtime.get("conditions") == "none":
            projected["condition_bank"] = {"items": []}
            for step in projected.get("steps", []):
                step["condition_refs"] = []
            projected["_ablation_note"] = "references disabled (F5b)"
        elif runtime.get("routing") == "all_steps":
            keys = [item.get("key") for item
                    in projected.get("condition_bank", {}).get("items", [])]
            for step in projected.get("steps", []):
                step["condition_refs"] = list(keys)
            projected["_ablation_note"] = "every reference routed to every step (F5)"
        return projected

    project_runtime_plan.__ablation__ = True
    _rebind_all("project_runtime_plan", project_runtime_plan,
                "runtime:" + ",".join(sorted(runtime)),
                "core.serialization", "baselines.umm_uniweaver_research",
                "baselines.umm_runner")


# --------------------------------------------------------------------------- #
# F3: one merged text field feeding both pixels and prose
# --------------------------------------------------------------------------- #
def _apply_merged_step_text_field(profile: P.Profile) -> None:
    if profile.runtime.get("merged_step_text_field") != "text":
        return
    from core import contract, render_inputs

    def merged_steps(plan2):
        """Give downstream stages the merged text under the production keys."""
        if not isinstance(plan2, dict):
            return plan2
        steps = plan2.get("steps")
        if not isinstance(steps, list):
            return plan2
        merged = dict(plan2)
        merged["steps"] = [
            {**step, "image_prompt": step["text"], "content": step["text"]}
            if isinstance(step, dict) and "text" in step else step
            for step in steps
        ]
        return merged

    original = render_inputs.compile_step_render_input

    def compile_step_render_input(plan, step):
        if isinstance(step, dict) and "text" in step:
            step = dict(step)
            step["image_prompt"] = step["text"]
            step["content"] = step["text"]
        return original(plan, step)

    original_plan3 = contract.build_plan3_request
    original_plan4 = contract.build_plan4_request

    def build_plan3_request(task_prompt, plan1, plan2):
        return original_plan3(task_prompt, plan1, merged_steps(plan2))

    def build_plan4_request(plan2):
        return original_plan4(merged_steps(plan2))

    for wrapper in (compile_step_render_input, build_plan3_request,
                    build_plan4_request):
        wrapper.__ablation__ = True
    _rebind_all("compile_step_render_input", compile_step_render_input,
                "merged_step_text_field:text",
                "core.render_inputs", "core.runner",
                "baselines.umm_uniweaver_research")
    _rebind_all("build_plan3_request", build_plan3_request,
                "merged_step_text_field:text", "core.contract", "core.runner")
    _rebind_all("build_plan4_request", build_plan4_request,
                "merged_step_text_field:text", "core.contract", "core.runner")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def expected_targets(profile: P.Profile) -> list[str]:
    """The ``module.attribute`` pairs this variant is required to rebind."""
    targets: list[str] = []
    for stage in profile.prompt:
        targets.append(f"core.contract.{stage.upper()}_PROMPT")
    if profile.request or profile.id in ("F1", "F2", "F3"):
        targets.extend(["core.contract.build_plan_request",
                        "core.runner.build_plan_request"])
    if profile.schema:
        for key in profile.schema:
            spec = _RELAXATIONS.get(key)
            if spec is None:
                continue
            name = str(spec["validator"])
            targets.extend([f"core.validation.{name}",
                            f"core.runner.{name}"])
    runtime = profile.runtime
    if any(key in _STAGE_PREDICATES for key in profile.schema):
        targets.append("core.runner.run_json_text_stage")
    if runtime.get("merged_step_text_field"):
        targets.extend([
            "core.render_inputs.compile_step_render_input",
            "core.runner.compile_step_render_input",
            "baselines.umm_uniweaver_research.compile_step_render_input",
            "core.contract.build_plan3_request",
            "core.contract.build_plan4_request",
        ])
    if any(key in runtime for key in ("conditions", "routing")):
        targets.extend([
            "core.serialization.project_runtime_plan",
            "baselines.umm_uniweaver_research.project_runtime_plan",
        ])
    return sorted(set(targets))


def apply(variant_id: str) -> P.Profile:
    profile = P.profile(variant_id)
    # Start from production so variants cannot stack in one interpreter.
    restore()
    if profile.id == "D3":
        return profile
    _apply_prompts(profile)
    _apply_schema_relaxations(profile)
    _apply_stage_predicate_relaxations(profile)
    _apply_runtime_overrides(profile)
    _apply_merged_step_text_field(profile)
    applied = set(_APPLIED)
    missing = [
        target for target in expected_targets(profile)
        if not any(entry.startswith(f"{target} <-") for entry in applied)]
    if missing:
        blocked = {
            module: reason for module, reason in _IMPORT_ERRORS.items()
            if any(target.startswith(f"{module}.") for target in missing)}
        if blocked:
            raise RuntimeError(
                f"{profile.id}: the patch layer could not import the modules "
                f"it has to patch ({blocked}); fix the environment before "
                f"running this variant.")
        raise RuntimeError(
            f"{profile.id}: the patch layer failed to rebind {missing}; this "
            "variant would not be a single-factor change. Refusing to run.")
    if not applied:
        raise RuntimeError(
            f"{profile.id}: no patch was applied; refusing to run a no-op "
            "variant.")
    return profile


def describe(variant_id: str) -> dict:
    profile = P.profile(variant_id)
    return {
        "variant": profile.id,
        "title": profile.title,
        "claim": profile.claim,
        "counted_as_experiment": profile.id != "D3",
        "prompt_overrides": dict(profile.prompt),
        "request_overrides": dict(profile.request),
        "schema_relaxations": list(profile.schema),
        "runtime": dict(profile.runtime),
        "notes": profile.notes,
    }
