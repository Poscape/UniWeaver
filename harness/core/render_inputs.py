"""Compile Planner artifacts into the only inputs allowed to affect rendering."""
from dataclasses import asdict, dataclass
from types import MappingProxyType

from .contract import _stringify
from .schema import ALL_LINEAGE_PATHS, CONDITION_TYPE, VISUAL_STYLE_FIELDS


@dataclass(frozen=True)
class StepRenderInput:
    semantic_image_prompt: str
    model_image_prompt: str
    content: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ConditionRenderInput:
    key: str
    item_type: str
    semantic_image_prompt: str
    model_image_prompt: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StepExecutionInput:
    prompt: str
    model_inputs: tuple
    condition_refs: tuple
    image_shape: tuple = (1024, 1024)


@dataclass(frozen=True)
class ConditionExecutionInput:
    prompt: str
    image_shape: tuple = (512, 512)


STEP_IMAGE_ENVELOPE = """###Human:
Image Generation. Your task is NOT to output text or JSON. Generate the image.

Image prompt: {image_prompt}

Generate the image. Show ONLY what is described in the image_prompt."""

CONDITION_EXECUTION_ENVELOPE = """###Human:
Image Generation. Generate one standalone reference image for a reusable visual target.

Element type: {item_type}
Key: {key}
Image prompt: {image_prompt}

The only supported type is "reference". Isolate the visible target on a plain neutral background.
Generate only the image; do not output analysis or text."""


# Semantic consumers can affect generation or decisions. Observers may only
# persist or display a field. This distinction prevents "present" from being
# mistaken for "used".
FIELD_LINEAGE = MappingProxyType({
    'task_prompt': {
        'producer': 'user_or_dataset',
        'consumers': ('plan1', 'plan2', 'plan3'),
        'observers': ('trace', 'ui'),
        'effect': 'immutable task scope'},
    'plan1.title': {
        'producer': 'plan1', 'consumers': ('report_renderer',),
        'observers': ('runtime_plan', 'trace'),
        'effect': 'report label only'},
    'plan1.endpoint': {
        'producer': 'plan1', 'consumers': ('plan2',),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'freezes the requested stopping boundary'},
    'plan1.steps[].image_subject': {
        'producer': 'plan1', 'consumers': ('plan2', 'plan3'),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': (
            'freezes the concrete visible subject authored by plan2 and '
            'exposes stable physical identity to plan3')},
    'plan1.steps[].content_goal': {
        'producer': 'plan1', 'consumers': ('plan2',),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'freezes the prose contribution and factual boundary'},
    'plan1.steps[].step_index': {
        'producer': 'plan1', 'consumers': ('plan2',),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'stable ordered Step identity'},
    'plan2.visual_style.*': {
        'producer': 'plan2',
        'consumers': ('compile_step_render_input',
                      'compile_condition_render_input'),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'shared rendering treatment; never scene semantics'},
    'plan2.steps[].image_prompt': {
        'producer': 'plan2',
        'consumers': ('plan3', 'plan4', 'compile_step_render_input'),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'final visible Step semantics'},
    'plan2.steps[].step_index': {
        'producer': 'plan2',
        'consumers': ('plan3', 'plan4', 'runtime_plan'),
        'observers': ('trace', 'ui'),
        'effect': 'routes final semantics and checks to one Step'},
    'plan2.steps[].content': {
        'producer': 'plan2',
        'consumers': ('compile_step_render_input', 'report_renderer'),
        'observers': ('runtime_plan', 'trace'),
        'effect': 'final report prose'},
    'step_cache': {
        'producer': 'compile_step_render_input',
        'consumers': ('restore_step_render_input',),
        'observers': ('trace', 'ui'),
        'effect': 'replays one previously compiled Step input without recomposition'},
    'reviewer.step_feedback': {
        'producer': 'image_reviewer',
        'consumers': ('compile_step_execution_prompt',),
        'observers': ('trace', 'ui'),
        'effect': 'retry-only instruction appended after the frozen model prompt'},
    'runtime.condition_images': {
        'producer': 'condition_generator_or_explicit_import',
        'consumers': ('compile_step_execution_input',),
        'observers': ('trace', 'ui'),
        'effect': 'reference pixels interleaved before the Step prompt'},
    'runtime.image_seed': {
        'producer': 'explicit_runtime_configuration',
        'consumers': ('per_request_initial_noise',),
        'observers': ('trace', 'ui'),
        'effect': 'controls independent initial noise without global RNG mutation'},
    'runtime.inference_hyperparameters': {
        'producer': 'explicit_runtime_configuration',
        'consumers': ('image_model',),
        'observers': ('trace',),
        'effect': 'controls image-model execution; target shape is compiler-owned'},
    'step_execution.image': {
        'producer': 'step_image_model',
        'consumers': ('image_reviewer', 'report_renderer', 'persistence'),
        'observers': ('trace',),
        'effect': 'final visual output for one report unit'},
    'condition_execution.image': {
        'producer': 'condition_image_model_or_explicit_import',
        'consumers': ('compile_step_execution_input', 'persistence'),
        'observers': ('trace', 'ui'),
        'effect': 'isolated reusable reference pixels'},
    'plan3.condition_bank.items[].key': {
        'producer': 'plan3',
        'consumers': ('compile_condition_render_input', 'step_reference_router'),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'stable identity of an optional reference'},
    'plan3.condition_bank.items[].type': {
        'producer': 'plan3',
        'consumers': ('compile_condition_render_input',),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'selects the Condition rendering envelope'},
    'plan3.condition_bank.items[].image_prompt': {
        'producer': 'plan3',
        'consumers': ('condition_text_reviewer',
                      'compile_condition_render_input'),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'visible semantics of the isolated reference'},
    'plan3.step_conditions': {
        'producer': 'plan3',
        'consumers': ('runtime_plan', 'step_reference_router'),
        'observers': ('trace', 'ui'),
        'effect': 'routes declared reference images to Steps'},
    'plan4.reviewer_checklist': {
        'producer': 'plan4', 'consumers': ('image_reviewer',),
        'observers': ('runtime_plan', 'trace', 'ui'),
        'effect': 'controls image review and retry decisions'},
    'runtime_plan.steps[].condition_refs': {
        'producer': 'runtime_projection',
        'consumers': ('compile_step_execution_input',),
        'observers': ('trace', 'ui'),
        'effect': 'sole runtime representation of projected Plan3 routing'},
})


def _format_visual_style(visual_style):
    """Format the closed rendering-style schema as one natural-language clause."""
    if not isinstance(visual_style, dict):
        return ''
    labels = {
        'medium': 'medium',
        'color_treatment': 'color',
        'lighting_character': 'lighting',
        'rendering_finish': 'finish'}
    parts = [
        f'{labels[field]}: {_stringify(visual_style.get(field)).strip()}'
        for field in VISUAL_STYLE_FIELDS
        if _stringify(visual_style.get(field)).strip()]
    return '; '.join(parts)


def _apply_visual_style(image_prompt, visual_style):
    """Compose one T2I instruction from semantics and shared style."""
    prompt = _stringify(image_prompt).strip()
    style = _format_visual_style(visual_style)
    if not style:
        return prompt
    style_clause = f'Shared visual style: {style}'
    if prompt.endswith(style_clause):
        return prompt
    return f'{prompt}\n\n{style_clause}'


def compile_step_render_input(plan, step):
    """Compile one Plan2 Step without adding or rewriting semantics."""
    if not isinstance(plan, dict):
        raise TypeError('Plan2 render source must be an object: got '
                        f'{type(plan).__name__}')
    if not isinstance(step, dict):
        raise TypeError('Plan2.steps[] render source must be an object: got '
                        f'{type(step).__name__}')
    step_index = step.get('step_index', '?')
    location = f'Plan2.steps[{step_index}]'
    image_prompt = step.get('image_prompt')
    content = step.get('content')
    if 'image_prompt' not in step:
        raise ValueError(f'{location}.image_prompt is missing')
    if not isinstance(image_prompt, str):
        raise ValueError(  # noqa: TRY004 - contract violations use one public error type
            f'{location}.image_prompt must be a string: '
            f'got {type(image_prompt).__name__}')
    if not image_prompt.strip():
        raise ValueError(f'{location}.image_prompt is empty or whitespace')
    if 'content' not in step:
        raise ValueError(f'{location}.content is missing')
    if not isinstance(content, str):
        raise ValueError(  # noqa: TRY004 - contract violations use one public error type
            f'{location}.content must be a string: got {type(content).__name__}')
    if not content.strip():
        raise ValueError(f'{location}.content is empty or whitespace')
    semantic_image_prompt = image_prompt.strip()
    return StepRenderInput(
        semantic_image_prompt=semantic_image_prompt,
        model_image_prompt=_apply_visual_style(
            semantic_image_prompt, plan.get('visual_style', {})),
        content=content.strip())


def compile_condition_render_input(plan, item):
    """Compile one Plan3 reference using the explicitly shared style."""
    if not isinstance(plan, dict):
        raise TypeError(
            'Plan3 render source must be an object: '
            f'got {type(plan).__name__}')
    if not isinstance(item, dict):
        raise TypeError(
            'Plan3.condition_bank.items[] must be an object: '
            f'got {type(item).__name__}')
    key = item.get('key')
    item_type = item.get('type')
    image_prompt = item.get('image_prompt')
    item_label = f'Plan3.condition_bank key={key!r}'
    if 'key' not in item:
        raise ValueError(f'{item_label}: key field is missing')
    if not isinstance(key, str):
        raise ValueError(  # noqa: TRY004 - contract violations use one public error type
            f'{item_label}: key must be a string: got {type(key).__name__}')
    if not key.strip():
        raise ValueError(f'{item_label}: key is empty or whitespace')
    if 'type' not in item:
        raise ValueError(f'{item_label}: type field is missing')
    if not isinstance(item_type, str):
        raise ValueError(  # noqa: TRY004 - contract violations use one public error type
            f'{item_label}: type must be a string: got '
            f'{type(item_type).__name__}')
    if not item_type.strip():
        raise ValueError(f'{item_label}: type is empty or whitespace')
    if item_type.strip() != CONDITION_TYPE:
        raise ValueError(
            f'{item_label}: unsupported Condition type={item_type.strip()!r}; '
            f'expected {CONDITION_TYPE!r}')
    if 'image_prompt' not in item:
        raise ValueError(f'{item_label}.image_prompt field is missing')
    if not isinstance(image_prompt, str):
        raise ValueError(  # noqa: TRY004 - contract violations use one public error type
            f'{item_label}.image_prompt must be a string: '
            f'got {type(image_prompt).__name__}')
    if not image_prompt.strip():
        raise ValueError(
            f'{item_label}.image_prompt is empty or whitespace')
    semantic_image_prompt = image_prompt.strip()
    return ConditionRenderInput(
        key=key.strip(), item_type=item_type.strip(),
        semantic_image_prompt=semantic_image_prompt,
        model_image_prompt=_apply_visual_style(
            semantic_image_prompt, plan.get('visual_style', {})))


def restore_step_render_input(cached):
    """Restore an exact compiled Step input; never infer or repair missing fields."""
    if not isinstance(cached, dict):
        raise TypeError('Step render cache must be an object')
    values = {}
    for field in ('semantic_image_prompt', 'model_image_prompt', 'content'):
        if field not in cached:
            raise ValueError(f'Step render cache field {field!r} is missing')
        value = cached[field]
        if not isinstance(value, str):
            raise ValueError(  # noqa: TRY004 - contract violations use one public error type
                f'Step render cache field {field!r} must be a string: '
                f'got {type(value).__name__}')
        if not value.strip():
            raise ValueError(
                f'Step render cache field {field!r} is empty or whitespace')
        values[field] = value.strip()
    return StepRenderInput(**values)


def compile_step_execution_prompt(render_input, feedback=''):
    """Compile the exact final text sent to the Step image model."""
    if not isinstance(render_input, StepRenderInput):
        raise TypeError('step execution requires StepRenderInput')
    prompt = render_input.model_image_prompt
    if feedback:
        prompt = f'{prompt}\n\nCRITICAL FIXES NEEDED: {str(feedback).strip()}'
    return STEP_IMAGE_ENVELOPE.replace('{image_prompt}', prompt)


def compile_step_execution_input(
        render_input, condition_images, condition_refs, feedback=''):
    """Compile the complete ordered multimodal input for one Step model call."""
    images = condition_images if isinstance(condition_images, dict) else {}
    model_inputs = []
    used_refs = []
    for key in dict.fromkeys(condition_refs or []):
        image = images.get(key)
        if image is None or not hasattr(image, 'size') or image.size[0] <= 0:
            continue
        model_inputs.extend((compile_condition_reference_instruction(key), image))
        used_refs.append(key)
    prompt = compile_step_execution_prompt(render_input, feedback)
    model_inputs.append(prompt)
    return StepExecutionInput(
        prompt=prompt, model_inputs=tuple(model_inputs),
        condition_refs=tuple(used_refs))


def compile_condition_execution_input(render_input):
    """Compile the exact final text and fixed target shape for one Condition."""
    if not isinstance(render_input, ConditionRenderInput):
        raise TypeError('condition execution requires ConditionRenderInput')
    prompt = (CONDITION_EXECUTION_ENVELOPE
              .replace('{item_type}', render_input.item_type)
              .replace('{key}', render_input.key)
              .replace('{image_prompt}', render_input.model_image_prompt))
    return ConditionExecutionInput(prompt=prompt)


def compile_condition_reference_instruction(key):
    """Compile the instruction paired with one Condition reference image."""
    key = str(key or '').strip()
    if not key:
        raise ValueError('Condition reference instruction requires a key')
    return (
        f'[Appearance/structure reference: {key}] Preserve its stable intrinsic '
        'appearance or structure. The current step controls state, action, and scene.')


def validate_field_lineage():
    """Fail when schema fields or registered runtime effects lack ownership."""
    registered = set(FIELD_LINEAGE)
    missing = sorted(ALL_LINEAGE_PATHS - registered)
    unexpected = sorted(registered - ALL_LINEAGE_PATHS)
    if missing or unexpected:
        raise ValueError(
            f'Field lineage does not match closed schema: '
            f'missing={missing}, unexpected={unexpected}')
    malformed = []
    for field, lineage in FIELD_LINEAGE.items():
        if set(lineage) != {'producer', 'consumers', 'observers', 'effect'}:
            malformed.append(field)
            continue
        if (not isinstance(lineage['producer'], str)
                or not lineage['producer']
                or not isinstance(lineage['consumers'], tuple)
                or not isinstance(lineage['observers'], tuple)
                or not isinstance(lineage['effect'], str)
                or not lineage['effect']
                or set(lineage['consumers']) & set(lineage['observers'])):
            malformed.append(field)
    if malformed:
        raise ValueError(f'Invalid field lineage entries: {sorted(malformed)}')
    return True
