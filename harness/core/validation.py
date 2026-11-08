"""Validation for the sole UniWeaver trajectory schema."""
import json
import re

from .contract import (
    PLANNER_SYSTEM,
    _stringify,
    build_plan1_request,
    build_plan_request,
    find_planner_placeholders,
)
from .decoding import _loads_with_duplicate_keys, condition_refs, read_cached_step_text
from .policy_registry import RENDER_ELIGIBLE_GATE
from .production import (
    planner_stage_config,
    require_trajectory_identity,
    trajectory_contract_view,
)
from .render_inputs import compile_step_render_input, restore_step_render_input
from .schema import (
    CONDITION_BANK_FIELDS,
    CONDITION_ITEM_FIELDS,
    CONDITION_TYPE,
    PLAN1_FIELDS,
    PLAN1_STEP_FIELDS,
    PLAN2_FIELDS,
    PLAN2_STEP_FIELDS,
    PLAN3_FIELDS,
    PLAN4_FIELDS,
    PLAN_STAGE_NAMES,
    VISUAL_STYLE_FIELDS,
)

_CONTENT_META = re.compile(
    r'\b(?:this (?:image|visual|prompt)|this step matters|needs? verification|'
    r'provided in the (?:text|content)|according to the visual|instructional tone|'
    r'prompt requirements?)\b', re.IGNORECASE)
_TEXT_READING = re.compile(
    r'\b(?:wording|spelled|letters?|words?|screen text|error code|departure time|'
    r'exact (?:text|label|signage|inscription)|'
    r'(?:readable|legible).{0,32}(?:text|label|signage|inscription|writing)|'
    r'(?:text|label|signage|inscription|writing).{0,32}(?:readable|legible))\b',
    re.IGNORECASE)
_CONDITION_KEY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$')

# Contract issues kept in the strict schema result that do not block render or
# execution: extra fields, an empty Plan1.title, and a single renderable unit.
# Zero units stays fatal.
NON_FATAL_CONTRACT_MESSAGE_MARKERS = (
    'unexpected fields',
    'title is missing or empty',
    'fewer than the minimum two',
)


def _is_non_fatal_contract_issue(issue):
    """Whether an issue belongs to the non-fatal contract marker set."""
    message = issue.get('message') if isinstance(issue, dict) else None
    return isinstance(message, str) and any(
        marker in message for marker in NON_FATAL_CONTRACT_MESSAGE_MARKERS)


def condition_item_is_executable(item):
    """A UniWeaver Condition is one isolated reusable reference."""
    return (
        isinstance(item, dict)
        and isinstance(item.get('key'), str)
        and bool(_CONDITION_KEY.fullmatch(item['key']))
        and item['key'] not in {'.', '..'}
        and item.get('type') == CONDITION_TYPE
        and isinstance(item.get('image_prompt'), str)
        and bool(item['image_prompt'].strip()))


def checklist_item_requires_text_reading(item):
    return isinstance(item, str) and bool(_TEXT_READING.search(item))


def json_only_response_issues(raw_output):
    text = _stringify(raw_output).strip()
    try:
        parsed, end = json.JSONDecoder().raw_decode(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ['response is not exactly one JSON object']
    if not isinstance(parsed, dict) or text[end:].strip():
        return ['response is not exactly one JSON object']
    try:
        _, duplicates = _loads_with_duplicate_keys(text)
    except (TypeError, ValueError):
        return ['response is not exactly one JSON object']
    return (['response contains duplicate JSON object keys; final value was used']
            if duplicates else [])


_TASK_SENTINEL = '\x00'


def task_text(trajectory):
    """Return the task statement recorded for one trajectory.

    A released corpus keeps no separate prompt field, so the statement is read
    back out of the recorded Plan1 user turn by removing the builder template.
    """
    if not isinstance(trajectory, dict):
        return ''
    prompt = trajectory.get('task_prompt')
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    attempts = trajectory.get('raw_attempts')
    if not isinstance(attempts, dict):
        return ''
    plan1 = attempts.get('plan1')
    if not isinstance(plan1, list) or not plan1 or not isinstance(plan1[0], dict):
        return ''
    raw = plan1[0].get('raw_input')
    if not isinstance(raw, str):
        return ''
    head, _, tail = build_plan1_request(_TASK_SENTINEL).partition(_TASK_SENTINEL)
    if not tail or not raw.endswith(tail):
        return ''
    body = raw[:len(raw) - len(tail)]
    start = body.find(head)
    if start < 0:
        return ''
    return body[start + len(head):].strip()


def plan1_output_issues(task_prompt, plan1):
    if not isinstance(plan1, dict):
        return ['plan1 is not a JSON object']
    issues = []
    if not isinstance(plan1.get('title'), str) or not plan1['title'].strip():
        issues.append('title is missing or empty')
    else:
        echoed = find_planner_placeholders(plan1['title'])
        if echoed:
            issues.append(
                f'title repeats the prompt example instead of the task: {echoed}')
    if not isinstance(plan1.get('endpoint'), str) or not plan1['endpoint'].strip():
        issues.append('endpoint is missing or empty')
    unexpected = sorted(set(plan1) - PLAN1_FIELDS)
    if unexpected:
        issues.append(f'Plan1 has unexpected fields: {unexpected}')
    steps = plan1.get('steps') if isinstance(plan1.get('steps'), list) else []
    if not steps:
        issues.append('Plan1 requires at least one distinct image-plus-content unit')
    elif len(steps) < 2:
        issues.append(
            'Plan1 has fewer than the minimum two distinct '
            'image-plus-content units')
    indexes = []
    for step in steps:
        if not isinstance(step, dict):
            issues.append('steps contains a non-object item')
            continue
        indexes.append(step.get('step_index'))
        extra = sorted(set(step) - PLAN1_STEP_FIELDS)
        if extra:
            issues.append(
                f"step {step.get('step_index', '?')} has unexpected fields: {extra}")
        for field in ('image_subject', 'content_goal'):
            if not isinstance(step.get(field), str) or not step[field].strip():
                issues.append(
                    f"step {step.get('step_index', '?')} has no {field}")
                continue
            echoed = find_planner_placeholders(step[field])
            if echoed:
                issues.append(
                    f"step {step.get('step_index', '?')} {field} repeats the "
                    f'prompt example instead of the task: {echoed}')
    expected = list(range(1, len(steps) + 1))
    if indexes != expected:
        issues.append(
            f'step indexes must be contiguous and ordered: expected {expected}, got {indexes}')
    return issues


def plan1_output_warnings(task_prompt, plan1):
    """Report non-blocking semantic observations with exact field locations."""
    if not isinstance(plan1, dict):
        return []
    occurrences = {}
    for step in plan1.get('steps', []):
        if not isinstance(step, dict):
            continue
        value = _stringify(step.get('image_subject')).strip()
        if value:
            occurrences.setdefault(value.casefold(), []).append(
                (step.get('step_index', '?'), value))
    warnings = []
    for normalized, entries in occurrences.items():
        if len(entries) < 2:
            continue
        locations = ', '.join(
            f'Plan1.steps[{index}].image_subject={value!r}'
            for index, value in entries)
        warnings.append(
            f'Plan1 has duplicate image_subject values after case-folding '
            f'(normalized={normalized!r}; locations: {locations})')
    # The endpoint does not become report content, so a copied example is
    # recorded without changing the validity of a plan whose units are
    # task-specific.
    echoed_endpoint = find_planner_placeholders(plan1.get('endpoint'))
    if echoed_endpoint:
        warnings.append(
            f'endpoint repeats the prompt example instead of the task: '
            f'{echoed_endpoint}')
    return warnings


def plan2_output_issues(plan1, plan2):
    if not isinstance(plan2, dict):
        return ['plan2 is not a JSON object']
    issues = []
    unexpected = sorted(set(plan2) - PLAN2_FIELDS)
    if unexpected:
        issues.append(f'Plan2 has unexpected fields: {unexpected}')
    visual_style = plan2.get('visual_style')
    if not isinstance(visual_style, dict):
        issues.append('Plan2 visual_style is missing or invalid')
    else:
        extra = sorted(set(visual_style) - set(VISUAL_STYLE_FIELDS))
        missing = [field for field in VISUAL_STYLE_FIELDS
                   if not isinstance(visual_style.get(field), str)
                   or not visual_style[field].strip()]
        if extra:
            issues.append(f'visual_style has unexpected fields: {extra}')
        if missing:
            issues.append(f'visual_style has missing or empty fields: {missing}')
        else:
            echoed = find_planner_placeholders(visual_style)
            if echoed:
                issues.append(
                    f'visual_style repeats the prompt example instead of the '
                    f'task: {echoed}')
    outline = plan1.get('steps', []) if isinstance(plan1, dict) else []
    steps = plan2.get('steps') if isinstance(plan2.get('steps'), list) else []
    expected_indexes = [s.get('step_index') for s in outline if isinstance(s, dict)]
    actual_indexes = [s.get('step_index') for s in steps if isinstance(s, dict)]
    if actual_indexes != expected_indexes:
        issues.append('Plan2 step indexes must exactly match Plan1')
    for step in steps:
        if not isinstance(step, dict):
            issues.append('Plan2 steps contains a non-object item')
            continue
        index = step.get('step_index', '?')
        extra = sorted(set(step) - PLAN2_STEP_FIELDS)
        if extra:
            issues.append(f'Plan2 step {index} has unexpected fields: {extra}')
        for field in ('image_prompt', 'content'):
            if not isinstance(step.get(field), str) or not step[field].strip():
                issues.append(f'Plan2 step {index} has no {field}')
                continue
            echoed = find_planner_placeholders(step[field])
            if echoed:
                issues.append(
                    f'Plan2 step {index} {field} repeats the prompt example '
                    f'instead of the task: {echoed}')
    return issues


def plan3_output_issues(plan1, plan3):
    issues = []
    plan3 = plan3 if isinstance(plan3, dict) else {}
    if set(plan3) - PLAN3_FIELDS:
        issues.append('Plan3 has unexpected fields')
    bank = plan3.get('condition_bank')
    if not isinstance(bank, dict) or set(bank) - CONDITION_BANK_FIELDS:
        issues.append('condition_bank must contain only items')
    items = bank.get('items', []) if isinstance(bank, dict) else []
    if len(items) > 2:
        issues.append('condition bank exceeds the 0-2 reference contract')
    key_values = [item.get('key') for item in items if isinstance(item, dict)]
    raw_keys = set(key_values)
    if len(key_values) != len(raw_keys):
        issues.append('condition keys must be unique')
    conditions = plan3.get('step_conditions', {})
    invalid_refs = sorted(set(condition_refs(conditions)) - raw_keys)
    if invalid_refs:
        issues.append(f'step_conditions contain unknown keys: {invalid_refs}')
    expected_steps = {
        f"step_{step.get('step_index')}" for step in plan1.get('steps', [])
        if isinstance(step, dict)}
    if not isinstance(conditions, dict) or set(conditions) != expected_steps:
        issues.append('step_conditions keys must exactly match Plan1 steps')
    for item in items:
        if not condition_item_is_executable(item):
            key = (
                item.get('key', '<missing>')
                if isinstance(item, dict) else '<missing>'
            )
            issues.append(f'condition {key} is invalid')
            continue
        if set(item) != CONDITION_ITEM_FIELDS:
            issues.append(f"condition {item['key']} has unexpected fields")
        echoed = find_planner_placeholders(item)
        if echoed:
            issues.append(
                f"condition {item['key']} repeats the prompt example instead "
                f'of the task: {echoed}')
        condition_values = conditions.values() if isinstance(conditions, dict) else ()
        uses = sum(item['key'] in refs for refs in condition_values
                   if isinstance(refs, list))
        if uses < 1:
            issues.append(f"condition {item['key']} must be routed to at least one step")
    return issues


def plan4_output_issues(plan1, plan4):
    checklist = plan4.get('reviewer_checklist') if isinstance(plan4, dict) else None
    if not isinstance(checklist, list) or not checklist:
        return ['reviewer_checklist is missing or invalid']
    issues = []
    if set(plan4) - PLAN4_FIELDS:
        issues.append('Plan4 has unexpected fields')
    indexes = {
        step.get('step_index') for step in plan1.get('steps', [])
        if isinstance(step, dict)}
    counts = {index: 0 for index in indexes}
    for item in checklist:
        if isinstance(item, str) and re.match(r'^general:\s*\S', item):
            for index in counts:
                counts[index] += 1
            if checklist_item_requires_text_reading(item):
                issues.append('general checklist must not require reading image text')
            continue
        match = re.match(r'^step_(\d+):\s*\S', item) if isinstance(item, str) else None
        if not match:
            issues.append(f'invalid checklist item: {item}')
            continue
        index = int(match.group(1))
        if index not in indexes:
            issues.append(f'checklist references unknown step_{index}')
            continue
        counts[index] += 1
        if checklist_item_requires_text_reading(item):
            issues.append(f'step_{index} checklist must not require reading image text')
        echoed = find_planner_placeholders(item)
        if echoed:
            issues.append(
                f'step_{index} checklist repeats the prompt example instead of '
                f'the task: {echoed}')
    for index, count in sorted(counts.items()):
        if count < 1 or count > 2:
            issues.append(f'step_{index} must have 1-2 checks, found {count}')
    return issues


def plan4_runtime_issues(plan1, plan4):
    """Return checklist errors other than the per-step count limit."""
    return [
        issue for issue in plan4_output_issues(plan1, plan4)
        if not re.match(r'^step_\d+ must have 1-2 checks, found \d+$', issue)
    ]


def cap_plan4_checks(plan4, limit=2):
    """Drop checklist entries beyond ``limit`` per step; return the count dropped."""
    checklist = plan4.get('reviewer_checklist') if isinstance(plan4, dict) else None
    if not isinstance(checklist, list):
        return 0
    general = sum(
        1 for item in checklist
        if isinstance(item, str) and re.match(r'^general:\s*\S', item))
    room = max(0, limit - general)
    seen = {}
    kept = []
    dropped = 0
    for item in checklist:
        match = re.match(r'^step_(\d+):\s*\S', item) if isinstance(item, str) else None
        if not match:
            kept.append(item)
            continue
        index = int(match.group(1))
        seen[index] = seen.get(index, 0) + 1
        if seen[index] <= room:
            kept.append(item)
        else:
            dropped += 1
    if dropped:
        plan4['reviewer_checklist'] = kept
    return dropped


def step_output_issues(parsed, step_context=None):
    if not isinstance(parsed, dict):
        return ['output is not a JSON object']
    return [f'{field} is empty' for field in ('image_prompt', 'content')
            if not isinstance(parsed.get(field), str) or not parsed[field].strip()]


def step_output_warnings(parsed, step_context=None):
    content = parsed.get('content') if isinstance(parsed, dict) else None
    if not isinstance(content, str):
        return []
    matches = list(_CONTENT_META.finditer(content))
    if not matches:
        return []
    stage = (
        f'{step_context}: ' if isinstance(step_context, str) and step_context
        else '')
    excerpt = ' '.join(content.split())
    if len(excerpt) > 160:
        excerpt = excerpt[:157] + '...'
    matched = [match.group(0) for match in matches]
    return [
        (
            f'{stage}content contains prompt/review meta-language '
            f'(matched={matched!r}; content={excerpt!r})'
        )
    ]


def _text_validation_issue(stage, issue_type, message, **details):
    issue = {'stage': stage, 'type': issue_type, 'message': message}
    issue.update(details)
    return issue


def _training_input_issues(trajectory):
    """Check exact recorded exchanges required for Planner supervision."""
    raw_attempts = trajectory.get('raw_attempts')
    if not isinstance(raw_attempts, dict):
        return ['trajectory has no raw_attempts for Planner training']
    issues = []
    for stage in PLAN_STAGE_NAMES:
        attempts = raw_attempts.get(stage)
        if not isinstance(attempts, list) or len(attempts) != 1:
            issues.append(
                f'trajectory must contain exactly one {stage} raw attempt')
            continue
        latest = attempts[-1]
        required = ('system_text', 'raw_input', 'raw_output')
        if (not isinstance(latest, dict)
                or any(not isinstance(latest.get(field), str)
                       or not latest[field].strip() for field in required)):
            issues.append(f'{stage} raw attempt is incomplete')
            continue
        raw_output = latest['raw_output'].strip()
        if latest['system_text'] != PLANNER_SYSTEM:
            issues.append(f'{stage} raw attempt uses a non-UniWeaver system prompt')
        # The builder can only be replayed when the parsed Plan is stored.
        stored_task = task_text(trajectory)
        if (isinstance(trajectory.get('plan'), dict) and trajectory['plan']
                and stored_task):
            expected_input = build_plan_request(
                stage, stored_task, trajectory['plan'])
            if latest['raw_input'] != expected_input:
                issues.append(
                    f'{stage} raw attempt does not match the UniWeaver builder')
        # Generation settings are recorded by the pipeline; a published corpus
        # may omit them, but a recorded value must still match the stage.
        if ('request_config' in latest
                and latest['request_config'] != planner_stage_config(stage)):
            issues.append(f'{stage} raw attempt uses non-UniWeaver generation settings')
            continue
        json_text = raw_output
        try:
            exact_target = json.loads(json_text)
        except (json.JSONDecodeError, TypeError, ValueError):
            issues.append(f'{stage} raw target is not exact JSON')
            continue
        final_plan = trajectory.get('plan')
        if (isinstance(final_plan, dict) and stage in final_plan
                and exact_target != final_plan[stage]):
            issues.append(
                f'{stage} raw target does not equal the final Plan block')
    return issues


def validate_text_trajectory(trajectory):
    if not isinstance(trajectory, dict):
        issue = _text_validation_issue(
            'trajectory', 'invalid_trajectory', 'trajectory is not a JSON object')
        return {
            'schema_valid': False,
            'render_eligible': False,
            'execution_eligible': False,
            'strict_execution_eligible': False,
            'training_eligible': False,
            'stages': {stage: False for stage in PLAN_STAGE_NAMES},
            'issues': [issue],
        }
    construction_issues = trajectory.get('_construction_issues', [])
    issues = list(construction_issues) if isinstance(construction_issues, list) else []
    # Classify by envelope and Plan block so a recovered Plan stays renderable
    # while schema validation still records the envelope violation.
    plan_blocks = (
        trajectory.get('plan') if isinstance(trajectory.get('plan'), dict)
        else {})
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        stage = issue.get('stage')
        if (issue.get('message') == 'response is not exactly one JSON object'
                and stage in plan_blocks
                and isinstance(plan_blocks.get(stage), dict)):
            issue['type'] = 'envelope_error'
    try:
        # A persisted trajectory carries view metadata the pipeline wrote when it
        # saved; identity is judged on the contract fields only.
        require_trajectory_identity(trajectory_contract_view(trajectory))
    except ValueError as error:
        issues.append(_text_validation_issue('trajectory', 'invalid_identity', str(error)))
    plan = trajectory.get('plan') if isinstance(trajectory.get('plan'), dict) else {}
    stage_checks = {stage: None for stage in PLAN_STAGE_NAMES}
    missing = [stage for stage in PLAN_STAGE_NAMES if stage not in plan]
    if not plan:
        # A text-SFT record keeps the recorded exchanges without the parsed
        # Plan; shape checks apply only when a Plan block is stored.
        pass
    elif missing:
        issues.append(_text_validation_issue(
            'trajectory', 'incomplete_plan', f'trajectory is missing plan blocks: {missing}'))
    else:
        plan1, plan2, plan3, plan4 = (
            plan['plan1'], plan['plan2'], plan['plan3'], plan['plan4'])
        stage_checks = {
            'plan1': plan1_output_issues(task_text(trajectory), plan1),
            'plan2': plan2_output_issues(plan1, plan2),
            'plan3': plan3_output_issues(plan1, plan3),
            'plan4': plan4_output_issues(plan1, plan4),
        }
        for stage, messages in stage_checks.items():
            issues.extend(_text_validation_issue(
                stage, 'contract_error', message) for message in messages)
        if trajectory.get('step_generation_expected', True):
            generations = trajectory.get('step_generations', {})
            for step in plan2.get('steps', []):
                if not isinstance(step, dict):
                    continue
                index = step.get('step_index')
                parsed = read_cached_step_text(generations.get(f'step_{index}'))
                pair = ({'image_prompt': parsed['model_image_prompt'],
                         'content': parsed['content']} if parsed else None)
                issues.extend(_text_validation_issue(
                    f'step_{index}', 'contract_error', message)
                    for message in step_output_issues(pair))
                if parsed:
                    try:
                        expected = compile_step_render_input(plan2, step)
                        actual = restore_step_render_input(parsed)
                        if actual != expected:
                            issues.append(_text_validation_issue(
                                f'step_{index}', 'contract_error',
                                'cached Step input does not match the current compiler'))
                    except ValueError as error:
                        issues.append(_text_validation_issue(
                            f'step_{index}', 'contract_error', str(error)))
    training_messages = _training_input_issues(trajectory)
    issues.extend(_text_validation_issue(
        'training', 'training_input_error', message)
        for message in training_messages)
    clean = []
    seen = set()
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        signature = (issue.get('stage'), issue.get('type'), issue.get('message'))
        if signature not in seen:
            seen.add(signature)
            clean.append(issue)
    schema_errors = [
        issue for issue in clean
        if issue.get('type') not in ('validation_warning', 'training_input_error')]
    # Extra fields are warnings; missing semantic fields, step-index mismatches,
    # incomplete visual_style and envelope/decoding failures stay fatal.
    relaxed_errors = [
        issue for issue in schema_errors
        if _is_non_fatal_contract_issue(issue)]
    render_errors = [
        issue for issue in schema_errors
        if issue.get('stage') not in RENDER_ELIGIBLE_GATE['ignored_stages']
        and issue.get('type') not in RENDER_ELIGIBLE_GATE['ignored_issue_types']
        and not _is_non_fatal_contract_issue(issue)]
    strict_execution_errors = [
        issue for issue in schema_errors
        if issue.get('type') != 'envelope_error']
    validation_warnings = [
        issue for issue in clean
        if issue.get('type') == 'validation_warning'] + relaxed_errors
    training_errors = schema_errors + [
        issue for issue in clean
        if issue.get('type') == 'training_input_error']
    stages = {
        stage: (
            stage in plan
            and stage_checks.get(stage) == []
            and not any(
                issue.get('stage') == stage
                and issue.get('type') not in ('validation_warning',)
                for issue in clean
            )
        )
        for stage in PLAN_STAGE_NAMES
    }
    return {
        'schema_valid': not schema_errors,
        # Rendering and execution consume the parsed Plan, so a record without
        # one is a training-only sample.
        'render_eligible': bool(plan) and not render_errors,
        # Execution consumes Plan3 references and the Plan4 checklist, so both
        # strict contracts gate image execution and report review.
        'execution_eligible': bool(plan) and not render_errors,
        'strict_execution_eligible': bool(plan) and not strict_execution_errors,
        'training_eligible': not schema_errors and not training_messages,
        'stages': stages,
        'acceptance': {
            'schema': [issue['message'] for issue in schema_errors],
            'render': [issue['message'] for issue in render_errors],
            'execution': [issue['message'] for issue in render_errors],
            'strict_execution': [issue['message'] for issue in strict_execution_errors],
            'training': [issue['message'] for issue in training_errors],
        },
        'warnings': [issue['message'] for issue in validation_warnings],
        'issues': clean,
    }


def check_text_validation(validation, criterion='execution'):
    """Project canonical validation through one explicit acceptance criterion."""
    acceptance_fields = {
        'schema': 'schema_valid',
        'render': 'render_eligible',
        'execution': 'execution_eligible',
        'strict_execution': 'strict_execution_eligible',
        'training': 'training_eligible',
    }
    if criterion not in acceptance_fields:
        raise ValueError(f'unknown text-validation criterion: {criterion}')
    acceptance = validation.get('acceptance', {})
    reasons = list(acceptance.get(criterion, []))
    passed = bool(validation.get(acceptance_fields[criterion], False))
    return {
        'passed': passed,
        'reasons': reasons,
        'retryable': False,
        'criterion': criterion,
        'schema_valid': bool(validation.get('schema_valid', False)),
        'render_eligible': bool(validation.get('render_eligible', False)),
        'execution_eligible': bool(validation.get('execution_eligible', False)),
        'strict_execution_eligible': bool(validation.get('strict_execution_eligible', False)),
        'training_eligible': bool(validation.get('training_eligible', False)),
        'details': [
            item for item in validation.get('issues', [])
            if isinstance(item, dict)
        ],
        'warnings': list(validation.get('warnings', [])),
    }
