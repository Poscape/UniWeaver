"""Raw exchange tracing and outward trajectory serialization."""
import copy
import json
import os
import re as _re
import tempfile
import time
from pathlib import Path

from .policy_registry import PLANNER_TRANSPORT_EVENT
from .production import CONTRACT_VERSION, TRAJECTORY_DESIGN


def atomic_write_text(path, text, encoding='utf-8'):
    """Publish one complete text file without exposing a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path, value):
    """Publish deterministic UTF-8 JSON through the shared atomic writer."""
    atomic_write_text(
        path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def record_raw_attempt(trajectory, label, raw_input, raw_output=None,
                       system_text='', request_config=None, transport=None,
                       status='completed', error=None):
    """Append one untouched exchange together with reproducibility metadata."""
    label = str(label or 'unlabeled')
    plan_match = _re.match(r'^(plan[1-4])\b', label)
    step_match = _re.match(r'^step\s+(\d+)\b', label)
    if plan_match:
        stage = plan_match.group(1)
    elif step_match:
        stage = f'step_{step_match.group(1)}'
    else:
        stage = _re.sub(r'[^a-zA-Z0-9_-]+', '_', label).strip('_').lower() or 'unlabeled'
    now = time.time()
    attempt = {
        'policy_id': PLANNER_TRANSPORT_EVENT['policy_id'],
        'label': label,
        'system_text': system_text,
        'raw_input': raw_input,
        'raw_output': raw_output,
        'request_config': copy.deepcopy(request_config or {}),
        'transport': copy.deepcopy(transport or {}),
        'status': status,
        'started_at': now,
        'finished_at': now if status != 'started' else None,
        'error': copy.deepcopy(error),
    }
    trajectory.setdefault('raw_attempts', {}).setdefault(stage, []).append(attempt)
    return attempt


def finish_raw_attempt(attempt, *, raw_output=None, transport=None, error=None):
    """Finish one already-persisted attempt without rewriting its raw input."""
    attempt['raw_output'] = raw_output
    attempt['transport'] = copy.deepcopy(transport or {})
    attempt['status'] = 'failed' if error is not None else 'completed'
    attempt['error'] = copy.deepcopy(error)
    attempt['finished_at'] = time.time()


def annotate_raw_attempt(attempt, **fields):
    """Record decode-time diagnostics on an already-persisted attempt."""
    if not isinstance(attempt, dict):
        return attempt
    for key, value in fields.items():
        attempt[key] = list(value) if isinstance(value, tuple) else value
    return attempt



def export_trajectory(trajectory):
    """Return an isolated copy of the canonical trajectory schema."""
    return copy.deepcopy(trajectory)


def attach_condition_refs(steps, step_conditions):
    """Copy steps and attach their canonical Plan3 condition references."""
    projected = copy.deepcopy(steps) if isinstance(steps, list) else []
    conditions = step_conditions if isinstance(step_conditions, dict) else {}
    for step in projected:
        if not isinstance(step, dict):
            continue
        step_index = step.get('step_index')
        refs = conditions.get(f'step_{step_index}', [])
        if isinstance(refs, list):
            step['condition_refs'] = copy.deepcopy(refs)
    return projected


def project_runtime_plan(
        plan_blocks, user_prompt='',
        *, trajectory_design, contract_version):
    """Project canonical Plan1-4 blocks into the one runtime plan shape."""
    if (trajectory_design != TRAJECTORY_DESIGN
            or contract_version != CONTRACT_VERSION):
        raise ValueError(
            'runtime plan must use '
            f'{TRAJECTORY_DESIGN}/{CONTRACT_VERSION}')
    blocks = plan_blocks if isinstance(plan_blocks, dict) else {}
    plan1 = blocks.get('plan1', {})
    plan2 = blocks.get('plan2', {})
    plan3 = blocks.get('plan3', {})
    plan4 = blocks.get('plan4', {})
    plan1 = plan1 if isinstance(plan1, dict) else {}
    plan2 = plan2 if isinstance(plan2, dict) else {}
    plan3 = plan3 if isinstance(plan3, dict) else {}
    plan4 = plan4 if isinstance(plan4, dict) else {}
    from .validation import plan3_output_issues, plan4_runtime_issues
    plan3_issues = plan3_output_issues(plan1, plan3)
    if plan3_issues:
        raise ValueError('Plan3 contract errors: ' + '; '.join(plan3_issues))
    step_conditions = copy.deepcopy(
        plan3.get('step_conditions', {}))
    plan4_issues = plan4_runtime_issues(plan1, plan4)
    if plan4_issues:
        raise ValueError('Plan4 contract errors: ' + '; '.join(plan4_issues))
    projected = {
        'title': plan1.get('title', ''),
        'steps': attach_condition_refs(
            plan2.get('steps', []),
            step_conditions),
        'condition_bank': copy.deepcopy(
            plan3.get('condition_bank', {'items': []})),
        'reviewer_checklist': copy.deepcopy(
            plan4.get('reviewer_checklist', [])),
        'reviewer_checklist_available': True,
        'trajectory_design': trajectory_design,
        'contract_version': contract_version,
        '_user_prompt': user_prompt,
        'raw_input': user_prompt,
        'errors': [],
        'warnings': [],
    }
    projected['visual_style'] = copy.deepcopy(
        plan2.get('visual_style', {}))
    return projected
