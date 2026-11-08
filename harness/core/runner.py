"""Canonical UniWeaver stage orchestration with one JSON-only call per stage."""
import copy
from pathlib import Path

from .contract import (
    PLANNER_SYSTEM,
    build_plan_request,
)
from .decoding import (
    decode_plan3,
    decode_plan4,
    decode_steps,
    detect_repetition_loop,
    detect_think_loop,
    split_think,
)
from .execution_recovery import (
    parser_recovery_provenance,
    repair_planner_stage,
)
from .production import (
    CONTRACT_VERSION,
    TRAJECTORY_DESIGN,
    TRAJECTORY_METADATA_FIELDS,
    planner_stage_config,
    require_trajectory_identity,
)
from .render_inputs import compile_step_render_input
from .serialization import (
    annotate_raw_attempt,
    atomic_write_json,
    export_trajectory,
    finish_raw_attempt,
    record_raw_attempt,
)
from .validation import (
    _text_validation_issue,
    cap_plan4_checks,
    json_only_response_issues,
    plan1_output_issues,
    plan1_output_warnings,
    plan2_output_issues,
    plan3_output_issues,
    plan4_runtime_issues,
    step_output_issues,
    step_output_warnings,
    validate_text_trajectory,
)


class TrajectoryError(Exception):
    """Raised when a trajectory task fails irrecoverably."""


def run_json_text_stage(user_text, label, max_tokens, send_backend, decoder,
                        required_keys, system_text=PLANNER_SYSTEM,
                        do_sample=True, temperature=0.7,
                        on_attempt=None, parsed_is_valid=None):
    """Execute exactly one canonical JSON-only Planner request."""
    raw_output = send_backend(
        user_text=user_text, label=label, max_tokens=max_tokens,
        system_text=system_text, do_sample=do_sample,
        temperature=temperature)
    think_text, after_think = split_think(raw_output)
    decoded = decoder(raw_output)
    parsed = decoded.get('parsed')
    parse_method = decoded.get('parse_method', 'parse_failed')
    if parsed is not None and parsed_is_valid and not parsed_is_valid(parsed):
        parsed = None
        parse_method = 'invalid_schema'
    parsed_for_missing = (
        decoded.get('candidate')
        if isinstance(decoded.get('candidate'), dict) else parsed)
    missing = [key for key in required_keys
               if not isinstance(parsed_for_missing, dict)
               or not parsed_for_missing.get(key)]
    if missing:
        parsed = None
        parse_method = 'missing_keys'
    # ``split_think`` returns the raw text as ``think_text`` when a transport
    # ships no reasoning block. Compare identity to decide which loop detector
    # applies instead of treating a non-empty ``think_text`` as reasoning.
    has_reasoning = think_text != raw_output
    loop_warning = (
        (detect_think_loop(think_text) if has_reasoning else None)
        or detect_repetition_loop(after_think)
    )
    result = {
        'label': label, 'raw_input': user_text, 'raw_output': raw_output,
        'think_text': think_text if think_text != raw_output else '',
        'after_think': after_think, 'parsed': parsed, 'decoded': decoded,
        'parse_method': parse_method, 'missing_keys': missing,
        'loop_warning': loop_warning, 'truncated': False,
        'retried': False, 'format_retried': False,
    }
    # Persist the decode verdict on the attempt that was already written, so the
    # artifacts explain *why* a stage failed instead of only showing the raw
    # transcript (think split, loop warning, parse method, missing keys).
    attempt = getattr(send_backend, 'last_attempt', None)
    if isinstance(attempt, dict):
        annotate_raw_attempt(
            attempt,
            think_text=result['think_text'],
            loop_warning=result['loop_warning'],
            parse_method=result['parse_method'],
            missing_keys=result['missing_keys'],
            retried=result['retried'],
            format_retried=result['format_retried'],
        )
        save_attempt = getattr(send_backend, 'save_attempt', None)
        if callable(save_attempt):
            save_attempt()
    if on_attempt:
        on_attempt(dict(result))
    return result


def run_trajectory_with_sender(task_key, task_prompt, output_path, send_backend,
                               extra_fields=None, include_step_generation=True,
                               on_text_attempt=None, mechanical_repairer=None):
    """Run the canonical UniWeaver text trajectory; only model transport is injected."""
    trajectory = {
        'task_key': task_key,
        'task_prompt': task_prompt,
        'raw_attempts': {},
        'plan': {},
        'step_generations': {},
        'step_generation_expected': bool(include_step_generation),
        '_construction_issues': [],
    }
    if extra_fields:
        unsupported = sorted(set(extra_fields) - TRAJECTORY_METADATA_FIELDS)
        if unsupported:
            raise ValueError(
                'extra_fields contains fields outside the closed metadata schema: '
                f'{unsupported}')
        trajectory.update(extra_fields)
    trajectory['trajectory_design'] = TRAJECTORY_DESIGN
    trajectory['contract_version'] = CONTRACT_VERSION
    require_trajectory_identity(trajectory)
    output_path = Path(output_path) if output_path is not None else None
    if output_path is not None and output_path.exists():
        raise FileExistsError(f'refusing to overwrite trajectory: {output_path}')

    def request_config(stage):
        return planner_stage_config(stage)

    def send(user_text, label, max_tokens, system_text=PLANNER_SYSTEM,
             do_sample=True, temperature=0.7):
        request = {
            'max_tokens': max_tokens,
            'do_sample': do_sample,
            'temperature': temperature,
        }
        attempt = record_raw_attempt(
            trajectory, label, user_text,
            system_text=system_text,
            request_config=request,
            status='started',
        )
        # Expose the in-flight attempt to run_json_text_stage, which annotates
        # it with the decode verdict once the response is parsed.
        send.last_attempt = attempt
        send.save_attempt = save
        save()
        try:
            raw_output = send_backend(
                user_text=user_text,
                label=label,
                max_tokens=max_tokens,
                system_text=system_text,
                do_sample=do_sample,
                temperature=temperature,
            )
        except BaseException as error:
            finish_raw_attempt(
                attempt,
                transport=getattr(send_backend, 'last_metadata', {}),
                error={
                    'type': type(error).__name__,
                    'message': str(error),
                },
            )
            save()
            raise
        finish_raw_attempt(
            attempt,
            raw_output=raw_output,
            transport=getattr(send_backend, 'last_metadata', {}),
        )
        save()
        return raw_output

    def save():
        if output_path is None:
            return
        atomic_write_json(output_path, export_trajectory(trajectory))

    def finalize_validation():
        trajectory['_text_validation'] = validate_text_trajectory(trajectory)

    def add_validation_issue(stage, issue_type, message, **details):
        trajectory['_construction_issues'].append(
            _text_validation_issue(
                stage, issue_type, message, **details))

    def fail(stage, message):
        add_validation_issue(
            stage, 'construction_failure', message)
        finalize_validation()
        save()
        raise TrajectoryError(message)

    def record_issues(stage, issue_type, issues):
        if not issues:
            return
        print(f'  {stage}: {issue_type} count={len(issues)}', flush=True)
        for message in issues:
            print(f'    - {message}', flush=True)
            add_validation_issue(stage, issue_type, message)

    def record_validation(stage, issues):
        # Retain the complete model output, then let the final validator state
        # whether it is schema-valid and/or safe to execute.
        record_issues(stage, 'contract_error', issues)

    def record_envelope(stage, envelope_messages):
        # Envelope violations are recorded separately so schema validation fails
        # while image eligibility follows the parsed Plan.
        record_issues(stage, 'envelope_error', envelope_messages)

    def record_warnings(stage, warnings):
        record_issues(stage, 'validation_warning', warnings)

    def envelope_issues(raw_output):
        return json_only_response_issues(raw_output)

    def apply_mechanical_repair(stage, result):
        repairer = mechanical_repairer or repair_planner_stage
        if repairer is None:
            return
        # The repairer also sees the frozen upstream blocks, because Plan2's
        # index realignment can only be decided against Plan1.
        repaired = repairer(stage, result, {'plan': trajectory.get('plan', {})})
        if not isinstance(repaired, dict) or not isinstance(
                repaired.get('parsed'), dict):
            return
        result['parsed'] = repaired['parsed']
        result['parse_method'] = 'mechanical_repair'
        result['repair_provenance'] = copy.deepcopy(
            repaired.get('provenance', {}))
        add_validation_issue(
            stage,
            'mechanical_repair',
            'Planner output required an audited mechanical repair for image execution',
            repair_provenance=result['repair_provenance'],
        )

    def record_parser_recovery(stage, result):
        provenance = parser_recovery_provenance(stage, result)
        if provenance is None:
            return
        add_validation_issue(
            stage,
            'parser_recovery',
            'Planner response required a provenance-recorded parser recovery for execution',
            repair_provenance=provenance,
        )

    print('\n=== Plan 1: Scope and Sequence ===', flush=True)
    user_msg = build_plan_request('plan1', task_prompt, trajectory['plan'])

    def valid_plan1(parsed):
        steps_value = parsed.get('steps') if isinstance(parsed, dict) else None
        return (
            isinstance(steps_value, list) and bool(steps_value) and
            all(
                isinstance(step, dict)
                and step.get('step_index') is not None
                and isinstance(step.get('image_subject'), str)
                and step['image_subject'].strip()
                and isinstance(step.get('content_goal'), str)
                and step['content_goal'].strip()
                for step in steps_value))

    stage_config = request_config('plan1')
    plan1_result = run_json_text_stage(
        user_msg, 'plan1', send_backend=send, decoder=decode_steps,
        required_keys=('steps',), on_attempt=on_text_attempt,
        parsed_is_valid=valid_plan1,
        **stage_config)
    record_parser_recovery('plan1', plan1_result)
    apply_mechanical_repair('plan1', plan1_result)
    raw1 = plan1_result['raw_output']
    decoded1 = plan1_result['decoded']
    plan1 = plan1_result['parsed']
    if not plan1:
        fail('plan1', 'plan1 did not return the required JSON object')
    steps = plan1['steps']
    trajectory['plan']['plan1'] = plan1
    record_issues('plan1', 'decoding_error', decoded1['issues'])
    plan1_issues = plan1_output_issues(
        task_prompt, plan1)
    plan1_warnings = (
        plan1_output_warnings(task_prompt, plan1))
    record_envelope('plan1', envelope_issues(raw1))
    record_validation('plan1', plan1_issues)
    record_warnings('plan1', plan1_warnings)
    print(f'  OK: {len(steps)} steps, title="{plan1.get("title", "")}"', flush=True)

    print('\n=== Plan 2: Text and Image Planning ===', flush=True)
    user_msg = build_plan_request('plan2', task_prompt, trajectory['plan'])
    stage_config = request_config('plan2')
    plan2_result = run_json_text_stage(
        user_msg, 'plan2', send_backend=send,
        decoder=decode_steps,
        required_keys=('steps',), on_attempt=on_text_attempt,
        **stage_config)
    record_parser_recovery('plan2', plan2_result)
    apply_mechanical_repair('plan2', plan2_result)
    raw2 = plan2_result['raw_output']
    decoded2 = plan2_result['decoded']
    plan2 = plan2_result['parsed']
    if not plan2 or not plan2.get('steps'):
        fail('plan2', 'plan2 did not return the required JSON object')
    trajectory['plan']['plan2'] = plan2
    record_issues('plan2', 'decoding_error', decoded2['issues'])
    plan2_issues = plan2_output_issues(
        plan1, plan2)
    record_envelope('plan2', envelope_issues(raw2))
    record_validation('plan2', plan2_issues)
    print(f'  OK: {len(plan2["steps"])} final steps', flush=True)
    execution_steps = plan2['steps']

    print('\n=== Plan 3: Selective Visual Conditioning ===', flush=True)
    user_msg = build_plan_request('plan3', task_prompt, trajectory['plan'])
    stage_config = request_config('plan3')
    plan3_result = run_json_text_stage(
        user_msg, 'plan3', send_backend=send,
        decoder=lambda raw: decode_plan3(raw, len(execution_steps)),
        required_keys=('condition_bank',), on_attempt=on_text_attempt,
        **stage_config)
    record_parser_recovery('plan3', plan3_result)
    plan3_result['plan1'] = plan1
    apply_mechanical_repair('plan3', plan3_result)
    raw3 = plan3_result['raw_output']
    decoded3 = plan3_result['decoded']
    plan3 = plan3_result['parsed']
    condition_items = decoded3.get('condition_items', [])
    trajectory['plan']['plan3'] = plan3
    record_issues('plan3', 'decoding_error', decoded3['issues'])
    plan3_issues = plan3_output_issues(plan1, plan3)
    if not isinstance(plan3, dict):
        plan3_issues = ['Plan3 did not return the required JSON object'] + plan3_issues
    record_envelope('plan3', envelope_issues(raw3))
    record_validation('plan3', plan3_issues)
    if plan3_issues:
        fail('plan3', 'Plan3 contract errors: ' + '; '.join(plan3_issues))
    print(f'  OK: {len(condition_items)} condition items', flush=True)

    print('\n=== Plan 4: Visual Verification Checklist ===', flush=True)
    user_msg = build_plan_request('plan4', task_prompt, trajectory['plan'])
    stage_config = request_config('plan4')
    plan4_result = run_json_text_stage(
        user_msg, 'plan4', send_backend=send, decoder=decode_plan4,
        required_keys=('reviewer_checklist',), on_attempt=on_text_attempt,
        **stage_config)
    record_parser_recovery('plan4', plan4_result)
    apply_mechanical_repair('plan4', plan4_result)
    raw4 = plan4_result['raw_output']
    decoded4 = plan4_result['decoded']
    plan4 = plan4_result['parsed']
    dropped_checks = cap_plan4_checks(plan4)
    if dropped_checks:
        add_validation_issue(
            'plan4',
            'mechanical_repair',
            f'dropped {dropped_checks} checklist entries beyond 2 per step',
            repair_provenance={
                'policy_id': 'plan4_check_limit_v1',
                'dropped': dropped_checks,
            },
        )
    trajectory['plan']['plan4'] = plan4
    record_issues('plan4', 'decoding_error', decoded4['issues'])
    plan4_issues = plan4_runtime_issues(plan1, plan4)
    if not isinstance(plan4, dict):
        plan4_issues = ['Plan4 did not return the required JSON object'] + plan4_issues
    record_envelope('plan4', envelope_issues(raw4))
    record_validation('plan4', plan4_issues)
    if plan4_issues:
        fail('plan4', 'Plan4 contract errors: ' + '; '.join(plan4_issues))
    print(f'  OK: {len(plan4.get("reviewer_checklist", []))} checklist items', flush=True)
    if include_step_generation:
        print('\n=== Execution: Step Generation ===', flush=True)
        for step in execution_steps:
            step_index = step['step_index']
            try:
                render_input = compile_step_render_input(plan2, step)
            except (TypeError, ValueError) as error:
                fail(f'step_{step_index}', str(error))
            semantic_image_prompt = render_input.semantic_image_prompt
            semantic_content = render_input.content
            final_image_prompt = render_input.model_image_prompt
            validation_step = {
                'image_prompt': semantic_image_prompt,
                'content': semantic_content,
            }
            semantic_issues = step_output_issues(validation_step)
            record_validation(f'step_{step_index}', semantic_issues)
            record_warnings(
                f'step_{step_index}',
                step_output_warnings(validation_step, f'Plan2.steps[{step_index}]'))
            trajectory['step_generations'][f'step_{step_index}'] = {
                'parsed': {
                    'semantic_image_prompt': semantic_image_prompt,
                    'model_image_prompt': final_image_prompt,
                    'content': semantic_content,
                },
            }
            print(
                f'  step {step_index}: deterministic assembly '
                f'(ip={len(final_image_prompt)}c '
                f'content={len(semantic_content)}c)',
                flush=True)

    finalize_validation()
    save()
    print('\n=== DONE ===', flush=True)
    if output_path is not None:
        print(f'Saved: {output_path}', flush=True)
    return trajectory
