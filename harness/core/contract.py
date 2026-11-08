"""Canonical UniWeaver prompts and stateless request builders."""
import json

MINIMUM_INTERLEAVE_CONTRACT = """An interleaved report requires at least two distinct,
useful image-plus-content units, so output at least two semantic steps. Every step must make a
different task-relevant contribution. Do not add steps merely to satisfy the minimum."""

# Literal example values the Planner prompts contain. A model that echoes its
# prompt returns exactly these strings inside a well-formed object, so decoding
# prefers a later real block and validation rejects the echo.
PLANNER_PLACEHOLDER_LITERALS = (
    'earliest completed outcome or last explicit event',
    'one concrete visible subject or state',
    'short reader-facing question',
    'complete independently renderable visible state',
    'finished report prose',
    # Plan4's skeleton renders "step_1: <check>"; a checklist copied from it
    # asks the reviewer to verify the literal word "check".
    '<check>',
    # Plan3's skeleton names the reference key and its prompt; both have been
# echoed verbatim into a real condition_bank item.
    'stable_identifier',
    'self-contained image of only the stable reusable target',
)


def _iter_strings(value):
    """Yield every string inside a nested JSON-like value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield from _iter_strings(key)
            yield from _iter_strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_strings(nested)


def find_planner_placeholders(value):
    """Return the prompt example literals echoed verbatim inside ``value``."""
    found = []
    for text in _iter_strings(value):
        lowered = text.casefold()
        for literal in PLANNER_PLACEHOLDER_LITERALS:
            if literal.casefold() in lowered and literal not in found:
                found.append(literal)
    return found


STEP_SUMMARY_FIELDS = frozenset(
    {'image_subject', 'content_goal', 'image_prompt', 'content'})

PLANNER_SYSTEM = """You construct useful illustrated reports in four stages. At the current
stage, derive decisions from the supplied task and frozen upstream output. Return exactly one
valid JSON object matching the requested schema, with no analysis or commentary."""

_PLAN1_BASE_PROMPT = """Task request:
{task}

Use task details, ordinary knowledge, and plausible scene detail. Keep an unspecified subject at
the category level instead of inventing a specific identity or exact event facts. Treat
source-aware or source-backed as a request for careful, credible subject matter, not prose about
sources or research work.

Partition the outcome into distinct useful image-plus-content units. For each unit separate:
- image_subject: one concrete physical subject or state that a single independent image can
  directly depict in one coherent view;
- content_goal: one short reader-facing question asking what useful explanation, interpretation,
  comparison, instruction, chronology, or decision the adjacent prose must answer. Ask about the
  subject itself.

Normally create one unit for each distinct requested physical subject, showing a view that best
supports all of its requested properties; put the properties and comparison criteria together in
that unit's content_goal. Create another unit for the same subject only when the Task explicitly
requires a real change of state, time, or location. A different evaluation criterion, viewpoint,
trade-off, comparison, recommendation, or prose section never creates a unit. Never create a
summary, diagram, plan, split view, collage, or information-display image. Audience, later use, and
presentation format do not add subjects.

Interpret the requested action literally. The endpoint is the earliest completed requested outcome
or the last event explicitly stated in a narrative. For set up, arrange, or organize, operating or
demonstrating the ready items is a later task. In a narrative, put the immediate reaction to the
last event in that event's content rather than adding an epilogue unit.

When the Task's main operation is comparing named alternatives, create exactly one unit for each
alternative unless the Task explicitly requests a real before/after or other state change within an
alternative. Each unit's one view must show that alternative in the use context that exposes the
most requested properties; its content_goal carries all criteria, trade-offs, and the decision.
Before returning, verify that unit count equals alternative count unless that exception applies.
For comparison tasks, an image_subject containing two or more named alternatives is invalid, and a
step whose only purpose is to synthesize or conclude the comparison is invalid. Put synthesis and
the decision in the content_goal of the last alternative's existing unit."""

_PLAN1_BASE_PROMPT += (
    "\n\n" + MINIMUM_INTERLEAVE_CONTRACT +
    """

Output exactly:
{"title":"...","endpoint":"earliest completed outcome or last explicit event",
"steps":[{"step_index":1,"image_subject":"one concrete visible subject or state",
"content_goal":"short reader-facing question"}]}
The top-level object has exactly these three keys. Every object inside steps has exactly three keys:
step_index, image_subject, and content_goal.""")

_PLAN2_BASE_PROMPT = """Task request:
{task}

Write concise finished prose for the reader. Directly answer each content_goal using task details,
ordinary knowledge, and plausible scene detail. Keep an unspecified subject at the category level
instead of inventing a specific identity or exact event facts. Discuss the subject, not the writing
process. Give useful information directly; do not assign research to the reader.

Frozen scope and sequence:
{steps_summary}
Frozen endpoint: {endpoint}
Frozen step indexes: {step_indexes}

Choose one shared visual_style for rendering treatment only: medium, color treatment, lighting
character, and rendering finish. Keep subjects, actions, locations, viewpoint, composition, and
prose voice in the individual steps. Follow an explicitly requested style; otherwise use a
restrained treatment appropriate to the subject.

Return one object for each frozen step index, in order. The complete task defines global scope and
each frozen unit separates its visible subject from its prose contribution. Apply every relevant
task constraint without changing the frozen sequence.

For each step output:
- the same step_index; do not repeat Plan1 fields in the output;
- image_prompt: one complete, self-contained rendering instruction for the observable subject,
  state, relations, and context required by this contribution;
- content: finished report prose carrying useful explanation that need not be visible in pixels.

Name required visible elements concretely rather than replacing them with generic collective
terms. Include only entities whose visible presence follows from the requested contribution. Keep
one coherent physical view per image. Depict the frozen image_subject rather than a container for
presenting information about it, unless that container is itself the requested subject.

Do not turn a comparison or synthesis into a split composition, collage, multi-panel sequence,
overlay, diagram, plan view, map, report page, label panel, or information display. The separate
images provide the concrete subjects; content carries relations across them. Across the content
fields, cover the complete task.

Output schema:
{"visual_style":{"medium":"...","color_treatment":"...",
"lighting_character":"...","rendering_finish":"..."},
"steps":{step_semantic_entries}}"""


_PLAN3_BASE_PROMPT = """Task:
{task}
Frozen semantic subjects:
{semantic_subjects}
Rendered Steps:
{steps_summary}

Select zero to two isolated identity references. Before writing JSON, silently enumerate every
concrete person, object, vessel, tool, carrier, and support explicitly visible in each rendered
Step, then group only occurrences established as the same physical individual. Do not group
repeated members of a category.

Treat identity as established when an ordered construction or setup keeps an explicitly visible
stable core in later Steps, even if later wording omits the phrase "the same": a completed artifact
retains its visibly described support frame, and a cumulative arrangement retains a named tool that
is still explicitly shown. Do not infer identity from category repetition across independent scenes.

For each candidate, silently perform this counterfactual test: imagine supplying one isolated
reference image unchanged to every proposed Step. Reject the candidate if any assigned Step needs
the candidate itself to have a different outline, topology, fold or crease pattern, construction,
surface state, or identity. Changes only to contents, attachments, pose, action, lighting, camera,
or surroundings do not alter identity when the unchanged core remains visible. A container can
therefore persist while its contents change, and a support frame can persist while panels are
attached; a sheet being cut or folded cannot persist as one unchanged visual reference.

A stable identity shown in only one Step may optionally receive a reference when its specific
identity matters. Every candidate shown in two or more Steps that passes the counterfactual test
must enter the bank and must be routed to every Step where it is explicitly visible. If more than
two pass, keep the two with greatest Step coverage, then greatest importance to the main artifact.
Return an empty bank only when no candidate passes. Do not create references for generic category
members, incidental props, scenes, backgrounds, landforms, layouts, spatial relations, routes, or
information carriers. Output only the final schema; do not expose the test or reasoning. A
reference prompt depicts only the invariant target in isolation, without a changing state, action,
scene, or background.

Each item contains only fields that the image runtime consumes:
- key: stable identifier;
- type: reference;
- image_prompt: a self-contained image of only the stable reusable target.

Every selected key must be referenced by at least one step_conditions entry. Include every step_N
and use [] when no reference applies. Each step_conditions array contains only key strings from
condition_bank.items, never image prompts or descriptions. When items is empty, every
step_conditions array must be [].

Default output when there is no qualifying cross-step identity reference:
{"condition_bank":{"items":[]},"step_conditions":{step_condition_entries}}

Only when a qualifying reference exists, an item has exactly this shape:
{"key":"stable_identifier","type":"reference",
"image_prompt":"self-contained image of only the stable reusable target"}"""





PLAN4_TEXT_FREE_CONTRACT = """
Checks must rely on directly visible non-text evidence. Do not require reading or recognizing text,
symbols, labels, numbers, screens, or inscriptions.
"""












PLAN4_PROMPT = """Steps:
{steps_summary}

Give each step 1-2 checks that a reviewer can verify from that image. Check the assigned observable
evidence and any exact visible count required by the step. Output only step_N checks.

{"reviewer_checklist": [{step_checklist_entries}]}"""







PLAN1_PROMPT = _PLAN1_BASE_PROMPT
PLAN2_PROMPT = _PLAN2_BASE_PROMPT
PLAN3_PROMPT = _PLAN3_BASE_PROMPT


def prompt_template(stage):
    selected = {
        'plan1': PLAN1_PROMPT,
        'plan2': PLAN2_PROMPT,
        'plan3': PLAN3_PROMPT,
        'plan4': PLAN4_PROMPT,
    }
    if stage not in selected:
        raise ValueError(f'unknown prompt stage: {stage}')
    return selected[stage]


def build_step_condition_entries(step_count):
    entries = []
    for i in range(1, step_count + 1):
        entries.append(f'"step_{i}": []')
    return '{\n      ' + ',\n      '.join(entries) + '\n      }'


def build_step_semantic_entries(steps):
    """Show every frozen Plan1 index in the Plan2 response skeleton."""
    entries = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        entries.append({
            'step_index': step.get('step_index'),
            'image_prompt': 'complete independently renderable visible state',
            'content': 'finished report prose',
        })
    return json.dumps(entries, ensure_ascii=False, separators=(',', ':'))


def build_step_checklist_entries(step_count):
    entries = []
    for i in range(1, step_count + 1):
        entries.append(f'"step_{i}: <check>", "step_{i}: <check>"')
    return ', '.join(entries)


def _stringify(value, default=''):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else default


def _steps_summary(steps, fields):
    fields = tuple(fields)
    unsupported = sorted(set(fields) - STEP_SUMMARY_FIELDS)
    if unsupported:
        raise ValueError(f'unsupported step summary fields: {unsupported}')
    lines = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        parts = [
            f'{field}={_stringify(step.get(field))}' for field in fields]
        suffix = f": {'; '.join(parts)}" if parts else ''
        lines.append(f"Step {step.get('step_index', '?')}{suffix}")
    return '\n'.join(lines) if lines else '(no valid steps)'


def build_plan1_request(task_prompt):
    return prompt_template('plan1').replace(
        '{task}', _stringify(task_prompt))


def build_plan2_request(task_prompt, plan1):
    plan1 = plan1 if isinstance(plan1, dict) else {}
    steps = (
        plan1.get('steps', [])
        if isinstance(plan1.get('steps'), list) else [])
    return (prompt_template('plan2')
            .replace('{task}', _stringify(task_prompt))
            .replace('{steps_summary}', _steps_summary(
                steps, ('image_subject', 'content_goal')))
            .replace('{endpoint}', _stringify(plan1.get('endpoint')))
            .replace('{step_indexes}', json.dumps(
                [step.get('step_index') for step in steps
                 if isinstance(step, dict)],
                ensure_ascii=False, separators=(',', ':')))
            .replace('{step_semantic_entries}',
                     build_step_semantic_entries(steps)))


def _stage_steps(plan1, plan2, step_source, stage):
    """Return the explicitly selected stage input; no implicit source selection."""
    sources = {'plan1': plan1, 'plan2': plan2}
    if step_source not in sources:
        raise TypeError(
            f'{stage} step_source must be explicitly set to plan1 or plan2')
    source = sources[step_source]
    steps = source.get('steps') if isinstance(source, dict) else None
    if not isinstance(steps, list):
        raise TypeError(f'{stage} selected {step_source}, but it has no steps list')
    return steps


def build_plan3_request(task_prompt, plan1, plan2):
    plan1 = plan1 if isinstance(plan1, dict) else {}
    plan2 = plan2 if isinstance(plan2, dict) else {}
    semantic_steps = _stage_steps(plan1, plan2, 'plan1', 'plan3')
    steps = _stage_steps(plan1, plan2, 'plan2', 'plan3')
    if ([step.get('step_index') for step in semantic_steps]
            != [step.get('step_index') for step in steps]):
        raise ValueError('plan3 requires matching Plan1 and Plan2 step indexes')
    return (prompt_template('plan3')
            .replace('{task}', _stringify(task_prompt))
            .replace('{semantic_subjects}', _steps_summary(
                semantic_steps, ('image_subject',)))
            .replace('{steps_summary}', _steps_summary(
                steps, ('image_prompt',)))
            .replace('{step_condition_entries}',
                     build_step_condition_entries(len(steps))))


def build_plan4_request(plan2):
    plan2 = plan2 if isinstance(plan2, dict) else {}
    steps = plan2.get('steps')
    if not isinstance(steps, list):
        raise TypeError('plan4 selected plan2, but it has no steps list')
    checklist_entries = build_step_checklist_entries(len(steps))
    return (PLAN4_TEXT_FREE_CONTRACT
            + prompt_template('plan4')
            .replace('{steps_summary}', _steps_summary(
                steps, ('image_prompt',)))
            .replace('{step_checklist_entries}', checklist_entries))


STAGE_HEADERS = {
    'plan1': 'Stage 1 — task scope and sequence.',
    'plan2': 'Stage 2 — final step semantics.',
    'plan3': 'Stage 3 — reusable image references.',
    'plan4': 'Stage 4 — image checklist.',
}


def _with_stage_header(stage, text):
    header = STAGE_HEADERS[stage] + '\n\n'
    if stage == 'plan4':
        index = text.find('Steps:')
        if index >= 0:
            return text[:index] + header + text[index:]
    return header + text


def build_plan_request(stage, task_prompt, plan):
    """Build one Planner-stage request from the accumulated parsed blocks."""
    blocks = plan if isinstance(plan, dict) else {}
    plan1 = blocks.get('plan1', {})
    plan2 = blocks.get('plan2', {})
    builders = {
        'plan1': lambda: build_plan1_request(task_prompt),
        'plan2': lambda: build_plan2_request(task_prompt, plan1),
        'plan3': lambda: build_plan3_request(task_prompt, plan1, plan2),
        'plan4': lambda: build_plan4_request(plan2),
    }
    if stage not in builders:
        raise ValueError(f'unknown plan stage: {stage}')
    return _with_stage_header(stage, builders[stage]())
