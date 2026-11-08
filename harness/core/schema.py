"""Closed field schemas for the sole UniWeaver production contract."""

PLAN_STAGE_NAMES = ('plan1', 'plan2', 'plan3', 'plan4')
PLAN1_FIELDS = frozenset({'title', 'endpoint', 'steps'})
PLAN1_STEP_FIELDS = frozenset(
    {'step_index', 'image_subject', 'content_goal'})
PLAN2_FIELDS = frozenset({'visual_style', 'steps'})
PLAN2_STEP_FIELDS = frozenset(
    {'step_index', 'image_prompt', 'content'})
VISUAL_STYLE_FIELDS = (
    'medium', 'color_treatment', 'lighting_character', 'rendering_finish')
PLAN3_FIELDS = frozenset({'condition_bank', 'step_conditions'})
CONDITION_BANK_FIELDS = frozenset({'items'})
CONDITION_ITEM_FIELDS = frozenset({'key', 'type', 'image_prompt'})
CONDITION_TYPE = 'reference'
PLAN4_FIELDS = frozenset({'reviewer_checklist'})

# Every semantic leaf emitted by Plan1-4. FIELD_LINEAGE must cover this exact
# set; adding a schema field without ownership therefore fails tests.
PLANNER_FIELD_PATHS = frozenset({
    'plan1.title',
    'plan1.endpoint',
    'plan1.steps[].step_index',
    'plan1.steps[].image_subject',
    'plan1.steps[].content_goal',
    'plan2.visual_style.*',
    'plan2.steps[].step_index',
    'plan2.steps[].image_prompt',
    'plan2.steps[].content',
    'plan3.condition_bank.items[].key',
    'plan3.condition_bank.items[].type',
    'plan3.condition_bank.items[].image_prompt',
    'plan3.step_conditions',
    'plan4.reviewer_checklist',
})

RUNTIME_FIELD_PATHS = frozenset({
    'task_prompt',
    'step_cache',
    'reviewer.step_feedback',
    'runtime.condition_images',
    'runtime.image_seed',
    'runtime.inference_hyperparameters',
    'runtime_plan.steps[].condition_refs',
    'step_execution.image',
    'condition_execution.image',
})

ALL_LINEAGE_PATHS = PLANNER_FIELD_PATHS | RUNTIME_FIELD_PATHS
