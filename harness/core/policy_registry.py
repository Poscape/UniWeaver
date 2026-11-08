"""Single machine-readable registry for recovery, runtime, and scoring policy.

Execution modules import policy identifiers and gate definitions from here.
"""

from __future__ import annotations

from collections.abc import Iterable

EVALUATION_SPEC_ID = "uniweaver-report-only"
REPORT_REQUEST_LAYERS = {
    "TC": "report_text_atomic",
    "WQ": "content_quality_atomic",
    "VC": "visual_atomic",
    "ITA": "report_interleaved_atomic",
}

RECOVERY_POLICIES = (
    {
        "policy_id": "json_payload_extraction_v1",
        "category": "parser_recovery",
        "methods": ("normalized_direct", "markdown", "text_prefix", "brace_match"),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": "continue only when the extracted object passes stage validation",
        "strict_result": "invalid envelope; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "json_syntax_closure_v1",
        "category": "parser_recovery",
        "methods": ("truncated_repaired", "stack_closed", "common_typo_repaired"),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": "continue only when repaired JSON passes stage validation",
        "strict_result": "invalid raw JSON; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "json_misordered_closure_v1",
        "category": "parser_recovery",
        "methods": ("misordered_closed",),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": (
            "continue only when one trailing mismatched closer is dropped, the "
            "stack closes uniquely and the repaired JSON passes stage validation"
        ),
        "strict_result": "invalid raw JSON; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "json_trailing_comma_recovery_v1",
        "category": "parser_recovery",
        "methods": ("trailing_comma_repaired",),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": "continue only when the repaired JSON passes stage validation",
        "strict_result": "invalid raw JSON; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "json_duplicate_key_last_value_v1",
        "category": "parser_recovery",
        "methods": ("duplicate_key_last_value",),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": "use JSON final-key semantics only; duplicate evidence is retained",
        "strict_result": "duplicate keys fail exact protocol/training validation",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "json_missing_comma_recovery_v1",
        "category": "parser_recovery",
        "methods": ("missing_comma_inserted",),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": (
            "continue only when the parser reports a missing comma at a member "
            "or element boundary, inserting one comma there yields valid JSON, "
            "and the result passes stage validation"
        ),
        "strict_result": "invalid raw JSON; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "json_split_object_merge_v1",
        "category": "parser_recovery",
        "methods": ("split_objects",),
        "stages": ("plan1", "plan2", "plan3", "plan4"),
        "step_gate": "continue only when the merged object passes stage validation",
        "strict_result": "invalid envelope; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "plan4_checklist_wrapper_v1",
        "category": "mechanical_repair",
        "methods": ("top_level_string_list_to_reviewer_checklist",),
        "stages": ("plan4",),
        "step_gate": "only a complete top-level list of strings may be wrapped",
        "strict_result": "schema-invalid repair; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "plan3_step_conditions_merge_v1",
        "category": "mechanical_repair",
        "methods": ("merge_complete_step_conditions_fragments",),
        "stages": ("plan3",),
        "step_gate": "complete, conflict-free and reference-closed; count is projected later",
        "strict_result": "schema-invalid repair; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "plan1_step_alias_v1",
        "category": "mechanical_repair",
        "methods": (
            "index_to_step_index",
            "subject_to_image_subject",
            "goal_to_content_goal",
        ),
        "stages": ("plan1",),
        "step_gate": "continue only when every step maps without conflict or missing values",
        "strict_result": "schema-invalid repair; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "plan2_step_index_realignment_v1",
        "category": "mechanical_repair",
        "methods": ("realign_step_index_by_position",),
        "stages": ("plan2",),
        "step_gate": (
            "continue only when Plan2 has exactly as many steps as the frozen "
            "Plan1, every step carries a complete image_prompt and content, and "
            "the only difference is the copied step_index value"
        ),
        "strict_result": "schema-invalid repair; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "plan2_step_alias_v1",
        "category": "mechanical_repair",
        "methods": (
            "plan2_subject_to_image_prompt",
            "plan2_index_to_step_index",
        ),
        "stages": ("plan2",),
        "step_gate": (
            "continue only when Plan2 has exactly as many steps as the frozen "
            "Plan1, every step's prose is complete and not a prompt example, and "
            "renaming the registered synonyms leaves every step with a complete "
            "image_prompt whose value is not a prompt example"
        ),
        "strict_result": "schema-invalid repair; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "plan2_visual_style_default_v1",
        "category": "mechanical_repair",
        "methods": ("plan2_missing_visual_style_to_default",),
        "stages": ("plan2",),
        "step_gate": (
            "continue only when Plan2's steps are complete, task-specific and "
            "free of prompt examples, and only the rendering metadata "
            "(visual_style) is missing or empty"
        ),
        "strict_result": "schema-invalid repair; never training-eligible",
        "report_score": "no direct penalty; independently score any completed final report",
    },
    {
        "policy_id": "judge_json_payload_extraction_v1",
        "category": "judge_parser_recovery",
        # Appends only the closers the open-bracket stack forces; the result
        # still has to pass the caller's verdict-schema check.
        "methods": (
            "judge_markdown_fence",
            "judge_stack_closure",
            "judge_object_extraction",
        ),
        "stages": ("judge",),
        "step_gate": "not applicable",
        "strict_result": "judge formatting recovery is retained as evaluation provenance",
        "report_score": "score only the validated recovered judgment",
    },
)


RUNTIME_PROJECTION_POLICIES = (
    {
        "policy_id": "runtime_condition_prefix_v1",
        "resource": "conditions",
        # The Plan3 contract asks for a 0-2 reference bank and the shipped 10K
        # corpus never exceeds two, so the execution default matches the
        # contract instead of carrying two extra slots of headroom.
        "default_limit": 2,
        "operation": "retain_declared_condition_prefix",
    },
    {
        "policy_id": "runtime_step_prefix_v1",
        "resource": "steps",
        "default_limit": 8,
        "operation": "retain_declared_step_prefix",
    },
    {
        "policy_id": "runtime_condition_reference_filter_v1",
        "resource": "condition_refs",
        "operation": "remove_unavailable_condition_references",
    },
    {
        "policy_id": "runtime_unreferenced_condition_filter_v1",
        "resource": "conditions",
        "operation": "remove_conditions_unreferenced_by_retained_steps",
    },
    {
        "policy_id": "runtime_checklist_step_reference_filter_v1",
        "resource": "reviewer_checklist",
        "operation": "remove_checks_for_unavailable_steps",
    },
    {
        "policy_id": "runtime_checklist_per_step_prefix_v1",
        "resource": "checks_per_step",
        "default_limit": 3,
        "operation": "retain_per_step_check_prefix",
        "character_limit": None,
    },
    {
        "policy_id": "runtime_general_check_application_v1",
        "resource": "reviewer_checklist",
        "operation": "apply_general_checks_to_each_step_within_step_budget",
    },
)


EVENT_POLICIES = (
    {
        "policy_id": "planner_transport_attempt_v1",
        "category": "model_call",
        "stages": ("plan1", "plan2", "plan3", "plan4"),
    },
    {
        "policy_id": "runtime_model_operation_v1",
        "category": "model_call",
        "stages": ("condition_generation", "step_generation", "step_review"),
    },
    {
        "policy_id": "minimal_step_operation_v1",
        "category": "model_call",
        "stages": ("minimal_step",),
    },
    {
        "policy_id": "step_attempt_persistence_v1",
        "category": "persistence",
        "stages": ("initial_step", "regenerated_step"),
    },
    {
        "policy_id": "review_variant_derivation_v1",
        "category": "evaluation_preparation",
        "stages": ("no_review", "reviewed"),
    },
    {
        "policy_id": "persistence_failure_v1",
        "category": "persistence",
        "stages": ("all",),
    },
    {
        "policy_id": "orchestration_failure_v1",
        "category": "orchestration",
        "stages": ("preflight", "batch", "parallel_stage"),
    },
    {
        "policy_id": "interruption_recovery_v1",
        "category": "interruption",
        "stages": ("resume",),
    },
    {
        "policy_id": "evaluation_judge_call_v1",
        "category": "evaluation_model_call",
        "stages": ("judge",),
    },
)


BOUNDARY_BINDINGS = (
    {"boundary": "planner_transport", "source": "core/serialization.py",
     "symbol": "PLANNER_TRANSPORT_EVENT", "policy_id": "planner_transport_attempt_v1"},
    {"boundary": "uniweaver_model_operation", "source": "inferencer_output_manager.py",
     "symbol": "RUNTIME_MODEL_OPERATION_EVENT", "policy_id": "runtime_model_operation_v1"},
    {"boundary": "minimal_step", "source": "baselines/umm_minimal_persistence.py",
     "symbol": "MINIMAL_STEP_OPERATION_EVENT", "policy_id": "minimal_step_operation_v1"},
    {"boundary": "step_attempt_persistence", "source": "inferencer_output_manager.py",
     "symbol": "STEP_ATTEMPT_PERSISTENCE_EVENT",
     "policy_id": "step_attempt_persistence_v1"},
    {"boundary": "review_variant_derivation",
     "source": "baselines/umm_evaluation_prepare.py",
     "symbol": "REVIEW_VARIANT_DERIVATION_EVENT",
     "policy_id": "review_variant_derivation_v1"},
    {"boundary": "persistence_failure", "source": "inferencer_output_manager.py",
     "symbol": "PERSISTENCE_FAILURE_EVENT", "policy_id": "persistence_failure_v1"},
    {"boundary": "orchestration_failure", "source": "baselines/umm_benchmark_runner.py",
     "symbol": "ORCHESTRATION_FAILURE_EVENT", "policy_id": "orchestration_failure_v1"},
    {"boundary": "interruption_recovery", "source": "inferencer_output_manager.py",
     "symbol": "INTERRUPTION_RECOVERY_EVENT", "policy_id": "interruption_recovery_v1"},
    {"boundary": "evaluation_judge", "source": "evaluation/uniweaver_eval/provider.py",
     "symbol": "EVALUATION_JUDGE_CALL_EVENT", "policy_id": "evaluation_judge_call_v1"},
    {"boundary": "judge_json_recovery", "source": "evaluation/uniweaver_eval/provider.py",
     "symbol": "JUDGE_JSON_EXTRACTION_POLICY", "policy_id": "judge_json_payload_extraction_v1"},
)

OPERATION_PROVENANCE_FIELDS = frozenset({
    "sequence", "policy_id", "operation", "input_sha256", "output_sha256",
})


GATE_POLICIES = (
    {
        "gate_id": "PLAN1_RENDER_CONTRACT",
        "kind": "continue",
        "tracks": ("step", "trajectory"),
    },
    {
        "gate_id": "PLAN2_RENDER_CONTRACT",
        "kind": "continue",
        "tracks": ("step", "trajectory"),
    },
    {
        "gate_id": "RENDER_ELIGIBLE",
        "kind": "continue",
        "tracks": ("step",),
        "validation_field": "render_eligible",
        "ignored_stages": (),
        "ignored_issue_types": (
            "envelope_error", "mechanical_repair", "parser_recovery",
        ),
    },
    {
        "gate_id": "REPORT_ORDERED_UNITS",
        "kind": "zero_score",
        "tracks": ("report",),
        "criterion": "nonempty_ordered_units",
        # Below ``minimum_units`` is recorded as a warning, not a failure: a run
        # that produced at least one real unit still reports something.
        "minimum_units": 2,
        "hard_minimum_units": 1,
        # The visual judge receives at most this many images per task
        # (judge configs: max_images_per_task), so a longer report cannot be
        # scored end to end and fails the structure gate.
        "max_units": 8,
    },
    {
        "gate_id": "REPORT_READER_CONTENT",
        "kind": "zero_score",
        "tracks": ("report",),
        "criterion": "nonempty_reader_content",
    },
    {
        "gate_id": "REPORT_IMAGE_PAIRING",
        "kind": "zero_score",
        "tracks": ("report",),
        "criterion": "one_to_one_existing_image_pairing",
    },
    {
        "gate_id": "REPORT_TERMINAL_STATUS",
        "kind": "diagnostic",
        "tracks": ("report",),
        "criterion": "explicit_terminal_completion",
        "accepted_statuses": ("completed", "done", "succeeded"),
        # A run that produced units but never announced completion is treated as
        # having finished at its last step, and flagged with a warning.
        "infer_from_units": True,
    },
    {
        "gate_id": "REPORT_FAILURE_PROVENANCE",
        "kind": "attribution",
        "tracks": ("report",),
        "zero_kinds": ("model_failure", "model_rejection", "run_failure"),
        "unresolved_kinds": (
            "artifact_collection", "file_omission", "path_error",
        ),
    },
    {
        "gate_id": "REPORT_INPUT_INTEGRITY",
        "kind": "resolution",
        "tracks": ("report",),
        "checks": (
            "frozen_task_prompt_hash", "unique_request_per_task_and_metric",
            "canonical_metric_layer", "request_content_hash",
            "judgment_request_binding", "raw_input_binding",
            "configured_judge_labels", "valid_repeat_index_binding",
        ),
    },
    {
        "gate_id": "PANEL_RESOLUTION",
        "kind": "resolution",
        "tracks": ("report",),
    },
)


def _index(records: Iterable[dict], key: str, registry_name: str) -> dict[str, dict]:
    indexed = {}
    for record in records:
        identifier = record.get(key)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"{registry_name} entry has no {key}")
        if identifier in indexed:
            raise ValueError(f"duplicate {registry_name} identifier: {identifier}")
        indexed[identifier] = record
    return indexed


RECOVERY_POLICY_BY_ID = _index(RECOVERY_POLICIES, "policy_id", "recovery policy")
RUNTIME_POLICY_BY_ID = _index(
    RUNTIME_PROJECTION_POLICIES, "policy_id", "runtime policy",
)
GATE_POLICY_BY_ID = _index(GATE_POLICIES, "gate_id", "gate")
EVENT_POLICY_BY_ID = _index(EVENT_POLICIES, "policy_id", "event policy")

PLAN1_STEP_ALIAS_POLICY = RECOVERY_POLICY_BY_ID["plan1_step_alias_v1"]
PLAN2_STEP_ALIAS_POLICY = RECOVERY_POLICY_BY_ID["plan2_step_alias_v1"]
PLAN2_VISUAL_STYLE_DEFAULT_POLICY = RECOVERY_POLICY_BY_ID[
    "plan2_visual_style_default_v1"
]
PLAN2_STEP_INDEX_REALIGNMENT_POLICY = RECOVERY_POLICY_BY_ID[
    "plan2_step_index_realignment_v1"
]
JSON_PAYLOAD_EXTRACTION_POLICY = RECOVERY_POLICY_BY_ID[
    "json_payload_extraction_v1"
]
JSON_SYNTAX_CLOSURE_POLICY = RECOVERY_POLICY_BY_ID["json_syntax_closure_v1"]
JSON_MISORDERED_CLOSURE_POLICY = RECOVERY_POLICY_BY_ID["json_misordered_closure_v1"]
JSON_TRAILING_COMMA_POLICY = RECOVERY_POLICY_BY_ID[
    "json_trailing_comma_recovery_v1"
]
JSON_MISSING_COMMA_POLICY = RECOVERY_POLICY_BY_ID[
    "json_missing_comma_recovery_v1"
]
JSON_DUPLICATE_KEY_POLICY = RECOVERY_POLICY_BY_ID[
    "json_duplicate_key_last_value_v1"
]
JSON_SPLIT_OBJECT_POLICY = RECOVERY_POLICY_BY_ID["json_split_object_merge_v1"]
PLAN3_FRAGMENT_MERGE_POLICY = RECOVERY_POLICY_BY_ID[
    "plan3_step_conditions_merge_v1"
]
PLAN4_CHECKLIST_WRAPPER_POLICY = RECOVERY_POLICY_BY_ID[
    "plan4_checklist_wrapper_v1"
]
JUDGE_JSON_EXTRACTION_POLICY = RECOVERY_POLICY_BY_ID[
    "judge_json_payload_extraction_v1"
]
RUNTIME_CONDITION_PREFIX_POLICY = RUNTIME_POLICY_BY_ID[
    "runtime_condition_prefix_v1"
]
RUNTIME_STEP_PREFIX_POLICY = RUNTIME_POLICY_BY_ID["runtime_step_prefix_v1"]
RUNTIME_CONDITION_REFERENCE_POLICY = RUNTIME_POLICY_BY_ID[
    "runtime_condition_reference_filter_v1"
]
RUNTIME_UNREFERENCED_CONDITION_POLICY = RUNTIME_POLICY_BY_ID[
    "runtime_unreferenced_condition_filter_v1"
]
RUNTIME_CHECKLIST_STEP_REFERENCE_POLICY = RUNTIME_POLICY_BY_ID[
    "runtime_checklist_step_reference_filter_v1"
]
RUNTIME_CHECKLIST_PREFIX_POLICY = RUNTIME_POLICY_BY_ID[
    "runtime_checklist_per_step_prefix_v1"
]
RUNTIME_GENERAL_CHECK_POLICY = RUNTIME_POLICY_BY_ID[
    "runtime_general_check_application_v1"
]

PLANNER_TRANSPORT_EVENT = EVENT_POLICY_BY_ID["planner_transport_attempt_v1"]
RUNTIME_MODEL_OPERATION_EVENT = EVENT_POLICY_BY_ID["runtime_model_operation_v1"]
MINIMAL_STEP_OPERATION_EVENT = EVENT_POLICY_BY_ID["minimal_step_operation_v1"]
STEP_ATTEMPT_PERSISTENCE_EVENT = EVENT_POLICY_BY_ID[
    "step_attempt_persistence_v1"
]
REVIEW_VARIANT_DERIVATION_EVENT = EVENT_POLICY_BY_ID[
    "review_variant_derivation_v1"
]
PERSISTENCE_FAILURE_EVENT = EVENT_POLICY_BY_ID["persistence_failure_v1"]
ORCHESTRATION_FAILURE_EVENT = EVENT_POLICY_BY_ID["orchestration_failure_v1"]
INTERRUPTION_RECOVERY_EVENT = EVENT_POLICY_BY_ID["interruption_recovery_v1"]
EVALUATION_JUDGE_CALL_EVENT = EVENT_POLICY_BY_ID["evaluation_judge_call_v1"]

RENDER_ELIGIBLE_GATE = GATE_POLICY_BY_ID["RENDER_ELIGIBLE"]
REPORT_ORDERED_UNITS_GATE = GATE_POLICY_BY_ID["REPORT_ORDERED_UNITS"]
REPORT_READER_CONTENT_GATE = GATE_POLICY_BY_ID["REPORT_READER_CONTENT"]
REPORT_IMAGE_PAIRING_GATE = GATE_POLICY_BY_ID["REPORT_IMAGE_PAIRING"]
REPORT_TERMINAL_STATUS_GATE = GATE_POLICY_BY_ID["REPORT_TERMINAL_STATUS"]
REPORT_FAILURE_PROVENANCE_GATE = GATE_POLICY_BY_ID[
    "REPORT_FAILURE_PROVENANCE"
]
REPORT_INPUT_INTEGRITY_GATE = GATE_POLICY_BY_ID["REPORT_INPUT_INTEGRITY"]


def recovery_policy_for_method(method: str) -> dict | None:
    matches = [
        policy for policy in RECOVERY_POLICIES
        if method in policy.get("methods", ())
    ]
    if len(matches) > 1:
        raise ValueError(f"recovery method has multiple policies: {method}")
    return matches[0] if matches else None


def validate_operation_provenance(operations: Iterable[dict]) -> None:
    """Require ordered, hash-bound operations owned by this registry."""
    known = set(RECOVERY_POLICY_BY_ID) | set(RUNTIME_POLICY_BY_ID)
    for sequence, operation in enumerate(operations, 1):
        missing = OPERATION_PROVENANCE_FIELDS - set(operation)
        if missing:
            raise ValueError(f"operation {sequence} is missing {sorted(missing)}")
        if operation["sequence"] != sequence:
            raise ValueError("operation sequence must be contiguous and ordered")
        if operation["policy_id"] not in known:
            raise ValueError(f"unregistered operation policy: {operation['policy_id']}")
        for field in ("input_sha256", "output_sha256"):
            value = operation[field]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"operation {sequence} has invalid {field}")


def validate_policy_registry() -> None:
    method_owners = {}
    for policy in RECOVERY_POLICIES:
        for method in policy["methods"]:
            if method in method_owners:
                raise ValueError(
                    f"recovery method {method} is owned by both "
                    f"{method_owners[method]} and {policy['policy_id']}"
                )
            method_owners[method] = policy["policy_id"]
    for policy in RUNTIME_PROJECTION_POLICIES:
        limit = policy.get("default_limit")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError(f"invalid runtime limit in {policy['policy_id']}")
    for event in EVENT_POLICIES:
        if not event.get("category") or not event.get("stages"):
            raise ValueError(f"invalid event policy in {event['policy_id']}")
    seen_boundaries = set()
    for binding in BOUNDARY_BINDINGS:
        if binding["boundary"] in seen_boundaries:
            raise ValueError(f"duplicate boundary binding: {binding['boundary']}")
        seen_boundaries.add(binding["boundary"])
        if (binding["policy_id"] not in EVENT_POLICY_BY_ID
                and binding["policy_id"] not in RECOVERY_POLICY_BY_ID):
            raise ValueError(
                f"boundary {binding['boundary']} uses an unregistered policy"
            )
    valid_tracks = {"step", "trajectory", "report"}
    for gate in GATE_POLICIES:
        if not gate.get("tracks") or set(gate["tracks"]) - valid_tracks:
            raise ValueError(f"invalid tracks in {gate['gate_id']}")
        minimum_units = gate.get("minimum_units")
        if minimum_units is not None and (
            not isinstance(minimum_units, int)
            or isinstance(minimum_units, bool)
            or minimum_units < 1
        ):
            raise ValueError(f"invalid minimum_units in {gate['gate_id']}")
        hard_minimum = gate.get("hard_minimum_units")
        if hard_minimum is not None and (
            not isinstance(hard_minimum, int)
            or isinstance(hard_minimum, bool)
            or hard_minimum < 1
            or (minimum_units is not None and hard_minimum > minimum_units)
        ):
            raise ValueError(
                f"invalid hard_minimum_units in {gate['gate_id']}"
            )
        if "infer_from_units" in gate and not isinstance(
            gate["infer_from_units"], bool
        ):
            raise ValueError(f"invalid infer_from_units in {gate['gate_id']}")


validate_policy_registry()
