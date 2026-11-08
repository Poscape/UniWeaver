"""Backend-neutral UniWeaver output transitions for unified UMM runs."""

from __future__ import annotations

from core.policy_registry import (
    PLANNER_TRANSPORT_EVENT,
    RUNTIME_MODEL_OPERATION_EVENT,
)
from core.production import CONTRACT_VERSION, TRAJECTORY_DESIGN
from inferencer_output_manager import OutputManager


def _execution_recoveries(trajectory):
    validation = trajectory.get("_text_validation", {})
    issues = validation.get("issues", []) if isinstance(validation, dict) else []
    recovery_types = {
        "parser_recovery", "mechanical_repair",
    }
    return [
        issue for issue in issues
        if isinstance(issue, dict) and issue.get("type") in recovery_types
    ]


def _empty_runtime_plan(task_prompt):
    return {
        "title": "",
        "visual_style": {},
        "steps": [],
        "condition_bank": {"items": []},
        "reviewer_checklist": [],
        "reviewer_checklist_available": False,
        "trajectory_design": TRAJECTORY_DESIGN,
        "contract_version": CONTRACT_VERSION,
        "_user_prompt": task_prompt,
        "raw_input": task_prompt,
        "errors": [],
        "warnings": [],
    }


def step_trace(result):
    """Record the honest execution trace of one Step image call."""
    return {
        "stage": "image_gen",
        "raw_input": result.get("raw_input", ""),
        "raw_output": "",
        "duration": result.get("duration", 0),
        "retry_round": result.get("retry_round"),
        "input_structured": result.get("input_structured", {}),
        "parsed": {
            "semantic_image_prompt": (
                result["render_input"].semantic_image_prompt),
            "model_image_prompt": result["render_input"].model_image_prompt,
            "content": result["render_input"].content,
        },
    }


class UniWeaverPersistenceTransition:
    """Own the state transitions shared by every UMM UniWeaver backend."""

    def __init__(self, manager: OutputManager, provenance: dict):
        self.manager = manager
        self.provenance = provenance
        self._persisted_conditions = set()
        self._persisted_steps = set()

    @staticmethod
    def _image_identity(provenance):
        execution = provenance.get("image_execution_identity")
        runtime = provenance.get("image_runtime_identity")
        if execution is None:
            from core.production import image_execution_identity
            execution = image_execution_identity()
        if runtime is None:
            from core.production import image_runtime_identity
            runtime = image_runtime_identity()
        return {
            "image_generation": (
                provenance.get("backend_image_config")
                or provenance.get("image_generation", {})
            ),
            "image_execution": execution,
            "image_runtime": runtime,
        }

    @classmethod
    def start(
        cls,
        output,
        task_key,
        task_prompt,
        provenance,
        runtime_plan=None,
        trajectory=None,
        resume=False,
        started_at=None,
    ):
        runtime_plan = runtime_plan or _empty_runtime_plan(task_prompt)
        step_generations = {
            key: value.get("parsed", {})
            for key, value in (trajectory or {}).get(
                "step_generations", {}).items()
        }
        manager = OutputManager(
            output,
            task_key,
            task_prompt,
            runtime_plan,
            step_generations,
            resume=resume,
            started_at=started_at,
        )
        transition = cls(manager, provenance)
        image_identity = cls._image_identity(provenance)
        identity = {
            "image_seed": provenance.get("base_seed"),
            "model_identity": provenance.get("model_identity", {}),
            "review_retry": bool(provenance.get("review_retry", False)),
            **image_identity,
        }
        if resume:
            existing_experiment = manager.state.get("image_experiment", {})
            if existing_experiment.get("identity") != identity:
                raise ValueError(
                    "resume image experiment settings do not match persisted state")
        done_conditions, done_steps = manager.completed_items() if resume else (set(), set())
        transition._persisted_conditions.update(done_conditions)
        transition._persisted_steps.update(done_steps)
        manager.record_image_experiment(
            identity=identity,
            provenance={
                "runtime_configuration": provenance,
                **image_identity,
            },
        )
        manager.update_state(
            trace={"runtime_configuration": provenance},
            image_experiment={
                **manager.state.get("image_experiment", {}),
                "planner_calls": True,
                "condition_generation": True,
                "review_retry": bool(provenance.get("review_retry", False)),
            },
        )
        manager.log("run", "start", "starting", mode="umm_uniweaver")
        return transition

    def prepare(self, trajectory, runtime_plan):
        self.manager.update_state(
            plan=runtime_plan,
            trajectory_design=runtime_plan["trajectory_design"],
            contract_version=runtime_plan["contract_version"],
            title=runtime_plan.get("title", ""),
            step_generations_raw={
                key: value.get("parsed", {})
                for key, value in trajectory.get("step_generations", {}).items()
            },
            trace={
                "runtime_configuration": self.provenance,
                "raw_attempts": trajectory.get("raw_attempts", {}),
                "planner_plan": trajectory.get("plan", {}),
                "text_validation": trajectory.get("_text_validation", {}),
                "execution_recoveries": _execution_recoveries(trajectory),
            },
        )

    def persist_planner_failure(self, trajectory):
        """Persist the complete failed Planner exchange before marking error."""
        self.manager.update_state(
            trajectory_design=trajectory.get("trajectory_design"),
            contract_version=trajectory.get("contract_version"),
            trace={
                "runtime_configuration": self.provenance,
                "raw_attempts": trajectory.get("raw_attempts", {}),
                "planner_plan": trajectory.get("plan", {}),
                "text_validation": trajectory.get("_text_validation", {}),
                "construction_issues": trajectory.get("_construction_issues", []),
                "partial_plan": trajectory.get("plan", {}),
                "execution_recoveries": _execution_recoveries(trajectory),
            },
        )

    def persist_condition(self, key, record, image):
        image_sha256 = self.manager.condition_done(
            key,
            {
                "image": image,
                "type": record.get("type", "reference"),
                "image_prompt": record.get("semantic_image_prompt", ""),
                "raw_input": record.get("raw_input", ""),
                "raw_output": record.get("raw_output", ""),
            },
        )
        self._persisted_conditions.add(key)
        return image_sha256

    def persist_step(self, step_index, result, runtime_step):
        render_input = result["render_input"]
        image_sha256 = self.manager.step_done(
            step_index,
            {
                "image": result["image"],
                "content": runtime_step.get("content", ""),
                "semantic_image_prompt": render_input.semantic_image_prompt,
                "model_image_prompt": render_input.model_image_prompt,
                "condition_refs": result["condition_refs"],
                # The execution-side split (passed/used/ignored) is persisted by
                # the output manager's step record so a resume cannot claim the
                # backend received references it never got.
                "condition_refs_passed": result.get("condition_refs_passed"),
                "condition_refs_used": result.get("condition_refs_used"),
                "condition_refs_ignored": result.get("condition_refs_ignored"),
                "image_seed": result["image_seed"],
                "retry_round": result.get("retry_round"),
                "raw_input": result.get("raw_input", ""),
                "feedback": result.get("feedback", ""),
                "img_trace": step_trace(result),
            },
        )
        self._persisted_steps.add(str(step_index))
        return image_sha256

    def persist_review(self, review):
        self.manager.review_done(review)

    def finish(self, trajectory, runtime_plan, execution):
        manager = self.manager
        if manager.state.get("plan") != runtime_plan:
            self.prepare(trajectory, runtime_plan)

        runtime_steps = {
            step["step_index"]: step for step in runtime_plan.get("steps", [])
        }
        for key, record in execution["condition_records"].items():
            if key not in self._persisted_conditions:
                self.persist_condition(
                    key, record, execution["condition_images"][key])
        for step_index in sorted(execution["steps"]):
            if str(step_index) not in self._persisted_steps:
                self.persist_step(
                    step_index,
                    execution["steps"][step_index],
                    runtime_steps[step_index],
                )

        manager.update_state(
            trace={
                "runtime_configuration": self.provenance,
                "raw_attempts": trajectory.get("raw_attempts", {}),
                "planner_plan": trajectory.get("plan", {}),
                "text_validation": trajectory.get("_text_validation", {}),
                "execution_recoveries": _execution_recoveries(trajectory),
                "execution": {
                    "conditions": execution["condition_records"],
                    "steps": {
                        str(index): {
                            key: value
                            for key, value in result.items()
                            if key not in {"image", "render_input"}
                        }
                        for index, result in execution["steps"].items()
                    },
                    "review": execution.get("review", {"enabled": False}),
                },
            }
        )
        manager.mark_done()
        return manager

    def fail(self, error, failure_kind="image_error"):
        classification = (
            "runtime_timeout" if failure_kind == "timed_out"
            else "model_or_runtime_failure"
        )
        try:
            self.manager.update_state(
                failure_kind=failure_kind,
                failure_provenance={
                    "policy_id": (
                        PLANNER_TRANSPORT_EVENT["policy_id"]
                        if failure_kind == "planner_error"
                        else RUNTIME_MODEL_OPERATION_EVENT["policy_id"]
                    ),
                    "classification": classification,
                    "resolved": True,
                    "source": "runtime_execution",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                },
            )
            self.manager.mark_error(f"{type(error).__name__}: {error}")
        except Exception as persistence_error:  # noqa: BLE001
            self.manager.record_persistence_failure(
                persistence_error, operation="persist_terminal_failure",
            )
            if persistence_error is not error and hasattr(error, "add_note"):
                error.add_note(
                    "terminal persistence also failed: "
                    f"{type(persistence_error).__name__}: {persistence_error}"
                )
