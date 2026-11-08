import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.run_eval import command_collect_inputs
from evaluation.uniweaver_eval.data import TaskSpec
from evaluation.uniweaver_eval.prompts import (
    build_report_request,
    build_report_requests,
)
from evaluation.uniweaver_eval.reports import (
    ReportArtifact,
    _from_sequential,
    _has_reader_content,
    _is_completion_boilerplate,
    _is_input_echo,
    _reader_text,
    input_echo_units,
    prose_completion_tasks,
    report_contract,
)
from evaluation.uniweaver_eval.rubric import dimensions_for

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCORE_TABLE = REPOSITORY_ROOT.parent / "scoring" / "scripts" / "score_table.py"


def _score_table():
    """Load the scoring CLI without leaving a bytecode cache behind."""
    module = types.ModuleType("score_table")
    module.__file__ = str(SCORE_TABLE)
    source = SCORE_TABLE.read_text(encoding="utf-8")
    exec(  # noqa: S102 - local CLI module, loaded without a bytecode cache
        compile(source, str(SCORE_TABLE), "exec"), module.__dict__)
    return module


_PLACEHOLDER_DIR = tempfile.TemporaryDirectory(prefix="report-units-")


def _placeholder_images():
    """Two existing files with distinct bytes; the contract only hashes them."""
    directory = Path(_PLACEHOLDER_DIR.name)
    for index in (1, 2):
        path = directory / f"unit_{index}.png"
        if not path.exists():
            path.write_bytes(b"unit %d\n" % index)
    return [directory / "unit_1.png", directory / "unit_2.png"]


class EvaluatorTests(unittest.TestCase):
    @staticmethod
    def _report_judgments(
        requests_by_metric,
        judge_labels_by_metric,
        required_repeats,
        tier_overrides=None,
    ):
        tier_overrides = tier_overrides or {}
        judgments_by_metric = {}
        for metric, requests in requests_by_metric.items():
            judgments = []
            for request in requests:
                for label in judge_labels_by_metric[metric]:
                    for repeat in range(1, required_repeats + 1):
                        raw_input = dict(request)
                        raw_input["request_id"] = (
                            f"{request['request_id']}:r{repeat}"
                        )
                        raw_input["repeat_index"] = repeat
                        tiers = {
                            criterion["id"]: {"tier": "MET", "evidence": "Visible."}
                            for criterion in request["criteria"]
                        }
                        tiers.update(
                            tier_overrides.get(
                                (metric, request["task_id"], label), {}
                            )
                        )
                        judgments.append({
                            "status": "ok",
                            "task_id": request["task_id"],
                            "run_id": request["run_id"],
                            "family": request["family"],
                            "layer": request["layer"],
                            "judge_label": label,
                            "repeat_index": repeat,
                            "request_id": f"{request['request_id']}:r{repeat}",
                            "request_content_sha256": (
                                request["request_content_sha256"]
                            ),
                            "raw_input": raw_input,
                            "judgment": {
                                "verdicts": {metric: tiers},
                            },
                        })
            judgments_by_metric[metric] = judgments
        return judgments_by_metric

    @staticmethod
    def _report_requests(tasks, artifacts):
        return {
            metric: build_report_requests(tasks, artifacts, layer)
            for metric, layer in {
                "TC": "report_text_atomic",
                "WQ": "content_quality_atomic",
                "VC": "visual_atomic",
                "ITA": "report_interleaved_atomic",
            }.items()
        }

    @staticmethod
    def _report_artifact(task, *, passing):
        source = Path(__file__)
        images = _placeholder_images()
        units = [
            {
                "unit_index": 1,
                "content": "The tray is ready to serve.",
                "image_path": str(images[0]),
            },
            {
                "unit_index": 2,
                "content": "The finished tray is on the table.",
                "image_path": str(images[1]),
            },
        ] if passing else []
        return ReportArtifact(
            task_id=task.task_id,
            run_id="run",
            prompt=task.prompt,
            units=units,
            image_paths=images if passing else [],
            source_path=source,
            declared_status="completed",
        )


    def test_report_contract_keeps_three_hard_gates_and_one_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "step_1.png"
            second = root / "step_2.png"
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt="task prompt",
                units=[
                    {
                        "unit_index": 1,
                        "content": "A complete reader-facing unit.",
                        "image_path": str(first),
                    },
                    {
                        "unit_index": 2,
                        "content": "A second reader-facing unit.",
                        "image_path": str(second),
                    },
                ],
                image_paths=[first, second],
                source_path=root / "run.json",
                declared_status="done",
            )
            result = report_contract(artifact)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["checks"]), 3)
        self.assertEqual(len(result["diagnostics"]), 1)
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_report_contract_rejects_duplicate_image_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "step_1.png"
            second = root / "step_2.png"
            first.write_bytes(b"same image")
            second.write_bytes(b"same image")
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt="task prompt",
                units=[
                    {
                        "unit_index": 1,
                        "content": "A complete reader-facing unit.",
                        "image_path": str(first),
                    },
                    {
                        "unit_index": 2,
                        "content": "A second reader-facing unit.",
                        "image_path": str(second),
                    },
                ],
                image_paths=[first, second],
                source_path=root / "run.json",
                declared_status="done",
            )
            result = report_contract(artifact)

        images = next(
            check for check in result["checks"]
            if check["criterion"] == "one_to_one_existing_image_pairing"
        )
        self.assertFalse(images["passed"])
        self.assertIn("unique_hashes=1", images["observation"])
        self.assertEqual(result["status"], "FAIL")

    def test_one_unit_passes_with_a_warning(self) -> None:
        """A thin report still reports something, so it warns rather than fails."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "step_1.png"
            image.write_bytes(b"png")
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt="task prompt",
                units=[
                    {
                        "unit_index": 1,
                        "content": "A single reader-facing unit.",
                        "image_path": str(image),
                    }
                ],
                image_paths=[image],
                source_path=root / "run.json",
                declared_status="completed",
            )
            result = report_contract(artifact)
        ordered = next(
            check for check in result["checks"]
            if check["criterion"] == "nonempty_ordered_units"
        )
        self.assertTrue(ordered["passed"])
        self.assertIn("unit_count=1", ordered["observation"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["warnings"], ["fewer_than_minimum_units"])

    def test_zero_units_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt="task prompt",
                units=[],
                image_paths=[],
                source_path=root / "run.json",
                declared_status="completed",
            )
            result = report_contract(artifact)
        ordered = next(
            check for check in result["checks"]
            if check["criterion"] == "nonempty_ordered_units"
        )
        self.assertFalse(ordered["passed"])
        self.assertEqual(result["status"], "FAIL")

    def test_completion_source_is_inferred_when_unmarked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "step_1.png"
            second = root / "step_2.png"
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt="task prompt",
                units=[
                    {"unit_index": 1, "content": "Gather the ingredients.",
                     "image_path": str(first)},
                    {"unit_index": 2, "content": "Cook the filling.",
                     "image_path": str(second)},
                ],
                image_paths=[first, second],
                source_path=root / "run.json",
                declared_status="incomplete",
            )
            result = report_contract(artifact)
        terminal = next(
            check for check in result["diagnostics"]
            if check["criterion"] == "explicit_terminal_completion"
        )
        self.assertTrue(terminal["passed"])
        self.assertIn("inferred from the last step", terminal["observation"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(
            result["warnings"], ["completion_inferred_from_last_step"]
        )

    def test_completion_boilerplate_detection(self) -> None:
        boilerplate = [
            "The task is complete.",
            "Task complete",
            "Step 3: The task is complete.",
            "I have completed the task.\nThe task is complete.",
            "The report is now complete.",
            "done",
        ]
        task_content = [
            "Introduce the observant student.\n\nThe task is complete.",
            "The task is to create an illustrated beginner guide for a tray dinner.",
            "Describe the concept and explain how it accommodates both uses.",
            "Step 7: I will create an illustrated event story.",
            "",
            None,
        ]
        for text in boilerplate:
            self.assertTrue(_is_completion_boilerplate(text), text)
        for text in task_content:
            self.assertFalse(_is_completion_boilerplate(text), text)

    def test_sequential_report_drops_boilerplate_at_any_step(self) -> None:
        """Boilerplate drops at any step without leaving a gap in unit_index."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [
                {"step_index": 1, "content": "Gather the ingredients.", "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "The task is complete.", "done": False,
                 "image_path": "step_2.png"},
                {"step_index": 3, "content": "Distribute the guide.", "done": False,
                 "image_path": "step_3.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "incomplete",
                 "task_prompt": "p"},
                root / "run.json",
                "run",
            )
        self.assertEqual([unit["unit_index"] for unit in artifact.units], [1, 2])
        self.assertEqual(
            [unit["content"] for unit in artifact.units],
            ["Gather the ingredients.", "Distribute the guide."],
        )
        self.assertEqual(
            [path.name for path in artifact.image_paths], ["step_1.png", "step_3.png"]
        )
        # The prose declaration resolves the terminal status the loop left open.
        self.assertEqual(artifact.declared_status, "completed")

    def test_sequential_report_drops_boilerplate_terminal_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [
                {"step_index": 1, "content": "Gather the ingredients.", "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "The task is complete.", "done": True,
                 "image_path": "step_2.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "completed", "task_prompt": "p"},
                root / "run.json",
                "run",
            )
            self.assertEqual([unit["unit_index"] for unit in artifact.units], [1])
            self.assertEqual([path.name for path in artifact.image_paths], ["step_1.png"])

    def test_sequential_report_captures_harness_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instruction = "Return one reader-facing paragraph for this step."
            artifact = _from_sequential(
                {
                    "task_key": "task",
                    "steps": [
                        {
                            "step_index": 1,
                            "content": "Gather the ingredients.",
                            "done": True,
                            "image_path": "step_1.png",
                        }
                    ],
                    "events": [
                        {"raw_input": instruction},
                        {"raw_input": instruction},
                    ],
                    "status": "completed",
                    "task_prompt": "p",
                },
                root / "run.json",
                "run",
            )
        self.assertEqual(artifact.harness_instructions, (instruction,))

    def test_prose_completion_is_accepted_without_the_marker(self) -> None:
        """A model that says it is done in prose passes the terminal gate."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "step_1.png"
            second = root / "step_2.png"
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            steps = [
                {"step_index": 1, "content": "Gather the ingredients.", "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "Cook the filling.", "done": False,
                 "image_path": "step_2.png"},
                {"step_index": 3, "content": "The task is complete.", "done": False,
                 "image_path": "step_3.png"},
                {"step_index": 4, "content": "I have completed the task.", "done": False,
                 "image_path": "step_4.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "incomplete",
                 "task_prompt": "p"},
                root / "run.json",
                "run",
            )
            result = report_contract(artifact)
        terminal = next(
            check for check in result["diagnostics"]
            if check["criterion"] == "explicit_terminal_completion"
        )
        self.assertTrue(terminal["passed"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual([unit["unit_index"] for unit in artifact.units], [1, 2])

    def test_sequential_report_keeps_terminal_step_with_task_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [
                {"step_index": 1, "content": "Gather the ingredients.", "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "Serve the finished tray dinner.", "done": True,
                 "image_path": "step_2.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "completed", "task_prompt": "p"},
                root / "run.json",
                "run",
            )
            self.assertEqual([unit["unit_index"] for unit in artifact.units], [1, 2])
            self.assertEqual(
                [path.name for path in artifact.image_paths], ["step_1.png", "step_2.png"]
            )

    def test_sequential_report_all_boilerplate_yields_no_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [
                {"step_index": 1, "content": "The task is complete.", "done": True,
                 "image_path": "step_1.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "completed", "task_prompt": "p"},
                root / "run.json",
                "run",
            )
        self.assertEqual(artifact.units, [])
        self.assertEqual(artifact.image_paths, [])

    def test_fence_markers_are_not_reader_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fence = "<beginning of the code>\n<end of the code>"
            steps = [
                {"step_index": 1, "content": fence * 40, "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "<end of the code>\nThe task is complete.",
                 "done": False, "image_path": "step_2.png"},
                {"step_index": 3, "content": "The task is complete.\n" * 30 + "The",
                 "done": False, "image_path": "step_3.png"},
                {"step_index": 4, "content": "<beginning of", "done": False,
                 "image_path": "step_4.png"},
                {"step_index": 5, "content": "Gather the ingredients.", "done": False,
                 "image_path": "step_5.png"},
                {"step_index": 6, "content": "Cook the filling.", "done": False,
                 "image_path": "step_6.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "incomplete",
                 "task_prompt": "p"},
                root / "run.json",
                "run",
            )
        # Only the two real sentences survive, renumbered without a gap.
        self.assertEqual(
            [unit["content"] for unit in artifact.units],
            ["Gather the ingredients.", "Cook the filling."],
        )
        self.assertEqual([unit["unit_index"] for unit in artifact.units], [1, 2])
        self.assertEqual(
            [path.name for path in artifact.image_paths],
            ["step_5.png", "step_6.png"],
        )

    def test_short_but_real_sentence_stays_a_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [
                {"step_index": 1, "content": "Chop the onion.", "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "Heat the pan.", "done": False,
                 "image_path": "step_2.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "completed",
                 "task_prompt": "p"},
                root / "run.json",
                "run",
            )
        self.assertEqual(len(artifact.units), 2)

    def test_marker_stripping_leaves_html_prose_intact(self) -> None:
        """A task about markup must not lose its angle-bracket prose."""
        samples = [
            "<h1>Welcome</h1><p>Add the chickpeas and simmer.</p>",
            "Use <b>bold</b> to mark the endpoint in your guide.",
            "The tag <section> groups related content on the page.",
            "Wrap the list in <ul> and each item in <li>.",
            '<html><body>Render the page with the heading.</body></html>',
        ]
        for text in samples:
            self.assertEqual(_reader_text(text), text)
            self.assertTrue(_has_reader_content(text), text)

    def test_marker_stripping_removes_backend_wrappers(self) -> None:
        wrappers = [
            "<beginning of the code>\n<end of the code>",
            "<think>\n</think>",
            "<|start_header_id|><|end_header_id|>",
            "<DONE>",
            "<beginning",
            "<end",
            '<beginning of="1">',
        ]
        for text in wrappers:
            self.assertEqual(_reader_text(text), "", text)
            self.assertFalse(_has_reader_content(text), text)

    def test_wrapped_completion_announcement_matches_the_bare_one(self) -> None:
        """Stripping a wrapper must not change how an announcement is judged."""
        self.assertTrue(_is_completion_boilerplate("The task is complete."))
        self.assertFalse(_has_reader_content("The task is complete."))
        for wrapped in [
            "<point> The task is complete.",
            "<Final step: The task is complete.>",
        ]:
            self.assertFalse(_has_reader_content(wrapped), wrapped)

    def test_wrapped_announcement_still_seals_the_run_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [
                {"step_index": 1, "content": "Gather the ingredients.", "done": False,
                 "image_path": "step_1.png"},
                {"step_index": 2, "content": "Cook the filling.", "done": False,
                 "image_path": "step_2.png"},
                {"step_index": 3, "content": "<point> The task is complete.",
                 "done": False, "image_path": "step_3.png"},
            ]
            artifact = _from_sequential(
                {"task_key": "task", "steps": steps, "status": "incomplete",
                 "task_prompt": "p"},
                root / "run.json",
                "run",
            )
        self.assertEqual(len(artifact.units), 2)
        self.assertEqual(artifact.declared_status, "completed")
        self.assertEqual(artifact.completion_source, "prose")

    def test_completion_source_separates_marker_from_prose(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def build(status, contents):
                steps = [
                    {"step_index": index, "content": content, "done": False,
                     "image_path": f"step_{index}.png"}
                    for index, content in enumerate(contents, 1)
                ]
                return _from_sequential(
                    {"task_key": "task", "steps": steps, "status": status,
                     "task_prompt": "p"},
                    root / "run.json",
                    "run",
                )

            marked = build("completed", ["Gather the ingredients.", "The task is complete."])
            prose = build("incomplete", ["Gather the ingredients.", "The task is complete."])
            unfinished = build("incomplete", ["Gather the ingredients.", "Cook the filling."])

        self.assertEqual(marked.completion_source, "marker")
        self.assertEqual(prose.completion_source, "prose")
        self.assertIsNone(unfinished.completion_source)
        self.assertEqual(prose_completion_tasks({"a": prose, "b": marked}), ["a"])

    def test_prompt_restatement_is_not_substantive(self) -> None:
        prompt = (
            "I have a short summer internship commute that includes walking, "
            "public transit, and an office with strong air conditioning."
        )
        # Pure restatement of the prompt: nothing substantive survives.
        self.assertTrue(_is_input_echo(prompt, prompt))
        self.assertTrue(_is_input_echo(prompt + " te.", prompt))
        self.assertTrue(_is_input_echo(
            prompt.replace("short", "brief"), prompt
        ))
        # Opens with the prompt but then says something new: not an echo.
        self.assertFalse(_is_input_echo(
            prompt + " Start by choosing the lightest breathable layer.", prompt
        ))
        self.assertFalse(_is_input_echo("Gather the ingredients.", prompt))
        self.assertFalse(_is_input_echo("", prompt))

    def test_prompt_restatement_is_reported_without_changing_the_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "step_1.png"
            second = root / "step_2.png"
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            prompt = (
                "Create an illustrated guide that helps a beginner cook a "
                "simple tray dinner without wasting ingredients tonight."
            )
            instructions = (
                "Return one reader-facing paragraph for the current step. "
                "Use a concise title, describe the visual action, and stop "
                "after the requested output."
            )
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt=prompt,
                units=[
                    {
                        "unit_index": 1,
                        "content": prompt,
                        "image_path": str(first),
                    },
                    {
                        "unit_index": 2,
                        "content": instructions,
                        "image_path": str(second),
                    },
                ],
                image_paths=[first, second],
                source_path=root / "run.json",
                declared_status="completed",
                harness_instructions=(instructions,),
            )
            echoes = input_echo_units(artifact)
            result = report_contract(artifact)
        self.assertEqual([entry["unit_index"] for entry in echoes], [1, 2])
        self.assertEqual(
            [entry["echo_source"] for entry in echoes],
            ["task_prompt", "harness_instructions"],
        )
        # Echoes are recorded only: the artifact still passes every gate.
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(all(check["passed"] for check in result["checks"]))

    def test_report_request_isolated_by_metric(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "step_1.png"
            image.write_bytes(b"png")
            task = TaskSpec(
                task_id="task",
                family="Recipe & Food Preparation",
                prompt="Make a dinner guide.",
                evaluation_reference={
                    "essential_coverage": ["a practical dinner"],
                    "endpoint": "a meal ready to serve",
                    "visible_evidence": ["a finished tray"],
                },
                difficulty="easy",
            )
            artifact = ReportArtifact(
                task_id="task",
                run_id="run",
                prompt=task.prompt,
                units=[
                    {
                        "unit_index": 1,
                        "content": "The tray is ready to serve.",
                        "image_path": str(image),
                    }
                ],
                image_paths=[image],
                source_path=Path(directory) / "run.json",
                declared_status="completed",
            )
            text_request = build_report_request(task, artifact, "report_text_atomic")
            visual_request = build_report_request(task, artifact, "visual_atomic")
        self.assertEqual(text_request["image_paths"], [])
        self.assertNotIn("visible_evidence", text_request["user_prompt"])
        self.assertNotIn("image_path", text_request["user_prompt"])
        self.assertEqual(visual_request["image_paths"], [str(image)])
        self.assertNotIn("semantic_image_prompt", visual_request["user_prompt"])
        self.assertEqual(
            dimensions_for("report_text_atomic", text_request["criteria"]),
            {"TC": ("text_coverage_01", "text_endpoint")},
        )

class CollectedTableTests(unittest.TestCase):
    """The release collects raw records and scores them in one place."""

    @staticmethod
    def _task(task_id):
        return TaskSpec(
            task_id=task_id,
            family="Recipe & Food Preparation",
            prompt="Make a dinner guide.",
            evaluation_reference={
                "essential_coverage": ["a practical dinner"],
                "endpoint": "a meal ready to serve",
                "visible_evidence": ["a finished tray"],
            },
        )

    def _collect(
        self, tasks, artifacts, root,
        object_id="run", display_name="Run", order=None,
    ):
        requests_dir = root / "requests"
        judgments_dir = root / "judgments"
        run_dir = root / "run"
        inputs = root / "inputs"
        for directory in (requests_dir, judgments_dir, run_dir):
            directory.mkdir(parents=True, exist_ok=True)
        (run_dir / "benchmark_task_matrix.json").write_text(
            "{}", encoding="utf-8")
        requests_by_metric = EvaluatorTests._report_requests(tasks, artifacts)
        labels = {
            metric: ["judge_a", "judge_b", "judge_c"]
            for metric in requests_by_metric
        }
        judgments_by_metric = EvaluatorTests._report_judgments(
            requests_by_metric, labels, 2)
        for metric, requests in requests_by_metric.items():
            (requests_dir / f"report_{metric.lower()}.requests.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in requests),
                encoding="utf-8")
            (judgments_dir / f"report_{metric.lower()}.judgments.jsonl").write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in judgments_by_metric[metric]),
                encoding="utf-8")
        with (
            patch(
                "evaluation.run_eval.load_matrix_tasks",
                return_value=tasks,
            ),
            patch(
                "evaluation.run_eval.load_report_artifacts",
                return_value=artifacts,
            ),
        ):
            command_collect_inputs(SimpleNamespace(
                run_dir=str(run_dir),
                out_dir=str(inputs),
                run="run",
                source="bagel",
                report_root=None,
                run_registry=None,
                benchmark_file=None,
                requests_dir=str(requests_dir),
                judgments_dir=str(judgments_dir),
                object_id=object_id,
                display_name=display_name,
                order=order,
            ))
        return inputs

    @staticmethod
    def _tables(inputs):
        out = inputs.parent / "out"
        module = _score_table()
        with patch.object(sys, "argv", [
            "score_table.py",
            "--inputs", str(inputs),
            "--targets", str(inputs / "targets.tsv"),
            "--out-dir", str(out),
            "--min-repeats", "2",
        ]):
            module.main()
        with (out / "main_table.tsv").open(encoding="utf-8", newline="") as handle:
            return next(csv.DictReader(handle, delimiter="\t"))

    def test_gate_failure_enters_the_family_validity_factor(self) -> None:
        tasks = {task_id: self._task(task_id) for task_id in ("pass", "fail")}
        artifacts = {
            "pass": EvaluatorTests._report_artifact(tasks["pass"], passing=True),
            "fail": EvaluatorTests._report_artifact(tasks["fail"], passing=False),
        }
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._collect(tasks, artifacts, Path(directory))
            table = self._tables(inputs)
            tasks_rows = [
                json.loads(line)
                for line in (inputs / "tasks.jsonl").read_text(
                    encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            {row["task_id"]: row["gate"] for row in tasks_rows},
            {"fail": 0, "pass": 1},
        )
        # 100 gate-passing quality times sqrt(0.5) validity.
        self.assertAlmostEqual(
            float(table["Overall|Total"]), 70.7107, places=4)
        for metric in ("TC", "WQ", "VC", "ITA"):
            self.assertAlmostEqual(
                float(table[f"Overall|{metric}"]), 70.7107, places=4)

    def test_dimension_columns_share_the_total_denominator(self) -> None:
        tasks = {task_id: self._task(task_id) for task_id in ("a", "b")}
        artifacts = {
            task_id: EvaluatorTests._report_artifact(task, passing=True)
            for task_id, task in tasks.items()
        }
        requests = EvaluatorTests._report_requests(tasks, artifacts)
        labels = {metric: ["judge_a", "judge_b", "judge_c"] for metric in requests}
        first = requests["TC"][0]["criteria"][0]["id"]
        judgments = EvaluatorTests._report_judgments(
            requests, labels, 2,
            tier_overrides={
                ("TC", "a", "judge_c"): {
                    first: {"tier": "NOT_MET", "evidence": "Absent."}
                }
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            inputs = self._collect(tasks, artifacts, Path(directory))
            (inputs.parent / "judgments" / "report_tc.judgments.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in judgments["TC"]),
                encoding="utf-8")
            with (
                patch(
                    "evaluation.run_eval.load_matrix_tasks",
                    return_value=tasks,
                ),
                patch(
                    "evaluation.run_eval.load_report_artifacts",
                    return_value=artifacts,
                ),
            ):
                command_collect_inputs(SimpleNamespace(
                    run_dir=str(inputs.parent / "run"),
                    out_dir=str(inputs),
                    run="run",
                    source="bagel",
                    report_root=None,
                    run_registry=None,
                    benchmark_file=None,
                    requests_dir=str(inputs.parent / "requests"),
                    judgments_dir=str(inputs.parent / "judgments"),
                    object_id="run",
                    display_name="Run",
                    order=1,
                ))
            table = self._tables(inputs)

        # a two-tier disagreement zeroes that criterion; TC half weight gives 75
        self.assertAlmostEqual(float(table["Overall|TC"]), 75.0, places=4)
        weighted = (
            0.45 * float(table["Overall|TC"])
            + 0.10 * float(table["Overall|WQ"])
            + 0.35 * float(table["Overall|VC"])
            + 0.10 * float(table["Overall|ITA"])
        )
        self.assertAlmostEqual(
            weighted, float(table["Overall|Total"]), places=3)

    def test_collection_accumulates_objects_without_duplicates(self) -> None:
        tasks = {"a": self._task("a")}
        artifacts = {"a": EvaluatorTests._report_artifact(tasks["a"], passing=True)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = self._collect(tasks, artifacts, root, object_id="a")
            self._collect(tasks, artifacts, root, object_id="b")
            self._collect(tasks, artifacts, root, object_id="a")
            task_rows = [
                json.loads(line)
                for line in (inputs / "tasks.jsonl").read_text(
                    encoding="utf-8").splitlines()
            ]
            targets = (inputs / "targets.tsv").read_text(
                encoding="utf-8").splitlines()

        self.assertEqual(sorted(row["object_id"] for row in task_rows), ["a", "b"])
        self.assertEqual(len(targets), 3)


if __name__ == "__main__":
    unittest.main()
