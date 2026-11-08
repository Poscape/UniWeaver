"""Regression lock for the clean UniWeaver production contract."""

import unittest
from pathlib import Path

from core.contract import (
    build_plan1_request,
    build_plan2_request,
    build_plan3_request,
    build_plan4_request,
)
from core.production import (
    CONTRACT_VERSION,
    TRAJECTORY_DESIGN,
    planner_stage_config,
)
from core.render_inputs import compile_step_render_input
from core.tasks import UNIWEAVER_TRAJECTORY_ROOT
from core.validation import (
    plan1_output_issues,
    plan1_output_warnings,
    plan2_output_issues,
    plan3_output_issues,
    plan4_output_issues,
    step_output_warnings,
)


class UNIWEAVERContractTest(unittest.TestCase):
    def test_plan1_rejects_placeholder_values(self):
        plan1 = {
            'title': 'short title',
            'endpoint': 'earliest completed outcome or last explicit event',
            'steps': [
                {'step_index': 1,
                 'image_subject': 'one concrete visible subject or state',
                 'content_goal': 'short reader-facing question'},
                {'step_index': 2,
                 'image_subject': 'second concrete visible subject or state',
                 'content_goal': 'short reader-facing question'},
            ],
        }
        issues = plan1_output_issues('Make a guide.', plan1)
        joined = ' | '.join(issues)
        self.assertIn('image_subject repeats the prompt example', joined)
        self.assertIn('content_goal repeats the prompt example', joined)

    def test_plan1_endpoint_is_advisory(self):
        plan1 = {
            'title': 'The Evolution of the Bicycle',
            'endpoint': 'earliest completed outcome or last explicit event',
            'steps': [
                {'step_index': 1, 'image_subject': '1817 Penny Farthing',
                 'content_goal': 'What is this bicycle?'},
                {'step_index': 2, 'image_subject': '1885 Safety Bicycle',
                 'content_goal': 'How did the design change?'},
            ],
        }
        self.assertEqual([], plan1_output_issues('Make a guide.', plan1))
        warnings = plan1_output_warnings('Make a guide.', plan1)
        self.assertTrue(
            any('endpoint repeats the prompt example' in w for w in warnings),
            warnings,
        )

    def test_plan1_accepts_task_specific_values(self):
        plan1 = {
            'title': 'The Evolution of the Bicycle',
            'endpoint': 'A rider understands the safety bicycle',
            'steps': [
                {'step_index': 1, 'image_subject': '1817 Penny Farthing',
                 'content_goal': 'What is this bicycle?'},
                {'step_index': 2, 'image_subject': '1885 Safety Bicycle',
                 'content_goal': 'How did the design change?'},
            ],
        }
        self.assertEqual([], plan1_output_issues('Make a guide.', plan1))

    def test_plan2_rejects_placeholder_values(self):
        plan1 = {
            'title': 'The Evolution of the Bicycle',
            'endpoint': 'A rider understands the safety bicycle',
            'steps': [
                {'step_index': 1, 'image_subject': '1817 Penny Farthing',
                 'content_goal': 'What is this bicycle?'},
                {'step_index': 2, 'image_subject': '1885 Safety Bicycle',
                 'content_goal': 'How did the design change?'},
            ],
        }
        plan2 = {
            'visual_style': {
                'medium': 'photograph', 'color_treatment': 'natural',
                'lighting_character': 'balanced', 'rendering_finish': 'clear',
            },
            'steps': [
                {'step_index': 1,
                 'image_prompt': 'complete independently renderable visible state',
                 'content': 'finished report prose'},
                {'step_index': 2,
                 'image_prompt': 'complete independently renderable visible state',
                 'content': 'finished report prose'},
            ],
        }
        issues = plan2_output_issues(plan1, plan2)
        joined = ' | '.join(issues)
        self.assertIn('image_prompt repeats the prompt example', joined)
        self.assertIn('content repeats the prompt example', joined)

    def test_plan4_requires_step_specific_checks(self):
        plan1 = {
            'title': 'T', 'endpoint': 'E',
            'steps': [
                {'step_index': 1, 'image_subject': 'A', 'content_goal': 'B'},
                {'step_index': 2, 'image_subject': 'C', 'content_goal': 'D'},
            ],
        }
        plan4 = {'reviewer_checklist': [
            'step_1: <check>', 'step_1: <check>',
            'step_2: <check>', 'step_2: <check>',
        ]}
        issues = plan4_output_issues(plan1, plan4)
        self.assertTrue(
            any('repeats the prompt example' in issue for issue in issues),
            issues,
        )

    def test_plan4_accepts_real_checks(self):
        plan1 = {
            'title': 'T', 'endpoint': 'E',
            'steps': [
                {'step_index': 1, 'image_subject': 'A', 'content_goal': 'B'},
                {'step_index': 2, 'image_subject': 'C', 'content_goal': 'D'},
            ],
        }
        plan4 = {'reviewer_checklist': [
            'step_1: a kettle is visible on the counter',
            'step_2: two mugs are placed beside the kettle',
        ]}
        self.assertEqual([], plan4_output_issues(plan1, plan4))

    def test_identity_and_single_attempt_default(self):
        self.assertEqual('uniweaver', TRAJECTORY_DESIGN)
        self.assertEqual('UniWeaver', CONTRACT_VERSION)
        self.assertEqual(2048, planner_stage_config('plan1')['max_tokens'])
        self.assertEqual(2560, planner_stage_config('plan2')['max_tokens'])
        self.assertEqual(768, planner_stage_config('plan3')['max_tokens'])
        self.assertEqual(1024, planner_stage_config('plan4')['max_tokens'])

    def test_plan1_has_only_consumed_top_level_fields(self):
        request = build_plan1_request('Make an illustrated report.')
        self.assertIn(
            '"title":"...","endpoint":"earliest completed outcome',
            request)
        self.assertNotIn('"guide_mode"', request)
        self.assertNotIn('"continuity_mode"', request)
        self.assertNotIn('<think>', request)

    def test_plan2_does_not_receive_retired_step_fields(self):
        plan1 = {
            'title': 'Example',
            'endpoint': 'Second useful view complete.',
            'steps': [
                {'step_index': 1, 'image_subject': 'Show the first useful view.', 'content_goal': 'Explain this visible contribution.'},
                {'step_index': 2, 'image_subject': 'Show the second useful view.', 'content_goal': 'Explain this visible contribution.'},
            ],
        }
        request = build_plan2_request('Make an illustrated report.', plan1)
        self.assertIn(
            'Step 1: image_subject=Show the first useful view.; '
            'content_goal=Explain this visible contribution.', request)
        self.assertIn('"step_index":1,"image_prompt"', request)
        self.assertNotIn('"step_index":1,"image_subject"', request)
        for field in (
                'step_role=', 'requirement_refs=', 'claim=',
                'visual_evidence=', 'prose_role=', 'necessity=',
                'visual_state=', 'visual_delta=', 'image_prompt=', 'content='):
            self.assertNotIn(field, request)

    def test_plan1_uses_irreplaceable_contribution_contract(self):
        request = build_plan1_request('Make an illustrated report.')
        self.assertIn('distinct useful image-plus-content units', request)
        self.assertIn('one concrete physical subject or state', request)
        self.assertNotIn('guidebook', request.lower())

    def test_plan3_selects_only_stable_reusable_identity(self):
        plan1 = {
            'title': 'Yogurt', 'endpoint': 'Ready to eat.',
            'steps': [
                {'step_index': 1, 'image_subject': 'A clear glass cup with yogurt.',
                 'content_goal': 'Explain the base.'},
                {'step_index': 2,
                 'image_subject': 'The same cup after fruit is added.',
                 'content_goal': 'Explain the addition.'},
            ],
        }
        plan2 = {
            'visual_style': {},
            'steps': [
                {
                    'step_index': 1,
                    'image_prompt': 'A clear glass cup containing yogurt.',
                    'content': 'First state.',
                },
                {
                    'step_index': 2,
                    'image_prompt': 'The same clear glass cup with fruit added.',
                    'content': 'Second state.',
                },
            ],
        }
        request = build_plan3_request('Make a yogurt bowl.', plan1, plan2)
        normalized = ' '.join(request.split())
        self.assertIn('Frozen semantic subjects:', request)
        self.assertIn('Rendered Steps:', request)
        self.assertIn('image_prompt=A clear glass cup containing yogurt.', request)
        self.assertNotIn('content=First state.', request)
        self.assertIn('counterfactual test', normalized)
        self.assertIn('different outline, topology, fold or crease pattern', normalized)
        self.assertIn('must enter the bank', normalized)
        self.assertNotIn('region mask', normalized)

    def test_plan3_requires_matching_semantic_and_rendered_indexes(self):
        plan1 = {'steps': [
            {'step_index': 1, 'image_subject': 'A cup.',
             'content_goal': 'Explain it.'}]}
        plan2 = {'steps': [
            {'step_index': 2, 'image_prompt': 'A cup.', 'content': 'Text.'}]}
        with self.assertRaisesRegex(ValueError, 'matching Plan1 and Plan2'):
            build_plan3_request('Task', plan1, plan2)

    def test_plan4_reads_plan2_image_prompts(self):
        plan2 = {'steps': [
            {'step_index': 2, 'image_prompt': 'A cup on a table.',
             'content': 'The cup rests on the table.'}]}
        request = build_plan4_request(plan2)
        self.assertIn('image_prompt=A cup on a table.', request)

    def test_single_step_identity_reference_is_valid_but_optional(self):
        plan3 = {
            'condition_bank': {'items': [{
                'key': 'lamp_a',
                'type': 'reference',
                'image_prompt': 'One isolated table lamp on a neutral background.',
            }]},
            'step_conditions': {'step_1': ['lamp_a'], 'step_2': []},
        }
        plan1 = {'steps': [
            {'step_index': 1}, {'step_index': 2},
        ]}
        self.assertEqual([], plan3_output_issues(plan1, plan3))

    def test_plan3_invalid_non_object_item_is_reported(self):
        issues = plan3_output_issues(
            {'steps': [{'step_index': 1}]},
            {
                'condition_bank': {'items': ['malformed']},
                'step_conditions': {'step_1': []},
            },
        )
        self.assertIn('condition <missing> is invalid', issues)

    def test_plan3_non_object_step_conditions_is_reported_without_crashing(self):
        issues = plan3_output_issues(
            {'steps': [{'step_index': 1}]},
            {
                'condition_bank': {'items': [{
                    'key': 'ref_a',
                    'type': 'reference',
                    'image_prompt': 'One object on a neutral background.',
                }]},
                'step_conditions': [{'step_1': ['ref_a']}],
            },
        )
        self.assertIn(
            'step_conditions keys must exactly match Plan1 steps', issues)
        self.assertIn(
            'condition ref_a must be routed to at least one step', issues)

        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(
            project_root / 'output_run' / 'runs' / 'uniweaver' / 'trajectories',
            UNIWEAVER_TRAJECTORY_ROOT,
        )

    def test_legacy_identity_is_not_accepted_as_compatibility_input(self):
        from core.production import require_trajectory_identity

        with self.assertRaises(ValueError):
            require_trajectory_identity({
                'trajectory_design': 'legacy',
                'contract_version': 'legacy',
            })

    def test_warnings_identify_exact_duplicate_subject_fields(self):
        warnings = plan1_output_warnings('', {
            'steps': [
                {'step_index': 2, 'image_subject': 'A cup'},
                {'step_index': 7, 'image_subject': 'a CUP'},
            ]})
        self.assertEqual(1, len(warnings))
        self.assertIn('Plan1.steps[2].image_subject=', warnings[0])
        self.assertIn('Plan1.steps[7].image_subject=', warnings[0])
        self.assertIn("normalized='a cup'", warnings[0])

    def test_warnings_identify_meta_language_and_content_excerpt(self):
        warnings = step_output_warnings(
            {'content': 'This image needs verification before release.'},
            'Plan2.steps[3]')
        self.assertEqual(1, len(warnings))
        self.assertIn('Plan2.steps[3]:', warnings[0])
        self.assertIn("matched=['This image', 'needs verification']", warnings[0])
        self.assertIn('content=', warnings[0])

    def test_step_render_error_identifies_missing_field_path(self):
        with self.assertRaisesRegex(
                ValueError, r'Plan2\.steps\[2\]\.content is missing'):
            compile_step_render_input(
                {'visual_style': {}},
                {'step_index': 2, 'image_prompt': 'A cup.'})


if __name__ == '__main__':
    unittest.main()
