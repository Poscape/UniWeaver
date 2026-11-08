"""API Planner transport used by ``baselines/run_umm.py``."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runner import TrajectoryError
from sft_prepare_data.trajectory_construct_driver import (
    construct_trajectory,
    run_constructor_cli,
    run_task_specs,
)
from sft_prepare_data.trajectory_providers import (
    API_CONFIGS,
    ApiStageSender,
    create_client,
)


def _run_one(task_key, task_prompt, output_path, api_provider, model,
             api_key_override, api_base_url, extra_fields=None):
    """Run full trajectory generation for a single task."""
    try:
        sender = ApiStageSender(
            api_provider, model, api_key_override, api_base_url)
    except (ValueError, RuntimeError) as exc:
        raise TrajectoryError(str(exc)) from exc

    print(f'API: {sender.config["label"]} | Model: {model}')

    trajectory = construct_trajectory(
        task_key, task_prompt, output_path, sender,
        extra_fields={
            **(extra_fields or {}),
            'model_provider': api_provider,
            'model_name': model,
        })
    totals = sender.totals
    print(
        f'Tokens: ~{int(totals["estimated_input_tokens"]):,} in + '
        f'~{int(totals["estimated_output_tokens"]):,} out '
        f'(real API: {int(totals["input_tokens"]):,} + '
        f'{int(totals["output_tokens"]):,})')
    print(f'Next: python baselines/run_umm.py --execution cached '
          f'--trajectory {output_path} --backend bagel')
    return trajectory

def run(args, parser):
    api_provider = args.api.lower()
    cfg_api = API_CONFIGS.get(api_provider, {})
    model = args.model or cfg_api.get('model', 'deepseek-v4-pro')
    label = cfg_api.get('label', 'api')
    try:
        cfg_api, _preflight_client = create_client(
            api_provider, args.api_key_override, args.api_base_url)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    del _preflight_client

    def execute_specs(specs):
        def run_one(key, prompt, path):
            return _run_one(
                key, prompt, path, api_provider, model,
                args.api_key_override, args.api_base_url,
                extra_fields={})

        return run_task_specs(specs, run_one, args.parallel)

    run_constructor_cli(args, parser, label, execute_specs)
