"""The complete and sole UniWeaver production identity and runner policy."""

import hashlib
import importlib.metadata
import json
import platform
from copy import deepcopy
from pathlib import Path

from .hashing import sha256_file

TRAJECTORY_DESIGN = 'uniweaver'
CONTRACT_VERSION = 'UniWeaver'
DEFAULT_IMAGE_SEED = 42
# Maximum review-retry rounds after the first generation.
REVIEW_MAX_RETRIES = 2

API_THINKING_MODE = 'disabled'
IMAGE_EXECUTION_REVISION = 'uniweaver-image-exec'
IMAGE_EXECUTION_SOURCES = (
    'bagel_model_loader.py',
    'data/__init__.py',
    'data/data_utils.py',
    'data/transforms.py',
    'inferencer.py',
    'inferencer_output_manager.py',
    'runtime_diagnostics.py',
    'sft_prepare_data/attention_backend.py',
    'core/contract.py',
    'core/decoding.py',
    'core/production.py',
    'core/render_inputs.py',
    'core/review.py',
    'core/runner.py',
    'core/uniweaver_execution.py',
    'baselines/umm_framework.py',
    'baselines/umm_uniweaver_research.py',
    'baselines/umm_uniweaver_persistence.py',
    'core/schema.py',
    'core/serialization.py',
    'core/validation.py',
    'core/inferencer_adapter.py',
    'modeling/autoencoder.py',
    'modeling/__init__.py',
    'modeling/bagel/__init__.py',
    'modeling/bagel/bagel.py',
    'modeling/bagel/modeling_utils.py',
    'modeling/bagel/qwen2_navit.py',
    'modeling/bagel/siglip_navit.py',
    'modeling/cache_utils/taylorseer.py',
    'modeling/qwen2/__init__.py',
    'modeling/qwen2/configuration_qwen2.py',
    'modeling/qwen2/modeling_qwen2.py',
    'modeling/qwen2/tokenization_qwen2.py',
    'modeling/qwen2/tokenization_qwen2_fast.py',
    'modeling/siglip/__init__.py',
    'modeling/siglip/configuration_siglip.py',
    'modeling/siglip/image_processing_siglip.py',
    'modeling/siglip/modeling_siglip.py',
    'modeling/siglip/processing_siglip.py',
    'modeling/siglip/tokenization_siglip.py',
)
IMAGE_GENERATION_CONFIG = {
    'cfg_text_scale': 3.0,
    'cfg_img_scale': 1.5,
    'cfg_interval': [0.4, 1.0],
    'timestep_shift': 3.0,
    'num_timesteps': 50,
    'cfg_renorm_min': 0.0,
    'cfg_renorm_type': 'global',
    'enable_taylorseer': False,
}
TRAJECTORY_METADATA_FIELDS = frozenset({
    'model_provider', 'model_name', 'model_artifact_sha256',
    'sft_checkpoint_sha256', 'sft_checkpoint_training_identity',
})
TRAJECTORY_FIELDS = frozenset({
    'task_key', 'task_prompt', 'raw_attempts',
    'trajectory_design', 'contract_version',
    # Written by the full pipeline; a text-SFT corpus carries raw_attempts only.
    'plan', 'step_generations', 'step_generation_expected',
    '_construction_issues', '_text_validation',
}) | TRAJECTORY_METADATA_FIELDS
REQUIRED_TRAJECTORY_FIELDS = frozenset({
    'task_key', 'raw_attempts',
})
# Pipeline-written persistence/view markers, outside the trajectory contract;
# rejecting them as unknown fields would reject the pipeline's own output.
TRAJECTORY_VIEW_FIELDS = frozenset({
    'status', 'derived_from', 'report_variant', 'review_derivation',
})


def trajectory_contract_view(trajectory):
    """Return a persisted trajectory as the closed contract sees it."""
    if not isinstance(trajectory, dict):
        return trajectory
    return {
        key: value for key, value in trajectory.items()
        if key not in TRAJECTORY_VIEW_FIELDS
    }

PLANNER_STAGE_CONFIGS = {
    'plan1': {'max_tokens': 2048, 'do_sample': False, 'temperature': 0},
    'plan2': {'max_tokens': 2560, 'do_sample': False, 'temperature': 0},
    'plan3': {'max_tokens': 768, 'do_sample': False, 'temperature': 0},
    'plan4': {'max_tokens': 1024, 'do_sample': False, 'temperature': 0},
}

# Every Planner request ends with the same closing instruction.
PLANNER_JSON_ONLY_SUFFIX = (
    "\n\nReturn the requested JSON object now. Output only that JSON object; "
    "do not repeat the prompt, add commentary, or wrap it in markdown fences."
)


def planner_stage_config(stage):
    """Return the sole generation configuration for one Planner stage."""
    try:
        return deepcopy(PLANNER_STAGE_CONFIGS[stage])
    except KeyError as error:
        raise ValueError(f'unknown Planner stage: {stage}') from error


def image_generation_config():
    """Return the sole production image-generation configuration."""
    return deepcopy(IMAGE_GENERATION_CONFIG)


def image_execution_identity(project_root=None, extra_sources=()):
    """Bind an image run to the bytes of its execution sources and adapter."""
    root = Path(project_root or Path(__file__).resolve().parents[1])
    files = []
    for relative in (*IMAGE_EXECUTION_SOURCES, *extra_sources):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'image execution source is missing: {path}')
        files.append({
            'path': relative,
            'sha256': sha256_file(path),
        })
    payload = {
        'revision': IMAGE_EXECUTION_REVISION,
        'files': files,
    }
    payload['sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def image_runtime_identity():
    """Identify the numerical runtime that may change generated pixels."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            'image execution requires PyTorch runtime identity') from error
    cuda = getattr(torch, 'cuda', None)
    cuda_available = bool(cuda and cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(cuda.device_count()):
            properties = cuda.get_device_properties(index)
            devices.append({
                'name': properties.name,
                'capability': list(cuda.get_device_capability(index)),
                'total_memory': int(properties.total_memory),
            })
    cudnn = getattr(getattr(torch, 'backends', None), 'cudnn', None)
    from sft_prepare_data.attention_backend import (
        attention_backend_identity,
    )

    payload = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'torch': str(getattr(torch, '__version__', 'unknown')),
        'torch_cuda': str(getattr(getattr(torch, 'version', None), 'cuda', None)),
        'cudnn': int(cudnn.version()) if cudnn and cudnn.version() else None,
        'cuda_available': cuda_available,
        'devices': devices,
        'nvidia_driver': (
            Path('/proc/driver/nvidia/version').read_text(
                encoding='utf-8', errors='replace').strip()
            if Path('/proc/driver/nvidia/version').is_file() else None),
        'packages': {},
        'attention_backend': attention_backend_identity(
            require_official=False, install_portable=False),
    }
    for package in ('flash-attn', 'numpy', 'pillow', 'safetensors'):
        try:
            payload['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            payload['packages'][package] = None
    payload['sha256'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def image_seed_for(base_seed, kind, identity):
    """Derive a stable per-image seed without touching process-global RNG state."""
    if base_seed is None:
        return None
    payload = f'{int(base_seed)}\0{kind}\0{identity}'.encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big')
    return (int(base_seed) + offset) % (2 ** 63 - 1)


def require_trajectory_identity(trajectory):
    """Reject every record that claims an identity other than UniWeaver."""
    if not isinstance(trajectory, dict):
        raise TypeError('trajectory must be an object')
    missing = sorted(REQUIRED_TRAJECTORY_FIELDS - set(trajectory))
    if missing:
        raise ValueError(
            f'UniWeaver trajectory is missing required fields: {missing}')
    # The pipeline stamps both markers; a released corpus may omit them, but a
    # recorded value must still be the canonical one.
    design = trajectory.get('trajectory_design', TRAJECTORY_DESIGN)
    version = trajectory.get('contract_version', CONTRACT_VERSION)
    if design != TRAJECTORY_DESIGN or version != CONTRACT_VERSION:
        raise ValueError(
            'trajectory must use '
            f'{TRAJECTORY_DESIGN}/{CONTRACT_VERSION}; '
            f'got {design!r}/{version!r}')
    unexpected = sorted(set(trajectory) - TRAJECTORY_FIELDS)
    if unexpected:
        raise ValueError(
            f'UniWeaver trajectory contains unexpected top-level fields: {unexpected}')
    if (not isinstance(trajectory['task_key'], str)
            or not trajectory['task_key'].strip()
            or not isinstance(trajectory['raw_attempts'], dict)):
        raise ValueError('UniWeaver trajectory has invalid required field types')
    optional_types = (
        ('plan', dict), ('step_generations', dict),
        ('step_generation_expected', bool), ('_construction_issues', list),
    )
    for field, kind in optional_types:
        if field in trajectory and not isinstance(trajectory[field], kind):
            raise ValueError(
                f'UniWeaver trajectory has invalid {field} type')
    return design, version
