"""Optimization policy for UniWeaver text SFT."""

import difflib
import json
import math
import re
from pathlib import Path

TRAINING_POLICY_VERSION = 'uniweaver-text-sft-policy'
TEXT_SFT_CHECKPOINT_TYPES = ('text_sft_resume', 'text_sft_final')
TRAINING_MODES = ('lora',)
DEFAULT_TRAINING_MODE = 'lora'
MAX_SEQ_LEN = 4096
PACK_SAMPLES = 2
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = (
    r'^(?!.*_moe_gen).*'
    r'(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$')
LEARNING_RATE = 5e-5
WARMUP_STEPS = 200
LOG_EVERY = 5
CHECKPOINT_EVERY = 100
KEEP_CHECKPOINTS = 3
KEEP_CHECKPOINT_STEPS = ()
KEEP_BEST_CE_STEPS = 0
MAX_GRAD_NORM = 1.0
SEED = 42
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WEIGHT_DECAY = 0.01
WARMUP_START_FACTOR = 0.01
TRAINING_MEMORY_RESERVE_BYTES = 4 * 1024 ** 3
DEFAULT_EPOCHS = 2.0

# Fields a per-run config may override.
POLICY_OVERRIDABLE = {
    'learning_rate': float,
    'warmup_steps': int,
    'max_grad_norm': float,
    'training_mode': str,
    'epochs': float,
    'max_seq_len': int,
    'pack_samples': int,
    'checkpoint_every': int,
    'keep_checkpoints': int,
    'keep_checkpoint_steps': list,
    'keep_best_ce_steps': int,
    'lora_rank': int,
    'lora_alpha': int,
    'lora_dropout': float,
}


def configure_lora_parameter_values(args, overrides=None):
    """Attach the LoRA policy values to ``args``."""
    if overrides is None:
        overrides = {}
    defaults = {
        'lora_rank': LORA_RANK,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT,
    }
    for key, default in defaults.items():
        setattr(args, key, overrides.get(key, default))
    return args


def configure_lora_model(model, rank=LORA_RANK, alpha=LORA_ALPHA,
                         dropout=LORA_DROPOUT):
    """Wrap the text path with the canonical PEFT LoRA policy."""
    from peft import LoraConfig, get_peft_model

    text_lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=LORA_TARGET_MODULES,
    )
    model = get_peft_model(model, text_lora)
    return model, {
        'decoder_layer_count': len(model.language_model.model.layers),
        'trainable_decoder_layers': len(model.language_model.model.layers),
        'trainable_names': tuple(
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad),
        'frozen_generation_names': tuple(
            name for name, parameter in model.named_parameters()
            if '_moe_gen' in name),
        'lora_rank': rank,
        'lora_alpha': alpha,
        'wrapped_count': sum(
            1 for name, parameter in model.named_parameters()
            if parameter.requires_grad) // 2,
    }


def lora_target_module_names(module_names):
    """The subset of ``module_names`` the LoRA adapter wraps."""
    pattern = re.compile(LORA_TARGET_MODULES)
    return tuple(name for name in module_names if pattern.fullmatch(name))


def _default_policy():
    return {
        'version': TRAINING_POLICY_VERSION,
        'max_seq_len': MAX_SEQ_LEN,
        'pack_samples': PACK_SAMPLES,
        'learning_rate': LEARNING_RATE,
        'warmup_steps': WARMUP_STEPS,
        'warmup_start_factor': WARMUP_START_FACTOR,
        'adamw_betas': list(ADAMW_BETAS),
        'adamw_eps': ADAMW_EPS,
        'adamw_weight_decay': ADAMW_WEIGHT_DECAY,
        'max_grad_norm': MAX_GRAD_NORM,
        'log_every': LOG_EVERY,
        'checkpoint_every': CHECKPOINT_EVERY,
        'keep_checkpoints': KEEP_CHECKPOINTS,
        'keep_checkpoint_steps': list(KEEP_CHECKPOINT_STEPS),
        'keep_best_ce_steps': KEEP_BEST_CE_STEPS,
        'seed': SEED,
        'lora_rank': LORA_RANK,
        'lora_alpha': LORA_ALPHA,
        'lora_dropout': LORA_DROPOUT,
        'single_device_memory_reserve_bytes': TRAINING_MEMORY_RESERVE_BYTES,
    }


def _validated_overrides(overrides):
    """Cast and range/finite-check a raw override dict against the policy."""
    if not isinstance(overrides, dict):
        raise TypeError('training policy overrides must be a dict')
    validated = {}
    for key, value in overrides.items():
        if key not in POLICY_OVERRIDABLE:
            raise ValueError(f'config field is not policy-overridable: {key}')
        # Retention selectors are validated structurally, not cast; empty or zero
        # is the documented default.
        if key == 'keep_checkpoint_steps':
            validated[key] = _validated_step_list(value)
            continue
        if key == 'keep_best_ce_steps':
            validated[key] = _validated_non_negative_int(key, value)
            continue
        if key == 'training_mode':
            if value not in TRAINING_MODES:
                raise ValueError(
                    f'config field \'training_mode\' must be one of '
                    f'{TRAINING_MODES}, got {value!r}')
            validated[key] = value
            continue
        cast = POLICY_OVERRIDABLE[key]
        try:
            value = cast(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f'config field {key!r} is not a valid {cast.__name__}: '
                f'{value!r}') from error
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f'config field {key!r} must be finite, got {value!r}')
        if isinstance(value, (int, float)) and value <= 0:
            raise ValueError(
                f'config field {key!r} must be positive, got {value!r}')
        validated[key] = value
    return validated


def _validated_step_list(value):
    """Validate an explicit list of checkpoint steps to pin."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError(
            f'config field \'keep_checkpoint_steps\' must be a list of step '
            f'numbers, got {value!r}')
    steps = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(
                f'config field \'keep_checkpoint_steps\' must contain only '
                f'integers, got {item!r}')
        if item <= 0:
            raise ValueError(
                f'config field \'keep_checkpoint_steps\' must be positive, '
                f'got {item!r}')
        steps.append(item)
    return sorted(set(steps))


def _validated_non_negative_int(key, value):
    """Validate a count that may legitimately be zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f'config field {key!r} is not a valid int: {value!r}')
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(
                f'config field {key!r} must be a whole number, got {value!r}')
        value = int(value)
    if value < 0:
        raise ValueError(f'config field {key!r} must not be negative: {value!r}')
    return value


def load_policy_config(path):
    """Load a per-run training config JSON and return validated overrides."""
    return _validated_overrides(load_policy_config_raw(path))


def load_policy_config_raw(path):
    """Load a per-run config JSON and return its raw policy-overridable fields."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise TypeError(f'training config must be a JSON object: {path}')
    # Reject a non-whitelisted key only when it looks like a typo of a policy
    # field, so config metadata still passes through.
    for key in data:
        if key not in POLICY_OVERRIDABLE:
            misspelled = difflib.get_close_matches(
                key, POLICY_OVERRIDABLE, n=1, cutoff=0.8)
            if misspelled:
                raise ValueError(
                    f'training config field {key!r} is not policy-overridable; '
                    f'did you mean {misspelled[0]!r}?')
    return {
        key: value for key, value in data.items()
        if key in POLICY_OVERRIDABLE
    }


def resolve_policy_overrides(config_path=None, keep_checkpoint_steps=None,
                             keep_best_ce_steps=None):
    """Resolve the effective policy overrides from a config and CLI selectors."""
    raw = load_policy_config_raw(config_path) if config_path else {}
    if keep_checkpoint_steps is not None:
        raw['keep_checkpoint_steps'] = list(keep_checkpoint_steps)
    if keep_best_ce_steps is not None:
        raw['keep_best_ce_steps'] = keep_best_ce_steps
    return _validated_overrides(raw)


def validate_retention_steps(args):
    """Reject pinned steps that the resolved schedule can never reach."""
    pinned = getattr(args, 'keep_checkpoint_steps', ()) or ()
    if not pinned:
        return args
    total_steps = getattr(args, 'total_steps', None)
    interval = getattr(args, 'ckpt_every', None)
    if not total_steps or not interval:
        return args
    beyond = [step for step in pinned if step > total_steps]
    if beyond:
        raise ValueError(
            f'keep_checkpoint_steps {beyond} exceed total_steps={total_steps}; '
            'checkpoint boundaries end at total_steps')
    off_grid = [step for step in pinned if step % interval]
    if off_grid:
        raise ValueError(
            f'keep_checkpoint_steps {off_grid} are not multiples of '
            f'checkpoint_every={interval}')
    return args


def training_policy(overrides=None):
    """Return the complete human-readable fixed optimization policy."""
    policy = _default_policy()
    if overrides:
        for key, value in _validated_overrides(overrides).items():
            policy[key] = value
    return policy


def apply_training_policy(args, overrides=None):
    """Attach the fixed optimization policy (with optional per-run overrides)."""
    if overrides is None:
        overrides = {}
    default = _default_policy()
    # value fields that a config may override (mapped from policy key -> arg)
    value_override = {
        'learning_rate': 'lr', 'warmup_steps': 'warmup_steps',
        'max_grad_norm': 'max_grad_norm', 'max_seq_len': 'max_seq_len',
        'pack_samples': 'pack_samples',
    }
    for policy_key, arg in value_override.items():
        if policy_key in overrides:
            setattr(args, arg, overrides[policy_key])
        else:
            setattr(args, arg, default[policy_key])
    # checkpoint_every is configurable so long runs can retain auditable
    # intermediate candidates without conflating them with resume retention.
    args.log_every = default['log_every']
    if 'checkpoint_every' in overrides:
        args.ckpt_every = overrides['checkpoint_every']
    else:
        args.ckpt_every = default['checkpoint_every']
    # seed remains fixed by the production policy.
    args.seed = default['seed']
    # training_mode / epochs / keep_checkpoints: CLI/config take effect
    if 'training_mode' in overrides:
        args.training_mode = overrides['training_mode']
    if 'epochs' in overrides:
        args.epochs = overrides['epochs']
    # keep_checkpoints always takes a concrete value (default or override) so
    # the launcher can pass it to the checkpoint pruner instead of drifting to
    # a hardcoded constant.
    args.keep_checkpoints = (
        overrides.get('keep_checkpoints', KEEP_CHECKPOINTS))
    # Retention selectors change which artifacts are kept, never how the model is
    # optimized, so the launcher passes them to the pruner.
    args.keep_checkpoint_steps = tuple(
        overrides.get('keep_checkpoint_steps', KEEP_CHECKPOINT_STEPS))
    args.keep_best_ce_steps = overrides.get(
        'keep_best_ce_steps', KEEP_BEST_CE_STEPS)
    configure_lora_parameter_values(args, overrides)
    args.optimization_policy = training_policy(overrides or None)
    return args


def require_training_memory_capacity(model, torch_module):
    """Check available CUDA memory before optimizer allocation."""
    if not torch_module.cuda.is_available():
        raise RuntimeError('UniWeaver text SFT requires a CUDA device')
    free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    trainable = [parameter for parameter in model.parameters()
                 if parameter.requires_grad]
    # PyTorch AdamW keeps one gradient and two moments at parameter dtype for
    # this bf16 path. Activations/kernels use the explicit reserve above.
    incremental_bytes = sum(
        parameter.numel() * parameter.element_size() * 3
        for parameter in trainable)
    required_bytes = incremental_bytes + TRAINING_MEMORY_RESERVE_BYTES
    if free_bytes < required_bytes:
        raise MemoryError(
            'selected UniWeaver text-SFT mode requires additional gradient/Adam '
            'state: '
            f'free={free_bytes / 1024**3:.1f} GiB, '
            f'required>={required_bytes / 1024**3:.1f} GiB')
    return {
        'free_gib': free_bytes / 1024 ** 3,
        'total_gib': total_bytes / 1024 ** 3,
        'minimum_incremental_gib': incremental_bytes / 1024 ** 3,
        'reserve_gib': TRAINING_MEMORY_RESERVE_BYTES / 1024 ** 3,
    }


def checkpoint_identity(args, checkpoint_type):
    """Return the complete immutable identity embedded in every checkpoint."""
    if checkpoint_type not in TEXT_SFT_CHECKPOINT_TYPES:
        raise ValueError(f'unknown text-SFT checkpoint type: {checkpoint_type}')
    from core.production import CONTRACT_VERSION, TRAJECTORY_DESIGN

    return {
        'checkpoint_type': checkpoint_type,
        'contract_version': CONTRACT_VERSION,
        'trajectory_design': TRAJECTORY_DESIGN,
        'training_policy_version': TRAINING_POLICY_VERSION,
        'training_mode': args.training_mode,
        'holdout_set_version': args.holdout_set_version,
        'holdout_manifest_sha256': args.holdout_manifest_sha256,
        'training_manifest_sha256': args.training_manifest_sha256,
        'training_provenance_sha256': args.training_provenance_sha256,
        'base_language_model_identity_sha256': (
            args.base_language_model_identity_sha256),
    }
