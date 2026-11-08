"""Shared BAGEL model construction for text-only and multimodal workloads."""
import gc
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache

from core.hashing import sha256_file

DEFAULT_BAGEL_MODEL_PATH = os.path.join(
    os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface')),
    'hub',
    'models--ByteDance-Seed--BAGEL-7B-MoT',
    'snapshots',
    '5019f57d168e5816e8f3f701b17cc816bb7cf24b',
)


def resolve_bagel_model_path(model_path=None):
    """Resolve the BAGEL checkpoint root used by UniWeaver."""
    return model_path or os.environ.get('BAGEL_MODEL_PATH', DEFAULT_BAGEL_MODEL_PATH)


@lru_cache(maxsize=32)
def _artifact_sha256(path, size, mtime_ns):
    """Hash one stable file version once per process."""
    blob_name = os.path.basename(path)
    if re.fullmatch(r'[0-9a-f]{64}', blob_name):
        return blob_name
    return sha256_file(path)


def _artifact_signature(path):
    """Identify exact bytes, reusing Hugging Face content-addressed blob names."""
    path = os.path.realpath(os.path.expanduser(str(path)))
    stat = os.stat(path)
    return {
        'path': path,
        'size': stat.st_size,
        'sha256': _artifact_sha256(path, stat.st_size, stat.st_mtime_ns),
    }


def bagel_model_identity(model_path=None, mode=1, sft_checkpoint=None):
    """Return a stable identity for the exact base and optional SFT artifacts."""
    root = os.path.realpath(resolve_bagel_model_path(model_path))
    required_artifact_names = (
        'ema.safetensors', 'ae.safetensors', 'llm_config.json',
        'vit_config.json',
    )
    missing = [name for name in required_artifact_names
               if not os.path.isfile(os.path.join(root, name))]
    if missing:
        raise FileNotFoundError(
            f'BAGEL model root lacks required artifacts: {missing}')
    tokenizer_artifact_names = (
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model',
    )
    if not any(os.path.isfile(os.path.join(root, name))
               for name in ('tokenizer.json', 'tokenizer.model', 'vocab.json')):
        raise FileNotFoundError(
            'BAGEL model root lacks tokenizer vocabulary artifacts')
    artifact_names = required_artifact_names + tokenizer_artifact_names
    artifacts = {
        name: _artifact_signature(os.path.join(root, name))
        for name in artifact_names if os.path.isfile(os.path.join(root, name))
    }
    if sft_checkpoint:
        artifacts['sft_checkpoint'] = _artifact_signature(sft_checkpoint)
    payload = {'model_root': root, 'mode': int(mode), 'artifacts': artifacts}
    digest_artifacts = {
        name: {'size': value['size'], 'sha256': value['sha256']}
        for name, value in artifacts.items()
    }
    digest = hashlib.sha256(json.dumps(
        {'mode': int(mode), 'artifacts': digest_artifacts},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'label': f'bagel-mode{mode}', 'sha256': digest, **payload}


def bagel_language_model_identity(model_path=None):
    """Identify only the artifacts consumed by text training and overlays."""
    root = os.path.realpath(resolve_bagel_model_path(model_path))
    required = ('ema.safetensors', 'llm_config.json')
    missing = [name for name in required
               if not os.path.isfile(os.path.join(root, name))]
    if missing:
        raise FileNotFoundError(
            f'BAGEL language model root lacks required artifacts: {missing}')
    tokenizer_names = (
        'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
        'added_tokens.json', 'vocab.json', 'merges.txt', 'tokenizer.model',
    )
    if not any(os.path.isfile(os.path.join(root, name))
               for name in ('tokenizer.json', 'tokenizer.model', 'vocab.json')):
        raise FileNotFoundError(
            'BAGEL language model root lacks tokenizer vocabulary artifacts')
    names = required + tokenizer_names
    artifacts = {
        name: _artifact_signature(os.path.join(root, name))
        for name in names if os.path.isfile(os.path.join(root, name))
    }
    portable = {
        name: {'size': value['size'], 'sha256': value['sha256']}
        for name, value in sorted(artifacts.items())
    }
    digest = hashlib.sha256(json.dumps(
        {'artifacts': portable}, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return {
        'label': 'bagel-language',
        'sha256': digest,
        'model_root': root,
        'artifacts': artifacts,
    }


def bagel_model_content_identity(identity):
    """Return the portable, path-independent identity used by experiments."""
    if not isinstance(identity, dict) or not identity.get('sha256'):
        raise ValueError('invalid BAGEL model identity')
    return {
        'label': identity.get('label', 'bagel'),
        'mode': identity.get('mode'),
        'sha256': identity['sha256'],
        'artifacts': {
            name: {
                'size': artifact.get('size'),
                'sha256': artifact.get('sha256'),
            }
            for name, artifact in sorted(
                identity.get('artifacts', {}).items())
        },
    }


def load_bagel_llm_config(model_path, use_cache=True):
    """Load the canonical BAGEL MoT language configuration."""
    from modeling.bagel import Qwen2Config

    config = Qwen2Config.from_json_file(os.path.join(model_path, 'llm_config.json'))
    config.qk_norm = True
    config.tie_word_embeddings = False
    config.use_cache = bool(use_cache)
    config.layer_module = 'Qwen2MoTDecoderLayer'
    return config


@dataclass
class BagelModelBundle:
    """Loaded BAGEL components required by one workload."""

    model: object
    tokenizer: object
    new_token_ids: dict = None
    vae_model: object = None
    vae_transform: object = None
    vit_transform: object = None
    sft_metadata: dict = None
    model_identity: dict = None


def prepare_bagel_environment(*, require_official_flash_attn=False):
    """Prioritize pinned local BAGEL sources and select an attention backend."""
    from sft_prepare_data.attention_backend import (
        attention_backend_identity,
        install_portable_attention_backend,
    )

    project_root = os.path.realpath(os.path.dirname(__file__))
    sys.path[:] = [
        entry for entry in sys.path
        if os.path.realpath(entry or os.curdir) != project_root]
    sys.path.insert(0, project_root)
    installed_portable = False
    if not require_official_flash_attn:
        installed_portable = install_portable_attention_backend()
    backend = attention_backend_identity(
        require_official=require_official_flash_attn,
        install_portable=False)
    if installed_portable:
        print('  [attention] PyTorch attention backend selected', flush=True)
    return backend


def load_sft_language_state(checkpoint_path):
    """Read the language-model portion of a UniWeaver text-SFT checkpoint."""
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f'invalid SFT checkpoint: {checkpoint_path}')
    state = checkpoint.get('model_state_dict', checkpoint)
    if not isinstance(state, dict):
        raise TypeError(f'invalid SFT checkpoint: {checkpoint_path}')
    language_state = {
        key: value for key, value in state.items()
        if key.startswith('language_model.')
    }
    if not language_state:
        raise ValueError(
            f'SFT checkpoint contains no language_model weights: {checkpoint_path}')
    metadata = {
        key: checkpoint.get(key)
        for key in (
            'step', 'checkpoint_type', 'training_mode',
            'training_manifest_sha256', 'training_provenance_sha256',
            'base_language_model_identity_sha256', 'contract_version',
            'trajectory_design', 'training_policy_version',
            'holdout_set_version', 'holdout_manifest_sha256')
    }
    del checkpoint, state
    return language_state, metadata


def validate_language_load_result(missing, unexpected, label):
    """Reject partial or incompatible language-state overlays."""
    missing_language = [
        key for key in missing if key.startswith('language_model.')]
    if missing_language or unexpected:
        raise ValueError(
            f'{label} is incomplete or incompatible: '
            f'{len(missing_language)} missing language tensors, '
            f'{len(unexpected)} unexpected tensors')


def _language_overlay_targets(model, language_state):
    """Resolve materialized language tensors and reject Accelerate offload."""
    offloaded_modules = []
    for module_name, module in model.named_modules():
        if not (module_name == 'language_model'
                or module_name.startswith('language_model.')):
            continue
        hook = getattr(module, '_hf_hook', None)
        if hook is not None and getattr(hook, 'offload', False):
            offloaded_modules.append(module_name)
    if offloaded_modules:
        raise RuntimeError(
            'SFT overlay cannot safely update Accelerate-offloaded language '
            f'modules: {offloaded_modules[:5]}')
    model_state = model.state_dict(keep_vars=True)
    missing_targets = sorted(set(language_state) - set(model_state))
    if missing_targets:
        raise ValueError(
            f'SFT overlay has no model targets for {missing_targets[:5]}')
    targets = {}
    for name, source in language_state.items():
        target = model_state[name]
        if getattr(target, 'is_meta', False):
            raise RuntimeError(
                f'SFT overlay target remains on the meta device: {name}')
        if tuple(target.shape) != tuple(source.shape):
            raise ValueError(
                f'SFT overlay shape mismatch for {name}: '
                f'{tuple(source.shape)} != {tuple(target.shape)}')
        targets[name] = target
    return targets


def _verify_language_overlay(targets, language_state):
    """Prove every checkpoint tensor is now exactly present in the model."""
    import torch

    mismatches = []
    with torch.no_grad():
        for name, target in targets.items():
            expected = language_state[name].to(
                device=target.device, dtype=target.dtype)
            if not torch.equal(target.detach(), expected):
                mismatches.append(name)
            del expected
    if mismatches:
        raise RuntimeError(
            'SFT checkpoint overlay did not update model tensors exactly: '
            f'{mismatches[:5]}')


def _resolve_checkpoint_holdout_provenance(metadata):
    """Validate the UniWeaver-Bench holdout identity stamped into a checkpoint."""
    from sft_train.benchmark_holdout import (
        HOLDOUT_MANIFEST_SHA256,
        HOLDOUT_SET_VERSION,
    )

    current_fields = ('holdout_set_version', 'holdout_manifest_sha256')
    has_current = [bool(metadata.get(key)) for key in current_fields]
    if any(has_current):
        if not all(has_current):
            raise ValueError('SFT checkpoint has an incomplete holdout identity')
        expected = (HOLDOUT_SET_VERSION, HOLDOUT_MANIFEST_SHA256)
        actual = tuple(metadata[key] for key in current_fields)
        if actual == expected:
            return {
                'kind': 'current_bench_120_holdout',
                'source_fields': list(current_fields),
                'set_version': actual[0],
                'manifest_sha256': actual[1],
                'bench_120_exclusion_status': 'verified',
            }
        raise ValueError(
            'SFT checkpoint does not match the UniWeaver-Bench holdout identity')
    raise ValueError('SFT checkpoint lacks a training holdout identity')


def apply_sft_checkpoint(
        model, checkpoint_path, *, assign=False, label='SFT',
        expected_base_language_model_sha256=None):
    """Overlay text-SFT language weights without touching visual components."""
    language_state, metadata = load_sft_language_state(checkpoint_path)
    required = (
        'checkpoint_type', 'training_mode', 'training_manifest_sha256',
        'training_provenance_sha256', 'base_language_model_identity_sha256',
        'contract_version', 'trajectory_design', 'training_policy_version')
    missing_metadata = [key for key in required if not metadata.get(key)]
    if missing_metadata:
        raise ValueError(
            f'SFT checkpoint lacks UniWeaver identity: {missing_metadata}')
    from sft_train.production import (
        TEXT_SFT_CHECKPOINT_TYPES,
        TRAINING_MODES,
        TRAINING_POLICY_VERSION,
    )

    if metadata['checkpoint_type'] not in TEXT_SFT_CHECKPOINT_TYPES:
        raise ValueError('SFT checkpoint has an unknown checkpoint_type')
    if metadata['training_mode'] not in TRAINING_MODES:
        raise ValueError('SFT checkpoint has an unknown training_mode')
    for key in (
            'training_manifest_sha256', 'training_provenance_sha256',
            'base_language_model_identity_sha256'):
        if not re.fullmatch(r'[0-9a-f]{64}', metadata[key]):
            raise ValueError(f'SFT checkpoint has an invalid {key}')
    holdout_provenance = _resolve_checkpoint_holdout_provenance(metadata)
    if not re.fullmatch(
            r'[0-9a-f]{64}', holdout_provenance['manifest_sha256']):
        raise ValueError('SFT checkpoint has an invalid holdout manifest hash')
    from core.production import CONTRACT_VERSION, TRAJECTORY_DESIGN

    expected_identity = {
        'contract_version': CONTRACT_VERSION,
        'trajectory_design': TRAJECTORY_DESIGN,
        'training_policy_version': TRAINING_POLICY_VERSION,
    }
    mismatches = [
        f'{key}: checkpoint={metadata.get(key)!r}, expected={value!r}'
        for key, value in expected_identity.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            'SFT checkpoint does not match the current UniWeaver training identity: '
            + '; '.join(mismatches))
    metadata['holdout_provenance'] = holdout_provenance
    if (expected_base_language_model_sha256 is not None
            and metadata['base_language_model_identity_sha256']
            != expected_base_language_model_sha256):
        raise ValueError('SFT checkpoint tokenizer/base-model identity mismatch')
    targets = _language_overlay_targets(model, language_state)
    missing, unexpected = model.load_state_dict(
        language_state, strict=False, assign=assign)
    validate_language_load_result(missing, unexpected, 'SFT checkpoint overlay')
    if assign:
        # assign=True replaces Parameter objects, so resolve the new targets.
        targets = _language_overlay_targets(model, language_state)
    _verify_language_overlay(targets, language_state)
    print(f'  [{label}] Applied {len(language_state)} language tensors from '
          f'{checkpoint_path}' + (
              f" (step {metadata['step']})"
              if metadata['step'] is not None else ''),
          flush=True)
    del language_state
    return metadata


def load_base_language_state(model_path):
    from safetensors.torch import load_file

    full_state = load_file(os.path.join(model_path, 'ema.safetensors'), device='cpu')
    language_state = {
        key: value for key, value in full_state.items()
        if key.startswith('language_model.')
    }
    del full_state
    return language_state


def _load_language_only_model(
        model_path, *, use_cache, require_official_flash_attn=False):
    """Construct and populate the sole BAGEL language-only model shape."""
    prepare_bagel_environment(
        require_official_flash_attn=require_official_flash_attn)
    from accelerate import init_empty_weights

    from modeling.bagel import Bagel, BagelConfig, Qwen2ForCausalLM
    from modeling.qwen2 import Qwen2Tokenizer

    model_path = resolve_bagel_model_path(model_path)
    llm_config = load_bagel_llm_config(model_path, use_cache=use_cache)
    config = BagelConfig(
        visual_gen=False, visual_und=False, llm_config=llm_config)
    with init_empty_weights():
        language_model = Qwen2ForCausalLM._from_config(llm_config)
        model = Bagel(language_model, None, config)

    base_state = load_base_language_state(model_path)
    missing, unexpected = model.load_state_dict(
        base_state, strict=False, assign=True)
    validate_language_load_result(
        missing, unexpected, 'base BAGEL language checkpoint')
    del base_state
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    return model_path, model, tokenizer


def load_text_training_model(model_path=None):
    """Load the exact language-only model shape used by UniWeaver text SFT."""
    import torch

    started_at = time.time()
    model_path, model, tokenizer = _load_language_only_model(
        model_path, use_cache=False, require_official_flash_attn=True)
    model = model.to(dtype=torch.bfloat16).cuda()
    gc.collect()
    torch.cuda.empty_cache()
    print(f'  [SFT] Text model loaded in {time.time() - started_at:.1f}s',
          flush=True)
    return BagelModelBundle(
        model=model, tokenizer=tokenizer,
        model_identity=bagel_model_content_identity(
            bagel_language_model_identity(model_path)))


def _multimodal_device_map(model, torch_module):
    from accelerate import infer_auto_device_map

    device_map = infer_auto_device_map(
        model,
        max_memory={index: '80GiB' for index in range(torch_module.cuda.device_count())},
        no_split_module_classes=['Bagel', 'Qwen2MoTDecoderLayer'],
    )
    colocated = [
        'language_model.model.embed_tokens', 'time_embedder', 'latent_pos_embed',
        'vae2llm', 'llm2vae', 'connector', 'vit_pos_embed',
    ]
    if torch_module.cuda.device_count() == 1:
        first_device = device_map.get(colocated[0], 'cuda:0')
        for key in colocated:
            device_map[key] = device_map.get(key, first_device)
    else:
        first_device = device_map.get(colocated[0])
        for key in colocated:
            if key in device_map:
                device_map[key] = first_device
    return device_map


def load_multimodal_model(model_path=None, mode=1, offload_folder='offload',
                          sft_checkpoint=None):
    """Load the full BAGEL understanding/generation model used online."""
    prepare_bagel_environment()
    if sft_checkpoint and mode != 1:
        raise ValueError('SFT checkpoint overlay currently requires mode=1')

    import torch
    from accelerate import (
        init_empty_weights,
        load_checkpoint_and_dispatch,
    )
    from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

    from data.data_utils import add_special_tokens
    from data.transforms import ImageTransform
    from modeling.autoencoder import load_ae
    from modeling.bagel import (
        Bagel,
        BagelConfig,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer

    model_path = resolve_bagel_model_path(model_path)
    llm_config = load_bagel_llm_config(model_path, use_cache=True)
    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(model_path, 'vit_config.json'))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1
    vae_model, vae_config = load_ae(
        local_path=os.path.join(model_path, 'ae.safetensors'))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
        latent_patch_size=2,
        max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    device_map = _multimodal_device_map(model, torch)
    sft_metadata = None
    if mode == 1:
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, 'ema.safetensors'),
            device_map=device_map,
            offload_buffers=True,
            offload_folder=offload_folder,
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()
        if sft_checkpoint:
            base_identity = bagel_model_content_identity(
                bagel_language_model_identity(model_path))
            sft_metadata = apply_sft_checkpoint(
                model, sft_checkpoint, label='ONLINE SFT',
                expected_base_language_model_sha256=base_identity['sha256'])
    elif mode == 2:
        quantization = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type='nf4',
        )
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, 'ema.safetensors'),
            bnb_quantization_config=quantization,
            device_map=device_map,
            offload_folder=offload_folder,
        ).eval()
    elif mode == 3:
        quantization = BnbQuantizationConfig(
            load_in_8bit=True, torch_dtype=torch.float32)
        model = load_and_quantize_model(
            model,
            weights_location=os.path.join(model_path, 'ema.safetensors'),
            bnb_quantization_config=quantization,
            device_map=device_map,
            offload_folder=offload_folder,
        ).eval()
    else:
        raise ValueError(f'unknown BAGEL mode: {mode}')

    return BagelModelBundle(
        model=model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        vae_model=vae_model,
        vae_transform=ImageTransform(1024, 512, 16),
        vit_transform=ImageTransform(980, 224, 14),
        sft_metadata=sft_metadata,
        model_identity=bagel_model_content_identity(
            bagel_model_identity(
                model_path, mode=mode, sft_checkpoint=sft_checkpoint)),
    )
