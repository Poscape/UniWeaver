"""Attention-backend detection and the portable PyTorch implementation."""
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import types


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **_kwargs):
    """Implement flash_attn's packed variable-length interface with PyTorch SDPA."""
    import torch
    from torch.nn import functional as F

    outputs = []
    for batch_index in range(int(cu_seqlens_q.numel()) - 1):
        q_start, q_end = (int(x) for x in cu_seqlens_q[batch_index:batch_index + 2])
        k_start, k_end = (int(x) for x in cu_seqlens_k[batch_index:batch_index + 2])
        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        key = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        value = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        attention_mask = None
        if causal:
            query_length, key_length = query.shape[-2], key.shape[-2]
            query_positions = torch.arange(query_length, device=q.device) + (
                key_length - query_length)
            key_positions = torch.arange(key_length, device=q.device)
            attention_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        kwargs = {
            "attn_mask": attention_mask,
            "dropout_p": dropout_p,
            "is_causal": False,
            "scale": softmax_scale,
        }
        try:
            attended = F.scaled_dot_product_attention(
                query, key, value, enable_gqa=query.shape[1] != key.shape[1], **kwargs)
        except TypeError:
            if query.shape[1] != key.shape[1]:
                repeat = query.shape[1] // key.shape[1]
                key = key.repeat_interleave(repeat, dim=1)
                value = value.repeat_interleave(repeat, dim=1)
            attended = F.scaled_dot_product_attention(query, key, value, **kwargs)
        outputs.append(attended.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else q.new_empty(q.shape)


def install_portable_attention_backend():
    """Install the portable attention implementation when no package is present."""
    if importlib.util.find_spec('flash_attn') is not None or 'flash_attn' in sys.modules:
        return False
    module = types.ModuleType('flash_attn')
    module.__spec__ = importlib.machinery.ModuleSpec('flash_attn', loader=None)
    module.__bagel_attention_backend__ = 'pytorch-sdpa'
    module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules['flash_attn'] = module
    return True


def attention_backend_identity(*, require_official=False, install_portable=False):
    """Return the active backend and enforce the requested implementation."""
    if install_portable:
        install_portable_attention_backend()
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func as active_function
    except (ImportError, OSError, RuntimeError) as error:
        if require_official:
            raise RuntimeError(
                'official flash_attn is required for formal BAGEL text SFT') \
                from error
        return {
            'implementation': 'not-installed',
            'package': 'flash-attn',
            'version': None,
            'official': False,
        }
    implementation = getattr(
        flash_attn, '__bagel_attention_backend__', 'official-flash-attn')
    official = implementation == 'official-flash-attn'
    if require_official and not official:
        raise RuntimeError(
            'official flash_attn is required for formal BAGEL text SFT; '
            f'active backend is {implementation}')
    try:
        version = importlib.metadata.version('flash-attn') if official else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    if official and version is None and require_official:
        raise RuntimeError(
            'formal training provenance requires flash_attn package metadata')
    if official:
        runtime_version = version
    else:
        import torch

        runtime_version = str(torch.__version__)
    return {
        'implementation': implementation,
        'package': 'flash-attn' if official else 'torch',
        'version': runtime_version,
        'official': official,
        'function_module': getattr(active_function, '__module__', None),
    }
