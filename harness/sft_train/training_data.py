"""Shared dataset selection and distributed sharding inputs for SFT trainers."""
import functools
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from bagel_model_loader import (
    bagel_language_model_identity,
    bagel_model_content_identity,
)
from core import validation as _VALIDATION
from core.hashing import sha256_bytes, sha256_file
from core.schema import PLAN_STAGE_NAMES
from core.serialization import atomic_write_json
from sft_train.benchmark_holdout import (
    HOLDOUT_SET_VERSION,
    holdout_manifest_sha256,
    load_benchmark_holdout_tasks,
    normalize_prompt,
)
from sft_train.production import (
    KEEP_CHECKPOINTS,
    training_policy,
    validate_retention_steps,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def open_training_logger(output_dir):
    """Return the sole flushed console/file logger used by SFT trainers."""
    # The caller owns this run-lifetime handle and closes it after training.
    log_file = open(os.path.join(output_dir, 'train.log'), 'a')  # noqa: SIM115

    def log(message):
        print(message, flush=True)
        log_file.write(message + '\n')
        log_file.flush()

    return log_file, log


def add_training_schedule_arguments(parser):
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument('--epochs', type=float)
    schedule.add_argument('--total-steps', type=int)


def resolve_training_schedule(args):
    """Preflight the corpus, freeze its identity, then resolve the schedule."""
    manifest, training_files = inspect_training_corpus(args.data_dir)
    args._training_files = training_files
    args._training_manifest = manifest
    args.training_manifest_sha256 = manifest['manifest_sha256']
    args.holdout_set_version = manifest['holdout_set']
    args.holdout_manifest_sha256 = manifest['holdout_manifest_sha256']
    if manifest['invalid']:
        report = persist_preflight_report(args.output_dir, manifest)
        raise ValueError(
            f"{len(manifest['invalid'])} training trajectories fail the UniWeaver "
            f'boundary; see {report}')
    if not training_files:
        raise ValueError('no UniWeaver training trajectories remain after benchmark holdout')
    if args.epochs is None and args.total_steps is None:
        args.epochs = 2.0
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError('epochs must be positive')
    args.trajectory_count = len(training_files)
    args.heldout_overlap_count = len(manifest['heldout'])
    examples = args.trajectory_count * len(PLAN_STAGE_NAMES)
    args.steps_per_epoch = examples / args.pack_samples
    if args.epochs is not None:
        args.total_samples = math.ceil(examples * args.epochs)
        args.total_steps = math.ceil(args.total_samples / args.pack_samples)
    else:
        args.total_samples = args.total_steps * args.pack_samples
    args.effective_epochs = args.total_samples / examples
    validate_retention_steps(args)
    return args


def inspect_training_corpus(data_dir):
    """Return a content-addressed UniWeaver corpus manifest and accepted paths."""
    root = Path(data_dir).resolve()
    files = sorted(root.glob('*.json'))
    holdout = load_benchmark_holdout_tasks()
    holdout_keys = {item['key'] for item in holdout}
    holdout_prompts = {item['prompt_sha256'] for item in holdout}
    accepted, heldout, invalid, accepted_paths = [], [], [], []
    seen_keys = set()
    for path in files:
        payload = path.read_bytes()
        file_hash = sha256_bytes(payload)
        relative = path.relative_to(root).as_posix()
        try:
            data = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            invalid.append({'path': relative, 'sha256': file_hash,
                            'reasons': [f'invalid JSON: {error}']})
            continue
        task_key = data.get('task_key')
        task_prompt = _VALIDATION.task_text(data)
        prompt_hash = (
            sha256_bytes(normalize_prompt(task_prompt).encode('utf-8'))
            if isinstance(task_prompt, str) and task_prompt.strip() else None)
        record = {'task_key': task_key, 'path': relative, 'sha256': file_hash,
                  'prompt_sha256': prompt_hash}
        if task_key in holdout_keys or prompt_hash in holdout_prompts:
            heldout.append(record)
            continue
        reasons = []
        if not isinstance(task_key, str) or not task_key.strip():
            reasons.append('missing task_key')
        elif task_key in seen_keys:
            reasons.append(f'duplicate training task_key: {task_key}')
        else:
            seen_keys.add(task_key)
        validation = _VALIDATION.validate_text_trajectory(data)
        if not validation['training_eligible']:
            reasons.extend(validation.get('acceptance', {}).get('training', []))
        if reasons:
            invalid.append(dict(record, reasons=reasons))
        else:
            accepted.append(record)
            accepted_paths.append(path)
    identity = {
        'manifest_version': 'uniweaver-training-corpus',
        'holdout_set': HOLDOUT_SET_VERSION,
        'holdout_manifest_sha256': holdout_manifest_sha256(),
        'accepted': accepted,
        'heldout': heldout,
        'invalid': invalid,
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    manifest = dict(identity, data_dir=str(root), file_count=len(files),
                    manifest_sha256=sha256_bytes(canonical.encode('utf-8')))
    return manifest, accepted_paths


def persist_preflight_report(output_dir, manifest):
    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.parent / f'{output.name}.training_preflight.json'
    atomic_write_json(path, manifest)
    return path


def expected_dataset_sha256(root, manifest):
    """Map each accepted trajectory's resolved path to its pinned content hash."""
    return {
        str((Path(root) / record['path']).resolve()): record['sha256']
        for record in manifest['accepted']
    }


def _git_record():
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT, check=True,
            text=True, capture_output=True).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=PROJECT_ROOT, check=True,
            text=True, capture_output=True).stdout.splitlines()
        return {'commit': commit, 'dirty': bool(status), 'status': status}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None, 'status': []}


def _training_source_hashes():
    paths = [
        PROJECT_ROOT / 'sft_train' / 'train_sft.py',
        PROJECT_ROOT / 'sft_train' / 'run_sft.sh',
        PROJECT_ROOT / 'sft_train' / 'production.py',
        PROJECT_ROOT / 'sft_train' / 'training_data.py',
        PROJECT_ROOT / 'sft_train' / 'benchmark_holdout.py',
        PROJECT_ROOT / 'datasets' / 'benchmarks' / 'uniweaver_bench' /
        'bench.json',
        PROJECT_ROOT / 'sft_train' / 'data' / 'sft_dataset.py',
        PROJECT_ROOT / 'bagel_model_loader.py',
        PROJECT_ROOT / 'core' / 'text_protocol.py',
        PROJECT_ROOT / 'bagel_text_backend.py',
        PROJECT_ROOT / 'bagel_multimodal_text_generation.py',
    ]
    paths.extend(sorted((PROJECT_ROOT / 'modeling' / 'qwen2').glob('*.py')))
    paths.extend(sorted((PROJECT_ROOT / 'modeling' / 'bagel').glob('*.py')))
    paths.extend(sorted((PROJECT_ROOT / 'core').glob('*.py')))
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
        for path in paths
    }


def build_training_provenance(args, torch_module):
    """Capture the exact model, code and numerical runtime used by training."""
    from sft_prepare_data.attention_backend import (
        attention_backend_identity,
    )

    cuda = torch_module.cuda
    gpu = None
    if cuda.is_available():
        props = cuda.get_device_properties(0)
        gpu = {
            'name': props.name,
            'total_memory': props.total_memory,
            'major': props.major,
            'minor': props.minor,
        }
    try:
        driver = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version',
             '--format=csv,noheader'], check=True, text=True,
            capture_output=True).stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        driver = None
    packages = {}
    for package in ('accelerate', 'flash-attn', 'safetensors', 'transformers'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    identity = {
        'provenance_version': 'uniweaver-text-sft',
        'training_mode': args.training_mode,
        # Record the override-applied policy already set on args; re-deriving the
        # default here would misreport a --config run.
        'optimization_policy': getattr(
            args, 'optimization_policy', None) or training_policy(),
        'benchmark_holdout': {
            'version': args.holdout_set_version,
            'manifest_sha256': args.holdout_manifest_sha256,
        },
        'base_language_model_identity': bagel_model_content_identity(
            bagel_language_model_identity(args.model_path)),
        'source_files': _training_source_hashes(),
        'git': _git_record(),
        'runtime': {
            'python': sys.version,
            'platform': platform.platform(),
            'torch': str(torch_module.__version__),
            'packages': packages,
            'attention_backend': attention_backend_identity(
                require_official=True, install_portable=False),
            'cuda': str(torch_module.version.cuda),
            'cudnn': (
                torch_module.backends.cudnn.version()
                if cuda.is_available() else None),
            'gpu': gpu,
            'nvidia_driver': driver,
            'environment': {
                key: os.environ.get(key)
                for key in (
                    'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                    'TORCH_CUDNN_SDPA_ENABLED')
            },
            'torch_backend': {
                'cudnn_benchmark': torch_module.backends.cudnn.benchmark,
                'cudnn_deterministic': torch_module.backends.cudnn.deterministic,
                'cuda_matmul_allow_tf32': (
                    torch_module.backends.cuda.matmul.allow_tf32
                    if cuda.is_available() else None),
            },
        },
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return dict(identity, model_path=str(Path(args.model_path).resolve()),
                identity_sha256=sha256_bytes(canonical.encode('utf-8')))


def resolve_training_data_arguments(args):
    positive_fields = (
        'max_seq_len', 'total_steps', 'pack_samples', 'log_every',
        'ckpt_every', 'max_grad_norm', 'lr')
    invalid = [
        field for field in positive_fields
        if getattr(args, field, None) is None or getattr(args, field) <= 0]
    if invalid:
        raise ValueError(
            'training arguments must be positive: ' + ', '.join(invalid))
    if not 0 <= args.warmup_steps < args.total_steps:
        raise ValueError(
            'warmup_steps must be non-negative and smaller than total_steps')
    return args


def latest_training_checkpoint(checkpoint_dir):
    """Return the numerically latest step_N.pt checkpoint, if one exists."""
    candidates = []
    for path in Path(checkpoint_dir).glob('step_*.pt'):
        match = re.fullmatch(r'step_(\d+)\.pt', path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


# Window for the trailing-mean CE that ranks retained checkpoints, expressed in
# steps rather than logged rows.
CE_WINDOW_STEPS = 2000
CE_TRACE_LENGTH = 20000


def trailing_mean_ce(ce_trace, step):
    """Mean training CE over the last CE_WINDOW_STEPS up to `step`."""
    cutoff = step - CE_WINDOW_STEPS
    window = [value for at_step, value in ce_trace if at_step > cutoff]
    if not window:
        return None
    return sum(window) / len(window)


def prune_checkpoints(checkpoint_dir, keep=KEEP_CHECKPOINTS, pinned=(),
                      best_ce_steps=0, ce_history=None, log=None):
    """Keep the newest `keep` checkpoints, every `pinned` step, and the best N."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return {'retained_steps': [], 'selected_steps': [],
                'newest_steps': [], 'pinned_steps': [],
                'best_ce_steps': [], 'deleted_steps': [],
                'delete_failed_steps': []}
    archived = []
    for path in checkpoint_dir.glob('step_*.pt'):
        try:
            archived.append((int(path.stem.split('_')[1]), path))
        except (ValueError, IndexError):
            continue
    if not archived:
        return {'retained_steps': [],
                'selected_steps': sorted(int(step) for step in pinned),
                'newest_steps': [],
                'pinned_steps': sorted(int(step) for step in pinned),
                'best_ce_steps': [], 'deleted_steps': [],
                'delete_failed_steps': []}
    archived.sort(key=lambda item: item[0])

    protected = {int(step) for step in pinned}
    best_steps = []
    if best_ce_steps > 0 and ce_history:
        ranked = sorted(ce_history.items(), key=lambda item: (item[1], item[0]))
        best_steps = [int(step) for step, _ in ranked[:best_ce_steps]]
        protected.update(best_steps)
    newest = {step for step, _ in archived[len(archived) - keep:]}
    removable = [(step, path) for step, path in archived
                 if step not in newest and step not in protected]
    if log is not None and removable:
        log(f"  prune: newest={sorted(newest)} "
            f"pinned={sorted(int(step) for step in pinned)} "
            f"best_ce={best_steps} -> deleting {[step for step, _ in removable]}")
    failed = []
    for _step, path in removable:
        try:
            path.unlink()
        except OSError as error:
            failed.append(_step)
            if log is not None:
                log(
                    f'  WARN checkpoint prune failed: path={path} '
                    f'operation=unlink error_type={type(error).__name__} '
                    f'error={error}')
    # Return and persist the complete retention decision so downstream
    # consumers read it rather than re-derive it. A step is retained when it is
    # still on disk after this call.
    retained = sorted(
        step for step in {*(s for s, _ in archived), *protected}
        if (Path(checkpoint_dir) / f'step_{step}.pt').exists())
    return {
        'retained_steps': retained,
        # Retention and selection differ by design: the rolling window supports
        # resume, while selected checkpoints are the explicit comparison points.
        'selected_steps': sorted(
            {int(step) for step in pinned} | set(best_steps)),
        'newest_steps': sorted(newest),
        'pinned_steps': sorted(int(step) for step in pinned),
        'best_ce_steps': sorted(best_steps),
        'deleted_steps': sorted(
            step for step, _ in removable if step not in failed),
        'delete_failed_steps': sorted(failed),
    }


def publish_retention_report(checkpoint_dir, *, step, keep, pinned,
                             best_ce_steps, ce_history, report):
    """Write ``retention.json`` beside the checkpoints and return its path."""
    path = Path(checkpoint_dir) / 'retention.json'
    atomic_write_json(path, {
        'step': int(step),
        'keep_checkpoints': int(keep),
        'keep_checkpoint_steps': sorted(int(item) for item in pinned),
        'keep_best_ce_steps': int(best_ce_steps),
        # The ranking itself, so the best-CE choice is auditable from uploaded
        # artifacts.
        'trailing_ce_by_step': {
            str(saved): round(value, 6)
            for saved, value in sorted((ce_history or {}).items())},
        **report,
    })
    return path


def advance_training_iterator(iterator, sample_count):
    """Restore the deterministic iterable-dataset cursor for exact resume."""
    if sample_count < 0:
        raise ValueError('sample_count must be non-negative')
    for _ in range(sample_count):
        try:
            next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                'training dataset ended before the saved resume cursor') from error
    return iterator


def restore_training_rng(torch_module, checkpoint):
    """Restore the random streams required for continuation-equivalent resume."""
    torch_state = checkpoint.get('torch_rng_state')
    cuda_states = checkpoint.get('cuda_rng_state_all')
    if torch_state is None or cuda_states is None:
        raise ValueError('resume checkpoint has no complete RNG state')
    torch_module.set_rng_state(torch_state)
    torch_module.cuda.set_rng_state_all(cuda_states)


def training_rng_state(torch_module):
    """Capture CPU and CUDA RNG streams in a weights-only-safe form."""
    return {
        'torch_rng_state': torch_module.get_rng_state(),
        'cuda_rng_state_all': torch_module.cuda.get_rng_state_all(),
    }


def _training_config(args):
    return {
        key.replace('_', '-'): (
            str(value) if isinstance(value, Path) else value)
        for key, value in sorted(vars(args).items()) if not key.startswith('_')
    }


def initialize_training_run(args, provenance, immutable_fields):
    """Atomically create or strictly verify the complete run identity."""
    output_dir = Path(args.output_dir)
    files = {
        'training_manifest.json': args._training_manifest,
        'training_provenance.json': provenance,
        'config.json': _training_config(args),
    }
    if args.resume:
        if not output_dir.is_dir():
            raise ValueError(f'--resume requires training directory: {output_dir}')
        previous_files = {}
        for name in files:
            path = output_dir / name
            if not path.is_file():
                raise ValueError(f'--resume requires complete run identity: {path}')
            previous_files[name] = json.loads(path.read_text(encoding='utf-8'))
        previous_manifest = previous_files['training_manifest.json']
        if (previous_manifest.get('manifest_sha256')
                != args.training_manifest_sha256):
            raise ValueError('training corpus changed since the original run')
        previous_provenance = previous_files['training_provenance.json']
        if (previous_provenance.get('identity_sha256')
                != provenance['identity_sha256']):
            raise ValueError('training model, code, or numerical runtime changed')
        previous = previous_files['config.json']
        config = files['config.json']
        mismatches = []
        for field in immutable_fields:
            key = field.replace('_', '-')
            if previous.get(key) != config.get(key):
                mismatches.append(
                    f'{key}: previous={previous.get(key)!r}, '
                    f'current={config.get(key)!r}')
        if mismatches:
            raise ValueError(
                'resume configuration does not match the original run: '
                + '; '.join(mismatches))
        return previous_files

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(
                f'training output is not empty; use --resume or a new path: '
                f'{output_dir}')
        output_dir.rmdir()
    staging = output_dir.parent / f'.{output_dir.name}.initializing-{os.getpid()}'
    if staging.exists():
        raise FileExistsError(f'stale training initialization exists: {staging}')
    try:
        staging.mkdir()
        for name, payload in files.items():
            (staging / name).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding='utf-8')
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return files


def create_training_dataset(args, tokenizer, *, repeat=True):
    """Load the selected dataset and pass one canonical distributed identity."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size != 1:
        raise RuntimeError(
            'UniWeaver text SFT requires WORLD_SIZE=1')
    module = importlib.import_module('sft_train.data.sft_dataset')
    root = Path(args.data_dir).resolve()
    expected_hashes = expected_dataset_sha256(root, args._training_manifest)
    return module.SFTTextDataset(
        args.data_dir,
        tokenizer,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        repeat=repeat,
        local_rank=int(os.environ.get('RANK', '0')),
        world_size=world_size,
        files=args._training_files,
        expected_sha256=expected_hashes,
    )


def preflight_training_window(args, tokenizer):
    """Tokenize every held-in exchange before allocating the training model."""
    dataset = create_training_dataset(args, tokenizer, repeat=False)
    actual = sum(1 for _ in dataset)
    expected = args.trajectory_count * len(PLAN_STAGE_NAMES)
    if actual != expected:
        raise ValueError(
            f'training-window preflight yielded {actual}/{expected} exchanges')
    return actual


def normalized_ce_loss(ce):
    """Match BAGEL training by averaging CE over response tokens."""
    if ce is None or ce.numel() == 0:
        raise ValueError('model returned no response-token CE values')
    return ce.sum() / ce.numel()


# The packed causal mask depends only on sequence length, so it is safe to cache;
# the bound keeps distinct lengths from retaining every mask.


@functools.lru_cache(maxsize=32)
def _causal_mask(sequence_length):
    import math

    import torch

    return torch.triu(torch.full(
        (sequence_length, sequence_length), -math.inf), diagonal=1)


def build_text_sft_batch(batch_ids, batch_prefix_lengths, device='cuda'):
    """Build the sole packed causal-LM batch consumed by BAGEL text SFT."""
    import torch

    if not batch_ids or len(batch_ids) != len(batch_prefix_lengths):
        raise ValueError('training batch ids/prefix lengths are empty or differ')
    total_tokens = sum(len(ids) for ids in batch_ids)
    sequence_offset = 0
    text_ids = []
    label_ids = []
    text_indexes = []
    position_ids = []
    masks = []
    ce_indexes = torch.zeros(total_tokens, dtype=torch.bool)
    for ids, prefix_length in zip(batch_ids, batch_prefix_lengths):
        sequence_length = len(ids)
        if not 0 < prefix_length < sequence_length:
            raise ValueError(
                f'invalid prefix length {prefix_length}/{sequence_length}')
        tokens = torch.tensor(ids, dtype=torch.long)
        text_ids.append(tokens)
        label_ids.append(tokens[prefix_length:])
        text_indexes.append(torch.arange(
            sequence_offset, sequence_offset + sequence_length))
        position_ids.append(torch.arange(sequence_length))
        masks.append(_causal_mask(sequence_length))
        # Position i predicts token i+1; supervise the complete assistant turn.
        ce_indexes[
            sequence_offset + prefix_length - 1:
            sequence_offset + sequence_length - 1] = True
        sequence_offset += sequence_length
    return {
        'sequence_length': total_tokens,
        'packed_text_ids': torch.cat(text_ids).to(device),
        'packed_text_indexes': torch.cat(text_indexes).to(device),
        'packed_position_ids': torch.cat(position_ids).to(device),
        'sample_lens': [len(ids) for ids in batch_ids],
        'nested_attention_masks': [mask.to(device) for mask in masks],
        'packed_label_ids': torch.cat(label_ids).to(device),
        'ce_loss_indexes': ce_indexes.to(device),
    }


def update_stage_loss_totals(totals, ce, stages, response_lengths):
    """Accumulate observational per-stage CE without changing optimization."""
    if len(stages) != len(response_lengths):
        raise ValueError('stage labels and response lengths differ')
    if sum(response_lengths) != ce.numel():
        raise ValueError('response lengths do not match returned CE values')
    offset = 0
    for stage, length in zip(stages, response_lengths):
        segment = ce[offset:offset + length]
        record = totals.setdefault(stage, {'loss_sum': 0.0, 'tokens': 0})
        record['loss_sum'] += float(segment.detach().float().sum().item())
        record['tokens'] += length
        offset += length


def stage_loss_means(totals):
    return {
        stage: record['loss_sum'] / record['tokens']
        for stage, record in totals.items() if record['tokens']
    }
