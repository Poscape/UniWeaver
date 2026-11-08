"""UniWeaver SFT dataset using the same request builders as inference."""
import hashlib
import json
import random
from pathlib import Path

from torch.utils.data import IterableDataset, get_worker_info

from core import validation as _VALIDATION
from core.production import require_trajectory_identity
from core.schema import PLAN_STAGE_NAMES
from core.text_protocol import encode_bagel_training_exchange

# Bounded memo of format_trajectory results keyed by (content sha256,
# max_seq_len, include_planner); the corpus is hash-pinned for the whole run.
_FORMATCACHE = {}


def shard_file_indices(indices, rank=0, world_size=1,
                       worker_id=0, num_workers=1):
    """Partition files exactly once across distributed ranks and loader workers."""
    if world_size < 1 or num_workers < 1:
        raise ValueError('world_size and num_workers must be positive')
    if not 0 <= rank < world_size:
        raise ValueError(f'rank {rank} is outside world_size {world_size}')
    if not 0 <= worker_id < num_workers:
        raise ValueError(
            f'worker_id {worker_id} is outside num_workers {num_workers}')
    shard_index = rank * num_workers + worker_id
    shard_count = world_size * num_workers
    return list(indices)[shard_index::shard_count]


def format_training_sample(tokenizer, messages, target_raw, max_seq_len):
    """One (input_ids, prefix_len) sample; None outside the bounds, never truncated."""
    roles = [message.get('role') for message in messages]
    if roles != ['system', 'user']:
        raise ValueError(
            f'UniWeaver training exchange requires system/user roles, got {roles}')
    total_ids, prefix_len = encode_bagel_training_exchange(
        tokenizer,
        messages[0].get('content', ''),
        messages[1].get('content', ''),
        target_raw,
    )
    prefix_ids = total_ids[:prefix_len]
    response_ids = total_ids[prefix_len:]

    if not prefix_ids or not response_ids:
        return
    if len(prefix_ids) + len(response_ids) > max_seq_len:
        return

    total_ids = prefix_ids + response_ids
    if len(total_ids) < 10:
        return
    return total_ids, len(prefix_ids)


def _latest_attempt(data, stage):
    attempts = data.get('raw_attempts', {}).get(stage, [])
    if not isinstance(attempts, list) or not attempts:
        return None
    attempt = attempts[-1]
    required = ('system_text', 'raw_input', 'raw_output')
    if (not isinstance(attempt, dict)
            or not all(isinstance(attempt.get(key), str)
                       and attempt.get(key) for key in required)):
        return None
    return attempt


def _training_exchange(system_text, user_text, target):
    return (
        [{"role": "system", "content": system_text},
         {"role": "user", "content": user_text}],
        target,
    )


def iter_training_examples(data, include_planner=True, validation=None):
    """Yield only exact UniWeaver Planner exchanges recorded at construction."""
    require_trajectory_identity(data)
    if not include_planner:
        return
    if validation is None:
        validation = _VALIDATION.validate_text_trajectory(data)
    if not validation['training_eligible']:
        raise ValueError(
            'UniWeaver trajectory is not training-eligible: '
            + '; '.join(validation['acceptance']['training']))
    for stage in PLAN_STAGE_NAMES:
        attempt = _latest_attempt(data, stage)
        if attempt is None:
            raise ValueError(f'UniWeaver trajectory has no exact {stage} raw attempt')
        messages, target = _training_exchange(
            attempt['system_text'], attempt['raw_input'], attempt['raw_output'])
        yield stage, messages, target


class SFTTextDataset(IterableDataset):
    """Read UniWeaver JSON files and yield exact Planner exchange targets."""

    def __init__(self, data_dir, tokenizer, *, max_seq_len, seed,
                 local_rank=0, world_size=1, repeat=True,
                 include_planner=True, files=None, expected_sha256=None):
        super().__init__()
        self.files = list(files) if files is not None else sorted(
            Path(data_dir).glob('*.json'))
        if not self.files:
            raise ValueError(
                f'no trajectory JSON training files found in {data_dir}')
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.repeat = repeat
        self.local_rank = local_rank
        self.world_size = world_size
        self.include_planner = include_planner
        self.expected_sha256 = dict(expected_sha256 or {})
        if world_size < 1 or not 0 <= local_rank < world_size:
            raise ValueError(
                f'local_rank {local_rank} is outside world_size {world_size}')

    def format_trajectory(self, data, validate=True, content_key=None):
        """Format and count one trajectory through the exact training boundary."""
        if content_key is not None:
            cache_key = (content_key, self.max_seq_len, self.include_planner)
            cached = _FORMATCACHE.get(cache_key)
            if cached is not None:
                return cached
        validation = _VALIDATION.validate_text_trajectory(data)
        if validate and not validation['training_eligible']:
            result = {
                'accepted': False,
                'validation': validation,
                'expected': {stage: 0 for stage in PLAN_STAGE_NAMES},
                'samples': {stage: [] for stage in PLAN_STAGE_NAMES},
            }
            if content_key is not None:
                _FORMATCACHE[cache_key] = result
            return result
        expected = {stage: 0 for stage in PLAN_STAGE_NAMES}
        samples = {stage: [] for stage in PLAN_STAGE_NAMES}
        for stage, messages, target in iter_training_examples(
                data, include_planner=self.include_planner,
                validation=validation):
            expected[stage] += 1
            sample = format_training_sample(
                self.tokenizer, messages, target, self.max_seq_len)
            if sample:
                samples[stage].append((*sample, stage))
        result = {
            'accepted': True,
            'validation': validation,
            'expected': expected,
            'samples': samples,
        }
        if content_key is not None:
            _FORMATCACHE[cache_key] = result
        return result

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        epoch = 0
        while True:
            rng = random.Random(self.seed + epoch)
            indices = list(range(len(self.files)))
            rng.shuffle(indices)
            indices = shard_file_indices(
                indices,
                rank=self.local_rank,
                world_size=self.world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            if self.files and not indices:
                raise RuntimeError(
                    'Dataset has fewer files than distributed ranks × workers; '
                    'reduce workers or add data')

            for idx in indices:
                path = self.files[idx]
                try:
                    payload = path.read_bytes()
                    expected_hash = self.expected_sha256.get(str(path.resolve()))
                    if (expected_hash is not None
                            and hashlib.sha256(payload).hexdigest() != expected_hash):
                        raise ValueError('content changed after training preflight')
                    data = json.loads(payload)
                except Exception as error:
                    raise RuntimeError(
                        f'failed to read training trajectory {path}: {error}') from error

                # expected_hash is the preflight content sha256: a stable,
                # immutable cache key for the tokenized result.
                formatted = self.format_trajectory(
                    data, content_key=expected_hash)
                if not formatted['accepted']:
                    reasons = formatted['validation'].get(
                        'acceptance', {}).get('training', [])
                    raise RuntimeError(
                        f'training trajectory is ineligible: {path}: '
                        + '; '.join(reasons))
                for stage in PLAN_STAGE_NAMES:
                    actual = len(formatted['samples'][stage])
                    expected = formatted['expected'][stage]
                    if actual != expected:
                        raise RuntimeError(
                            f'training trajectory exceeds or violates the '
                            f'{self.max_seq_len}-token boundary: {path}: '
                            f'{stage} samples {actual}/{expected}')
                for stage in PLAN_STAGE_NAMES:
                    yield from formatted['samples'][stage]

            epoch += 1
            if not self.repeat:
                break
