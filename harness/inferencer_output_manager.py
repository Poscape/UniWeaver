"""Atomic incremental JSON and image persistence for trajectory generation."""
import json
import os
import shutil
import socket
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from core.hashing import sha256_file
from core.policy_registry import (
    INTERRUPTION_RECOVERY_EVENT,
    PERSISTENCE_FAILURE_EVENT,
    RUNTIME_MODEL_OPERATION_EVENT,
    STEP_ATTEMPT_PERSISTENCE_EVENT,
)
from core.serialization import atomic_write_json, atomic_write_text

DEFAULT_LEASE_SECONDS = 300.0


class OutputManager:
    """Manage incremental state, image persistence, and diagnostics."""

    def __init__(self, out_dir: Path, task_key: str, task_prompt: str,
                 plan: dict, step_generations: dict, resume: bool = False,
                 started_at: float | None = None):
        """Create the manager for one task."""
        self.out_dir = Path(out_dir)
        if (not resume and self.out_dir.exists()
                and any(self.out_dir.iterdir())):
            raise FileExistsError(
                f'refusing to mix a new run into non-empty output: {self.out_dir}')
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.out_dir / 'state.json'
        self.log_path = self.out_dir / 'gen.log'
        self._state_lock = threading.RLock()

        if resume and self.state_path.exists():
            with open(self.state_path, encoding='utf-8') as handle:
                self.state = json.load(handle)
            if (self.state.get('task_key') != task_key
                    or self.state.get('task_prompt') != task_prompt
                    or self.state.get('plan') != plan):
                raise ValueError(
                    'resume state does not match the requested task and runtime plan')
            if self.state.get('status') == 'done':
                return
            owner = self.state.get('owner', {})
            if (self.state.get('status') == 'running'
                    and owner == self._owner_identity()):
                return
            self._recover_interrupted_run()
            self.state['status'] = 'running'
            self.state['resumed_at'] = time.time()
            self.state['owner'] = self._owner_identity()
            self.state['heartbeat_at'] = time.time()
            self.state['lease_expires_at'] = (
                self.state['heartbeat_at'] + DEFAULT_LEASE_SECONDS
            )
            self._write_state()
            return

        title = plan.get('title', plan.get('steps', [{}])[0].get('image_prompt', '')[:40] if plan.get('steps') else '')
        self.state = {
            'task_key': task_key,
            'task_prompt': task_prompt,
            'trajectory_design': plan.get('trajectory_design', ''),
            'contract_version': plan.get('contract_version', ''),
            'title': title,
            'plan': plan,
            'step_generations_raw': step_generations,
            'conditions': {},
            'steps': {},
            'started_at': time.time() if started_at is None else started_at,
            'status': 'running',
            'owner': self._owner_identity(),
            'heartbeat_at': time.time(),
            'lease_expires_at': time.time() + DEFAULT_LEASE_SECONDS,
            'active_operations': {},
        }
        self._write_state()

    @staticmethod
    def _owner_identity():
        return {'hostname': socket.gethostname(), 'pid': os.getpid()}

    @staticmethod
    def _pid_is_alive(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return False
        return True

    def _recover_interrupted_run(self):
        """Record an abandoned running state before an explicit resume."""
        if self.state.get('status') != 'running':
            return
        owner = self.state.get('owner', {})
        same_host = owner.get('hostname') == socket.gethostname()
        if (same_host and owner.get('pid') != os.getpid()
                and self._pid_is_alive(owner.get('pid'))):
            raise RuntimeError(
                f"refusing to resume a live run owned by pid {owner.get('pid')}")
        now = time.time()
        if not same_host and self.state.get('lease_expires_at', 0) > now:
            raise RuntimeError(
                "refusing to resume a run whose cross-host lease is still active"
            )
        self.state.setdefault('interruptions', []).append({
            'policy_id': INTERRUPTION_RECOVERY_EVENT['policy_id'],
            'status': 'interrupted',
            'detected_at': now,
            'previous_owner': owner,
            'last_heartbeat_at': self.state.get('heartbeat_at'),
            'active_operations': self.state.get('active_operations', {}),
        })

    def heartbeat(self, operation=None, detail=None, deadline_seconds=None):
        """Persist a lease checkpoint around a potentially long operation."""
        with self._state_lock:
            now = time.time()
            self.state['heartbeat_at'] = now
            if operation is not None:
                token = uuid.uuid4().hex
                self.state.setdefault('active_operations', {})[token] = {
                    'policy_id': RUNTIME_MODEL_OPERATION_EVENT['policy_id'],
                    'operation': operation,
                    'detail': detail,
                    'started_at': now,
                    'deadline_at': (
                        now + deadline_seconds if deadline_seconds else None
                    ),
                }
                self._refresh_lease(now)
                self._write_state()
                return token
            self._refresh_lease(now)
            self._write_state()
            return None

    def _refresh_lease(self, now):
        deadlines = [
            item.get('deadline_at')
            for item in self.state.get('active_operations', {}).values()
            if isinstance(item, dict)
            and isinstance(item.get('deadline_at'), (int, float))
        ]
        self.state['lease_expires_at'] = max(
            [now + DEFAULT_LEASE_SECONDS, *deadlines]
        )

    def finish_operation(self, token, *, outcome='completed', error=None):
        if token is None:
            return
        with self._state_lock:
            record = self.state.setdefault('active_operations', {}).pop(
                token, None)
            self.state['heartbeat_at'] = time.time()
            self._refresh_lease(self.state['heartbeat_at'])
            if record is not None:
                self.state.setdefault('operation_history', []).append({
                    **record,
                    'finished_at': self.state['heartbeat_at'],
                    'outcome': outcome,
                    'error': (
                        f'{type(error).__name__}: {error}'
                        if error is not None else None
                    ),
                })
            self._write_state()

    @contextmanager
    def operation(self, operation, detail=None, deadline_seconds=None):
        token = self.heartbeat(operation, detail, deadline_seconds)
        try:
            yield
        except BaseException as error:
            self.finish_operation(token, outcome='failed', error=error)
            raise
        else:
            self.finish_operation(token)

    def completed_items(self):
        """Return the records whose persisted pixels are unchanged."""
        recovered = self._recover_step_attempt_sidecars()
        condition_dir = self.out_dir / 'conditions'
        step_dir = self.out_dir / 'steps'
        conditions = set()
        for key, record in self.state.get('conditions', {}).items():
            if not isinstance(record, dict) or record.get('generated') is not True:
                continue
            path = condition_dir / f'{Path(key).name}.png'
            if self._persisted_image_matches(path, record):
                conditions.add(key)
                continue
            if not self._record_has_hash(record):
                # Incomplete record: leave it out so Condition generation reruns.
                continue
            raise ValueError(f'condition resume image hash mismatch for {key!r}')
        steps = set()
        for key, record in self.state.get('steps', {}).items():
            if not isinstance(record, dict) or record.get('generated') is not True:
                continue
            path = step_dir / f'step_{key}.png'
            if self._persisted_image_matches(path, record):
                steps.add(key)
                continue
            try:
                attempt = self._selected_attempt_record(key, record)
            except ValueError:
                if not self._record_has_hash(record):
                    # Incomplete record: leave the Step out so it reruns.
                    continue
                raise ValueError(f'Step resume image hash mismatch at {key}')
            source = self.out_dir / attempt['image_path']
            self._copy_file_atomic(
                source, path, expected_sha256=record['image_sha256'])
            recovered.append({
                'policy_id': STEP_ATTEMPT_PERSISTENCE_EVENT['policy_id'],
                'operation': 'rebuild_canonical_step',
                'step_index': int(key),
                'attempt_index': attempt['attempt_index'],
                'image_sha256': record['image_sha256'],
            })
            steps.add(key)
        if recovered:
            self.state.setdefault('persistence_recoveries', []).extend(recovered)
            self._write_state()
        return conditions, steps

    def _selected_attempt_record(self, step_index, step_record):
        selected = step_record.get('selected_attempt')
        matches = [
            item for item in self.state.get('step_attempts', {}).get(
                str(step_index), [])
            if isinstance(item, dict) and item.get('attempt_index') == selected
        ]
        if len(matches) != 1:
            raise ValueError(
                f'Step {step_index} has no unique selected immutable attempt')
        attempt = matches[0]
        if attempt.get('image_sha256') != step_record.get('image_sha256'):
            raise ValueError(
                f'Step {step_index} selected attempt hash conflicts with state')
        return attempt

    def _recover_step_attempt_sidecars(self):
        """Merge durable attempt commits left ahead of state.json after a crash."""
        recoveries = []
        for path in sorted(self.out_dir.glob('attempts/step_*/attempt_*.json')):
            with path.open(encoding='utf-8') as handle:
                payload = json.load(handle)
            attempt = payload.get('attempt')
            step = payload.get('selected_step')
            step_index = payload.get('step_index')
            if not isinstance(attempt, dict) or not isinstance(step, dict):
                raise TypeError(f'invalid Step attempt sidecar: {path}')
            key = str(step_index)
            expected_dir = f'step_{key}'
            expected_name = f"attempt_{attempt.get('attempt_index')}.json"
            if path.parent.name != expected_dir or path.name != expected_name:
                raise ValueError(f'Step attempt sidecar identity mismatch: {path}')
            image_path = self.out_dir / str(attempt.get('image_path', ''))
            try:
                image_path.resolve().relative_to(self.out_dir.resolve())
            except ValueError as exc:
                raise ValueError(
                    f'Step attempt image escapes output directory: {image_path}'
                ) from exc
            expected_hash = attempt.get('image_sha256')
            if (not image_path.is_file() or not isinstance(expected_hash, str)
                    or sha256_file(image_path) != expected_hash):
                raise ValueError(f'Step attempt sidecar image mismatch: {path}')
            attempts = self.state.setdefault('step_attempts', {}).setdefault(
                key, [])
            matching = [
                item for item in attempts
                if item.get('attempt_index') == attempt.get('attempt_index')
            ]
            if len(matching) > 1 or (matching and matching[0] != attempt):
                raise ValueError(
                    f'conflicting durable provenance for Step {key} '
                    f"attempt {attempt.get('attempt_index')}")
            if not matching:
                attempts.append(attempt)
                attempts.sort(key=lambda item: item['attempt_index'])
                current = self.state.get('steps', {}).get(key, {})
                current_index = current.get('selected_attempt', -1)
                if not isinstance(current_index, int):
                    current_index = -1
                if attempt['attempt_index'] >= current_index:
                    self.state.setdefault('steps', {})[key] = step
                recoveries.append({
                    'policy_id': STEP_ATTEMPT_PERSISTENCE_EVENT['policy_id'],
                    'operation': 'recover_attempt_sidecar',
                    'step_index': int(key),
                    'attempt_index': attempt['attempt_index'],
                    'image_sha256': expected_hash,
                })
        return recoveries

    @staticmethod
    def _persisted_image_matches(path, record):
        expected = record.get('image_sha256')
        if not isinstance(expected, str) or len(expected) != 64:
            return False
        if not path.is_file():
            return False
        return sha256_file(path) == expected

    @staticmethod
    def _record_has_hash(record):
        expected = record.get('image_sha256')
        return isinstance(expected, str) and len(expected) == 64

    @staticmethod
    def _save_png_atomic(image, path):
        """Persist one complete PNG and return the hash of its exact bytes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        try:
            image.save(temporary, format='PNG')
            digest = sha256_file(temporary)
            os.replace(temporary, path)
            return digest
        finally:
            if temporary.exists():
                temporary.unlink()

    @classmethod
    def _save_png_immutable(cls, image, path):
        """Create one immutable attempt image, tolerating an identical crash replay."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            return cls._save_png_atomic(image, path)
        existing = sha256_file(path)
        temporary = path.with_suffix(path.suffix + '.verify.tmp')
        try:
            image.save(temporary, format='PNG')
            candidate = sha256_file(temporary)
        finally:
            if temporary.exists():
                temporary.unlink()
        if candidate != existing:
            raise FileExistsError(
                f'refusing to overwrite immutable Step attempt: {path}')
        return existing

    @staticmethod
    def _write_json_immutable(value, path):
        """Create immutable JSON metadata, allowing an identical replay."""
        if path.exists():
            with path.open(encoding='utf-8') as handle:
                if json.load(handle) != value:
                    raise FileExistsError(
                        f'refusing to overwrite immutable metadata: {path}')
            return
        atomic_write_json(path, value)

    @staticmethod
    def _copy_file_atomic(source, destination, *, expected_sha256):
        """Atomically materialize a canonical file from immutable evidence."""
        if not source.is_file() or sha256_file(source) != expected_sha256:
            raise ValueError(f'immutable source hash mismatch: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != expected_sha256:
                raise ValueError(f'canonical copy hash mismatch: {destination}')
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    def record_image_experiment(self, **configuration):
        """Persist explicit image-intervention settings before execution starts."""
        self.state['image_experiment'] = self._json_safe(configuration)
        self._write_state()

    @staticmethod
    def _json_safe(value):
        """Drop non-serializable runtime objects from persisted diagnostics."""
        if isinstance(value, dict):
            return {
                str(key): OutputManager._json_safe(item)
                for key, item in value.items() if key != 'image'
            }
        if isinstance(value, (list, tuple)):
            return [OutputManager._json_safe(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _write_markdown(self):
        """Write the compact human-readable guide beside the canonical state."""
        plan = self.state.get('plan', {})
        lines = [f'# {plan.get("title", "Guide")}', '', '## Steps', '']
        for step_index in sorted(self.state.get('steps', {}), key=int):
            step = self.state['steps'][step_index]
            lines.extend([
                f'### {step_index}. {step.get("content", "")}', '',
                f'![Step {step_index}](steps/step_{step_index}.png)', '',
            ])
        atomic_write_text(self.out_dir / 'guide.md', '\n'.join(lines) + '\n')

    def publish_state(self):
        """Publish canonical runtime state through the shared atomic writer."""
        with self._state_lock:
            atomic_write_json(self.state_path, self.state)

    def record_persistence_failure(self, error, *, operation=None):
        """Write an auxiliary recovery record for persistence diagnostics."""
        record = {
            'policy_id': PERSISTENCE_FAILURE_EVENT['policy_id'],
            'status': 'unresolved',
            'operation': operation,
            'exception_type': type(error).__name__,
            'message': str(error),
            'recorded_at': time.time(),
        }
        try:
            atomic_write_json(self.out_dir / 'persistence_error.json', record)
        except Exception:  # noqa: BLE001
            return False
        return True

    def update_state(self, **fields):
        """Apply one persisted state transition atomically."""
        with self._state_lock:
            self.state.update({
                key: self._json_safe(value) for key, value in fields.items()
            })
            self.state['heartbeat_at'] = time.time()
            self._refresh_lease(self.state['heartbeat_at'])
            self.publish_state()

    def _write_state(self):
        self.publish_state()

    def log(self, stage: str, key: str, status: str, **kwargs):
        """Write diagnostic log entry directly to log file (flushed immediately)."""
        parts = [f'stage={stage}', f'key={key}', f'status={status}']
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f'{k}={v:.1f}')
            else:
                parts.append(f'{k}={v}')
        line = ' | '.join(parts)
        ts = datetime.now().astimezone().strftime('%H:%M:%S')
        with open(self.log_path, 'a') as lf:
            lf.write(f'{ts} | {line}\n')
            lf.flush()
        # Also print to stderr for real-time monitoring
        import sys
        print(f'[LOG] {line}', file=sys.stderr, flush=True)

    def condition_done(self, key: str, result: dict):
        """Called after each condition image is generated."""
        from core.validation import condition_item_is_executable

        candidate = {
            'key': key,
            'type': result.get('type'),
            'image_prompt': result.get('image_prompt'),
        }
        if not condition_item_is_executable(candidate):
            raise ValueError(f'unsafe or invalid Condition key: {key!r}')
        if key in self.state['conditions']:
            raise ValueError(f'duplicate Condition key: {key!r}')
        img = result.get('image')
        image_sha256 = None
        if img:
            image_dir = self.out_dir / 'conditions'
            image_sha256 = self._save_png_atomic(
                img, image_dir / f'{key}.png')
        self.state['conditions'][key] = {
            'type': result.get('type', ''),
            'image_prompt': result.get('image_prompt', ''),
            'generated': img is not None,
            'error': result.get('error', ''),
            'image_path': f'conditions/{key}.png',
            'image_sha256': image_sha256,
            'img_size': list(img.size) if img else [],
            'think_text': result.get('think_text', ''),
            'raw_input': result.get('raw_input', ''),
            'raw_output': result.get('raw_output', ''),
        }
        self._write_state()
        self.log('condition', key, 'PASS' if img else 'FAIL',
                 img_size=f'{img.size[0]}x{img.size[1]}' if img else '',
                 img_type=result.get('type', ''))
        return image_sha256

    def step_done(self, step_index: int, result: dict):
        """Persist one Step image with its honest execution trace."""
        img = result.get('image')
        image_sha256 = None
        retry_round = result.get('retry_round')
        attempt_index = 0 if retry_round is None else int(retry_round)
        attempt_record = None
        if img:
            attempt_path = (
                self.out_dir / 'attempts' / f'step_{step_index}'
                / f'attempt_{attempt_index}.png'
            )
            attempt_sha256 = self._save_png_immutable(img, attempt_path)
            attempt_record = {
                'policy_id': STEP_ATTEMPT_PERSISTENCE_EVENT['policy_id'],
                'attempt_index': attempt_index,
                'round': attempt_index,
                'retry_round': retry_round,
                'image_path': str(attempt_path.relative_to(self.out_dir)),
                'image_sha256': attempt_sha256,
                'image_size': list(img.size),
                'prompt': result.get('raw_input', ''),
                'feedback': result.get('feedback', ''),
                'seed': result.get('image_seed'),
                'semantic_image_prompt': result.get('semantic_image_prompt', ''),
                'model_image_prompt': result.get('model_image_prompt', ''),
            }
            attempts = self.state.get('step_attempts', {}).get(
                str(step_index), [])
            matching = [
                item for item in attempts
                if item.get('attempt_index') == attempt_index
            ]
            if len(matching) > 1 or (matching and matching[0] != attempt_record):
                raise ValueError(
                    f'conflicting provenance for Step {step_index} '
                    f'attempt {attempt_index}')
            image_sha256 = attempt_sha256

        content = result.get('content', '')
        semantic_image_prompt = result.get('semantic_image_prompt', '')
        model_image_prompt = result.get('model_image_prompt', '')
        img_trace = self._json_safe(result.get('img_trace', {}))
        step_record = {
            'generated': img is not None,
            'content': content,
            'model_image_prompt': model_image_prompt,
            'semantic_image_prompt': semantic_image_prompt,
            'condition_refs': result.get('condition_refs', []),
            # Execution-side split of the declared references. Backends that
            # consume nothing (or only a single source) must not be recorded as
            # having received everything, including after a resume.
            'condition_refs_passed': result.get(
                'condition_refs_passed', result.get('condition_refs', [])),
            'condition_refs_used': result.get(
                'condition_refs_used', result.get('condition_refs', [])),
            'condition_refs_ignored': result.get(
                'condition_refs_ignored', []),
            'image_seed': result.get('image_seed'),
            'retry_round': retry_round,
            'selected_attempt': attempt_index,
            'image_path': f'steps/step_{step_index}.png',
            'image_sha256': image_sha256,
            'img_size': list(img.size) if img else [],
            'img_trace': img_trace,
        }
        if attempt_record is not None:
            sidecar_path = attempt_path.with_suffix('.json')
            self._write_json_immutable({
                'schema_version': 'umm-step-attempt-v1',
                'policy_id': STEP_ATTEMPT_PERSISTENCE_EVENT['policy_id'],
                'step_index': int(step_index),
                'attempt': attempt_record,
                'selected_step': step_record,
            }, sidecar_path)
        self.state['steps'][str(step_index)] = step_record
        if attempt_record is not None:
            attempts = self.state.setdefault('step_attempts', {}).setdefault(
                str(step_index), [])
            if not matching:
                attempts.append(attempt_record)
                attempts.sort(key=lambda item: item['attempt_index'])
        self._write_state()
        if attempt_record is not None:
            self._copy_file_atomic(
                attempt_path,
                self.out_dir / 'steps' / f'step_{step_index}.png',
                expected_sha256=image_sha256,
            )
        cond_refs = result.get('condition_refs', [])
        self.log('step', str(step_index), 'PASS' if img else 'FAIL',
                 img_size=f'{img.size[0]}x{img.size[1]}' if img else '',
                 semantic_prompt_len=len(semantic_image_prompt),
                 model_prompt_len=len(model_image_prompt),
                 content_len=len(content),
                 gen_img_dur=img_trace.get('duration', 0),
                 cond_refs=','.join(cond_refs) if cond_refs else 'none')
        return image_sha256

    def review_done(self, review):
        """Persist one canonical UMM review round."""
        self.state.setdefault('reviews', []).append(self._json_safe(review))
        self._write_state()
        self.log(
            'review', f'round_{review.get("round", 0)}', 'done',
            approved=str(review.get('approved', False)),
            retry_steps=','.join(
                str(index) for index in review.get('retry_step_indexes', []))
                or 'none')

    def mark_done(self):
        with self._state_lock:
            self.state['status'] = 'done'
            self.state['finished_at'] = time.time()
            self.state['heartbeat_at'] = self.state['finished_at']
            self.state['active_operations'] = {}
            self._write_state()
        elapsed = self.state['finished_at'] - self.state['started_at']
        self.log('run', 'done', 'completed', elapsed_sec=elapsed)

    def mark_error(self, error: str):
        with self._state_lock:
            self.state['status'] = 'error'
            self.state['error'] = error
            self.state['finished_at'] = time.time()
            self.state['heartbeat_at'] = self.state['finished_at']
            self.state['active_operations'] = {}
            self._write_state()
        self.log(
            'run', 'error', error,
            elapsed_sec=self.state['finished_at'] - self.state['started_at'])
