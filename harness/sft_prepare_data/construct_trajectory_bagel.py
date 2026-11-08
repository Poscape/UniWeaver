"""Canonical UniWeaver trajectory transport used by ``baselines/run_umm.py``.

The public entrypoint is ``run_umm.py --trajectory-only --planner bagel``.
"""
import gc
import queue
import sys
import threading
import time
from concurrent.futures import Future
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bagel_model_loader import (
    load_multimodal_model,
)
from bagel_text_backend import BagelTextBackend
from core.contract import PLANNER_SYSTEM
from core.hashing import sha256_file
from sft_prepare_data.trajectory_construct_driver import (
    construct_trajectory,
    run_constructor_cli,
    run_task_specs,
)


class BagelBatchDispatcher:
    """Own the GPU generation loop and pack concurrently submitted requests."""

    def __init__(self, model, tokenizer, batch_window=0.08, max_batch_size=2):
        self.model = model
        self.tokenizer = tokenizer
        self.text_backend = BagelTextBackend(model, tokenizer)
        self.batch_window = batch_window
        self.max_batch_size = max(1, max_batch_size)
        self.requests = queue.Queue()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def send(self, user_text, system, max_tokens, do_sample, temperature):
        future = Future()
        signature = (system, max_tokens, do_sample, temperature)
        self.requests.put((signature, user_text, future))
        return future.result()

    def close(self):
        self.requests.put(None)
        self.worker.join()

    def _run(self):
        pending = []
        while True:
            item = pending.pop(0) if pending else self.requests.get()
            if item is None:
                return
            signature = item[0]
            batch = [item]
            deadline = time.monotonic() + self.batch_window
            while len(batch) < self.max_batch_size and time.monotonic() < deadline:
                try:
                    candidate = self.requests.get(
                        timeout=max(0.0, deadline - time.monotonic()))
                except queue.Empty:
                    break
                if candidate is None:
                    pending.append(candidate)
                    break
                if candidate[0] == signature:
                    batch.append(candidate)
                else:
                    pending.append(candidate)
            try:
                system, max_tokens, do_sample, temperature = signature
                print(f'    [BATCH] size={len(batch)} max_tokens={max_tokens}',
                      flush=True)
                outputs = self.text_backend.send_many(
                    system,
                    [request[1] for request in batch],
                    do_sample=do_sample, temperature=temperature,
                    max_tokens=max_tokens,
                )
                for request, output in zip(batch, outputs):
                    request[2].set_result(output)
            except Exception as error:  # noqa: BLE001 - dispatcher boundary
                for request in batch:
                    request[2].set_exception(error)
            finally:
                # send_many owns a full per-request KV cache. Release the completed
                # batch before the dispatcher admits the next one on a 40GB GPU.
                if 'outputs' in locals():
                    del outputs
                gc.collect()
                try:
                    import torch
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except (ImportError, RuntimeError) as error:
                    print(
                        f'    [BATCH] CUDA cleanup unavailable: {error}',
                        flush=True)


def _run_one_impl(model, tokenizer, task_key, task_prompt, output_path, extra_fields=None,
                  dispatcher=None):
    """Run the canonical UniWeaver trajectory with BAGEL as the transport backend."""
    text_backend = BagelTextBackend(model, tokenizer)

    def _send(user_text, label='', max_tokens=2048,
              system_text=PLANNER_SYSTEM, do_sample=True, temperature=0.7):
        start_time = time.time()
        if dispatcher is not None:
            response = dispatcher.send(
                user_text, system_text, max_tokens, do_sample, temperature)
        else:
            response = text_backend.send(
                system_text, user_text, max_tokens=max_tokens,
                do_sample=do_sample, temperature=temperature)
        duration = time.time() - start_time
        _send.last_metadata = {
            'provider': 'bagel',
            'model_name': (extra_fields or {}).get(
                'model_name', type(model).__name__),
            'model_artifact_sha256': (extra_fields or {}).get(
                'model_artifact_sha256'),
            'sft_checkpoint_sha256': (extra_fields or {}).get(
                'sft_checkpoint_sha256'),
            'duration_seconds': duration,
            'input_tokens': text_backend.input_token_count(
                system_text, user_text),
            'output_tokens': text_backend.output_token_count(response),
            'reasoning_tokens': 0,
            'token_count_scope': (
                'runtime_prefix_including_injected_start;returned_content'),
        }
        print(
            f'    [{duration:.1f}s] {label}: '
            f'{len(response)} chars', flush=True)
        return response

    trajectory = construct_trajectory(
        task_key, task_prompt, output_path, _send,
        extra_fields={
            **(extra_fields or {}),
            'model_provider': 'bagel',
            'model_name': (extra_fields or {}).get(
                'model_name', type(model).__name__),
        },
        include_step_generation=False)
    return trajectory


def _run_loaded_specs(specs, sft_ckpt=None, base_only=False, parallel=1,
                      bagel_mode=1, offload_folder='offload', model_path=None):
    """Load BAGEL once and execute serial or packed concurrent trajectories."""
    import torch

    bundle = load_multimodal_model(
        model_path=model_path,
        mode=bagel_mode,
        offload_folder=offload_folder,
        sft_checkpoint=None if base_only else sft_ckpt,
    )
    model = bundle.model
    tokenizer = bundle.tokenizer
    model_artifact_sha256 = bundle.model_identity['sha256']
    sft_checkpoint_sha256 = None
    if not base_only:
        sft_checkpoint_sha256 = sha256_file(sft_ckpt)
    worker_count = max(1, min(parallel, len(specs)))
    dispatcher = (
        BagelBatchDispatcher(model, tokenizer) if worker_count > 1 else None)
    def run_one(key, prompt, path):
        model_label = (
            'bagel-base' if base_only
            else f'bagel-sft:{Path(sft_ckpt).name}')
        return _run_one_impl(
            model, tokenizer, key, prompt, path,
            extra_fields={
                'model_provider': 'bagel',
                'model_name': model_label,
                'model_artifact_sha256': model_artifact_sha256,
                **({'sft_checkpoint_sha256': sft_checkpoint_sha256}
                   if sft_checkpoint_sha256 else {}),
                **({'sft_checkpoint_training_identity': bundle.sft_metadata}
                   if bundle.sft_metadata else {}),
            },
            dispatcher=dispatcher)

    try:
        return run_task_specs(specs, run_one, worker_count)
    finally:
        if dispatcher is not None:
            dispatcher.close()
        gc.collect()
        torch.cuda.empty_cache()


def run(args, parser):
    base_only = args.base_only or not args.sft_checkpoint
    label = 'bagel_base' if base_only else 'bagel_sft'
    if not base_only and not Path(args.sft_checkpoint).is_file():
        parser.error(f'SFT checkpoint does not exist: {args.sft_checkpoint}')

    def execute_specs(specs):
        return _run_loaded_specs(
            specs, sft_ckpt=args.sft_checkpoint, base_only=base_only,
            parallel=args.parallel,
            bagel_mode=getattr(args, 'bagel_mode', 1),
            offload_folder=getattr(args, 'offload_folder', 'offload'),
            model_path=args.model_path)

    run_constructor_cli(args, parser, label, execute_specs)
