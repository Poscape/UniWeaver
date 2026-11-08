#!/usr/bin/env python3
"""BAGEL text SFT for the canonical UniWeaver JSON-only trajectory contract."""
import argparse
import gc
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import torch
from transformers import set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bagel_model_loader import (
    load_text_training_model,
    resolve_bagel_model_path,
)
from core.serialization import atomic_write_json
from modeling.qwen2 import Qwen2Tokenizer
from sft_train.production import (
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_WEIGHT_DECAY,
    DEFAULT_TRAINING_MODE,
    TRAINING_MODES,
    WARMUP_START_FACTOR,
    apply_training_policy,
    checkpoint_identity,
    configure_lora_model,
    require_training_memory_capacity,
    resolve_policy_overrides,
)
from sft_train.training_data import (
    CE_TRACE_LENGTH,
    add_training_schedule_arguments,
    advance_training_iterator,
    build_text_sft_batch,
    build_training_provenance,
    create_training_dataset,
    initialize_training_run,
    latest_training_checkpoint,
    normalized_ce_loss,
    open_training_logger,
    preflight_training_window,
    prune_checkpoints,
    publish_retention_report,
    resolve_training_data_arguments,
    resolve_training_schedule,
    restore_training_rng,
    stage_loss_means,
    trailing_mean_ce,
    training_rng_state,
    update_stage_loss_totals,
)


def atomic_torch_save(payload, path):
    """Publish a checkpoint only after torch has written it completely."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    temporary.replace(path)


class _SamplePrefetcher:
    """Single-producer background prefetch of dataset samples."""

    _SENTINEL = object()

    def __init__(self, iterator, max_prefetch=8):
        self._queue = queue.Queue(maxsize=max_prefetch)
        self._daemon = threading.Thread(
            target=self._produce, args=(iterator,), daemon=True, name='sft-prefetch')
        self._daemon.start()

    def _produce(self, iterator):
        try:
            while True:
                self._queue.put(next(iterator))
        except StopIteration:
            self._queue.put(self._SENTINEL)
        except BaseException as error:  # noqa: BLE001 - re-raised to consumer
            self._queue.put(error)

    def next(self):
        item = self._queue.get()
        if item is self._SENTINEL:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item


def _gpu_monitor_thread(mon_file, stop_event):
    """Background poller writing nvidia-smi rows to the CSV monitor file."""
    unavailable = ('', 'N/A', '[N/A]')
    while not stop_event.is_set():
        try:
            r = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=utilization.gpu,temperature.gpu,memory.used,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, check=True)
            parts = [
                value.strip() for value
                in r.stdout.strip().splitlines()[0].split(',')]
            if len(parts) != 4:
                raise ValueError(f'unexpected nvidia-smi row: {r.stdout!r}')
            parts = ['' if value in unavailable else value for value in parts]
            memory = f'{float(parts[2]) / 1024:.1f}' if parts[2] else ''
            mon_file.write(
                f"{time.strftime('%H:%M:%S')},{parts[0]},{parts[1]},"
                f"{memory},{parts[3]}\n")
            mon_file.flush()
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            # GPU monitoring is diagnostic and does not alter the training step.
            pass
        stop_event.wait(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--model-path')
    add_training_schedule_arguments(parser)
    parser.add_argument(
        '--training-mode', choices=TRAINING_MODES, default=DEFAULT_TRAINING_MODE,
        help=(
            'LoRA over the complete text/understanding path; the frozen '
            'generation-only MoT experts stay byte-identical.'))
    parser.add_argument(
        '--config',
        help='path to a per-run training config JSON under sft_train/configs/; '
             'overrides whitelisted policy values (lr/warmup/grad_norm/mode/'
             'epochs/max_seq/pack/checkpoint_every/keep_checkpoints/'
             'keep_checkpoint_steps/keep_best_ce_steps) and becomes the run record')
    parser.add_argument(
        '--keep-checkpoint-steps', type=int, nargs='+', metavar='STEP',
        help=(
            'step numbers that must survive pruning no matter where they sit, '
            'e.g. a mid-run comparison point the rolling window would delete; '
            'overrides the config value'))
    parser.add_argument(
        '--keep-best-ce-steps', type=int, metavar='N',
        help=(
            'in addition to the newest and pinned checkpoints, retain the N '
            'with the lowest trailing training CE; overrides the config value'))
    parser.add_argument(
        '--output-dir',
        default=str(Path(__file__).resolve().parents[1]
                    / 'output_run' / 'runs' / 'training' / 'sft_real'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    overrides = resolve_policy_overrides(
        args.config,
        keep_checkpoint_steps=args.keep_checkpoint_steps,
        keep_best_ce_steps=args.keep_best_ce_steps)

    apply_training_policy(args, overrides)
    resolve_training_schedule(args)
    resolve_training_data_arguments(args)
    model_path = resolve_bagel_model_path(args.model_path)
    args.model_path = model_path

    # Complete data/token/model provenance preflight before creating the run.
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    preflight_training_window(args, tokenizer)
    provenance = build_training_provenance(args, torch)
    args.training_provenance_sha256 = provenance['identity_sha256']
    args.base_language_model_identity_sha256 = (
        provenance['base_language_model_identity']['sha256'])
    initialize_training_run(args, provenance, (
        'data_dir', 'lr', 'warmup_steps', 'total_steps', 'epochs',
        'trajectory_count', 'heldout_overlap_count', 'steps_per_epoch',
        'effective_epochs', 'total_samples', 'training_manifest_sha256',
        'max_seq_len',
        'pack_samples', 'training_mode', 'seed', 'max_grad_norm', 'model_path',
        'optimization_policy', 'training_provenance_sha256',
        'base_language_model_identity_sha256', 'holdout_set_version',
        'holdout_manifest_sha256'))

    set_seed(args.seed)

    # ── Log file (opened before any diagnostic so the whole run, including
    #    model-load/memory preflight, lands in train.log) ──
    log_file, log = open_training_logger(args.output_dir)

    # ── Dataset ──
    ds = create_training_dataset(args, tokenizer)
    ds_iter = iter(ds)

    # ── Model ──
    log("Creating model (LLM only, no VAE/ViT)...")
    t0 = time.time()
    bundle = load_text_training_model(model_path)
    model = bundle.model
    log(f"  Loaded in {time.time() - t0:.1f}s ({sum(p.numel() for p in model.parameters())/1e9:.2f}B total)")
    log(f"  GPU mem: {torch.cuda.memory_allocated()/1024**3:.1f} GiB")

    # Select only the text/understanding path. Image-generation MoT experts are
    # never supervised by this corpus and must remain byte-identical.
    model, parameter_policy = configure_lora_model(
        model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    decoder_layer_count = parameter_policy['decoder_layer_count']
    n_layers_train = parameter_policy['trainable_decoder_layers']

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    mode_desc = (
        f"[LoRA over {n_layers_train} text-path layers; "
        f"wrapped={parameter_policy['wrapped_count']} projections]")
    log(f"  Trainable: {trainable/1e9:.2f}B / {total/1e9:.2f}B ({trainable/total*100:.2f}%) "
        f"{mode_desc}")
    memory_capacity = require_training_memory_capacity(model, torch)
    log(
        '  Memory preflight: '
        f"free={memory_capacity['free_gib']:.1f} GiB, "
        f"optimizer/grad>={memory_capacity['minimum_incremental_gib']:.1f} "
        f"GiB, reserve={memory_capacity['reserve_gib']:.1f} GiB")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, betas=ADAMW_BETAS,
        eps=ADAMW_EPS, weight_decay=ADAMW_WEIGHT_DECAY)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer, start_factor=WARMUP_START_FACTOR,
                total_iters=args.warmup_steps),
            CosineAnnealingLR(optimizer, T_max=args.total_steps - args.warmup_steps),
        ],
        milestones=[args.warmup_steps])

    # ── Resume ──
    start_step = 0
    resume_loss = None
    stage_loss_totals = {}
    # Trailing-mean CE per checkpointed step; it must outlive pruning, so it
    # travels inside the checkpoint and is restored on resume.
    trailing_ce_by_step = {}
    ce_trace = deque(maxlen=CE_TRACE_LENGTH)
    prior_elapsed_seconds = 0.0
    prior_tokens_processed = 0
    prior_peak_mem_gib = 0.0
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')
    if args.resume:
        latest = latest_training_checkpoint(ckpt_dir)
        if latest is None:
            if (Path(args.output_dir) / 'model.pt').exists():
                raise ValueError('training run is already complete; do not resume it')
            log('  Run identity exists but no checkpoint was published; '
                'restarting optimization from step 0')
        else:
            # Plain load, not mmap: load_state_dict copies every tensor to the
            # device, so a surviving mmap view would pin the whole file in RAM.
            ckpt = torch.load(latest, map_location='cpu', weights_only=True)
            if not isinstance(ckpt, dict):
                raise TypeError(f'invalid text-SFT resume checkpoint: {latest}')
            expected_resume = checkpoint_identity(args, 'text_sft_resume')
            mismatches = [
                f'{key}: checkpoint={ckpt.get(key)!r}, expected={value!r}'
                for key, value in expected_resume.items()
                if ckpt.get(key) != value
            ]
            if mismatches:
                raise ValueError(
                    'text-SFT resume checkpoint is incompatible: '
                    + '; '.join(mismatches))
            model.load_state_dict(ckpt['model_state_dict'], strict=True)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_step = ckpt['step']
            resume_loss = ckpt.get('loss')
            if not isinstance(resume_loss, (int, float)):
                raise ValueError('resume checkpoint has no numeric loss')
            samples_consumed = ckpt.get('samples_consumed')
            if not isinstance(samples_consumed, int) or not (
                    0 <= samples_consumed <= args.total_samples):
                raise ValueError('resume checkpoint has invalid samples_consumed')
            stage_loss_totals = ckpt.get('stage_loss_totals', {})
            restored_ce = ckpt.get('trailing_ce_by_step')
            if isinstance(restored_ce, dict):
                trailing_ce_by_step = {
                    int(saved_step): float(value)
                    for saved_step, value in restored_ce.items()
                }
            prior_elapsed_seconds = ckpt.get('cumulative_elapsed_seconds')
            prior_tokens_processed = ckpt.get('cumulative_tokens_processed')
            prior_peak_mem_gib = ckpt.get('peak_mem_gib')
            if (not isinstance(prior_elapsed_seconds, (int, float))
                    or prior_elapsed_seconds < 0):
                raise ValueError(
                    'resume checkpoint has invalid cumulative_elapsed_seconds')
            if (not isinstance(prior_tokens_processed, int)
                    or prior_tokens_processed < 0):
                raise ValueError(
                    'resume checkpoint has invalid cumulative_tokens_processed')
            if (not isinstance(prior_peak_mem_gib, (int, float))
                    or prior_peak_mem_gib < 0):
                raise ValueError('resume checkpoint has invalid peak_mem_gib')
            advance_training_iterator(ds_iter, samples_consumed)
            restore_training_rng(torch, ckpt)
            log(f"  Resumed from step {start_step} ({latest.name})")
            # Release the host-side checkpoint copy. Every model and optimizer
            # tensor has already been copied onto the training device.
            del ckpt
            gc.collect()

    log(f"\n{'─'*65}")
    log(f"Training: {args.total_steps} steps | lr={args.lr} | mode={args.training_mode}")
    log(f"Schedule: {args.effective_epochs:.3f} epochs | {args.steps_per_epoch} steps/epoch")
    log(f"Data: {args.data_dir} | pack={args.pack_samples} | max_seq_len={args.max_seq_len}")
    log(
        f"Trajectories: {args.trajectory_count} train | "
        f"{args.heldout_overlap_count} UniWeaver-Bench benchmark overlaps held out")
    log("Loss weighting: uniform over every response token")
    log(f"{'─'*72}")
    log(f"{'Step':>6} | {'Train CE':>8} | {'Step/s':>7} | {'Tok/s':>9} | {'Mem(GiB)':>9} | {'LR':>10}")
    log(f"{'─'*72}")

    # ── GPU monitor (background poll) ──
    # A slow nvidia-smi probe runs on a thread so it never stalls a training step.
    mon_file = open(os.path.join(args.output_dir, 'gpu_monitor.csv'), 'a')  # noqa: SIM115
    if os.path.getsize(os.path.join(args.output_dir, 'gpu_monitor.csv')) == 0:
        mon_file.write("time,gpu_util_pct,gpu_temp_c,mem_gib,power_w\n")
    # Loss trace as data: the training curve backs the best-CE selection, and
    # progress is readable without parsing train.log.
    trace_file = open(  # noqa: SIM115
        os.path.join(args.output_dir, 'loss_trace.csv'), 'a')
    if os.path.getsize(os.path.join(args.output_dir, 'loss_trace.csv')) == 0:
        trace_file.write(
            "step,train_ce,plan1,plan2,plan3,plan4,mem_gib,lr,"
            "elapsed_s,tokens_per_s\n")
    mon_stop = threading.Event()
    mon_thread = threading.Thread(
        target=_gpu_monitor_thread, args=(mon_file, mon_stop), daemon=True,
        name='gpu-monitor')
    mon_thread.start()

    # ── Training ──
    model.train()
    step = start_step
    samples_consumed = 0 if start_step == 0 else samples_consumed
    peak_mem = float(prior_peak_mem_gib)
    t_start = time.time()
    session_tokens = 0
    loss = None

    # Prefetch samples on a thread to overlap I/O and tokenization with GPU
    # steps; created after any resume advance to keep the dataset order exact.
    use_prefetch = os.environ.get('BAGEL_NO_PREFETCH', '') != '1'
    if use_prefetch:
        prefetcher = _SamplePrefetcher(ds_iter)
    else:
        prefetcher = None

    while samples_consumed < args.total_samples:
        # Collect pack_n samples
        batch_ids, batch_pfx, batch_stages = [], [], []
        batch_tokens = 0
        pack_count = min(
            args.pack_samples, args.total_samples - samples_consumed)
        for _ in range(pack_count):
            try:
                ids, pfx, stage = (
                    prefetcher.next() if prefetcher is not None
                    else next(ds_iter))
            except StopIteration:
                break
            batch_ids.append(ids)
            batch_pfx.append(pfx)
            batch_stages.append(stage)
            batch_tokens += len(ids)
        if not batch_ids:
            break

        data = build_text_sft_batch(batch_ids, batch_pfx, device='cuda')

        t_step = time.time()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss_dict = model(**data)

        ce = loss_dict["ce"]
        loss = normalized_ce_loss(ce)
        update_stage_loss_totals(
            stage_loss_totals, ce, batch_stages,
            [len(ids) - pfx for ids, pfx in zip(batch_ids, batch_pfx)])

        loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024**3)
        session_tokens += batch_tokens
        step += 1
        samples_consumed += len(batch_ids)
        dt = time.time() - t_step

        # Log
        if step % args.log_every == 0 or step <= 3:
            ce_trace.append((step, float(loss.item())))
            mem = torch.cuda.max_memory_allocated() / 1024**3
            tok_s = batch_tokens / max(dt, 1e-6)
            means = stage_loss_means(stage_loss_totals)
            trace_file.write(
                f"{step},{loss.item():.6f},"
                + ",".join(f"{means.get(stage, float('nan')):.6f}"
                           for stage in ('plan1', 'plan2', 'plan3', 'plan4'))
                + f",{mem:.1f},{scheduler.get_last_lr()[0]:.6e},"
                f"{prior_elapsed_seconds + time.time() - t_start:.1f},"
                f"{tok_s:.0f}\n")
            trace_file.flush()
            log(f"{step:6d} | {loss.item():8.3f} | {1/dt:5.2f} | {tok_s:9.0f} | {mem:9.1f} | {scheduler.get_last_lr()[0]:.2e}")
            log("         stage CE | " + " | ".join(
                f'{stage}={means.get(stage, float("nan")):.3f}'
                for stage in ('plan1', 'plan2', 'plan3', 'plan4')))

        # Checkpoint
        if step % args.ckpt_every == 0 and step > start_step:
            os.makedirs(ckpt_dir, exist_ok=True)
            p = os.path.join(ckpt_dir, f'step_{step}.pt')
            # Record this step's trailing CE before building the payload, so a
            # resumed run ranks this checkpoint from its own value rather than
            # the previous one.
            trailing_mean = trailing_mean_ce(ce_trace, step)
            if trailing_mean is not None:
                trailing_ce_by_step[step] = trailing_mean
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'samples_consumed': samples_consumed,
                'stage_loss_totals': stage_loss_totals,
                'cumulative_elapsed_seconds': (
                    prior_elapsed_seconds + time.time() - t_start),
                'cumulative_tokens_processed': (
                    prior_tokens_processed + session_tokens),
                'peak_mem_gib': peak_mem,
                'trailing_ce_by_step': dict(trailing_ce_by_step),
            }
            checkpoint.update(checkpoint_identity(args, 'text_sft_resume'))
            checkpoint.update(training_rng_state(torch))
            atomic_torch_save(checkpoint, p)
            retention = prune_checkpoints(
                ckpt_dir, keep=args.keep_checkpoints,
                pinned=args.keep_checkpoint_steps,
                best_ce_steps=args.keep_best_ce_steps,
                ce_history=trailing_ce_by_step, log=log)
            # Persist the retention decision so the launcher knows which points
            # exist; overwritten every time to describe the current on-disk set.
            publish_retention_report(
                ckpt_dir, step=step, keep=args.keep_checkpoints,
                pinned=args.keep_checkpoint_steps,
                best_ce_steps=args.keep_best_ce_steps,
                ce_history=trailing_ce_by_step, report=retention)
            log(f"  -> CKPT saved: {p}")

    # -- Final --
    if loss is None and resume_loss is None:
        raise RuntimeError(
            f'no optimization step ran (start_step={start_step}, '
            f'total_steps={args.total_steps})')
    final_loss = float(loss.item()) if loss is not None else float(resume_loss)
    session_time = time.time() - t_start
    cumulative_time = prior_elapsed_seconds + session_time
    cumulative_tokens = prior_tokens_processed + session_tokens

    summary = {
        'total_steps': step,
        'samples_consumed': samples_consumed,
        'start_step': start_step,
        'cumulative_time_s': round(cumulative_time, 1),
        'session_time_s': round(session_time, 1),
        'avg_step_time_s': round(cumulative_time / max(step, 1), 2),
        'peak_mem_gib': round(peak_mem, 1),
        'cumulative_tokens_processed': cumulative_tokens,
        'session_tokens_processed': session_tokens,
        'avg_tokens_per_sec': round(
            cumulative_tokens / max(cumulative_time, 0.001)),
        'final_loss': round(final_loss, 4),
        'trainable_params_b': round(trainable / 1e9, 2),
        'total_params_b': round(total / 1e9, 2),
        'training_mode': args.training_mode,
        'optimization_policy': args.optimization_policy,
        'training_manifest_sha256': args.training_manifest_sha256,
        'training_provenance_sha256': args.training_provenance_sha256,
        'base_language_model_identity_sha256': (
            args.base_language_model_identity_sha256),
        'holdout_set_version': args.holdout_set_version,
        'holdout_manifest_sha256': args.holdout_manifest_sha256,
        'decoder_layer_count': decoder_layer_count,
        'trainable_decoder_layers': n_layers_train,
        'single_device_memory_preflight': memory_capacity,
        'lr': args.lr,
        'max_seq_len': args.max_seq_len,
        'pack_samples': args.pack_samples,
        'trajectory_count': args.trajectory_count,
        'heldout_overlap_count': args.heldout_overlap_count,
        'steps_per_epoch': args.steps_per_epoch,
        'effective_epochs': args.effective_epochs,
        'cumulative_training_stage_ce': stage_loss_means(stage_loss_totals),
        'data_dir': args.data_dir,
    }
    summary_path = Path(args.output_dir) / 'summary.json'
    atomic_write_json(summary_path, summary)

    log(f"\n{'='*55}")
    log("TRAINING COMPLETE")
    log(f"{'='*55}")
    for k, v in summary.items():
        log(f"  {k}: {v}")

    # Save a self-identifying checkpoint. The LoRA delta is folded back into
    # the base weights first so model.pt remains a plain Qwen2 state_dict
    # compatible with the SFT overlay contract.
    log('  Merging LoRA deltas into base weights for final model.pt '
        '(PEFT merge_and_unload)')
    from peft import PeftModel
    if not isinstance(model, PeftModel):
        raise TypeError('text SFT expects a PEFT-wrapped model')
    model = model.merge_and_unload()
    log('  LoRA unloaded; final model is a bare text model')
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'step': step,
        'cumulative_elapsed_seconds': cumulative_time,
        'cumulative_tokens_processed': cumulative_tokens,
        'peak_mem_gib': peak_mem,
    }
    final_checkpoint.update(checkpoint_identity(args, 'text_sft_final'))
    atomic_torch_save(final_checkpoint, Path(args.output_dir) / 'model.pt')
    log(f"\n  Model: {args.output_dir}/model.pt (step {step})")

    log(f"  Logs: {args.output_dir}/train.log")
    log(f"  GPU monitor: {args.output_dir}/gpu_monitor.csv")
    log(f"  Loss trace: {args.output_dir}/loss_trace.csv")

    mon_stop.set()
    mon_thread.join(timeout=10)
    mon_file.close()
    trace_file.close()
    log_file.close()


if __name__ == '__main__':
    main()
