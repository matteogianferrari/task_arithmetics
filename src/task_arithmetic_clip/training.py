"""Independent CLIP vision fine-tuning with resumable optimizer-step training."""

import json
import math
from pathlib import Path
from typing import Iterator

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from task_arithmetic_clip.arithmetic import TaskVector
from task_arithmetic_clip.checkpoints import (
    load_backbone,
    load_payload,
    save_payload,
    snapshot,
    state_identity,
    write_json,
)
from task_arithmetic_clip.config import ExperimentConfig
from task_arithmetic_clip.evaluation import evaluate_dataset, load_task
from task_arithmetic_clip.runtime import (
    autocast_context,
    capture_rng,
    resolve_device,
    restore_rng,
    seed_everything,
    to_cpu,
)

RESUME_KEYS = (
    "seed",
    "batch_size",
    "accumulation_steps",
    "steps",
    "warmup_steps",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "precision",
    "augment",
    "gradient_checkpointing",
)


class InfiniteBatchSampler(Sampler[list[int]]):
    """InfiniteBatchSampler class for deterministic, resumable shuffled training batches.

    Incomplete batches at the end of an epoch are omitted. Progress is counted by
    consumed batches in the training loop, independent of DataLoader prefetching.

    Attributes:
        size: Number of training examples.
        batch_size: Number of examples per microbatch.
        seed: Base seed for independent per-epoch shuffles.
        start_batch: Number of already consumed microbatches.
    """

    def __init__(self, size: int, batch_size: int, seed: int, start_batch: int = 0) -> None:
        """Initialize the training batch sampler.

        Args:
            size: Number of training examples.
            batch_size: Microbatch size.
            seed: Training shuffle seed.
            start_batch: Microbatch cursor loaded from a training checkpoint.
        Returns:
            None.
        """
        if size < batch_size or batch_size < 1 or start_batch < 0:
            raise ValueError("Training needs at least one full batch and a nonnegative cursor.")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.start_batch = start_batch

    def __iter__(self) -> Iterator[list[int]]:
        """Yield a reproducible sequence of full training microbatches.

        Args:
            None.
        Returns:
            An iterator of shuffled lists of training indices.
        """
        batches_per_epoch = self.size // self.batch_size
        epoch, offset = divmod(self.start_batch, batches_per_epoch)
        while True:
            generator = torch.Generator().manual_seed(self.seed + epoch)
            indices = torch.randperm(self.size, generator=generator).tolist()
            for batch in range(offset, batches_per_epoch):
                start = batch * self.batch_size
                yield indices[start : start + self.batch_size]
            epoch += 1
            offset = 0


def learning_rate_factor(step: int, warmup: int, total: int) -> float:
    """Compute linear warm-up followed by cosine learning rate decay.

    Args:
        step: Zero-based index of the next successful optimizer update.
        warmup: Number of warm-up updates.
        total: Total number of optimizer updates.
    Returns:
        A nonnegative multiplier for the peak learning rate.
    """
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def train_task(task: str, config: ExperimentConfig, resume: str | Path | None = None) -> Path:
    """Fine-tune one task from the common base with a frozen CLIP classification head.

    Args:
        task: Classification task to train.
        config: Training, dataset, and checkpoint settings.
        resume: Optional resumable checkpoint for this same task and training configuration.
    Returns:
        The final fine-tuned backbone checkpoint path.
    """
    config.validate()
    device = resolve_device(config.device, config.precision)
    seed_everything(config.seed)
    root = Path(config.output_root)
    base_path = root / "base.pt"
    model, base = load_backbone(base_path)
    if state_identity(base["state_dict"], base["vision_config"]) != base["base_id"]:
        raise ValueError("base.pt must contain the original pretrained initialization.")
    bundle, head = load_task(task, config, base)
    head_id = state_identity(head.state_dict(), {"class_names": bundle.class_names})
    if bundle.train is None:
        raise ValueError(f"{task} has no training partition.")
    task_root = root / task
    task_root.mkdir(parents=True, exist_ok=True)
    history_path = task_root / "history.jsonl"
    if resume is None and (
        (task_root / "last.pt").exists()
        or (task_root / "finetuned.pt").exists()
        or history_path.exists()
    ):
        raise FileExistsError(
            f"{task} already has a training run. Use --resume or a new output_root."
        )
    model.to(device).train()
    head.to(device).eval()
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and config.precision == "fp16"
    )
    step, micro_batches = 0, 0
    if resume is not None:
        saved = load_payload(resume, "backbone")
        metadata = saved["metadata"]
        training = saved.get("training")
        if (
            training is None
            or saved["base_id"] != base["base_id"]
            or metadata.get("task") != task
            or metadata.get("split_id") != bundle.split_id
            or metadata.get("head_id") != head_id
        ):
            raise ValueError("Resume requires a training checkpoint from this same base and task.")
        changed = [
            name for name in RESUME_KEYS if metadata["config"][name] != config.to_dict()[name]
        ]
        if changed:
            raise ValueError(f"Resume training settings changed: {changed}")
        model.load_state_dict(saved["state_dict"], strict=True)
        optimizer.load_state_dict(training["optimizer"])
        scaler.load_state_dict(training["scaler"])
        step = training["step"]
        micro_batches = training["micro_batches"]
        restore_rng(training["rng"])
        del saved
        if history_path.exists():
            rows = [
                json.loads(line) for line in history_path.read_text().splitlines() if line.strip()
            ]
            with history_path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    if row["step"] <= step:
                        stream.write(json.dumps(row) + "\n")
    sampler = InfiniteBatchSampler(len(bundle.train), config.batch_size, config.seed, micro_batches)
    loader = DataLoader(
        bundle.train,
        batch_sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    batches = iter(loader)
    # The template carries architecture and preprocessing; the base weights stay on disk.
    template = {key: value for key, value in base.items() if key != "state_dict"}
    del base
    skipped = 0
    validation = None
    progress = tqdm(total=config.steps, initial=step, desc=task, unit="update")
    try:
        while step < config.steps:
            model.train()
            rate = config.learning_rate * learning_rate_factor(
                step, config.warmup_steps, config.steps
            )
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(config.accumulation_steps):
                images, labels = next(batches)
                micro_batches += 1
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast_context(device, config.precision):
                    logits = head(model(images))
                    loss = F.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Training loss is non-finite. Reduce the learning rate."
                    )
                loss_sum += loss.detach().item()
                scaler.scale(loss / config.accumulation_steps).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise FloatingPointError("Training gradients are non-finite.")
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped += 1
                if skipped >= 50:
                    raise FloatingPointError("Too many fp16 overflow skips. Use bf16 or fp32.")
                continue
            skipped = 0
            step += 1
            row = {
                "step": step,
                "micro_batches": micro_batches,
                "learning_rate": rate,
                "train_loss": loss_sum / config.accumulation_steps,
            }
            if step % config.eval_every == 0 or step == config.steps:
                validation = evaluate_dataset(model, head, bundle.val, config, device)
                row["validation"] = validation
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            progress.update(1)
            progress.set_postfix(loss=f"{row['train_loss']:.4f}")
            if step % config.save_every == 0 or step == config.steps:
                checkpoint = snapshot(
                    model,
                    template,
                    {
                        "task": task,
                        "split_id": bundle.split_id,
                        "config": config.to_dict(),
                        "step": step,
                        "head_id": head_id,
                    },
                )
                checkpoint["training"] = {
                    "step": step,
                    "micro_batches": micro_batches,
                    "optimizer": to_cpu(optimizer.state_dict()),
                    "scaler": scaler.state_dict(),
                    "rng": capture_rng(),
                }
                save_payload(task_root / "last.pt", checkpoint)
                del checkpoint
    finally:
        progress.close()
    if validation is None:
        validation = evaluate_dataset(model, head, bundle.val, config, device)
    final_path = task_root / "finetuned.pt"
    save_payload(
        final_path,
        snapshot(
            model,
            template,
            {
                "task": task,
                "split_id": bundle.split_id,
                "config": config.to_dict(),
                "step": step,
                "head_id": head_id,
            },
        ),
    )
    TaskVector.from_checkpoints(base_path, final_path).save(task_root / "vector.pt")
    write_json(task_root / "validation.json", validation)
    return final_path
