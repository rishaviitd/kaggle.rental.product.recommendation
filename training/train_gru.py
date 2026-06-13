from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from training.config import CONFIG, PAD_IDX, PipelineConfig
from training.model import GRURecDual
from training.utils import (
    ensure_directory,
    get_logger,
    load_json,
    load_pickle,
    log_summary,
    save_json,
    seed_everything,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)


class TemporalSessionDataset(Dataset):
    """Compact GRU sessions paired with their age in days."""

    def __init__(
        self,
        sessions: list[tuple[list[tuple[int, int, int]], int]],
        minimum_length: int,
    ) -> None:
        self.sessions = [
            session
            for session in sessions
            if len(session[0]) >= minimum_length
        ]

    def __len__(self) -> int:
        return len(self.sessions)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[list[tuple[int, int, int]], int]:
        return self.sessions[index]


def collate_temporal_sequences(
    batch: list[tuple[list[tuple[int, int, int]], int]],
    minimum_length: int,
    use_sample_weighting: bool,
    sample_weight_decay_rate: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    valid_batch = [
        session
        for session in batch
        if len(session[0]) >= minimum_length
    ]
    if not valid_batch:
        return (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1,), dtype=torch.float),
        )

    batch_size = len(valid_batch)
    sequence_steps = max(len(sequence) for sequence, _ in valid_batch) - 1

    x = torch.full(
        (batch_size, sequence_steps),
        PAD_IDX,
        dtype=torch.long,
    )
    tier = torch.zeros((batch_size, sequence_steps), dtype=torch.long)
    category = torch.zeros(
        (batch_size, sequence_steps),
        dtype=torch.long,
    )
    target = torch.full(
        (batch_size, sequence_steps),
        PAD_IDX,
        dtype=torch.long,
    )
    sample_weights = torch.ones((batch_size,), dtype=torch.float)

    for batch_index, (sequence, days_old) in enumerate(valid_batch):
        if use_sample_weighting:
            sample_weights[batch_index] = float(
                np.exp(-sample_weight_decay_rate * days_old)
            )

        for step in range(len(sequence) - 1):
            x[batch_index, step] = sequence[step][0]
            tier[batch_index, step] = sequence[step][1]
            category[batch_index, step] = sequence[step][2]
            target[batch_index, step] = sequence[step + 1][0]

    return x, tier, category, target, sample_weights


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    mask = targets != pad_idx
    if mask.sum() == 0:
        return torch.tensor(
            0.0,
            device=logits.device,
            requires_grad=True,
        )

    logits_flat = logits[mask]
    targets_flat = targets[mask]
    batch_size, sequence_steps = mask.shape
    expanded_weights = weights.unsqueeze(1).expand(
        batch_size,
        sequence_steps,
    )
    weights_flat = expanded_weights[mask]
    losses = functional.cross_entropy(
        logits_flat,
        targets_flat,
        reduction="none",
    )
    return (losses * weights_flat).sum() / weights_flat.sum()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable. Run training outside "
            "restricted environments or set training.device explicitly."
        )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(requested_device)


def build_model(
    metadata: dict[str, Any],
    config: PipelineConfig,
) -> GRURecDual:
    model_config = config.model
    return GRURecDual(
        num_items=int(metadata["num_items"]),
        num_categories=int(metadata["num_categories"]),
        item_emb_dim=model_config.item_embedding_dim,
        tier_emb_dim=model_config.tier_embedding_dim,
        cat_emb_dim=model_config.category_embedding_dim,
        hidden_dim=model_config.hidden_dim,
        cat_hidden_dim=model_config.category_hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        use_tier=model_config.use_price_tier,
    )


def train_model(config: PipelineConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    intermediate_dir = ensure_directory(config.paths.intermediate_dir)
    sequences = load_pickle(intermediate_dir / "gru_sequences.pkl")
    metadata = load_json(intermediate_dir / "gru_metadata.json")

    dataset = TemporalSessionDataset(
        sequences,
        config.gru_data.minimum_session_products,
    )
    collate = partial(
        collate_temporal_sequences,
        minimum_length=config.gru_data.minimum_session_products,
        use_sample_weighting=config.training.use_sample_weighting,
        sample_weight_decay_rate=(
            config.training.sample_weight_decay_rate
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    device = resolve_device(config.training.device)
    model = build_model(metadata, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
    )

    model_path = intermediate_dir / "model.pt"
    best_loss = float("inf")
    epoch_losses = []

    log_summary(
        logger,
        "Training Setup",
        {
            "Device": device,
            "Samples": f"{len(dataset):,}",
            "Batches per epoch": f"{len(loader):,}",
            "Parameters": f"{sum(parameter.numel() for parameter in model.parameters()):,}",
            "Epochs": f"{config.training.epochs}",
            "Batch size": f"{config.training.batch_size}",
        },
    )

    for epoch in range(config.training.epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for x, tier, category, target, weights in loader:
            x = x.to(device)
            tier = tier.to(device)
            category = category.to(device)
            target = target.to(device)
            weights = weights.to(device)

            optimizer.zero_grad()
            logits = model(x, tier, category)
            loss = weighted_cross_entropy(logits, target, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip_norm,
            )
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        average_loss = total_loss / max(batch_count, 1)
        scheduler.step(average_loss)
        epoch_losses.append(average_loss)

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(model.state_dict(), model_path)

        logger.info(
            f"  Epoch {epoch + 1:>2}/{config.training.epochs}    "
            f"loss  {average_loss:.4f}    "
            f"best  {best_loss:.4f}    "
            f"lr  {optimizer.param_groups[0]['lr']:.6f}"
        )

    metrics = {
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
        "epochs": config.training.epochs,
        "device": str(device),
        "training_samples": len(dataset),
        "batches_per_epoch": len(loader),
        "parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    save_json(intermediate_dir / "training_metrics.json", metrics)
    log_summary(
        logger,
        "Result",
        {"Best loss": f"{best_loss:.4f}", "Model": model_path},
    )
    return metrics


def train_gru(config: PipelineConfig = CONFIG) -> dict[str, Any]:
    stage_start(logger, "Train GRU")
    metrics = train_model(config)
    stage_complete(logger, config.paths.intermediate_dir / "model.pt")
    return metrics


if __name__ == "__main__":
    train_gru()
