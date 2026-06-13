from collections.abc import Iterable

import torch
import torch.nn as nn

from training.config import PAD_IDX


class GRURecDual(nn.Module):
    """Dual-path GRU over product and category sequences."""

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        item_emb_dim: int = 128,
        tier_emb_dim: int = 4,
        cat_emb_dim: int = 8,
        hidden_dim: int = 128,
        cat_hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.2,
        use_tier: bool = True,
    ) -> None:
        super().__init__()
        self.num_items = num_items
        self.num_categories = num_categories
        self.use_tier = use_tier

        self.item_emb = nn.Embedding(
            num_items + 1,
            item_emb_dim,
            padding_idx=PAD_IDX,
        )
        nn.init.xavier_uniform_(self.item_emb.weight.data)

        if self.use_tier:
            self.tier_emb = nn.Embedding(
                6,
                tier_emb_dim,
                padding_idx=PAD_IDX,
            )

        self.cat_emb = nn.Embedding(
            num_categories + 1,
            cat_emb_dim,
            padding_idx=PAD_IDX,
        )

        item_input_dim = item_emb_dim + (
            tier_emb_dim if self.use_tier else 0
        )
        self.item_proj = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.item_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.cat_proj = nn.Sequential(
            nn.Linear(cat_emb_dim, cat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cat_gru = nn.GRU(
            input_size=cat_hidden_dim,
            hidden_size=cat_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.out = nn.Linear(
            hidden_dim + cat_hidden_dim,
            num_items + 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        tier: torch.Tensor | None = None,
        cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        item_embedding = self.item_emb(x)
        item_features = [item_embedding]

        if self.use_tier:
            if tier is None:
                tier = torch.zeros_like(x)
            item_features.append(self.tier_emb(tier))

        item_hidden = self.item_proj(torch.cat(item_features, dim=-1))
        item_hidden, _ = self.item_gru(item_hidden)

        if cat is None:
            cat = torch.zeros_like(x)
        category_hidden = self.cat_proj(self.cat_emb(cat))
        category_hidden, _ = self.cat_gru(category_hidden)

        fused = torch.cat([item_hidden, category_hidden], dim=-1)
        return self.out(fused)

    @torch.no_grad()
    def predict_topk(
        self,
        session_tokens: list[int],
        session_tier: list[int],
        session_cat: list[int],
        k: int = 6,
        device: str | torch.device = "cpu",
        banned: Iterable[int] | None = None,
    ) -> list[int]:
        if not session_tokens:
            return []

        x = torch.tensor(
            session_tokens,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        tier = torch.tensor(
            session_tier,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        cat = torch.tensor(
            session_cat,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        logits = self.forward(x, tier, cat)[0, -1]

        if banned:
            for token_id in banned:
                if 0 < token_id < len(logits):
                    logits[token_id] = -float("inf")

        logits[PAD_IDX] = -float("inf")
        candidate_count = min(
            k + len(session_tokens),
            len(logits),
        )
        top_indices = torch.topk(logits, candidate_count).indices

        predictions = []
        seen = set(session_tokens)
        for token_id in top_indices.tolist():
            if token_id > PAD_IDX and token_id not in seen:
                predictions.append(token_id)
                if len(predictions) >= k:
                    break

        return predictions
