from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from training.config import CONFIG, PipelineConfig
from training.utils import (
    ensure_directory,
    get_logger,
    load_pickle,
    log_summary,
    save_json,
    save_pickle,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)

GruAction = tuple[str, int, int, int]
EncodedItem = tuple[int, int, int]
EncodedSession = tuple[list[EncodedItem], int]


@dataclass(frozen=True)
class GruDataArtifacts:
    gru_sequences: list[EncodedSession]
    pid2idx: dict[int, int]
    idx2pid: dict[int, int]
    metadata: dict[str, Any]


def deduplicate_consecutive_actions(
    actions: list[GruAction],
) -> list[GruAction]:
    """Remove consecutive views of the same product."""
    if not actions:
        return []

    deduplicated = [actions[0]]
    for action in actions[1:]:
        if action[1] != deduplicated[-1][1]:
            deduplicated.append(action)
    return deduplicated


def extract_gru_actions(
    merged: pd.DataFrame,
    slug_map: dict[str, int],
    pid_to_tier: dict[int, int],
    pid_to_cat_idx: dict[int, int],
) -> pd.DataFrame:
    """Extract product, price-tier, and category sequences per visit."""
    relevant = merged[merged["page_type"] == "PRODUCT"].copy()
    relevant = relevant.dropna(subset=["visit_id", "slug"])
    relevant["product_id"] = relevant["slug"].map(slug_map)
    relevant = relevant.dropna(subset=["product_id"])

    rows = []
    for visit_id, group in relevant.groupby("visit_id"):
        ordered = group.sort_values("date_time").reset_index(drop=True)
        actions = [
            (
                "product",
                int(product_id),
                int(pid_to_tier.get(int(product_id), 0)),
                int(pid_to_cat_idx.get(int(product_id), 0)),
            )
            for product_id in ordered["product_id"].tolist()
        ]
        actions = deduplicate_consecutive_actions(actions)

        if actions:
            rows.append(
                {
                    "session_id": str(visit_id),
                    "user_actions": actions,
                    "timestamp": ordered["date_time"].iloc[0],
                }
            )

    return pd.DataFrame(
        rows,
        columns=["session_id", "user_actions", "timestamp"],
    )


def encode_sequence(
    actions: list[GruAction],
    pid2idx: dict[int, int],
) -> list[EncodedItem]:
    encoded = []
    for _, product_id, tier_index, category_index in actions:
        token_id = pid2idx.get(product_id, 0)
        if token_id > 0:
            encoded.append(
                (token_id, int(tier_index), int(category_index))
            )
    return encoded


def _feature_coverage(actions: pd.DataFrame) -> tuple[float, float]:
    total = 0
    tier_nonzero = 0
    category_nonzero = 0

    for session_actions in actions["user_actions"]:
        for _, _, tier_index, category_index in session_actions:
            total += 1
            tier_nonzero += int(tier_index > 0)
            category_nonzero += int(category_index > 0)

    if total == 0:
        return 0.0, 0.0
    return tier_nonzero / total, category_nonzero / total


def build_gru_data_artifacts(
    config: PipelineConfig,
) -> GruDataArtifacts:
    intermediate_dir = config.paths.intermediate_dir
    slug_map = load_pickle(intermediate_dir / "slug_map.pkl")
    pid_to_tier = load_pickle(intermediate_dir / "pid_to_tier.pkl")
    pid_to_cat_idx = load_pickle(
        intermediate_dir / "pid_to_cat_idx.pkl"
    )
    cat2idx = load_pickle(intermediate_dir / "cat2idx.pkl")
    train_merged = pd.read_parquet(
        intermediate_dir / "train_merged.parquet"
    )
    test_merged = pd.read_parquet(
        intermediate_dir / "test_merged.parquet"
    )

    train_actions = extract_gru_actions(
        train_merged,
        slug_map,
        pid_to_tier,
        pid_to_cat_idx,
    )
    test_actions = extract_gru_actions(
        test_merged,
        slug_map,
        pid_to_tier,
        pid_to_cat_idx,
    )
    all_actions = pd.concat(
        [train_actions, test_actions],
        ignore_index=True,
    )
    all_actions["product_count"] = all_actions["user_actions"].apply(len)
    training_actions = all_actions[
        all_actions["product_count"]
        >= config.gru_data.minimum_session_products
    ].copy()

    all_products = {
        product_id
        for actions in training_actions["user_actions"]
        for _, product_id, _, _ in actions
    }
    pid2idx = {
        product_id: index + 1
        for index, product_id in enumerate(sorted(all_products))
    }
    idx2pid = {index: product_id for product_id, index in pid2idx.items()}

    max_date = pd.Timestamp(train_merged["visit_start"].max())
    training_actions["timestamp"] = pd.to_datetime(
        training_actions["timestamp"]
    )

    gru_sequences: list[EncodedSession] = []
    for row in training_actions.itertuples(index=False):
        encoded = encode_sequence(row.user_actions, pid2idx)
        if len(encoded) >= config.gru_data.minimum_session_products:
            days_old = int((max_date - row.timestamp).days)
            gru_sequences.append((encoded, days_old))

    lengths = [len(sequence) for sequence, _ in gru_sequences]
    ages = [days_old for _, days_old in gru_sequences]
    train_tier_coverage, train_category_coverage = _feature_coverage(
        train_actions
    )
    test_tier_coverage, test_category_coverage = _feature_coverage(
        test_actions
    )

    metadata = {
        "num_items": len(pid2idx),
        "num_categories": len(cat2idx),
        "max_date": max_date.isoformat(),
        "train_sessions": len(train_actions),
        "test_sessions": len(test_actions),
        "training_sessions": len(gru_sequences),
        "minimum_session_products": config.gru_data.minimum_session_products,
        "sequence_length": {
            "mean": float(np.mean(lengths)),
            "median": float(np.median(lengths)),
            "min": int(min(lengths)),
            "max": int(max(lengths)),
        },
        "session_age_days": {
            "mean": float(np.mean(ages)),
            "min": int(min(ages)),
            "max": int(max(ages)),
        },
        "feature_coverage": {
            "train_tier": train_tier_coverage,
            "train_category": train_category_coverage,
            "test_tier": test_tier_coverage,
            "test_category": test_category_coverage,
        },
    }

    log_summary(
        logger,
        "GRU Dataset",
        {
            "Train sessions": f"{len(train_actions):,}",
            "Test sessions": f"{len(test_actions):,}",
            "Eligible sessions": f"{len(gru_sequences):,}",
            "Vocabulary items": f"{len(pid2idx):,}",
            "Categories": f"{len(cat2idx):,}",
            "Mean sequence length": f"{metadata['sequence_length']['mean']:.1f}",
            "Median sequence length": f"{metadata['sequence_length']['median']:.0f}",
            "Maximum sequence length": f"{metadata['sequence_length']['max']:,}",
        },
    )

    return GruDataArtifacts(
        gru_sequences=gru_sequences,
        pid2idx=pid2idx,
        idx2pid=idx2pid,
        metadata=metadata,
    )


def save_gru_data_artifacts(
    artifacts: GruDataArtifacts,
    config: PipelineConfig,
) -> None:
    output_dir = ensure_directory(config.paths.intermediate_dir)
    save_pickle(output_dir / "gru_sequences.pkl", artifacts.gru_sequences)
    save_pickle(output_dir / "pid2idx.pkl", artifacts.pid2idx)
    save_pickle(output_dir / "idx2pid.pkl", artifacts.idx2pid)
    save_json(output_dir / "gru_metadata.json", artifacts.metadata)
    log_summary(
        logger,
        "Output",
        {"Files": "4", "Directory": output_dir},
    )


def prepare_gru_data(
    config: PipelineConfig = CONFIG,
) -> GruDataArtifacts:
    stage_start(logger, "Prepare GRU Data")
    artifacts = build_gru_data_artifacts(config)
    save_gru_data_artifacts(artifacts, config)
    stage_complete(logger, config.paths.intermediate_dir)
    return artifacts


if __name__ == "__main__":
    prepare_gru_data()
