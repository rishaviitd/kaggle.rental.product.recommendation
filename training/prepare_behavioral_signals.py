from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from training.config import CONFIG, PipelineConfig
from training.prepare_sessions import extract_product_sessions
from training.utils import (
    ensure_directory,
    get_logger,
    load_pickle,
    log_summary,
    save_pickle,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class BehavioralArtifacts:
    coocc: dict[str, Counter]
    p2p: dict[str, Counter]
    trigrams: dict[tuple[str, str], Counter]
    cat2p: dict[str, Counter]
    order_cooccur: dict[str, Counter]


def build_cooccurrence(
    sessions: list[list[str]],
    window_size: int,
    session_weights: list[float],
) -> defaultdict[str, Counter]:
    coocc = defaultdict(Counter)

    for sequence, session_weight in zip(sessions, session_weights):
        if len(sequence) < 2:
            continue

        for index, item in enumerate(sequence):
            start = max(0, index - window_size)
            end = min(len(sequence), index + window_size + 1)
            for neighbor_index in range(start, end):
                if index == neighbor_index:
                    continue
                distance = abs(index - neighbor_index)
                neighbor = sequence[neighbor_index]
                coocc[item][neighbor] += (
                    1.0 / distance
                ) * float(session_weight)

    return coocc


def build_transitions(
    sessions: Iterable[list[str]],
) -> tuple[defaultdict[str, Counter], defaultdict[tuple[str, str], Counter]]:
    p2p = defaultdict(Counter)
    trigrams = defaultdict(Counter)

    for sequence in sessions:
        for index in range(len(sequence) - 1):
            p2p[sequence[index]][sequence[index + 1]] += 1
            if index >= 1:
                key = (sequence[index - 1], sequence[index])
                trigrams[key][sequence[index + 1]] += 1

    return p2p, trigrams


def build_cat2p(
    merged: pd.DataFrame,
    slug_map: dict[str, int],
) -> defaultdict[str, Counter]:
    cat2p = defaultdict(Counter)
    relevant = merged[
        merged["page_type"].isin(["PRODUCT", "CATEGORY"])
    ].copy()
    relevant = relevant.dropna(subset=["visit_id", "slug"])
    relevant["product_id"] = relevant["slug"].map(slug_map)

    for _, group in relevant.groupby("visit_id"):
        last_category = None
        for row in group.sort_values("date_time").itertuples(index=False):
            if row.page_type == "CATEGORY":
                last_category = row.slug
            elif (
                row.page_type == "PRODUCT"
                and last_category is not None
                and pd.notna(row.product_id)
            ):
                cat2p[last_category][str(int(row.product_id))] += 1

    return cat2p


def _add_order_pairs(
    order_cooccur: defaultdict[str, Counter],
    product_ids: list[str],
) -> None:
    unique_ids = list(set(product_ids))
    for index, first in enumerate(unique_ids):
        for second in unique_ids[index + 1 :]:
            order_cooccur[first][second] += 1
            order_cooccur[second][first] += 1


def build_order_cooccurrence(
    config: PipelineConfig,
    old_to_new: dict,
    valid_product_ids: set[str],
) -> defaultdict[str, Counter]:
    order_cooccur = defaultdict(Counter)

    new_orders = pd.read_parquet(
        config.paths.data_dir / "new_site_orders.parquet"
    )
    for _, group in new_orders.groupby("id"):
        product_ids = [
            str(int(product_id))
            for product_id in group["product_id"].unique()
            if str(int(product_id)) in valid_product_ids
        ]
        _add_order_pairs(order_cooccur, product_ids)

    old_orders = pd.read_parquet(
        config.paths.data_dir / "old_site_orders.parquet"
    )
    for _, group in old_orders.groupby("id"):
        product_ids = []
        for old_product_id in group["product_id"].unique():
            new_product_id = old_to_new.get(int(old_product_id))
            if (
                new_product_id
                and str(int(new_product_id)) in valid_product_ids
            ):
                product_ids.append(str(int(new_product_id)))
        _add_order_pairs(order_cooccur, product_ids)

    log_summary(
        logger,
        "Order Data",
        {
            "New-site rows": f"{len(new_orders):,}",
            "Old-site rows": f"{len(old_orders):,}",
        },
    )
    return order_cooccur


def _sequence_lists(sessions: pd.DataFrame) -> list[list[str]]:
    return [list(sequence) for sequence in sessions["product_sequence"]]


def build_behavioral_artifacts(
    config: PipelineConfig,
) -> BehavioralArtifacts:
    intermediate_dir = config.paths.intermediate_dir
    slug_map = load_pickle(intermediate_dir / "slug_map.pkl")
    old_to_new = load_pickle(intermediate_dir / "old_to_new.pkl")
    new_products = pd.read_parquet(
        intermediate_dir / "new_products.parquet"
    )
    train_merged = pd.read_parquet(
        intermediate_dir / "train_merged.parquet"
    )
    test_merged = pd.read_parquet(
        intermediate_dir / "test_merged.parquet"
    )
    test_sessions = pd.read_parquet(
        intermediate_dir / "test_sessions.parquet"
    )

    recent_train_merged = train_merged[
        train_merged["visit_start"] >= config.discovery.cutoff_date
    ]
    recent_train_sessions = extract_product_sessions(
        recent_train_merged,
        slug_map,
    )

    minimum_length = config.sessions.minimum_product_sequence_length
    recent_sequences = _sequence_lists(
        recent_train_sessions[
            recent_train_sessions["sequence_length"] >= minimum_length
        ]
    )
    test_sequences = _sequence_lists(
        test_sessions[test_sessions["sequence_length"] >= minimum_length]
    )
    combined_sequences = recent_sequences + test_sequences
    combined_weights = (
        [config.behavioral_signals.recent_train_weight] * len(recent_sequences)
        + [config.behavioral_signals.test_weight] * len(test_sequences)
    )

    coocc = build_cooccurrence(
        combined_sequences,
        config.behavioral_signals.cooccurrence_window_size,
        combined_weights,
    )
    p2p, trigrams = build_transitions(combined_sequences)

    cat2p = build_cat2p(recent_train_merged, slug_map)
    test_cat2p = build_cat2p(test_merged, slug_map)
    for category_slug, counts in test_cat2p.items():
        cat2p[category_slug] += counts

    valid_product_ids = {
        str(int(product_id))
        for product_id in new_products["id"].astype(int).unique()
    }
    order_cooccur = build_order_cooccurrence(
        config,
        old_to_new,
        valid_product_ids,
    )

    log_summary(
        logger,
        "Behavioral Signals",
        {
            "Recent train sessions": f"{len(recent_sequences):,}",
            "Test sessions": f"{len(test_sequences):,}",
            "Co-occurrence anchors": f"{len(coocc):,}",
            "P2P anchors": f"{len(p2p):,}",
            "Trigram contexts": f"{len(trigrams):,}",
            "Category contexts": f"{len(cat2p):,}",
            "Order anchors": f"{len(order_cooccur):,}",
        },
    )

    return BehavioralArtifacts(
        coocc=dict(coocc),
        p2p=dict(p2p),
        trigrams=dict(trigrams),
        cat2p=dict(cat2p),
        order_cooccur=dict(order_cooccur),
    )


def save_behavioral_artifacts(
    artifacts: BehavioralArtifacts,
    config: PipelineConfig,
) -> None:
    output_dir = ensure_directory(config.paths.intermediate_dir)
    outputs = {
        "coocc.pkl": artifacts.coocc,
        "p2p.pkl": artifacts.p2p,
        "trigrams_dict.pkl": artifacts.trigrams,
        "cat2p.pkl": artifacts.cat2p,
        "order_cooccur.pkl": artifacts.order_cooccur,
    }

    for filename, value in outputs.items():
        save_pickle(output_dir / filename, value)

    log_summary(
        logger,
        "Output",
        {"Files": f"{len(outputs)}", "Directory": output_dir},
    )


def prepare_behavioral_signals(
    config: PipelineConfig = CONFIG,
) -> BehavioralArtifacts:
    stage_start(logger, "Prepare Behavioral Signals")
    artifacts = build_behavioral_artifacts(config)
    save_behavioral_artifacts(artifacts, config)
    stage_complete(logger, config.paths.intermediate_dir)
    return artifacts


if __name__ == "__main__":
    prepare_behavioral_signals()
