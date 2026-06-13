from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from training.config import CONFIG, PipelineConfig
from training.utils import (
    ensure_directory,
    get_logger,
    load_pickle,
    log_summary,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)


@dataclass(frozen=True)
class SessionArtifacts:
    train_merged: pd.DataFrame
    test_merged: pd.DataFrame
    train_sessions: pd.DataFrame
    test_sessions: pd.DataFrame


def build_sessions_merge_asof(
    hits_path: Path,
    visits_path: Path,
    tolerance: pd.Timedelta,
) -> pd.DataFrame:
    """Match hits to the latest preceding visit for the same client."""
    hits = pd.read_parquet(hits_path)
    hits["date_time"] = pd.to_datetime(
        hits["date_time"],
        format="ISO8601",
        errors="coerce",
    )
    hits = hits.dropna(subset=["date_time", "client_id"])

    visits = pd.read_parquet(visits_path)
    visits["date_time"] = pd.to_datetime(
        visits["date_time"],
        format="ISO8601",
        errors="coerce",
    )
    visits = visits.dropna(subset=["date_time", "client_id", "visit_id"])
    visits = visits.rename(columns={"date_time": "visit_start"})

    hits_sorted = hits.sort_values(
        ["client_id", "date_time"]
    ).reset_index(drop=True)
    visits_sorted = visits.sort_values(
        ["client_id", "visit_start"]
    ).reset_index(drop=True)

    merged = pd.merge_asof(
        hits_sorted.sort_values("date_time"),
        visits_sorted.sort_values("visit_start")[
            ["client_id", "visit_start", "visit_id", "project_id"]
        ],
        by="client_id",
        left_on="date_time",
        right_on="visit_start",
        direction="backward",
        tolerance=tolerance,
    )

    matched = int(merged["visit_id"].notna().sum())
    total = len(merged)
    match_rate = matched / total if total else 0.0
    log_summary(
        logger,
        hits_path.stem,
        {
            "Hits": f"{len(hits):,}",
            "Visits": f"{len(visits):,}",
            "Matched hits": f"{matched:,} ({match_rate:.2%})",
        },
    )
    return merged


def extract_product_sessions(
    merged: pd.DataFrame,
    slug_map: dict[str, int],
) -> pd.DataFrame:
    """Create one ordered product sequence per matched visit."""
    products = merged[merged["page_type"] == "PRODUCT"].copy()
    products["product_id"] = products["slug"].map(slug_map)
    products = products.dropna(subset=["visit_id", "product_id"])
    products["product_id"] = products["product_id"].astype(int).astype(str)

    rows = []
    for visit_id, group in products.groupby("visit_id"):
        sequence = group.sort_values("date_time")["product_id"].tolist()
        if sequence:
            rows.append(
                {
                    "session_id": str(visit_id),
                    "product_sequence": sequence,
                    "sequence_length": len(sequence),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["session_id", "product_sequence", "sequence_length"],
    )


def build_session_artifacts(config: PipelineConfig) -> SessionArtifacts:
    data_dir = config.paths.data_dir
    slug_map = load_pickle(config.paths.intermediate_dir / "slug_map.pkl")

    train_merged = build_sessions_merge_asof(
        data_dir / "metrika_hits.parquet",
        data_dir / "metrika_visits.parquet",
        config.sessions.tolerance,
    )
    test_merged = build_sessions_merge_asof(
        data_dir / "metrika_hits_test.parquet",
        data_dir / "metrika_visits_test.parquet",
        config.sessions.tolerance,
    )

    train_sessions = extract_product_sessions(train_merged, slug_map)
    test_sessions = extract_product_sessions(test_merged, slug_map)

    eligible_train = int(
        (
            train_sessions["sequence_length"]
            >= config.sessions.minimum_product_sequence_length
        ).sum()
    )
    eligible_test = int(
        (
            test_sessions["sequence_length"]
            >= config.sessions.minimum_product_sequence_length
        ).sum()
    )
    log_summary(
        logger,
        "Sessions",
        {
            "Train sessions": f"{len(train_sessions):,}",
            "Test sessions": f"{len(test_sessions):,}",
            "Eligible sessions": f"{eligible_train + eligible_test:,}",
        },
    )

    return SessionArtifacts(
        train_merged=train_merged,
        test_merged=test_merged,
        train_sessions=train_sessions,
        test_sessions=test_sessions,
    )


def save_session_artifacts(
    artifacts: SessionArtifacts,
    config: PipelineConfig,
) -> None:
    output_dir = ensure_directory(config.paths.intermediate_dir)
    outputs = {
        "train_merged.parquet": artifacts.train_merged,
        "test_merged.parquet": artifacts.test_merged,
        "train_sessions.parquet": artifacts.train_sessions,
        "test_sessions.parquet": artifacts.test_sessions,
    }

    for filename, dataframe in outputs.items():
        dataframe.to_parquet(output_dir / filename, index=False)

    log_summary(
        logger,
        "Output",
        {"Files": f"{len(outputs)}", "Directory": output_dir},
    )


def prepare_sessions(
    config: PipelineConfig = CONFIG,
) -> SessionArtifacts:
    stage_start(logger, "Prepare Sessions")
    artifacts = build_session_artifacts(config)
    save_session_artifacts(artifacts, config)
    stage_complete(logger, config.paths.intermediate_dir)
    return artifacts


if __name__ == "__main__":
    prepare_sessions()
