import shutil
from pathlib import Path
from typing import Any

from training.config import CONFIG, PAD_IDX, PipelineConfig
from training.utils import (
    ensure_directory,
    get_logger,
    load_json,
    log_summary,
    save_json,
    stage_complete,
    stage_start,
)


logger = get_logger(__name__)

COPY_ARTIFACTS = (
    "model.pt",
    "old_to_new.pkl",
    "slug_map.pkl",
    "pid_to_cat_idx.pkl",
    "pid_to_tier.pkl",
    "pid2idx.pkl",
    "idx2pid.pkl",
    "p2p.pkl",
    "coocc.pkl",
    "trigrams_dict.pkl",
    "cat2p.pkl",
    "order_cooccur.pkl",
    "search_index.pkl",
    "slug_to_cat_map.pkl",
    "global_top.json",
)


def _validate_sources(source_dir: Path) -> None:
    missing = [
        filename
        for filename in COPY_ARTIFACTS
        if not (source_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required intermediate artifacts: " + ", ".join(missing)
        )


def build_inference_config(
    metadata: dict[str, Any],
    config: PipelineConfig,
) -> dict[str, Any]:
    return {
        "pad_idx": PAD_IDX,
        "data_dir": str(config.paths.data_dir),
        "model_path": str(config.paths.final_dir / "model.pt"),
        "w_new": config.discovery.new_site_weight,
        "session_tolerance": str(config.sessions.tolerance),
        "cutoff_date": str(config.discovery.cutoff_date),
        "use_sample_weighting": config.training.use_sample_weighting,
        "use_price_tier": config.model.use_price_tier,
        "sample_weight_decay_rate": (
            config.training.sample_weight_decay_rate
        ),
        "num_items": int(metadata["num_items"]),
        "num_categories": int(metadata["num_categories"]),
    }


def package_artifacts(config: PipelineConfig = CONFIG) -> Path:
    stage_start(logger, "Package Artifacts")
    source_dir = config.paths.intermediate_dir
    final_dir = ensure_directory(config.paths.final_dir)

    if config.packaging.validate_required_artifacts:
        _validate_sources(source_dir)

    for filename in COPY_ARTIFACTS:
        shutil.copy2(source_dir / filename, final_dir / filename)

    metadata = load_json(source_dir / "gru_metadata.json")
    inference_config = build_inference_config(metadata, config)
    save_json(final_dir / "config.json", inference_config)

    packaged_files = [final_dir / filename for filename in COPY_ARTIFACTS]
    packaged_files.append(final_dir / "config.json")
    total_bytes = sum(path.stat().st_size for path in packaged_files)
    log_summary(
        logger,
        "Final Bundle",
        {
            "Artifacts": f"{len(packaged_files)}",
            "Total size": f"{total_bytes / (1024 * 1024):.2f} MB",
            "Directory": final_dir,
        },
    )
    stage_complete(logger, final_dir)
    return final_dir


if __name__ == "__main__":
    package_artifacts()
