from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from training.utils import get_logger, log_summary, stage_start


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"
PAD_IDX = 0
logger = get_logger(__name__)


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    artifacts_dir: Path
    intermediate_dir: Path
    final_dir: Path


@dataclass(frozen=True)
class ProductsConfig:
    price_tiers: dict[str, int]


@dataclass(frozen=True)
class DiscoveryConfig:
    reference_date: pd.Timestamp
    lookback_days: int
    new_site_weight: float
    category_min_votes: int
    search_min_token_length: int
    global_top_k: int

    @property
    def cutoff_date(self) -> pd.Timestamp:
        return self.reference_date - pd.Timedelta(days=self.lookback_days)


@dataclass(frozen=True)
class SessionsConfig:
    tolerance: pd.Timedelta
    minimum_product_sequence_length: int


@dataclass(frozen=True)
class BehavioralSignalsConfig:
    cooccurrence_window_size: int
    recent_train_weight: float
    test_weight: float


@dataclass(frozen=True)
class GruDataConfig:
    minimum_session_products: int


@dataclass(frozen=True)
class ModelConfig:
    item_embedding_dim: int
    tier_embedding_dim: int
    category_embedding_dim: int
    hidden_dim: int
    category_hidden_dim: int
    num_layers: int
    dropout: float
    use_price_tier: bool


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    use_sample_weighting: bool
    sample_weight_decay_rate: float
    scheduler_factor: float
    scheduler_patience: int


@dataclass(frozen=True)
class PackagingConfig:
    validate_required_artifacts: bool


@dataclass(frozen=True)
class PipelineConfig:
    seed: int
    paths: PathsConfig
    products: ProductsConfig
    discovery: DiscoveryConfig
    sessions: SessionsConfig
    behavioral_signals: BehavioralSignalsConfig
    gru_data: GruDataConfig
    model: ModelConfig
    training: TrainingConfig
    packaging: PackagingConfig


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _require_section(params: dict[str, Any], name: str) -> dict[str, Any]:
    section = params.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in {PARAMS_PATH}")
    return section


def _validate_config(config: PipelineConfig) -> None:
    positive_values = {
        "seed": config.seed,
        "discovery.lookback_days": config.discovery.lookback_days,
        "discovery.global_top_k": config.discovery.global_top_k,
        "sessions.minimum_product_sequence_length": (
            config.sessions.minimum_product_sequence_length
        ),
        "behavioral_signals.cooccurrence_window_size": (
            config.behavioral_signals.cooccurrence_window_size
        ),
        "gru_data.minimum_session_products": config.gru_data.minimum_session_products,
        "model.num_layers": config.model.num_layers,
        "training.epochs": config.training.epochs,
        "training.batch_size": config.training.batch_size,
        "training.learning_rate": config.training.learning_rate,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(
            "Configuration values must be positive: " + ", ".join(invalid)
        )

    if not 0 <= config.model.dropout < 1:
        raise ValueError("model.dropout must be in the range [0, 1)")
    if not 0 < config.training.scheduler_factor < 1:
        raise ValueError("training.scheduler_factor must be in the range (0, 1)")
    if config.sessions.tolerance <= pd.Timedelta(0):
        raise ValueError("sessions.tolerance must be greater than zero")
    if config.training.device not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(
            "training.device must be one of: auto, cpu, mps, cuda"
        )


def load_config(params_path: Path = PARAMS_PATH) -> PipelineConfig:
    if not params_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_path}")

    with params_path.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    if not isinstance(params, dict):
        raise ValueError(f"Expected a YAML mapping in {params_path}")

    paths = _require_section(params, "paths")
    products = _require_section(params, "products")
    discovery = _require_section(params, "discovery")
    sessions = _require_section(params, "sessions")
    behavioral = _require_section(params, "behavioral_signals")
    gru_data = _require_section(params, "gru_data")
    model = _require_section(params, "model")
    training = _require_section(params, "training")
    packaging = _require_section(params, "packaging")

    config = PipelineConfig(
        seed=int(params["seed"]),
        paths=PathsConfig(
            data_dir=_resolve_project_path(paths["data_dir"]),
            artifacts_dir=_resolve_project_path(paths["artifacts_dir"]),
            intermediate_dir=_resolve_project_path(paths["intermediate_dir"]),
            final_dir=_resolve_project_path(paths["final_dir"]),
        ),
        products=ProductsConfig(
            price_tiers={
                str(name): int(index)
                for name, index in products["price_tiers"].items()
            }
        ),
        discovery=DiscoveryConfig(
            reference_date=pd.Timestamp(discovery["reference_date"]),
            lookback_days=int(discovery["lookback_days"]),
            new_site_weight=float(discovery["new_site_weight"]),
            category_min_votes=int(discovery["category_min_votes"]),
            search_min_token_length=int(discovery["search_min_token_length"]),
            global_top_k=int(discovery["global_top_k"]),
        ),
        sessions=SessionsConfig(
            tolerance=pd.Timedelta(sessions["tolerance"]),
            minimum_product_sequence_length=int(
                sessions["minimum_product_sequence_length"]
            ),
        ),
        behavioral_signals=BehavioralSignalsConfig(
            cooccurrence_window_size=int(behavioral["cooccurrence_window_size"]),
            recent_train_weight=float(behavioral["recent_train_weight"]),
            test_weight=float(behavioral["test_weight"]),
        ),
        gru_data=GruDataConfig(
            minimum_session_products=int(gru_data["minimum_session_products"])
        ),
        model=ModelConfig(
            item_embedding_dim=int(model["item_embedding_dim"]),
            tier_embedding_dim=int(model["tier_embedding_dim"]),
            category_embedding_dim=int(model["category_embedding_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            category_hidden_dim=int(model["category_hidden_dim"]),
            num_layers=int(model["num_layers"]),
            dropout=float(model["dropout"]),
            use_price_tier=bool(model["use_price_tier"]),
        ),
        training=TrainingConfig(
            device=str(training["device"]).lower(),
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            use_sample_weighting=bool(training["use_sample_weighting"]),
            sample_weight_decay_rate=float(
                training["sample_weight_decay_rate"]
            ),
            scheduler_factor=float(training["scheduler_factor"]),
            scheduler_patience=int(training["scheduler_patience"]),
        ),
        packaging=PackagingConfig(
            validate_required_artifacts=bool(
                packaging["validate_required_artifacts"]
            )
        ),
    )
    _validate_config(config)
    return config


def log_config_summary(config: PipelineConfig) -> None:
    stage_start(logger, "Pipeline Configuration")
    log_summary(
        logger,
        "Paths",
        {
            "Parameters": PARAMS_PATH.name,
            "Data": config.paths.data_dir,
            "Intermediate": config.paths.intermediate_dir,
            "Final": config.paths.final_dir,
        },
    )
    log_summary(
        logger,
        "Run Settings",
        {
            "Device": config.training.device,
            "Seed": config.seed,
            "Recency cutoff": config.discovery.cutoff_date.date(),
            "Session tolerance": config.sessions.tolerance,
            "Epochs": config.training.epochs,
            "Batch size": config.training.batch_size,
            "Sample weighting": config.training.use_sample_weighting,
            "Price tier feature": config.model.use_price_tier,
        },
    )


CONFIG = load_config()

# Compatibility constants keep stage code concise while train.py is modularized.
DATA_DIR = CONFIG.paths.data_dir
ARTIFACTS_DIR = CONFIG.paths.artifacts_dir
INTERMEDIATE_DIR = CONFIG.paths.intermediate_dir
FINAL_DIR = CONFIG.paths.final_dir
W_NEW = CONFIG.discovery.new_site_weight
SESSION_TOLERANCE = CONFIG.sessions.tolerance
CUTOFF_DATE = CONFIG.discovery.cutoff_date
USE_SAMPLE_WEIGHTING = CONFIG.training.use_sample_weighting
USE_PRICE_TIER = CONFIG.model.use_price_tier
SAMPLE_WEIGHT_DECAY_RATE = CONFIG.training.sample_weight_decay_rate
EPOCHS = CONFIG.training.epochs


if __name__ == "__main__":
    log_config_summary(CONFIG)
