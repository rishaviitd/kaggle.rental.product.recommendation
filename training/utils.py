import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPORT_WIDTH = 78


class ConsoleFormatter(logging.Formatter):
    """Keep INFO output clean while making warnings and errors unmistakable."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
            if record.args:
                message = f"{message} {record.args}"
        if record.levelno >= logging.WARNING:
            return f"[{record.levelname}] {message}"
        return message


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted console logger."""
    named_logger = logging.getLogger(name)
    named_logger.setLevel(level)
    named_logger.propagate = False

    if not named_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(ConsoleFormatter())
        named_logger.addHandler(console_handler)

    return named_logger


logger = get_logger(__name__)


def stage_start(logger: logging.Logger, title: str) -> None:
    logger.info(
        "\n%s\n  %s\n%s",
        "=" * REPORT_WIDTH,
        title.upper(),
        "=" * REPORT_WIDTH,
    )


def _display_value(value: Any) -> str:
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(value)
    return str(value)


def log_summary(
    logger: logging.Logger,
    title: str,
    values: dict[str, Any],
) -> None:
    label_width = max(len(label) for label in values)
    lines = [f"\n  {title}", f"  {'-' * (REPORT_WIDTH - 2)}"]
    lines.extend(
        f"  {label:<{label_width}}  {_display_value(value)}"
        for label, value in values.items()
    )
    logger.info("\n".join(lines))


def stage_complete(
    logger: logging.Logger,
    output: str | Path | None = None,
) -> None:
    message = "\n  DONE"
    if output is not None:
        message += f"  ->  {_display_value(output)}"
    logger.info(f"{message}\n{'=' * REPORT_WIDTH}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("Random seed set to %d", seed)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory and return its resolved Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_pickle(path: str | Path, value: Any) -> Path:
    """Serialize a Python object using the highest pickle protocol."""
    output_path = Path(path)
    ensure_directory(output_path.parent)

    with output_path.open("wb") as file:
        pickle.dump(value, file, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug("Saved pickle: %s", output_path)
    return output_path


def load_pickle(path: str | Path) -> Any:
    """Load a trusted local pickle artifact."""
    input_path = Path(path)
    with input_path.open("rb") as file:
        value = pickle.load(file)

    logger.debug("Loaded pickle: %s", input_path)
    return value


def save_json(path: str | Path, value: Any) -> Path:
    """Write a human-readable UTF-8 JSON artifact."""
    output_path = Path(path)
    ensure_directory(output_path.parent)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=str)
        file.write("\n")

    logger.debug("Saved JSON: %s", output_path)
    return output_path


def load_json(path: str | Path) -> Any:
    """Load a UTF-8 JSON artifact."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as file:
        value = json.load(file)

    logger.debug("Loaded JSON: %s", input_path)
    return value
