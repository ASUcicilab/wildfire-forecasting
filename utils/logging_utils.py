"""Logging utilities for CMPF experiments."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
) -> logging.Logger:
    """Setup logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(log_path / f"{name}_{timestamp}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Logger for experiment results and metrics."""

    def __init__(self, output_dir: str, experiment_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name

        self.results = {
            "experiment": experiment_name,
            "start_time": datetime.now().isoformat(),
            "metrics": {},
            "config": {},
        }

        self.logger = setup_logger(experiment_name, str(self.output_dir))

    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.results["config"] = config
        self.logger.info(f"Config: {json.dumps(config, indent=2)}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if name not in self.results["metrics"]:
            self.results["metrics"][name] = []

        entry = {"value": value}
        if step is not None:
            entry["step"] = step

        self.results["metrics"][name].append(entry)
        self.logger.info(f"{name}: {value:.4f}" + (f" (step {step})" if step else ""))

    def log_results(self, results: dict):
        """Log multiple results."""
        for name, value in results.items():
            self.log_metric(name, value)

    def save(self):
        """Save all logged results to file."""
        self.results["end_time"] = datetime.now().isoformat()

        output_path = self.output_dir / f"{self.experiment_name}_results.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
