import argparse
import csv
import fcntl
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

# Support direct execution from scripts/ without installing the project.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_snapshot import capture_snapshot, finish_snapshot
from src.models import MODEL_REGISTRY
from src.segmentation_experiment import run_segmentation

RESULTS_DIR = PROJECT_ROOT / "results"
LEADERBOARD_FIELDS = ("timestamp", "model_name", "config_path", "metrics", "train_seconds")


def append_result(path, row):
    with path.open("a", encoding="utf-8", newline="") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        writer = csv.DictWriter(file, fieldnames=LEADERBOARD_FIELDS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(config_path):
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("The YAML config must contain a mapping.")

    for field in ("model_name", "random_seed", "test_size"):
        if field not in config:
            raise ValueError(f"Missing config field {field!r}")

    model_name = config["model_name"]
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_name!r}. Choose from {sorted(MODEL_REGISTRY)}.")

    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("The params field must contain a mapping.")
    timestamp = datetime.now(UTC)
    run_id = f"{model_name}-{timestamp:%Y%m%dT%H%M%S%fZ}-{uuid4().hex}"
    runs_dir = RESULTS_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = runs_dir / run_id
    with TemporaryDirectory(dir=runs_dir, prefix=".run-") as temporary_dir:
        snapshot = Path(temporary_dir)
        runtime = capture_snapshot(PROJECT_ROOT, snapshot, config_bytes)
        model = MODEL_REGISTRY[model_name](random_seed=config["random_seed"], **params)
        metrics, train_seconds, artifact_path = run_segmentation(config, model, RESULTS_DIR, PROJECT_ROOT)
        result = {
            "timestamp": timestamp.isoformat(timespec="microseconds"),
            "model_name": model_name,
            "config_path": str(config_path),
            "metrics": metrics,
            "train_seconds": train_seconds,
        }
        finish_snapshot(
            snapshot, result, runtime, str(getattr(model, "device", "cpu")),
            artifact_path.relative_to(RESULTS_DIR),
        )
        snapshot.rename(snapshot_path)

    append_result(
        RESULTS_DIR / "leaderboard.csv",
        {**result, "metrics": json.dumps(metrics, sort_keys=True)},
    )
    print(f"{model_name}: {json.dumps(metrics, sort_keys=True)}")
    print(f"Saved output to {artifact_path}")
    print(f"Saved run snapshot to {snapshot_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate a configured model.")
    parser.add_argument("--config", default=PROJECT_ROOT / "configs/dinov3_detail_decoder.yaml", type=Path, help="Path to a YAML config.")
    args = parser.parse_args()
    try:
        run_experiment(args.config)
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
