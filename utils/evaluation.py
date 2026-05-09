from __future__ import annotations

import json
from pathlib import Path


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(model_dir: str) -> dict | None:
    return _load_json(Path(model_dir) / "metrics.json")


def load_importances(model_dir: str) -> dict | None:
    return _load_json(Path(model_dir) / "feature_importances.json")


def load_model_comparison(model_dir: str) -> list[dict] | None:
    return _load_json(Path(model_dir) / "model_comparison.json")
