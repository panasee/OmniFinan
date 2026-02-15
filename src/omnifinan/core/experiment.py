"""Experiment record and comparison helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyomnix.consts import OMNIX_PATH


@dataclass
class ExperimentRecorder:
    run_id: str
    base_dir: Path | None = None

    @property
    def file_path(self) -> Path:
        root = self.base_dir or (OMNIX_PATH / "financial" / "experiments")
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{self.run_id}.json"

    def persist(self, payload: dict[str, Any]) -> Path:
        with open(self.file_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return self.file_path


def compare_experiments(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    """Compare key metrics from two experiment payloads."""
    first_metrics = first.get("metrics", {})
    second_metrics = second.get("metrics", {})
    keys = sorted(set(first_metrics.keys()) | set(second_metrics.keys()))
    diffs = {}
    for key in keys:
        a = first_metrics.get(key)
        b = second_metrics.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diffs[key] = {"first": a, "second": b, "delta": b - a}
        else:
            diffs[key] = {"first": a, "second": b}
    return {"metric_diffs": diffs}
