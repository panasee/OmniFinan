"""Append-only run scratchpad for auditing and debugging."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pyomnix.consts import OMNIX_PATH


@dataclass
class Scratchpad:
    run_id: str
    base_dir: Path | None = None

    @property
    def file_path(self) -> Path:
        root = self.base_dir or (OMNIX_PATH / "omnifinan" / "logs" / "scratchpad")
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{self.run_id}.jsonl"

    def append(self, entry_type: str, payload: dict[str, Any]) -> None:
        line = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": entry_type,
            "payload": payload,
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
