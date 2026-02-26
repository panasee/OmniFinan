"""Observability helpers for workflow runs."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pyomnix.consts import OMNIX_PATH


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metrics: dict[str, Any] = field(default_factory=dict)
    nodes: list[dict[str, Any]] = field(default_factory=list)
    tool_call_metrics: list[dict[str, Any]] = field(default_factory=list)
    token_usage: list[dict[str, Any]] = field(default_factory=list)
    cost_estimate: dict[str, float] = field(
        default_factory=lambda: {"input": 0.0, "output": 0.0, "total": 0.0}
    )

    def mark_node(self, node_name: str, status: str, payload: dict[str, Any] | None = None) -> None:
        self.nodes.append(
            {
                "node": node_name,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
                "payload": payload or {},
            }
        )

    def finish(self) -> None:
        self.metrics["elapsed_seconds"] = round(
            time.time() - datetime.fromisoformat(self.started_at).timestamp(), 3
        )
        self.metrics["tool_calls"] = len(self.tool_call_metrics)
        self.metrics["llm_calls"] = len(self.token_usage)

    def mark_tool_call(
        self,
        tool_name: str,
        elapsed_seconds: float,
        input_summary: str = "",
        output_summary: str = "",
    ) -> None:
        self.tool_call_metrics.append(
            {
                "tool_name": tool_name,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "input_summary": input_summary[:500],
                "output_summary": output_summary[:500],
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    def mark_llm_usage(
        self,
        agent_name: str,
        model_name: str,
        provider_api: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.token_usage.append(
            {
                "agent_name": agent_name,
                "model_name": model_name,
                "provider_api": provider_api,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Coarse default estimator in USD per 1k tokens.
        in_cost = (input_tokens / 1000.0) * 0.001
        out_cost = (output_tokens / 1000.0) * 0.002
        self.cost_estimate["input"] += in_cost
        self.cost_estimate["output"] += out_cost
        self.cost_estimate["total"] = (
            self.cost_estimate["input"] + self.cost_estimate["output"]
        )

    def persist(self, base_dir: Path | None = None) -> Path:
        target_dir = base_dir or (OMNIX_PATH / "omnifinan" / "logs" / "runs")
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{self.run_id}.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "started_at": self.started_at,
                    "metrics": self.metrics,
                    "nodes": self.nodes,
                    "tool_call_metrics": self.tool_call_metrics,
                    "token_usage": self.token_usage,
                    "cost_estimate": self.cost_estimate,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return target_file
