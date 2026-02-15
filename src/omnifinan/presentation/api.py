"""REST API layer for OmniFinan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, request

from ..core.workflow import run_hedge_fund


@dataclass
class AnalyzeRequest:
    tickers: list[str]
    start_date: str
    end_date: str
    portfolio: dict[str, Any] | None = None
    model_name: str = "deepseek-chat"
    provider_api: str = "deepseek"
    language: str = "Chinese"
    temperature: float | None = None
    llm_seed: int | None = None
    deterministic_mode: bool | None = None
    data_provider: str | None = None


def create_app() -> Flask:
    app = Flask(__name__)

    @app.post("/analyze")
    def analyze() -> Any:
        payload = request.get_json(force=True)
        req = AnalyzeRequest(**payload)
        result = run_hedge_fund(
            tickers=req.tickers,
            start_date=req.start_date,
            end_date=req.end_date,
            portfolio=req.portfolio,
            model_name=req.model_name,
            provider_api=req.provider_api,
            language=req.language,
            temperature=req.temperature,
            llm_seed=req.llm_seed,
            deterministic_mode=req.deterministic_mode,
            data_provider=req.data_provider,
        )
        return jsonify(result)

    @app.get("/healthz")
    def healthz() -> Any:
        return jsonify({"status": "ok"})

    return app
