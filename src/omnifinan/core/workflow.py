"""Workflow construction and execution for OmniFinan."""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage
from pyomnix.consts import OMNIX_PATH
from pyomnix.omnix_logger import get_logger

from ..agents.graphs import create_trading_graph
from ..agents.prompts import PROMPTS
from ..data.cache import DataCache
from ..data.providers.factory import create_data_provider
from ..data.unified_service import UnifiedDataService
from ..utils.progress import progress
from ..utils.scratchpad import Scratchpad
from .config import RuntimeConfig
from .experiment import ExperimentRecorder
from .observability import RunTrace

logger = get_logger(__name__)


def parse_hedge_fund_response(response: str) -> dict[str, Any] | None:
    try:
        return json.loads(response)
    except json.JSONDecodeError as exc:
        logger.error("JSON decoding error while parsing hedge fund response: %s", exc)
        return None
    except TypeError as exc:
        logger.error("Invalid response type while parsing hedge fund response: %s", exc)
        return None
    except Exception as exc:  # pragma: no cover - defensive branch
        logger.error("Unexpected error while parsing hedge fund response: %s", exc)
        return None
def create_workflow(selected_analysts: list[str] | None = None):
    return create_trading_graph(selected_analysts=selected_analysts)


def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict | None,
    show_reasoning: bool = False,
    selected_analysts: list[str] | None = None,
    model_name: str = "deepseek-chat",
    provider_api: str = "deepseek",
    language: str = "Chinese",
    temperature: float | None = None,
    llm_seed: int | None = None,
    deterministic_mode: bool | None = None,
    data_provider: str | None = None,
):
    config_path = os.getenv("OMNIFINAN_CONFIG_PATH")
    runtime_config = (
        RuntimeConfig.from_file(config_path) if config_path else RuntimeConfig.from_env()
    )
    active_data_provider = data_provider or runtime_config.data_provider
    trace = RunTrace()
    scratchpad = Scratchpad(run_id=trace.run_id)
    data_service = UnifiedDataService(
        provider=create_data_provider(active_data_provider),
        cache=DataCache(),
        ttl_seconds=runtime_config.data_cache_ttl_seconds,
    )
    progress.start()
    try:
        active_analysts = selected_analysts or runtime_config.enabled_analysts or None
        workflow = create_workflow(active_analysts)
        agent = workflow.compile()
        trace.mark_node("workflow", "started", {"tickers": tickers})

        final_state = agent.invoke(
            {
                "messages": [HumanMessage(content=PROMPTS.workflow_start)],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name or runtime_config.llm.model_name,
                    "provider_api": provider_api or runtime_config.llm.provider_api,
                    "language": language or runtime_config.llm.language,
                    "temperature": (
                        runtime_config.llm.temperature if temperature is None else temperature
                    ),
                    "llm_max_retries": runtime_config.llm.max_retries,
                    "deterministic_mode": (
                        runtime_config.deterministic_mode
                        if deterministic_mode is None
                        else deterministic_mode
                    ),
                    "llm_seed": runtime_config.llm_seed if llm_seed is None else llm_seed,
                    "run_id": trace.run_id,
                    "data_service": data_service,
                    "trace": trace,
                    "scratchpad": scratchpad,
                    "max_debate_rounds": runtime_config.debate_rounds,
                    "max_risk_discuss_rounds": runtime_config.debate_rounds,
                    "data_provider": active_data_provider,
                },
            }
        )

        out_dir = OMNIX_PATH / "financial"
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(final_state, open(out_dir / "hedge_fund_output.json", "w", encoding="utf-8"))

        if runtime_config.enable_scratchpad:
            scratchpad.append(
                "result",
                {
                    "tickers": tickers,
                    "decision": final_state["messages"][-1].content if final_state["messages"] else "",
                },
            )

        trace.mark_node("workflow", "completed")
        trace.finish()
        if runtime_config.enable_observability:
            trace.persist()
            ExperimentRecorder(trace.run_id).persist(
                {
                    "run_id": trace.run_id,
                    "params": {
                        "tickers": tickers,
                        "start_date": start_date,
                        "end_date": end_date,
                        "selected_analysts": active_analysts or [],
                    },
                    "metrics": trace.metrics,
                    "cost_estimate": trace.cost_estimate,
                }
            )

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
            "run_id": trace.run_id,
        }
    finally:
        progress.stop()
