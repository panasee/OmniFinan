from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


def _build_portfolio(tickers: list[str], cash: float = 100000.0) -> dict[str, Any]:
    return {
        "cash": cash,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            ticker: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for ticker in tickers
        },
        "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
    }


def _price_summary(price_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not price_rows:
        return {"available": False}
    df = pd.DataFrame(price_rows).copy()
    if df.empty or "close" not in df.columns:
        return {"available": False}

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df.dropna(subset=["time"]).sort_values("time")
    if df.empty:
        return {"available": False}

    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if close.empty:
        return {"available": False}

    ret = close.pct_change().dropna()
    rolling_max = close.cummax()
    drawdown = (close / rolling_max) - 1
    last = float(close.iloc[-1])
    first = float(close.iloc[0])
    total_return = (last / first - 1.0) if first != 0 else 0.0
    annual_vol = float(ret.std() * np.sqrt(252)) if not ret.empty else 0.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    window = min(20, len(close))
    ma20 = float(close.tail(window).mean())

    return _to_jsonable(
        {
            "available": True,
            "bars": int(len(df)),
            "start": str(df["time"].iloc[0].date()) if "time" in df.columns else None,
            "end": str(df["time"].iloc[-1].date()) if "time" in df.columns else None,
            "first_close": first,
            "last_close": last,
            "total_return": total_return,
            "annualized_volatility": annual_vol,
            "max_drawdown": max_drawdown,
            "ma20": ma20,
            "above_ma20": bool(last > ma20),
        }
    )


def _run_step(
    name: str,
    fn,
    state: dict[str, Any],
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    try:
        return fn(state)
    except Exception as exc:  # pragma: no cover - runtime safety
        errors.append(
            {
                "step": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
        print(f"[WARN] step failed: {name} -> {type(exc).__name__}: {exc}")
        return state


def run_msft_analysis(
    *,
    start_date: str,
    end_date: str,
    data_provider: str = "yfinance",
    include_sentiment: bool = False,
    include_valuation: bool = False,
) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage

    from omnifinan.agents.fundamentals import fundamentals_agent
    from omnifinan.agents.macro import macro_analyst_agent
    from omnifinan.agents.sentiment import sentiment_agent
    from omnifinan.agents.technicals import technical_analyst_agent
    from omnifinan.agents.valuation import valuation_agent
    from omnifinan.data.cache import DataCache
    from omnifinan.data.providers.factory import create_data_provider
    from omnifinan.data.unified_service import UnifiedDataService

    ticker = "MSFT"
    errors: list[dict[str, str]] = []

    provider = create_data_provider(data_provider)
    data_service = UnifiedDataService(provider=provider, cache=DataCache())

    # Pre-collect stable datasets first (avoid hard dependency on endpoints known to be brittle).
    try:
        prices = data_service.get_prices(ticker, start_date, end_date)
    except Exception as exc:  # pragma: no cover - runtime safety
        prices = []
        errors.append(
            {
                "step": "collect_prices",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )

    try:
        financial_metrics = data_service.get_financial_metrics(
            ticker=ticker, end_date=end_date, period="ttm", limit=10
        )
    except Exception as exc:  # pragma: no cover - runtime safety
        financial_metrics = []
        errors.append(
            {
                "step": "collect_financial_metrics",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )

    try:
        line_items = data_service.get_line_items(ticker=ticker, period="ttm", limit=10)
    except Exception as exc:  # pragma: no cover - runtime safety
        line_items = []
        errors.append(
            {
                "step": "collect_line_items",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )

    try:
        macro_indicators = data_service.get_macro_indicators(start_date=start_date, end_date=end_date)
    except Exception as exc:  # pragma: no cover - runtime safety
        macro_indicators = {"series": {}, "latest": {}, "snapshot_at": None}
        errors.append(
            {
                "step": "collect_macro_indicators",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )

    # Build state compatible with agent functions.
    state: dict[str, Any] = {
        "messages": [HumanMessage(content="MSFT comprehensive analysis")],
        "data": {
            "tickers": [ticker],
            "start_date": start_date,
            "end_date": end_date,
            "portfolio": _build_portfolio([ticker]),
            "prices": {ticker: prices},
            "financial_metrics": {ticker: financial_metrics},
            "financial_line_items": {ticker: line_items},
            "macro_indicators": macro_indicators,
            "analyst_signals": {},
            # Keep this small to avoid LLM path in sentiment agent by default.
            "num_of_news": 2,
        },
        "metadata": {
            "show_reasoning": False,
            "model_name": "deepseek-chat",
            "provider_api": "deepseek",
            "language": "Chinese",
            "temperature": 0.2,
            "llm_seed": 7,
            "llm_max_retries": 1,
            "data_service": data_service,
        },
    }

    pipeline = [
        ("technical_analyst_agent", technical_analyst_agent),
        ("fundamentals_agent", fundamentals_agent),
        ("macro_analyst_agent", macro_analyst_agent),
    ]
    if include_valuation:
        pipeline.append(("valuation_agent", valuation_agent))
    if include_sentiment:
        pipeline.append(("sentiment_agent", sentiment_agent))

    for step_name, step_fn in pipeline:
        print(f"[RUN] {step_name}")
        state = _run_step(step_name, step_fn, state, errors)

    prices = state.get("data", {}).get("prices", {}).get(ticker, [])
    metrics = state.get("data", {}).get("financial_metrics", {}).get(ticker, [])
    line_items = state.get("data", {}).get("financial_line_items", {}).get(ticker, [])
    macro_indicators = state.get("data", {}).get("macro_indicators", {})
    analyst_signals = state.get("data", {}).get("analyst_signals", {})

    report = {
        "meta": {
            "ticker": ticker,
            "data_provider": data_provider,
            "start_date": start_date,
            "end_date": end_date,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pipeline_steps": [name for name, _ in pipeline],
            "failed_steps": [item["step"] for item in errors],
        },
        "market_data_summary": {
            "price_summary": _price_summary(prices),
            "financial_metrics_count": len(metrics),
            "line_items_count": len(line_items),
            "market_cap": None,
            "macro_series_count": len((macro_indicators or {}).get("series", {})),
        },
        "analyst_signals": _to_jsonable(analyst_signals),
        "errors": errors,
    }
    return _to_jsonable(report)


def _default_dates() -> tuple[str, str]:
    end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.strptime(end, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
    return start, end


def main() -> None:
    # Avoid Windows console encoding crashes from third-party loggers.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="ignore")

    d_start, d_end = _default_dates()
    parser = argparse.ArgumentParser(description="Comprehensive MSFT analysis (robust mode).")
    parser.add_argument("--start-date", default=d_start, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=d_end, help="YYYY-MM-DD")
    parser.add_argument(
        "--data-provider",
        default="yfinance",
        choices=["akshare", "finnhub", "yfinance"],
        help="Data provider backend",
    )
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Include sentiment agent (may involve news/LLM path depending on data).",
    )
    parser.add_argument(
        "--include-valuation",
        action="store_true",
        help="Include valuation agent (may depend on provider line items/market-cap completeness).",
    )
    args = parser.parse_args()

    report = run_msft_analysis(
        start_date=args.start_date,
        end_date=args.end_date,
        data_provider=args.data_provider,
        include_sentiment=args.include_sentiment,
        include_valuation=args.include_valuation,
    )

    out_dir = ROOT / "tests" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"msft_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== MSFT Comprehensive Analysis ===")
    print(json.dumps(report["meta"], ensure_ascii=False, indent=2))
    print("\nPrice summary:")
    print(json.dumps(report["market_data_summary"]["price_summary"], ensure_ascii=False, indent=2))
    print("\nAnalyst signal keys:", list(report["analyst_signals"].keys()))
    print("Errors:", len(report["errors"]))
    print("Artifact:", out_path)
    if report["errors"]:
        print("\nError details:")
        for item in report["errors"]:
            print(f"- [{item['step']}] {item['error']}")


if __name__ == "__main__":
    main()
