"""CLI entry for OmniFinan presentation layer."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime

import questionary
from colorama import Fore, Style
from dateutil.relativedelta import relativedelta
from pyomnix.consts import OMNIX_PATH

from ..core.workflow import create_workflow, run_hedge_fund
from ..utils.analysts import ANALYST_ORDER
from ..utils.display import print_trading_output
from ..visualize import save_graph_as_png


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--margin-requirement", type=float, default=0.0)
    parser.add_argument("--tickers", type=str, required=True)
    parser.add_argument("--start-date", type=str)
    parser.add_argument("--end-date", type=str)
    parser.add_argument("--language", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--llm-seed", type=int)
    parser.add_argument("--deterministic-mode", action="store_true")
    parser.add_argument("--non-deterministic-mode", action="store_true")
    parser.add_argument("--show-reasoning", action="store_true")
    parser.add_argument("--show-agent-graph", action="store_true")
    parser.add_argument(
        "--data-provider",
        type=str,
        default="akshare",
        choices=["akshare", "finnhub", "yfinance", "sec_edgar"],
    )
    args = parser.parse_args()

    tickers = [ticker.strip() for ticker in args.tickers.split(",")]
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[questionary.Choice(display, value=value) for display, value in ANALYST_ORDER],
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
    ).ask()
    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    selected_analysts = choices

    llm_choice = "deepseek-chat"
    provider_api = "deepseek"
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if args.start_date:
        start_date = args.start_date
    else:
        start_date = (datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)).strftime(
            "%Y-%m-%d"
        )

    portfolio = {
        "cash": args.initial_cash,
        "margin_requirement": args.margin_requirement,
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

    if args.show_agent_graph:
        workflow = create_workflow(selected_analysts)
        app = workflow.compile()
        file_path = OMNIX_PATH / "financial" / "agent-graph"
        file_path.mkdir(parents=True, exist_ok=True)
        save_graph_as_png(app, file_path / "graph.png")
    print(
        f"Selected model: {Fore.GREEN}{llm_choice}{Style.RESET_ALL} ({provider_api})"
    )
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=llm_choice,
        provider_api=provider_api,
        language=args.language or "Chinese",
        temperature=args.temperature,
        llm_seed=args.llm_seed,
        deterministic_mode=(False if args.non_deterministic_mode else (True if args.deterministic_mode else None)),
        data_provider=args.data_provider,
    )
    print_trading_output(result)
