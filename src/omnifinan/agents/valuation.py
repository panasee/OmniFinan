"""Enhanced Valuation Agent

Combines the best of both versions:
- Keeps the first version's interface and multi-method approach
- Incorporates the second version's improved algorithms
- Maintains console output and progress tracking from first version
"""

import json
from statistics import median

from langchain_core.messages import HumanMessage

from .. import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning


def valuation_agent(state: AgentState):
    """Run enhanced valuation across tickers and write signals back to `state`."""

    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    valuation_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status("valuation_agent", ticker, "Fetching financial data")

        # --- Historical financial metrics (pull 8 latest TTM snapshots for medians) ---
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=8,
        )
        if not financial_metrics:
            progress.update_status(
                "valuation_agent", ticker, "Failed: No financial metrics found"
            )
            continue
        most_recent_metrics = financial_metrics[0]

        # --- Fine‑grained line‑items (need two periods to calc WC change) ---
        progress.update_status("valuation_agent", ticker, "Gathering line items")
        line_items = search_line_items(ticker, period="ttm", limit=2)
        if len(line_items) < 2:
            progress.update_status(
                "valuation_agent", ticker, "Failed: Insufficient financial line items"
            )
            continue
        li_curr, li_prev = line_items[0], line_items[1]

        # ------------------------------------------------------------------
        # Enhanced Valuation models
        # ------------------------------------------------------------------
        wc_change = li_curr.working_capital - li_prev.working_capital
        growth_rate = most_recent_metrics.earnings_growth or 0.05

        # Improved Owner Earnings with dynamic growth decay
        owner_val = calculate_owner_earnings_value(
            net_income=li_curr.net_income,
            depreciation=li_curr.depreciation_and_amortization,
            capex=li_curr.capital_expenditure,
            working_capital_change=wc_change,
            growth_rate=growth_rate,
            required_return=0.15,
            margin_of_safety=0.25,
            num_years=5,
        )

        # Enhanced DCF with growth rate constraints
        dcf_val = calculate_intrinsic_value(
            free_cash_flow=li_curr.free_cash_flow,
            growth_rate=growth_rate,
            discount_rate=0.10,
            terminal_growth_rate=0.03,
            num_years=5,
        )

        # EV/EBITDA multiple valuation (from first version)
        ev_ebitda_val = calculate_ev_ebitda_value(financial_metrics)

        # Residual Income Model (from first version)
        rim_val = calculate_residual_income_value(
            market_cap=most_recent_metrics.market_cap,
            net_income=li_curr.net_income,
            price_to_book_ratio=most_recent_metrics.price_to_book_ratio,
            book_value_growth=most_recent_metrics.book_value_growth or 0.03,
        )

        # ------------------------------------------------------------------
        # Aggregate & signal with asymmetric thresholds
        # ------------------------------------------------------------------
        market_cap = get_market_cap(ticker, end_date)
        if not market_cap:
            progress.update_status(
                "valuation_agent", ticker, "Failed: Market cap unavailable"
            )
            continue

        # Method weights (sum to 1)
        method_values = {
            "dcf": {"value": dcf_val, "weight": 0.35},
            "owner_earnings": {"value": owner_val, "weight": 0.35},
            "ev_ebitda": {"value": ev_ebitda_val, "weight": 0.20},
            "residual_income": {"value": rim_val, "weight": 0.10},
        }

        # Calculate gaps with asymmetric thresholds
        for v in method_values.values():
            if v["value"] > 0:
                v["gap"] = (v["value"] - market_cap) / market_cap
                # Adjusted signal thresholds (10% undervalued for bullish, 20% overvalued for bearish)
                v["signal"] = (
                    "bullish"
                    if v["gap"] > 0.10
                    else "bearish"
                    if v["gap"] < -0.20
                    else "neutral"
                )
            else:
                v["gap"] = None
                v["signal"] = "neutral"

        # Calculate weighted gap (only include methods with valid values)
        valid_methods = [v for v in method_values.values() if v["gap"] is not None]
        total_weight = sum(v["weight"] for v in valid_methods)

        if total_weight == 0:
            progress.update_status(
                "valuation_agent", ticker, "Failed: All valuation methods zero"
            )
            continue

        weighted_gap = sum(v["weight"] * v["gap"] for v in valid_methods) / total_weight

        # Final signal determination
        signal = (
            "bullish"
            if weighted_gap > 0.10
            else "bearish"
            if weighted_gap < -0.20
            else "neutral"
        )
        confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100))

        # Prepare reasoning output
        reasoning = {
            m: {
                "signal": vals["signal"],
                "details": (
                    f"Value: ${vals['value']:,.2f}, Market Cap: ${market_cap:,.2f}, "
                    f"Gap: {vals['gap']:.1%}, Weight: {vals['weight'] * 100:.0f}%"
                ),
            }
            for m, vals in method_values.items()
            if vals["value"] > 0
        }

        valuation_analysis[ticker] = {
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        progress.update_status("valuation_agent", ticker, "Done")

    # ---- Emit message (for LLM tool chain) ----
    msg = HumanMessage(content=json.dumps(valuation_analysis), name="valuation_agent")
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(valuation_analysis, "Enhanced Valuation Analysis Agent")
    state["data"]["analyst_signals"]["valuation_agent"] = valuation_analysis
    return state | {"messages": [msg], "data": data}


#############################
# Enhanced Valuation Functions
#############################


def calculate_owner_earnings_value(
    net_income: float | None,
    depreciation: float | None,
    capex: float | None,
    working_capital_change: float | None,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    num_years: int = 5,
) -> float:
    """Improved owner-earnings valuation with dynamic growth decay."""
    # Data validation
    if not all(
        isinstance(x, (int, float))
        for x in [net_income, depreciation, capex, working_capital_change]
    ):
        return 0

    owner_earnings = net_income + depreciation - capex - working_capital_change
    if owner_earnings <= 0:
        return 0

    # Constrain growth rate to reasonable bounds (0-25%)
    growth_rate = min(max(growth_rate, 0), 0.25)

    # Calculate present value of forecast period with growth decay
    pv = 0.0
    for yr in range(1, num_years + 1):
        # Growth decays linearly over forecast period
        year_growth = growth_rate * (1 - yr / (2 * num_years))
        future = owner_earnings * (1 + year_growth) ** yr
        pv += future / (1 + required_return) ** yr

    # Terminal value with constrained terminal growth
    terminal_growth = min(growth_rate * 0.4, 0.03)  # Max 3% terminal growth
    term_val = (
        owner_earnings * (1 + growth_rate) ** num_years * (1 + terminal_growth)
    ) / (required_return - terminal_growth)
    pv_term = term_val / (1 + required_return) ** num_years

    intrinsic = pv + pv_term
    return intrinsic * (1 - margin_of_safety)


def calculate_intrinsic_value(
    free_cash_flow: float | None,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth_rate: float = 0.02,
    num_years: int = 5,
) -> float:
    """Enhanced DCF with growth rate constraints."""
    if free_cash_flow is None or free_cash_flow <= 0:
        return 0

    # Constrain growth rates
    growth_rate = min(max(growth_rate, 0), 0.25)  # 0-25% range
    terminal_growth_rate = min(growth_rate * 0.4, 0.03)  # Max 3% terminal growth

    # Forecast period with growth decay
    pv = 0.0
    for yr in range(1, num_years + 1):
        year_growth = growth_rate * (1 - yr / (2 * num_years))
        fcft = free_cash_flow * (1 + year_growth) ** yr
        pv += fcft / (1 + discount_rate) ** yr

    # Terminal value
    term_val = (
        free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)
    ) / (discount_rate - terminal_growth_rate)
    pv_term = term_val / (1 + discount_rate) ** num_years

    return pv + pv_term


def calculate_ev_ebitda_value(financial_metrics: list) -> float:
    """Implied equity value via median EV/EBITDA multiple (from first version)."""
    if not financial_metrics:
        return 0
    m0 = financial_metrics[0]
    if not (m0.enterprise_value and m0.enterprise_value_to_ebitda_ratio):
        return 0
    if m0.enterprise_value_to_ebitda_ratio == 0:
        return 0

    ebitda_now = m0.enterprise_value / m0.enterprise_value_to_ebitda_ratio
    med_mult = median(
        [
            m.enterprise_value_to_ebitda_ratio
            for m in financial_metrics
            if m.enterprise_value_to_ebitda_ratio
        ]
    )
    ev_implied = med_mult * ebitda_now
    net_debt = (m0.enterprise_value or 0) - (m0.market_cap or 0)
    return max(ev_implied - net_debt, 0)


def calculate_residual_income_value(
    market_cap: float | None,
    net_income: float | None,
    price_to_book_ratio: float | None,
    book_value_growth: float = 0.03,
    cost_of_equity: float = 0.10,
    terminal_growth_rate: float = 0.03,
    num_years: int = 5,
) -> float:
    """Residual Income Model (from first version with added validation)."""
    if not (
        market_cap and net_income and price_to_book_ratio and price_to_book_ratio > 0
    ):
        return 0

    book_val = market_cap / price_to_book_ratio
    ri0 = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0

    # Constrain growth rates
    book_value_growth = min(max(book_value_growth, 0), 0.15)
    terminal_growth_rate = min(book_value_growth * 0.4, 0.03)

    pv_ri = 0.0
    for yr in range(1, num_years + 1):
        ri_t = ri0 * (1 + book_value_growth) ** yr
        pv_ri += ri_t / (1 + cost_of_equity) ** yr

    term_ri = (
        ri0
        * (1 + book_value_growth) ** (num_years + 1)
        / (cost_of_equity - terminal_growth_rate)
    )
    pv_term = term_ri / (1 + cost_of_equity) ** num_years

    intrinsic = book_val + pv_ri + pv_term
    return intrinsic * 0.8  # 20% margin of safety
