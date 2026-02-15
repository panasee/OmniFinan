"""Fundamental valuation helpers used by analyst agents."""

from __future__ import annotations


def dcf_intrinsic_value(
    free_cash_flow: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.02,
    years: int = 5,
) -> float:
    """Compute a simplified DCF intrinsic value."""
    if discount_rate <= terminal_growth:
        raise ValueError("discount_rate must be greater than terminal_growth")
    value = 0.0
    fcf = free_cash_flow
    for year in range(1, years + 1):
        fcf *= 1 + growth_rate
        value += fcf / ((1 + discount_rate) ** year)
    terminal = (fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    value += terminal / ((1 + discount_rate) ** years)
    return value


def valuation_signal(current_price: float, intrinsic_value: float, margin_threshold: float = 0.15) -> str:
    """Map valuation gap to bullish/bearish/neutral signal."""
    if intrinsic_value <= 0:
        return "neutral"
    gap = (intrinsic_value - current_price) / intrinsic_value
    if gap >= margin_threshold:
        return "bullish"
    if gap <= -margin_threshold:
        return "bearish"
    return "neutral"
