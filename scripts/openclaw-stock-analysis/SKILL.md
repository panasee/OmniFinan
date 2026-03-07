---
name: openclaw-stock-analysis
description: End-to-end single-stock analysis workflow for swing/medium-term decisions in 2026, using OmniFinan-first price, fundamentals, options, macro, and news signals. Designed for OpenClaw tool-using LLMs.
---

# OpenClaw Stock Analysis Skill (OmniFinan-first)

## 1) Scope and Goal

Use this skill to produce a structured single-stock analysis for discretionary trading or research review.

Core paths:
1. Price structure and technical state
2. Fundamentals and valuation context
3. Option-implied positioning and risk
4. News, catalyst, and macro overlay

Output target:
- `stance`: `bullish` | `neutral` | `bearish`
- `confidence`: 0-1
- `signals`: structured diagnostics by path
- `risk_flags`: list of thesis-breaking risks

## 2) Data Source Policy (Important)

### 2.1 Preferred source order

Always try OmniFinan APIs first. If missing/incomplete:
1. Search on the web
2. Use other available finance skills/tools
3. Merge into a single normalized structure before analysis

Never stop the full workflow only because one feed is missing. Mark gaps explicitly.

### 2.2 OmniFinan-first endpoints

Primary:
- `UnifiedDataService.get_prices(...)`
- `UnifiedDataService.get_financial_metrics(...)`
- `UnifiedDataService.get_line_items(...)`
- `UnifiedDataService.get_company_news(...)`
- `UnifiedDataService.get_macro_indicators(...)`
- `UnifiedDataService.get_stock_option_chain_analytics(...)`
- `UnifiedDataService.get_stock_option_gex(...)`

Notes:
- A-share/HK equities: stock options unsupported in current provider stack; expect explicit unavailable payload.
- Option `iv` inputs are normalized to decimal volatility before Greeks/GEX computation.

### 2.3 Missing-data fallback rules

If OmniFinan lacks required data:
- Technicals:
  - compute from available price history or fetch external EOD data
- Fundamentals:
  - use company filings, investor relations, or high-quality market data pages
- Options:
  - fall back to index/sector ETF option proxy only when clearly labeled
- News/catalysts:
  - use recent reliable news sources and mark inferred impact explicitly

## 3) Workflow

## Step A: Price Structure and Technical State

Inputs:
- daily price history
- optional benchmark/sector ETF prices

Compute:
- moving averages: `MA20`, `MA100`, `MA200`
- `RSI` (default period = 14)
- `MACD` (default EMA params = 12, 26, 9)
- trend state (short/medium horizon)
- recent returns and realized volatility
- support/resistance or congestion zones
- relative strength vs benchmark when available

Outputs:
- technical bias
- momentum state
- key levels
- technical indicator snapshot

## Step B: Fundamentals and Valuation Context

Inputs:
- financial metrics
- line items
- optional valuation outputs or external references

Compute:
- growth quality
- profitability / balance-sheet resilience
- valuation stretch vs growth quality

Outputs:
- fundamental score
- valuation posture
- primary balance-sheet or earnings risks

## Step C: Option-Implied Positioning and Risk

Inputs:
- option analytics payload
- GEX payload when available

Compute:
- ATM IV / IV-HV relationship
- skew and term structure
- GEX regime and key strikes
- gamma flip / call wall / put wall when available

Outputs:
- options sentiment
- positioning pressure
- volatility regime

## Step D: News, Catalyst, and Macro Overlay

Inputs:
- company news
- relevant macro indicators
- event calendar context if available

Compute:
- recent catalyst map
- macro sensitivity
- narrative alignment or divergence vs price

Outputs:
- catalyst bias
- macro headwind/tailwind assessment
- event-risk flags

## Step E: Fusion and Final Stance

Fusion suggestion (base weights):
- technicals: 0.30
- fundamentals/valuation: 0.30
- options: 0.20
- news/macro: 0.20

Produce:
- `stance`
- `confidence`
- `top_driver`
- `rationale`
- `missing_data`

## 4) Output JSON Contract

```json
{
  "as_of": "YYYY-MM-DD",
  "analysis_type": "stock_analysis",
  "subject": {
    "ticker": "AAPL"
  },
  "ticker": "AAPL",
  "summary": {
    "primary_call": "bullish|neutral|bearish",
    "confidence": 0.0,
    "top_driver": "technicals|fundamentals|options|news_macro|null"
  },
  "stance": "bullish|neutral|bearish",
  "confidence": 0.0,
  "top_driver": "technicals|fundamentals|options|news_macro|null",
  "signals": {
    "technicals": {
      "trend_short": null,
      "trend_medium": null,
      "realized_volatility": null,
      "moving_averages": {
        "ma20": null,
        "ma100": null,
        "ma200": null
      },
      "rsi": null,
      "macd": {
        "macd_line": null,
        "signal_line": null,
        "histogram": null
      },
      "support_levels": [],
      "resistance_levels": []
    },
    "fundamentals": {
      "growth_quality": null,
      "profitability_quality": null,
      "balance_sheet_risk": null,
      "valuation_posture": "cheap|fair|rich|unknown"
    },
    "options": {
      "atm_iv": null,
      "iv_minus_hv": null,
      "gamma_regime": "positive_gamma|negative_gamma|unknown",
      "gamma_flip_price": null,
      "call_wall": null,
      "put_wall": null
    },
    "news_macro": {
      "news_bias": null,
      "macro_bias": null,
      "key_catalysts": []
    }
  },
  "risk_flags": [
    {
      "code": "event_risk",
      "severity": "low|medium|high",
      "message": ""
    }
  ],
  "missing_data": [
    {
      "field": "signals.options.gamma_flip_price",
      "reason": "",
      "impact": "low|medium|high"
    }
  ],
  "sources": [
    {
      "name": "omnifinan",
      "type": "internal|external|proxy",
      "target": "AAPL",
      "note": ""
    }
  ]
}
```

## 5) Script Helpers

Use `scripts/openclaw_stock_technicals.py` (relative to this skill folder) for quick technical indicator processing:
- `simple_moving_average(closes, window)`
- `moving_average_pack(closes)` -> `ma20`, `ma100`, `ma200`
- `relative_strength_index(closes, period=14)`
- `ema_series(closes, span)`
- `macd(closes, fast_span=12, slow_span=26, signal_span=9)`
- `technical_snapshot(closes, ...)`

Example:
```bash
python scripts/openclaw_stock_technicals.py --mode demo
```

## 6) Execution Discipline for OpenClaw

1. Try OmniFinan endpoints first.
2. Keep analysis ticker-specific; do not drift into broad market commentary unless needed for context.
3. If one path is missing, continue with explicit gaps.
4. Keep outputs structured and deterministic.
5. Mark proxy-based option or macro interpretations clearly.
6. Separate observed facts from inferred thesis statements.
