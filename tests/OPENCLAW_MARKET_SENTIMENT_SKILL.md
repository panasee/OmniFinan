---
name: openclaw-market-sentiment
description: End-to-end market sentiment workflow for swing/medium-term trading in 2026, using option-implied signals, KG-style news diffusion, and breadth/flow diagnostics. Designed for OpenClaw tool-using LLMs.
---

# OpenClaw Market Sentiment Skill (OmniFinan-first)

## 1) Scope and Goal

Use this skill to produce a daily/weekly sentiment regime assessment for medium-term and swing trading.

Core paths:
1. Option-implied sentiment (skew, IV risk premium, term structure)
2. KG-style news diffusion (entity impact with exponential time decay)
3. Market breadth and flow divergence (participation + institutional flow proxies)

Output target:
- `regime`: `risk_on` | `neutral` | `risk_off`
- `confidence`: 0-1
- `signals`: structured diagnostics from all three paths
- `risk_flags`: list of divergence / stress warnings

## 2) Data Source Policy (Important)

### 2.1 Preferred source order

Always try OmniFinan APIs first. If missing/incomplete:
1. Search on the web (browser or search tools)
2. Use other available finance skills/tools
3. Merge into a single normalized structure before analysis

Never stop the full workflow only because one feed is missing. Mark gaps explicitly.

### 2.2 OmniFinan-first endpoints

Primary:
- `UnifiedDataService.get_stock_option_chain_analytics(...)`
- `UnifiedDataService.get_futures_option_chain_analytics(...)`
- `UnifiedDataService.get_macro_indicators(...)` (includes `us_vix`)
- `UnifiedDataService.get_company_news(...)`
- `UnifiedDataService.get_prices(...)`

Notes:
- A-share/HK equities: stock options unsupported in current provider stack; expect explicit unavailable payload.
- Crypto options routing is normalized to base (`BTC-USDT` -> `BTC`, `ETH-USD` -> `ETH`) in current code.

### 2.3 Missing-data fallback rules

If OmniFinan lacks required data:
- VIX/SVIX/term structure:
  - search official exchange/data pages or high-quality market data providers
- Breadth:
  - search index constituents + EOD prices to compute `% above MA200`
- Block flow:
  - use institution/block trade summaries when available; otherwise clearly mark as proxy/unavailable
- News KG:
  - if no graph data source, extract entities/relations directly from raw news with LLM and build temporary graph in memory

## 3) Workflow

## Step A: Option-Implied Sentiment

Inputs:
- option analytics payload (`analytics.summary`, `skew_by_expiry`, `term_structure`, `smile_by_expiry`, `implied_vs_realized`)
- VIX level series
- (optional) VIX futures front/next term structure (external fallback)

Compute:
- `RR25` = `risk_reversal_25d`
- `BF25` = `butterfly_25d`
- `iv_minus_hv`, `iv_to_hv_ratio`
- term state proxy: contango/backwardation (if futures available)

Signal logic:
- High positive crash-hedge demand: deep negative RR25 / steep put wing -> risk aversion
- IV >> HV and rising short-end stress -> risk_off tendency
- Futures curve flips to backwardation -> strong risk_off flag

## Step B: KG News Diffusion

Inputs:
- company/index news stream (timestamped)
- extracted entities/relations per article

Core formula:

`S_total(t) = Σ_i (weight_i * sentiment_i * exp(-lambda * (t - t_i)))`

Recommended daily lambda:
- `lambda = 0.23 ~ 0.46` (roughly 1.5-3 day half-life)

Propagation:
- map upstream -> midstream -> downstream entities
- apply lag penalty (e.g., 1-2 days) and attenuation factor

Outputs:
- per-entity decayed sentiment
- sector/chain aggregate sentiment
- top propagation edges contributing to current score

## Step C: Breadth & Flow

Inputs:
- universe close prices (or sampled proxies)
- optional block flow / institutional flow feed

Compute:
- `% above MA200`
- advance/decline proxy if available
- divergence flags:
  - index up but breadth down
  - index up but block flow net outflow

Outputs:
- breadth score (0-1)
- divergence severity

## Step D: Fusion and Regime

Fusion suggestion:
- option-implied: 0.45
- news diffusion: 0.30
- breadth/flow: 0.25

Produce:
- `regime`
- `confidence`
- `rationale` (short and structured)
- `missing_data` list

## 4) Output JSON Contract

```json
{
  "as_of": "YYYY-MM-DD",
  "universe": ["SPY", "QQQ"],
  "regime": "risk_on|neutral|risk_off",
  "confidence": 0.0,
  "signals": {
    "option_implied": {
      "rr25": null,
      "bf25": null,
      "iv_minus_hv": null,
      "iv_to_hv_ratio": null,
      "vix_level": null,
      "vix_term_state": "contango|backwardation|unknown"
    },
    "news_diffusion": {
      "lambda": 0.3,
      "score_total": 0.0,
      "top_entities": [],
      "top_edges": []
    },
    "breadth_flow": {
      "pct_above_ma200": null,
      "block_flow_polar": null,
      "divergence_flags": []
    }
  },
  "risk_flags": [],
  "missing_data": [],
  "sources": []
}
```

## 5) Script Helpers

Use `tests/openclaw_market_math.py` for quick numeric processing:
- decayed sentiment (`S_total`)
- IV percentile
- IV/HV relation
- breadth `% above MA200`
- simple term-structure state

Example:
```bash
python tests/openclaw_market_math.py --mode demo
```

## 6) Execution Discipline for OpenClaw

1. Try OmniFinan endpoints first.
2. If any required field missing, search the web and/or use finance skills.
3. Normalize and continue; do not abort entire workflow.
4. Mark all inferred/proxy values and data gaps in output.
5. Keep outputs deterministic and structured.

