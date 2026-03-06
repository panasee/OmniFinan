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

Data-quality guardrail:
- For index options, ensure `underlying_price` is sourced from explicit quote fields (`underlyingPrice` / provider meta) before trusting analytics; strike-median fallback should be treated as degraded mode.

Compute:
- `RR25` = `risk_reversal_25d`
- `BF25` = `butterfly_25d`
- `iv_minus_hv`, `iv_to_hv_ratio`
- term state proxy: contango/backwardation (if futures available)
- `vvix_vix_ratio` (if VVIX available)
- `gamma_regime` (if net-GEX estimate available)

Signal logic:
- High positive crash-hedge demand: deep negative RR25 / steep put wing -> risk aversion
- IV >> HV and rising short-end stress -> risk_off tendency
- Futures curve flips to backwardation -> strong risk_off flag
- VIX low but VVIX rising (ratio uptrend) -> tail-risk early warning
- Negative gamma regime -> volatility amplification risk
- Total pressure scale + purity:
  - `total_abs_gex = sum(abs(gex_row))` captures total hedge pressure magnitude.
  - `gamma_purity = net_gex_nominal / total_abs_gex` captures directional purity vs cancellation.
  - small `|gamma_purity|` with large `total_abs_gex` implies strong call/put tug-of-war and potential regime instability.
- Gamma concentration index: `CI = sum(|g(K)|_top5) / sum(|g(K)|_all)`
  - `CI > 0.6`: strike pressure highly concentrated (magnet-like pin risk)
  - `CI < 0.2`: pressure distributed, volatility path tends to be smoother
- Gamma skew and tail warning:
  - `gamma_skew = |PutGEX| / |CallGEX|`
  - `put_gex_share = |PutGEX| / (|CallGEX| + |PutGEX|)`
  - if `put_gex_share > 0.7` and `net_gex_nominal < 0`, trigger `tail_risk_warning`
- Vanna/Charm estimated exposures (BS-derived, vectorized):
  - `vanna = phi(d1) * (-d2 / sigma)`
  - `vanna_exposure* = sign * vanna * OI * multiplier * spot * 0.01`
  - `charm = -phi(d1) * (r/(sigma*sqrt(T)) - d2/(2*T))`
  - `charm_exposure* = sign * charm * OI * multiplier * spot * (1/365)`
  - sign convention follows GEX (`call=+`, `put=-`) for dealer-flow style aggregation.
  - output now includes call/put split fields and `flow_interpretation` regime hints for downstream decision layers.

## Step B: KG News Diffusion

Inputs:
- company/index news stream (timestamped)
- extracted entities/relations per article

Core formula:

`S_total(t) = Σ_i (weight_i * centrality_i * sentiment_i * exp(-lambda * (t - t_i)))`

Recommended daily lambda:
- `lambda = 0.23 ~ 0.46` (roughly 1.5-3 day half-life)
- event-adaptive multiplier for macro days (e.g., NFP/FOMC): `lambda_event = lambda * m`, `m ∈ [1.1, 1.8]`

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

Fusion suggestion (base weights):
- option-implied: 0.45
- news diffusion: 0.30
- breadth/flow: 0.25

Implementation notes:
- Compute per-path normalized subscore in `[-1, +1]`:
  - `S_opt`, `S_news`, `S_breadth`
- Composite score:
  - `S = 0.45*S_opt + 0.30*S_news + 0.25*S_breadth`
- Map to regime:
  - `S >= +0.25 -> risk_on`
  - `S <= -0.25 -> risk_off`
  - otherwise `neutral`
- Confidence:
  - `confidence = min(1.0, abs(S))`
- `top_driver`:
  - choose path with max absolute weighted contribution among:
    - `0.45*S_opt`, `0.30*S_news`, `0.25*S_breadth`

Option-implied path enhancement (when available):
- If `vvix_vix_ratio` exists, fold into `S_opt` as tail-risk adjustment.
- If `gamma_regime == negative_gamma`, apply additional downside penalty to `S_opt`.

Produce:
- `regime`
- `confidence`
- `top_driver`
- `rationale` (short and structured)
- `missing_data` list

## 4) Output JSON Contract

```json
{
  "as_of": "YYYY-MM-DD",
  "data_cutoff_time": "YYYY-MM-DDTHH:mm:ssZ",
  "universe": ["SPY", "QQQ"],
  "regime": "risk_on|neutral|risk_off",
  "confidence": 0.0,
  "top_driver": "option_implied|news_diffusion|breadth_flow|null",
  "signals": {
    "option_implied": {
      "rr25": null,
      "bf25": null,
      "iv_minus_hv": null,
      "iv_to_hv_ratio": null,
      "vix_level": null,
      "vvix_level": null,
      "vvix_vix_ratio": null,
      "gamma_regime": "positive_gamma|negative_gamma|unknown",
      "vix_term_state": "contango|backwardation|unknown"
    },
    "news_diffusion": {
      "lambda": 0.3,
      "lambda_event_multiplier": 1.0,
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

Use `scripts/openclaw_market_math.py` (relative to this skill folder) for quick numeric processing:
- decayed sentiment (`S_total`)
- centrality-weighted decayed sentiment (with clamp)
- event-adaptive lambda helpers
- IV percentile
- IV/HV relation (annualization-aligned)
- VVIX/VIX ratio
- gamma regime
- breadth `% above MA200`
- simple term-structure state
- top-driver selection

Example:
```bash
python scripts/openclaw_market_math.py --mode demo
```

## 6) Execution Discipline for OpenClaw

1. Try OmniFinan endpoints first.
2. If any required field missing, search the web and/or use finance skills.
3. Normalize and continue; do not abort entire workflow.
4. Mark all inferred/proxy values and data gaps in output.
5. Keep outputs deterministic and structured.
6. Always emit `data_cutoff_time` and align option/news/breadth snapshots to avoid time-misalignment bias.
7. If option data is unsupported for the target universe, use explicit proxy map (e.g., index options) and annotate in `missing_data` / `sources`.

