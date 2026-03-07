---
name: openclaw-quanta-dig
description: End-to-end quantitative factor mining workflow for 2026, using OmniFinan-first panel data, factor generation, IC diagnostics, and cross-sectional backtests. Designed for OpenClaw tool-using LLMs.
---

# OpenClaw Quanta Dig Skill (OmniFinan-first)

## 1) Scope and Goal

Use this skill to build, test, and review quantitative alpha factors for cross-sectional research.

Core paths:
1. Panel data preparation
2. Factor generation and normalization
3. Research route selection (`non_ml` default, `ml` optional)
4. IC / RankIC diagnostics
5. Cross-sectional backtest and robustness review

Output target:
- `research_verdict`: `promising` | `mixed` | `rejected`
- `confidence`: 0-1
- `factor_report`: structured diagnostics for selected factors
- `risk_flags`: list of data-mining or implementation risks

## 2) Data Source Policy (Important)

### 2.1 Preferred source order

Always try OmniFinan APIs first. If missing/incomplete:
1. Search local code/docs for reusable panel builders
2. Use external data only when explicitly required
3. Normalize all factor inputs before evaluation

Never over-claim a factor from partial data. Mark gaps explicitly.

Default route rule:
- If the user does not explicitly request machine learning, use the `non_ml` route.
- Only use the `ml` route when the user asks for machine learning, model training, feature importance, or prediction-based ranking.

### 2.2 OmniFinan-first endpoints and modules

Primary:
- `UnifiedDataService.get_prices(...)`
- `omnifinan.analysis.factor_mining`
- `omnifinan.analysis.factor_backtest`
- `omnifinan.research.factors`

Useful functions:
- `add_candidate_factors(...)`
- `zscore_by_date(...)`
- `daily_ic(...)`
- `evaluate_factors(...)`
- `apply_custom_factors(...)`
- `build_cross_sectional_weights(...)`
- `run_daily_backtest(...)`
- `perf_stats(...)`

Qlib-style mapping:
- `data handler / processor` -> clean panel, date filtering, normalization, leakage guards
- `feature set` -> built-in candidates + custom factors + DSL expressions
- `label` -> forward return such as `fwd_ret_5`
- `model` -> optional route, typically gradient boosting / tree models
- `strategy + backtest` -> cross-sectional ranking, quantile portfolio, delayed execution, transaction costs

### 2.3 Missing-data fallback rules

If OmniFinan lacks required data:
- panel prices:
  - fetch external EOD data only if user scope allows
- factor references:
  - inspect local research modules before introducing new formulas
- backtest assumptions:
  - prefer simple, explicit assumptions over hidden heuristics

## 2.4 Qlib-style best practices to follow

Follow these common Qlib-style habits unless the user asks otherwise:
- Build a clean daily panel first; do not mix data cleaning with factor evaluation.
- Keep the feature pipeline explicit:
  - raw prices/volume
  - derived factors
  - normalization / winsorization / missing handling
- Use train / validation / test splits by date, not random row shuffling.
- Keep the label horizon explicit, such as `fwd_ret_5` or `fwd_ret_10`.
- Evaluate both signal quality and tradability:
  - IC / RankIC
  - turnover
  - transaction-cost-aware backtest
  - drawdown / Sharpe / win-rate
- Guard against leakage:
  - use only information available at prediction time
  - preserve one-period signal delay in backtest
- Prefer simple baselines first, then more complex models only if they add clear value.
- Record universe, benchmark, date split, cost rate, and rebalance convention in output.

## 3) Workflow

## Step A: Panel Preparation

Inputs:
- universe list
- date range
- daily OHLCV panel

Compute:
- clean panel
- survivorship / missingness review
- forward return label setup
- optional benchmark alignment
- optional winsorization / clipping / z-score preparation

Outputs:
- ready-to-mine panel
- data quality summary

## Step B: Factor Generation

Inputs:
- cleaned panel
- built-in factor functions
- optional custom factor definitions

Compute:
- candidate factors
- cross-sectional z-score normalization
- factor metadata table
- optional qlib-style expression factors from `research.factors`

Outputs:
- factor matrix
- normalized factor columns

## Step C: Route Selection

Select one of two research routes:

### Route 1: `non_ml` (default)

Use when:
- user asks for factor mining, IC testing, or quantile backtest
- user does not explicitly ask for model training
- interpretability and faster iteration are preferred

Workflow:
- generate factor candidates
- normalize cross-sectionally
- compute IC / RankIC
- pick a small factor set
- form score by simple aggregation or ranked combination
- run long-only or long-short backtest

Typical score construction:
- single-factor ranking
- equal-weighted multi-factor z-score sum
- manually signed factor basket

### Route 2: `ml`

Use only when:
- user explicitly asks for machine learning, prediction models, or feature importance
- enough history and universe breadth are available

Preferred mainstream model family:
- default first choice:
  - tree-based tabular models
  - gradient boosting / LightGBM-style ranking or regression
- secondary choice:
  - MLP / TabNet-style tabular deep models
- advanced choice only when user explicitly wants it:
  - sequence or transformer-style models for multi-horizon temporal structure

Default model stance:
- For structured cross-sectional alpha research, start with `LightGBM`-style models before deep sequence models.
- Treat deep models as an escalation path, not the baseline.

Workflow:
- generate feature matrix
- split by date into train / validation / test
- fit model to forward returns or cross-sectional ranks
- produce prediction score per date-symbol
- backtest prediction ranking with the same cost-aware framework

ML guardrails:
- no random split across dates
- no target leakage from future returns
- always compare against `non_ml` baseline
- report feature importance only as supportive evidence, not final proof
- prefer rolling or walk-forward retraining over one-shot static fitting when enough data exists
- for panel data, preserve date-wise cross-sectional structure during evaluation

Output:
- `route_selected`
- `route_reason`
- `baseline_comparison`

## Step D: IC Diagnostics

Inputs:
- factor matrix
- forward return labels

Compute:
- IC mean/std/IR
- RankIC mean/std/IR
- observation counts and stability checks

Outputs:
- ranked factor report
- persistence / fragility notes

## Step E: Backtest and Robustness

Inputs:
- chosen factor scores
- portfolio construction assumptions

Compute:
- long-only or long-short weights
- turnover and cost-aware performance
- drawdown / Sharpe / win-rate

Outputs:
- backtest summary
- implementation friction notes
- robustness warnings

## Step F: Research Verdict

Decision guidance:
- `promising`: IC and backtest both hold with acceptable turnover/cost
- `mixed`: some signal quality, but unstable or implementation-heavy
- `rejected`: weak IC, weak backtest, or obvious overfit behavior

Produce:
- `research_verdict`
- `confidence`
- `top_driver`
- `next_actions`

## 4) Qlib-style Route Templates

### 4.1 Non-ML template

Recommended default:
1. build panel
2. run `add_candidate_factors(...)`
3. choose label such as `fwd_ret_5`
4. run `zscore_by_date(...)`
5. run `evaluate_factors(...)`
6. select top interpretable factors
7. combine scores
8. run `build_cross_sectional_weights(...)`
9. run `run_daily_backtest(...)`
10. summarize with `perf_stats(...)`

#### 4.1.1 Default non-ML screening rules

Use these as the default filter unless the user specifies custom thresholds:
- minimum `obs_days >= 60`
- keep factors with positive signal quality:
  - `rank_ic_mean > 0`
- prefer stable factors:
  - `rank_ic_ir > 0.2`
- de-prioritize weak factors:
  - reject if `abs(rank_ic_mean) < 0.01` and `abs(ic_mean) < 0.01`
- if too many factors survive, keep the top `3-5` by `rank_ic_mean`

If no factor passes the preferred threshold:
- keep the best `1-2` factors by `rank_ic_mean`
- mark the route as weak / exploratory in `risk_flags`

#### 4.1.2 Default factor basket construction

When multiple factors are selected, prefer this order:
1. align factor sign using `rank_ic_mean`
   - if `rank_ic_mean < 0`, multiply the factor by `-1`
2. use z-scored factors
3. build combined score as equal-weight average of signed z-scores

Recommended formula:

`score_ticker,date = mean(signed_factor_1_z, signed_factor_2_z, ...)`

Fallback options:
- single-factor ranking if only one factor survives
- manually weighted basket only if the user explicitly requests it

#### 4.1.3 Default backtest assumptions

Use these defaults unless the user specifies otherwise:
- rebalance frequency: daily
- signal delay: 1 day (already enforced by backtest implementation)
- portfolio style:
  - `long_short=True` if universe breadth is sufficient
  - otherwise `long_short=False`
- quantile: `0.2`
- `min_universe: 8`
- `cost_rate: 0.001`

Always report:
- `annual_return`
- `annual_vol`
- `sharpe`
- `max_drawdown`
- `win_rate`
- average or median turnover impression from backtest output

#### 4.1.4 Default non-ML verdict mapping

Use the following default interpretation:

- `promising`
  - at least one selected factor has `rank_ic_mean > 0.02`
  - and `rank_ic_ir > 0.3`
  - and backtest `sharpe > 0.8`
  - and `max_drawdown > -0.25`

- `mixed`
  - factor IC is positive but weak or unstable
  - or backtest is profitable but turnover / drawdown is unattractive
  - or only one weak factor survives fallback selection

- `rejected`
  - selected factors show weak or inconsistent IC
  - or backtest metrics are clearly unattractive
  - or signal quality disappears after cost-aware testing

Confidence guidance:
- start from signal quality:
  - stronger `rank_ic_mean` and `rank_ic_ir` -> higher confidence
- then reduce confidence for:
  - low `obs_days`
  - concentrated dependence on one factor
  - excessive turnover
  - weak out-of-sample narrative

### 4.2 ML template

Recommended optional route:
1. build panel
2. create factor/features matrix
3. date split into train / validation / test
4. train a `LightGBM`-style baseline model
5. tune lightly on validation
6. freeze model and score test period
7. convert predictions to cross-sectional weights
8. run the same cost-aware backtest
9. compare against `non_ml` baseline
10. only if justified, escalate to deeper models

#### 4.2.1 Default ML baseline

Use this as the default ML route unless the user requests a different model:
- model family: gradient boosting decision trees
- default mental model: `LightGBM`
- target style:
  - regression on `fwd_ret_5` or `fwd_ret_10`
  - or ranking-oriented objective when available
- feature input:
  - built-in candidates from `add_candidate_factors(...)`
  - custom factor columns
  - optional DSL-derived features from `research.factors`

Why this is the default:
- strong baseline for tabular panel data
- robust on medium-size quantitative datasets
- easier to interpret and debug than deep sequence models
- aligns with common Qlib benchmark practice

#### 4.2.2 Date split and retraining policy

Default evaluation policy:
- use chronological split only
- preferred split:
  - `train`
  - `validation`
  - `test`
- if enough history exists, prefer walk-forward or rolling retraining

Recommended pattern:
1. fit on train
2. validate on next segment
3. freeze hyperparameters
4. score test segment
5. optionally repeat in rolling windows and aggregate

Avoid:
- random row shuffle
- mixing future dates into training folds
- evaluating on the same segment used for model selection

#### 4.2.3 Prediction-to-portfolio conversion

Default conversion:
- model output becomes cross-sectional `score_col`
- rank securities by prediction within each date
- use the same portfolio construction logic as `non_ml`
- keep costs, turnover, and signal delay rules identical for fair comparison

This is important:
- the ML route should differ in score generation
- not in relaxed backtest assumptions

#### 4.2.4 Escalation path beyond GBDT

Only escalate beyond the default baseline if the user explicitly requests it or if the baseline is clearly insufficient.

Escalation order:
1. `LightGBM` / GBDT baseline
2. tabular deep model (`MLP`, `TabNet`-style)
3. temporal deep model (`LSTM`, `GRU`, `Transformer`-style)

Require explicit justification before escalation:
- longer history
- richer feature set
- clear non-linear or temporal interaction hypothesis
- baseline underperformance that cannot be explained by data quality

#### 4.2.5 Default ML verdict interpretation

Use these defaults:
- `promising`
  - ML route beats `non_ml` baseline on test-period Sharpe or annual return
  - without meaningfully worse drawdown or turnover
  - and model behavior is stable across date segments

- `mixed`
  - ML route improves one metric but worsens stability, drawdown, or turnover
  - or advantage vs baseline is too small / fragile

- `rejected`
  - ML route fails to beat baseline
  - or signs of leakage / instability / regime dependence are too strong

### 4.3 Common feature families

Good default families to mention or build from:
- momentum:
  - short / medium returns
  - moving-average gaps
- reversal:
  - short-term mean reversion
- volatility:
  - rolling std
  - range / amplitude
- liquidity / participation:
  - volume ratio
  - turnover spike proxies
- relative strength:
  - asset vs benchmark / sector

Avoid claiming Qlib-level “Alpha158/Alpha360 parity” unless the feature coverage is actually implemented.

## 5) Output JSON Contract

```json
{
  "as_of": "YYYY-MM-DD",
  "analysis_type": "quanta_dig",
  "subject": {
    "universe": []
  },
  "universe": [],
  "route_selected": "non_ml|ml",
  "route_reason": "",
  "summary": {
    "primary_call": "promising|mixed|rejected",
    "confidence": 0.0,
    "top_driver": "ic|rank_ic|backtest|robustness|null"
  },
  "research_verdict": "promising|mixed|rejected",
  "confidence": 0.0,
  "top_driver": "ic|rank_ic|backtest|robustness|null",
  "factor_report": {
    "label": "fwd_ret_5",
    "candidates": [],
    "selected": [],
    "ic_summary": [],
    "score_construction": {
      "method": "equal_weight_signed_zscore",
      "selected_factors": [],
      "sign_map": {}
    },
    "backtest_summary": {
      "annual_return": null,
      "annual_vol": null,
      "sharpe": null,
      "max_drawdown": null,
      "win_rate": null,
      "turnover_comment": null
    },
    "baseline_comparison": {}
  },
  "risk_flags": [
    {
      "code": "overfit_risk",
      "severity": "low|medium|high",
      "message": ""
    }
  ],
  "missing_data": [
    {
      "field": "factor_report.baseline_comparison",
      "reason": "",
      "impact": "low|medium|high"
    }
  ],
  "sources": [
    {
      "name": "omnifinan",
      "type": "internal|external|proxy",
      "target": "panel_prices",
      "note": ""
    }
  ]
}
```

## 6) Execution Discipline for OpenClaw

1. Prefer existing OmniFinan factor and backtest modules before inventing new infrastructure.
2. If the user does not request ML, stay on the `non_ml` route.
3. Keep factor definitions explicit and reproducible.
4. Separate in-sample evidence from robustness claims.
5. Report turnover, costs, and drawdown; do not rely on Sharpe alone.
6. For ML route, always include a simple non-ML baseline comparison.
7. Mark external data, custom formulas, and proxy assumptions clearly.
8. Default to concise, machine-readable summaries first; deeper commentary second.
9. In the default `non_ml` route, prefer a small, interpretable factor basket over a large unstable combination.
10. In the `ml` route, default to a `LightGBM`-style tabular baseline before considering deeper sequence models.
