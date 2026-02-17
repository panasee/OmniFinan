# OmniFinan Human Guide

## Scope
This guide reflects the current implementation in code (not historical docs).
It focuses on macro data ingestion, cache behavior, structured output, and daily usage.

## Core Macro Data Policy
- China fixed sources: AkShare official endpoints.
- International fixed sources: FRED / IMF / World Bank.
- Singapore is allowed to use stable non-identical fixed sources where needed, but still follows one reliable source per metric.

## Storage Layout
- Repository policy:
  - This repo stores code and curated cold/static assets only.
  - Runtime hot data must not be written into the repo.
- Request cache: `OMNIX_PATH/omnifinan/request_cache/`
  - Fast response cache for concrete requests.
- Dataset store: `OMNIX_PATH/omnifinan/datasets/`
  - Persistent historical snapshots and reusable master data.
- Runtime reports/logs:
  - `OMNIX_PATH/omnifinan/reports/`
  - `OMNIX_PATH/omnifinan/logs/`
- Master macro dataset file:
  - `OMNIX_PATH/omnifinan/datasets/macro_indicators_history/fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank__master.json`

## Master-First Mechanism
`UnifiedDataService.get_macro_indicators` uses a master-scope key.
- Query windows are filtered subsets from the same master payload.
- Service first tries request cache.
- If request cache is missing, it restores from latest dataset master snapshot.
- Master snapshot history is appended for longitudinal tracking.

## Refresh / Stale Logic
Stale detection is series-level:
- If cycle can be inferred from observation intervals:
  - stale threshold = `3 * cycle_days`.
- If cycle cannot be inferred:
  - stale threshold = `30 days` fallback.

To avoid repeated expensive pulls:
- If cache has non-empty data and file was refreshed within 24h, stale refresh is skipped and local data is returned.

## Structured Output for Analysis
`get_macro_indicators_structured` returns:
- `meta`
- `dimensions`
- `metrics`
- `chart_data`

Each metric includes analysis-ready fields such as:
- `latest_value`, `latest_date`, `obs_count`, `frequency`
- `mom`, `qoq`, `yoy`
- `trend_short`, `trend_medium`, `volatility`
- `source`, `error`

## Naming Compatibility (Important)
Current code supports compatibility aliases for common naming differences:
- `us_term_spread_10y2y` (alias of `us_term_spread_10y_2y`)
- `us_real_interest_rate` (alias of `us_real_rate_10y` source)
- `sg_gdp_yoy` (alias of `sg_gdp_growth` source)
- `sg_cpi_yoy` (alias of `sg_inflation_cpi` source)

For unavailable metrics under fixed source policy, service returns explicit error payload instead of silent NA.

## Typical Usage
1. Pull macro raw:
- `get_macro_indicators(start_date, end_date)`
2. Pull structured macro for AI/human analysis:
- `get_macro_indicators_structured(start_date, end_date)`
3. Build reports from structured `metrics` fields (`yoy/mom/trend`) directly.

## Operational Notes
- If some series remain stale but source has no updates, this is expected.
- Prefer examining `source`, `latest_date`, and `error` before changing logic.
- Keep one source per metric to avoid duplicate/conflicting values.
