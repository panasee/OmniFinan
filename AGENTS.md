# Repository Guidelines

## Scope
This file is the single operating guide for AI and human contributors in `omnifinan`.  
It reflects current code behavior, especially macro data sourcing, cache semantics, output stability, and verification rules.

## Source Policy Boundary
- Do not change macro fixed-source policy unless explicitly requested.
- China macro baseline: AkShare official channels.
- International macro baseline: FRED / IMF / World Bank.
- Singapore may use stable fixed sources not identical to CN/US mix.
- Keep one reliable provider per metric; avoid duplicate parallel providers.

## Runtime Data Boundary
- Repository stores code and cold/static assets only.
- Runtime hot data must not be written into the repo.
- Runtime base path: `OMNIX_PATH/omnifinan/`.
- Required runtime directories:
- `request_cache/`
- `datasets/`
- `reports/`
- `logs/`
- Master macro dataset path:
- `OMNIX_PATH/omnifinan/datasets/macro_indicators_history/fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank__master.json`

## Macro Cache and Refresh Rules
- `UnifiedDataService.get_macro_indicators` is master-first:
- use request cache first;
- if missing, restore from latest dataset master snapshot;
- query windows are filtered subsets of one master payload.
- Stale policy is series-level:
- threshold = `3 * inferred_cycle_days`;
- fallback = `30 days` if cycle cannot be inferred.
- Anti-loop guard:
- if non-empty cache refreshed within 24h, skip stale refresh and return local data.
- Keep history snapshots for longitudinal tracking.

## Output and Compatibility Contract
- Preserve unified API external structures.
- Structured macro output keys must stay stable:
- `meta`, `dimensions`, `metrics`, `chart_data`.
- Preserve derived metric fields used downstream:
- `yoy`, `mom`, `qoq`, `trend_short`, `trend_medium`, `volatility`.
- Maintain backward-compatible aliases; do not silently break existing keys.
- Prefer alias cloning over duplicate pulls.
- For unavailable fixed-source metrics, return explicit errors:
- `source = fixed_sources_unavailable`
- clear `error` message
- never silent NA.

## Operational and Modification Constraints
- Do not broaden scope beyond the user request.
- Do not revert unrelated local changes.
- Keep edits minimal, deterministic, and technically concise.
- Avoid repeated full refresh loops when sources have no delta.
- Prefer subset refresh when supported.
- Prefer local cached payload for repeated analysis/report generation.

## Required Verification
After macro logic changes, run:
- `pytest tests/test_macro_source_policy.py`
- `pytest tests/test_macro_structured.py`
- `pytest tests/test_macro_visualize.py`

## Typical Macro Usage
1. `get_macro_indicators(start_date, end_date)`
2. `get_macro_indicators_structured(start_date, end_date)`
3. Build reports from `metrics` fields (`yoy`, `mom`, trends, volatility, source/error).
