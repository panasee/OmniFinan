# AI Boundary Spec

This document defines the coding boundary and execution rules for AI agents working in this repository.
It is intentionally strict and should be treated as an operating contract.

## 1) Source Policy Boundary
- Do not change macro fixed source policy unless explicitly requested.
- China macro source baseline: AkShare official channels.
- International macro source baseline: FRED / IMF / World Bank.
- Singapore can use stable fixed sources not required to be identical to CN/US source mix.
- One reliable provider per metric; avoid duplicate parallel providers for same metric.

## 1.1) Runtime Data Location Boundary
- Repo content scope:
  - code + static/cold embedded reference data only.
- Runtime hot data (forbidden in repo):
  - reports, request cache, datasets snapshots, temporary exports, runtime logs.
- Required runtime base:
  - `OMNIX_PATH/omnifinan/`
- Required runtime categories under base:
  - `request_cache/`, `datasets/`, `reports/`, `logs/`.

## 2) Data Model / Output Stability
- Preserve existing external output structure from unified APIs.
- For macro structured output, keep keys stable:
  - `meta`, `dimensions`, `metrics`, `chart_data`
- Preserve metric-level derived fields used by downstream analysis:
  - `yoy`, `mom`, `qoq`, `trend_short`, `trend_medium`, `volatility`.

## 3) Naming Compatibility Rules
- Do not silently break existing keys.
- When introducing canonical names, maintain backward-compatible aliases.
- Prefer aliasing via cloned payloads over duplicated data pulls.

## 4) Cache and Refresh Rules
- Request cache and dataset store have different roles and must both remain valid.
- Master macro scope must be authoritative for all query windows.
- Stale policy:
  - threshold = `3 * inferred cycle`;
  - fallback = `30 days` when cycle cannot be inferred.
- Anti-loop protection:
  - if non-empty master cache was refreshed within 24h, skip stale refresh and return local data.

## 5) Failure Semantics
- Do not return silent missing values when source is known unavailable.
- For unavailable fixed-source metrics, return explicit error payload:
  - `source = fixed_sources_unavailable`
  - clear `error` message.

## 6) Modification Constraints
- Do not broaden scope beyond user request.
- Do not revert unrelated user changes.
- Keep edits minimal and deterministic.
- Prefer additive compatibility over breaking refactors.
- Keep comments concise and technical.

## 7) Test / Verification Rules
- After macro logic changes, run at least:
  - `tests/test_macro_source_policy.py`
  - `tests/test_macro_structured.py`
  - `tests/test_macro_visualize.py`
- If tests fail due intentional policy change, update tests to match new policy semantics.

## 8) Performance Guardrails
- Avoid repeated full refresh loops when source returns no delta.
- Favor subset refresh only when necessary.
- Use local cached payload for repeated analysis/report generation.

## 9) Documentation Rules
- Human-facing doc and AI-facing boundary doc must be maintained as the two canonical markdown files.
- Regenerate content from current code behavior, not historical narrative docs.
