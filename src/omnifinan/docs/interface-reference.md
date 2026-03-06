# Interface Reference

This document is the canonical interface contract reference for non-macro APIs.
For macro series definitions and macro structured schema details, see
[`macro-reference.md`](macro-reference.md).

Read this file when tasks require:
- Provider method contracts (`DataProvider` / `UnifiedDataService`)
- Option chain and option analytics return schema
- Compatibility rules for unavailable options sources
- Supported orchestration entrypoints

## Provider Registry

| Provider | Name strings | Markets | Capabilities |
|----------|-------------|---------|-------------|
| AkShare | `"akshare"` | CN, US, HK | Prices, financials, A-share raw news, macro (CN+intl) |
| YFinance | `"yfinance"`, `"yf"`, `"yahoo"` | US, global, crypto | Prices, financials, news |
| SEC EDGAR | `"sec_edgar"`, `"sec"` | US | Financial metrics, line items from XBRL |

Provider factory:
- `create_data_provider("akshare")`

## DataProvider Contract

All providers implement:
- `get_prices(ticker, start_date, end_date, interval) -> list[Price]`
- `get_financial_metrics(ticker, end_date, period, limit) -> list[FinancialMetrics]`
- `search_line_items(ticker, period, limit) -> list[LineItem]`
- `get_company_news_raw(ticker, start_date, end_date, limit) -> list[CompanyNews]`
  - Provider-level contract only. This is raw article discovery, not integrated event output.
- `get_insider_trades(ticker, end_date, start_date, limit) -> list[InsiderTrade]`
- `get_market_cap(ticker, end_date) -> float | None`
- `get_macro_indicators(start_date, end_date) -> dict`

## UnifiedDataService API Path

For non-LLM analytics tasks, prefer direct `UnifiedDataService` APIs:
- `get_prices`
- `get_financial_metrics`
- `get_line_items`
- `get_company_news`
- `get_insider_trades`
- `get_macro_indicators`
- `get_macro_indicators_structured`
- `get_stock_option_chain`
- `get_futures_option_chain`
- `get_stock_option_chain_analytics`
- `get_futures_option_chain_analytics`

## Option Analytics Contract

Use first when tasks request IV/skew/term-structure/Greeks:
- `UnifiedDataService.get_stock_option_chain_analytics(...)`
- `UnifiedDataService.get_futures_option_chain_analytics(...)`

Market compatibility:
- China A-share / HK equity tickers currently have no options support in provider stack.
- In that case, stock-option APIs return explicit unavailability:
  - `meta.source = "fixed_sources_unavailable"`
  - `meta.error` with clear reason
- Do not treat this as a fatal workflow error.
- Futures option chains are also explicit unavailable in the current moomoo-only provider stack.
- Crypto pair symbols are normalized to base asset for options endpoints:
  - `BTC-USDT`, `BTC-USD`, `BTCUSDT` -> `BTC`
  - `ETH-USDT`, `ETH-USD`, `ETHUSDT` -> `ETH`

Return contract:
- `meta` (inherits chain metadata + `analytics_version`)
- `data` (raw option rows)
- `raw` (provider raw payload)
- `analytics.summary` (`option_count`, `enriched_count`, `underlying_price`, `median_iv`)
- `analytics.surface` (per-contract normalized metrics with IV/Greeks)
- `analytics.term_structure` (ATM IV by expiry)
- `analytics.skew_by_expiry` (`risk_reversal_25d`, `butterfly_25d`, ATM IV)
- `analytics.smile_by_expiry` (IV smile points by strike/moneyness per expiry)
- `analytics.max_pain` (overall and per-expiry max pain strike)
- `analytics.levels` (primary support/resistance from put/call OI walls)
- `analytics.implied_vs_realized` (`current_atm_iv`, `historical_volatility`, `iv_minus_hv`, `iv_to_hv_ratio`)
- `analytics.summary.iv_historical_percentile` (requires `iv_history` input)
- `analytics.errors` (explicit calculation issues)

## Orchestration Interfaces

These remain supported when task intent is end-to-end orchestration:
- `omnifinan.run_hedge_fund(...)`
- `omnifinan.core.workflow.run_hedge_fund(...)`
- `omnifinan.backtester.Backtester`
- `omnifinan.presentation.api.create_app()`
- `omnifinan.visualize.StockFigure`
- `omnifinan.visualize.create_macro_figure`

## Compatibility Rules

Keep external contracts stable:
- Top-level result keys (for example `decisions`, `analyst_signals`)
- Macro structured keys (`meta`, `dimensions`, `metrics`, `chart_data`)

## Company News Contract

Public `get_company_news(...)` now returns integrated news events instead of raw article rows.

- Signature:
  - `get_company_news(ticker, start_date=None, end_date=None, limit=10) -> list[IntegratedNewsEvent]`
- Internal raw discovery path:
  - provider-layer `get_company_news_raw(...) -> list[CompanyNews]`

- A-shares:
  - raw discovery from AkShare
- US / HK:
  - `Tavily` search first
  - `Brave Search` as supplemental recall
- Cross-verification:
  - weight by actual publisher / domain, not by search engine
  - event clustering uses deterministic URL/title/time/token rules
  - output includes `weighted_source_score`, `consensus_passed`, `official_confirmed`

Primary output model:
- `IntegratedNewsEvent`
  - `event_id`, `ticker`, `headline`, `summary`, `published_at`
  - `primary_source`, `source_count`, `high_weight_source_count`
  - `weighted_source_score`, `consensus_passed`, `official_confirmed`
  - `sources`, `urls`, `tags`

Compatibility notes:
- Event rows still expose alias fields used by older code:
  - `title`, `source`, `date`, `url`, `publish_time`, `content`, `keyword`
- No sentiment inference or LLM processing occurs in `get_company_news`.
- Configure data-source API keys in `OMNIX_PATH/finn_api.json` under:
  - `FRED.api_key`
  - `tavily.api_key`
  - `brave.api_key`
