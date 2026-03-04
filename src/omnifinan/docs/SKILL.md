---
name: omnifinan
description: Guide AI to use the OmniFinan Python library for financial analysis, including multi-agent hedge fund workflow, data fetching (AkShare/YFinance/FRED/SEC EDGAR), macro indicators, technical/fundamental/sentiment/valuation analysis, bull-bear debate, risk management, portfolio decisions, backtesting, visualization, and report pipelines. Use when working with OmniFinan code, running analyses, modifying agents, or adding features to this financial analysis system.
---

# OmniFinan Library Guide

OmniFinan is an AI-driven multi-agent hedge fund analysis system supporting US and Chinese markets. It uses LangGraph for agent orchestration, LangChain for LLM calls, and multiple data providers (AkShare, YFinance, Finnhub, SEC EDGAR, FRED, World Bank).

## Project Structure

```
src/omnifinan/
├── __init__.py              # Exports: MarketType, run_hedge_fund
├── main.py                  # CLI entrypoint -> presentation.cli.run_cli()
├── unified_api.py           # Low-level AkShare/FRED/WorldBank data functions (3800+ lines)
├── data_models.py           # Pydantic models: Price, FinancialMetrics, LineItem, etc.
├── visualize.py             # Plotly charting: StockFigure, macro dashboards
├── backtester.py            # Backtester class for strategy evaluation
├── agents/                  # LangGraph agent nodes
│   ├── graphs.py            # Graph builders: create_trading_graph()
│   ├── state.py             # AgentState TypedDict
│   ├── nodes.py             # Node factories
│   ├── edges.py             # Conditional routing functions
│   ├── prompts.py           # Prompt constants
│   ├── market_data.py       # Data collection agent
│   ├── technicals.py        # Technical analysis agent
│   ├── fundamentals.py      # Fundamental analysis agent
│   ├── macro.py             # Macro analyst agent
│   ├── sentiment.py         # Sentiment analysis agent (uses LLM)
│   ├── valuation.py         # DCF/multiples valuation agent
│   ├── researcher_bull.py   # Bullish thesis generator (uses LLM)
│   ├── researcher_bear.py   # Bearish thesis generator (uses LLM)
│   ├── debate_room.py       # Bull-bear debate judge (uses LLM)
│   ├── risk_manager.py      # Position sizing / risk constraints
│   └── portfolio_manager.py # Final buy/sell/hold decisions (uses LLM)
├── core/
│   ├── config.py            # RuntimeConfig (env vars / YAML / JSON)
│   ├── workflow.py          # run_hedge_fund() orchestration
│   ├── observability.py     # RunTrace for metrics/cost tracking
│   └── experiment.py       # ExperimentRecorder for run comparison
├── data/
│   ├── cache.py             # DataCache: file-based request + dataset cache
│   ├── unified_service.py   # UnifiedDataService: cached provider wrapper
│   ├── symbols.py           # is_crypto_ticker() helper
│   └── providers/
│       ├── base.py          # DataProvider ABC
│       ├── factory.py       # create_data_provider(name)
│       ├── akshare_provider.py
│       ├── yfinance_provider.py
│       ├── finnhub_provider.py
│       ├── sec_edgar_provider.py
│       └── marketdata_provider.py   # options-only provider
├── analysis/
│   ├── indicators.py        # XMA, cross_over, cross_under (TA-Lib wrappers)
│   ├── transform.py         # Feature engineering: returns, rolling features
│   ├── factor_mining.py     # CustomFactorSpec, IC evaluation
│   └── factor_backtest.py   # Cross-sectional factor backtesting
├── research/
│   ├── valuation.py         # dcf_intrinsic_value(), valuation_signal()
│   ├── factors.py           # Qlib-style DSL: ref, mean, std, rank, apply_factor
│   ├── report_pipeline.py   # PDF report -> LLM synthesis
│   └── report_parser.py     # ParsedReport via pypdf
├── llm/
│   ├── client.py            # call_llm(): unified LLM call with cache + retry
│   └── providers.py         # PROVIDER_REGISTRY: gpt/claude/gemini/deepseek
├── presentation/
│   ├── cli.py               # argparse CLI with questionary analyst selector
│   └── api.py               # Flask REST API: POST /analyze, GET /healthz
└── utils/
    ├── analysts.py          # ANALYST_CONFIG registry, get_analyst_nodes()
    ├── holidays.py          # Trading calendar filtering
    ├── normalization.py     # confidence_to_unit()
    ├── progress.py          # Progress tracking
    ├── display.py           # Console output formatting
    ├── scratchpad.py        # Scratchpad for run artifacts
    └── llm.py               # Convenience LLM wrapper
```

## Core Concepts

### AgentState

All agents receive and return `AgentState`:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]
```

- `data` holds tickers, dates, prices, financial_metrics, macro_indicators, analyst_signals
- `metadata` holds show_reasoning, model_name, provider_api, language, data_service, trace, scratchpad

### Trading Graph Pipeline

```
start_node -> market_data_agent -> [analyst agents in parallel] -> investment_debate_start
-> researcher_bull_agent <-> researcher_bear_agent -> debate_room_agent
-> risk_start -> risk_management_agent -> execution_start -> portfolio_management_agent -> END
```

Analyst agents (run in parallel after market_data_agent):
- `technical_analyst_agent` - trend, momentum, mean reversion, volatility, stat arb
- `fundamentals_agent` - profitability, growth, financial health, valuation ratios
- `macro_analyst_agent` - central bank rates, inflation, employment
- `sentiment_agent` - news keyword scoring + LLM sentiment analysis
- `valuation_agent` - DCF, owner earnings, residual income, comparable multiples

### Data Providers

| Provider | Name strings | Markets | Capabilities |
|----------|-------------|---------|-------------|
| AkShare | `"akshare"` | CN, US, HK | Prices, financials, news, macro (CN+intl) |
| YFinance | `"yfinance"`, `"yf"`, `"yahoo"` | US, global, crypto | Prices, financials, news |
| Finnhub | `"finnhub"` | US | Prices, financials, news, insider trades |
| SEC EDGAR | `"sec_edgar"`, `"sec"` | US | Financial metrics, line items from XBRL |
| MarketData (options-only) | n/a (direct options APIs) | US stocks, futures options | Option chain fetch only |

Create with: `create_data_provider("akshare")`

### DataProvider ABC

All providers implement:
- `get_prices(ticker, start_date, end_date, interval) -> list[Price]`
- `get_financial_metrics(ticker, end_date, period, limit) -> list[FinancialMetrics]`
- `search_line_items(ticker, period, limit) -> list[LineItem]`
- `get_company_news(ticker, start_date, end_date, limit) -> list[CompanyNews]`
- `get_insider_trades(ticker, end_date, start_date, limit) -> list[InsiderTrade]`
- `get_market_cap(ticker, end_date) -> float | None`
- `get_macro_indicators(start_date, end_date) -> dict`

### UnifiedDataService

Wraps a `DataProvider` with intelligent caching via `DataCache`. Key behaviors:
- **Price data**: Incremental fetch - backfills gaps, appends new data, avoids redundant downloads
- **Financial metrics/line items**: Refetch when latest report is >30 days stale
- **Macro indicators**: Master-first strategy with series-level staleness, subset refresh, anti-loop guard
- **Company news**: Incremental forward/backward fetch, deduplicated by URL/title
- **Insider trades**: Incremental fetch by filing_date
- **Crypto**: Auto-routes to YFinance for crypto tickers
- **Options**:
  - `get_stock_option_chain()` defaults to `provider="auto"` (MarketData first, then YFinance fallback)
  - `get_futures_option_chain()` uses MarketData
  - default snapshot mode is previous-business-day close (`snapshot_mode="prev_close"`)

```python
from omnifinan.data.cache import DataCache
from omnifinan.data.providers.factory import create_data_provider
from omnifinan.data.unified_service import UnifiedDataService

service = UnifiedDataService(
    provider=create_data_provider("akshare"),
    cache=DataCache(),
    ttl_seconds=3600,
)
prices = service.get_prices("600519", "2025-01-01", "2025-12-31")
macro = service.get_macro_indicators("2025-01-01", "2025-12-31")
structured = service.get_macro_indicators_structured("2025-01-01", "2025-12-31")
stock_opts = service.get_stock_option_chain("AAPL", expiration="2026-06-19")
fut_opts = service.get_futures_option_chain("ES", expiration="2026-06-19")
```

### Macro Data Architecture

- Source policy: `fixed_sources_v1_china_akshare_official__intl_fred_imf_worldbank`
- Master payload stored as dataset with snapshot history
- Staleness detection: per-series based on `cycle_days * 3` threshold
- Refetch cooldown: `fetched_at` + frequency-based minimum interval
- Subset refresh when provider supports `get_macro_indicators_subset()`
- Structured output keys: `meta`, `dimensions`, `metrics`, `chart_data`

### Structured Macro Output

`get_macro_indicators_structured()` returns:
- `meta`: snapshot_at, source_policy, counts
- `dimensions`: growth, inflation, liquidity, credit, market_feedback
- `metrics`: per-series cards with yoy, mom, qoq, trend_short, trend_medium, volatility
- `chart_data.long`: flattened list for plotting

## LLM Execution Guidance

This section is for LLM runtime orchestration guidance only.

### Preferred Data-First API Path

For non-LLM analytics tasks, prefer direct `UnifiedDataService` APIs instead of full
multi-agent orchestration:

- `get_prices`
- `get_financial_metrics`
- `get_line_items`
- `get_company_news`
- `get_insider_trades`
- `get_macro_indicators`
- `get_macro_indicators_structured`
- `get_stock_option_chain`
- `get_futures_option_chain`
- `get_stock_option_chain_analytics` (new; non-LLM)
- `get_futures_option_chain_analytics` (new; non-LLM)

### Option Analytics (Non-LLM) Guidance

When the task asks for IV/skew/term-structure/Greeks, use analytics APIs first:

- `UnifiedDataService.get_stock_option_chain_analytics(...)`
- `UnifiedDataService.get_futures_option_chain_analytics(...)`

Market compatibility rule:
- China A-share / HK equity tickers do not have options support in current provider stack.
- For those tickers, stock-option APIs return `meta.source = "fixed_sources_unavailable"` with explicit `meta.error`.
- LLM should continue the broader analysis flow without treating this as a fatal error.
- Crypto pair symbols are normalized to base asset for options endpoints:
  - `BTC-USDT`, `BTC-USD`, `BTCUSDT` -> `BTC`
  - `ETH-USDT`, `ETH-USD`, `ETHUSDT` -> `ETH`

Return contract from analytics APIs:

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

### LLM-as-Glue Fallback Pattern

If a step requires semantic generation but nested runtime model calls are not desired:

1. Pause at the step and collect exact upstream state.
2. Generate the structured output in the current LLM context.
3. Write back to the exact state path expected downstream.
4. Resume remaining deterministic steps.

Required compatibility write-back path example:

- `state["data"]["analyst_signals"]["sentiment_agent"][ticker]`
- fields: `signal`, `confidence`, `reasoning`

Keep external contracts stable:

- Top-level result keys (e.g., `decisions`, `analyst_signals`)
- Macro structured keys (`meta`, `dimensions`, `metrics`, `chart_data`)

### Orchestration Interfaces (Still Available)

These orchestration interfaces remain supported; use them when task intent is
end-to-end execution rather than atomic data/analytics calls.

- `omnifinan.run_hedge_fund(...)` (full multi-agent workflow)
- `omnifinan.core.workflow.run_hedge_fund(...)` (workflow module entry)
- `omnifinan.backtester.Backtester` (backtest runner)
- `omnifinan.presentation.api.create_app()` (REST API entry)
- `omnifinan.visualize.StockFigure` / `omnifinan.visualize.create_macro_figure` (charting)

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OMNIFINAN_DATA_PROVIDER` | `akshare` | Data provider name |
| `OMNIFINAN_MARKET_TYPE` | `china` | Market: us, china, hongkong |
| `OMNIFINAN_DATA_CACHE_TTL` | `3600` | Cache TTL in seconds |
| `OMNIFINAN_MODEL_NAME` | `deepseek-chat` | LLM model |
| `OMNIFINAN_PROVIDER_API` | `deepseek` | LLM provider |
| `OMNIFINAN_LANGUAGE` | `Chinese` | Output language |
| `OMNIFINAN_MODEL_TEMPERATURE` | `0.2` | LLM temperature |
| `OMNIFINAN_DEBATE_ROUNDS` | `1` | Bull-bear debate rounds |
| `OMNIFINAN_DETERMINISTIC_MODE` | `1` | Enable LLM response caching |
| `OMNIFINAN_LLM_SEED` | `7` | LLM seed for reproducibility |
| `OMNIFINAN_ENABLED_ANALYSTS` | (all) | Comma-separated analyst keys |
| `OMNIFINAN_CONFIG_PATH` | (none) | Path to YAML/JSON config file |

### Config File (YAML/JSON)

```yaml
data_provider: akshare
market_type: china
data_cache_ttl_seconds: 3600
debate_rounds: 2
deterministic_mode: true
enabled_analysts:
  - technical_analyst
  - fundamentals_analyst
  - macro_analyst
llm:
  model_name: deepseek-chat
  provider_api: deepseek
  temperature: 0.2
  max_retries: 3
  language: Chinese
```

### Available Analyst Keys

- `technical_analyst` - Technical indicators and pattern analysis
- `fundamentals_analyst` - Financial statement analysis
- `macro_analyst` - Macroeconomic indicators analysis
- `sentiment_analyst` - News and insider trading sentiment
- `valuation_analyst` - Intrinsic value estimation

## Data Models

### Key Pydantic Models

| Model | Key Fields |
|-------|-----------|
| `Price` | open, close, high, low, volume, time, market |
| `FinancialMetrics` | ticker, report_period, period, currency, market_cap, PE/PB/PS ratios, margins, ROE/ROA, growth rates |
| `LineItem` | ticker, report_period, net_income, operating_revenue, free_cash_flow (extra="allow") |
| `CompanyNews` | ticker, title, source, date, url, sentiment |
| `InsiderTrade` | ticker, filing_date, transaction_date, transaction_shares, transaction_value |
| `MarketType` | US, CHINA, CHINA_SZ, CHINA_SH, HK, UNKNOWN |

## Runtime Data Paths

All runtime data under `OMNIX_PATH/omnifinan/`:
- `request_cache/` - API response cache (hashed JSON files)
- `datasets/` - Persistent datasets (prices, financials, macro history)
- `reports/` - Output reports and experiment records
- `logs/` - Application logs

## LLM Integration

```python
from omnifinan.llm.client import call_llm

# Plain text response
text = call_llm(
    prompt="Analyze this data...",
    model_name="deepseek-chat",
    provider_api="deepseek",
)

# Structured Pydantic response
from pydantic import BaseModel
class Analysis(BaseModel):
    signal: str
    confidence: float

result = call_llm(
    prompt="...",
    model_name="deepseek-chat",
    provider_api="deepseek",
    pydantic_model=Analysis,
    deterministic_mode=True,  # enables response caching
    trace=trace,              # optional RunTrace
    scratchpad=scratchpad,    # optional Scratchpad
)
```

Supported providers: `deepseek`, `openai` (gpt), `anthropic` (claude), `google` (gemini)

## Testing

Run verification tests after changes:

```bash
# Core macro logic
pytest tests/test_macro_source_policy.py
pytest tests/test_macro_structured.py
pytest tests/test_macro_visualize.py

# Other test suites
pytest tests/test_agent_graphs.py
pytest tests/test_agent_edges.py
pytest tests/test_data_cache.py
pytest tests/test_factor_mining.py
pytest tests/test_factor_backtest.py
pytest tests/test_llm_client.py
pytest tests/test_runtime_config.py
pytest tests/test_sec_edgar_provider.py
pytest tests/test_symbols.py
```

## Key Constraints (from AGENTS.md)

1. **Source policy**: Do not change macro source policy unless explicitly requested
2. **Runtime data**: Never write hot data into the repo; use `OMNIX_PATH/omnifinan/`
3. **Output stability**: Preserve unified API structures (`meta`, `dimensions`, `metrics`, `chart_data`)
4. **Minimal edits**: Keep changes deterministic and technically concise
5. **Anti-loop**: Avoid repeated full refresh loops when sources have no delta
6. **Local-first**: Prefer cached data for repeated analysis/report generation

## Adding a New Analyst Agent

1. Create `src/omnifinan/agents/your_analyst.py` with function signature:
   ```python
   def your_analyst_agent(state: AgentState) -> AgentState:
       # Read from state["data"], state["metadata"]["data_service"]
       # Write signals to state["data"]["analyst_signals"][ticker]["your_analyst_agent"]
       return {"messages": state["messages"], "data": {...}, "metadata": state["metadata"]}
   ```
2. Register in `src/omnifinan/utils/analysts.py` ANALYST_CONFIG
3. The graph builder (`graphs.py`) auto-wires registered analysts

## Adding a New Data Provider

1. Create `src/omnifinan/data/providers/your_provider.py` implementing `DataProvider` ABC
2. Register in `src/omnifinan/data/providers/factory.py`
3. All 7 abstract methods must be implemented

## Ticker Format

- China A-shares: 6-digit code (e.g., `600519`, `000001`)
- Hong Kong: 5-digit zero-padded (e.g., `00700`)
- US: Standard symbols (e.g., `AAPL`, `MSFT`)
- Crypto: Pair format (e.g., `BTC-USD`, `ETHUSDT`) - auto-routed to YFinance

For detailed macro series reference and structured output schema, see [macro-reference.md](macro-reference.md).
