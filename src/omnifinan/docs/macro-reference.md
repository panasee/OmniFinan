# Macro Data Reference

## Series Keys by Country and Source

### US (FRED)

| Key | FRED ID | Description |
|-----|---------|-------------|
| `fed_policy_rate` | FEDFUNDS | Federal Funds Rate |
| `sofr` | SOFR | Secured Overnight Financing Rate |
| `us_cpi_yoy` | CPIAUCSL (yoy) | CPI Year-over-Year |
| `us_core_cpi_yoy` | CPILFESL (yoy) | Core CPI YoY (ex food & energy) |
| `us_core_pce_price` | PCEPILFE (yoy) | Core PCE Price Index YoY |
| `us_unemployment_rate` | UNRATE | Unemployment Rate |
| `us_non_farm_payrolls` | PAYEMS | Non-Farm Payrolls |
| `us_initial_jobless_claims` | ICSA | Initial Jobless Claims |
| `us_retail_sales` | RSAFS | Retail Sales |
| `us_industrial_production` | INDPRO | Industrial Production Index |
| `us_real_gdp_latest_quarter` | GDPC1 | Real GDP Level |
| `us_real_gdp_qoq_annualized` | derived GDPC1 | Real GDP QoQ Annualized |
| `us_real_gdp_yoy` | derived GDPC1 | Real GDP YoY |
| `us_gdp_growth` | alias us_real_gdp_yoy | GDP Growth (alias) |
| `us_breakeven_10y` | T10YIE | 10Y Breakeven Inflation |
| `us_m2` | M2SL | M2 Money Supply |
| `us_central_bank_total_assets` | WALCL | Fed Total Assets |
| `us_real_rate_10y` | DFII10 | 10Y Real Rate (TIPS) |
| `us_real_interest_rate` | alias/REAINTRATREARAT10Y | Real Interest Rate |
| `us_treasury_2y` | DGS2 | 2Y Treasury Yield |
| `us_treasury_10y` | DGS10 | 10Y Treasury Yield |
| `us_term_spread_10y_2y` | T10Y2Y | 10Y-2Y Term Spread |
| `us_term_spread_10y2y` | alias T10Y2Y | Term Spread (alias) |
| `us_corporate_bbb_oas` | BAMLC0A4CBBB | BBB Corporate OAS |
| `us_business_loan_delinquency_rate` | DRBLACBS | Business Loan Delinquency |
| `us_bank_loan_growth_yoy` | derived BUSLOANS | Bank Loan Growth YoY |
| `us_equity_sp500` | SP500 | S&P 500 Index |
| `us_vix` | VIXCLS | VIX Volatility Index |
| `us_dollar_index_broad` | DTWEXBGS | US Dollar Index (Broad) |
| `commodity_wti_crude` | DCOILWTICO | WTI Crude Oil |
| `commodity_copper` | PCOPPUSDM | Copper Price |

### US (AkShare)

| Key | AkShare Function | Description |
|-----|-----------------|-------------|
| `us_consumer_confidence_cb` | macro_usa_cb_consumer_confidence | CB Consumer Confidence |
| `us_consumer_sentiment_michigan` | macro_usa_michigan_consumer_sentiment | Michigan Consumer Sentiment |
| `us_pmi_manufacturing` | macro_usa_ism_pmi (+ local fallback) | ISM Manufacturing PMI |
| `us_pmi_services` | macro_usa_ism_non_pmi (+ local fallback) | ISM Services PMI |

### China (AkShare Official)

| Key | Source | Description |
|-----|--------|-------------|
| `pboc_policy_rate` | PBOC | Policy benchmark rate (fallback to LPR 1Y) |
| `china_lpr_1y` | PBOC | Loan Prime Rate 1Y |
| `china_shibor_3m` | CFETS | SHIBOR 3-Month |
| `china_cpi_yoy` | NBS | CPI Year-over-Year |
| `china_cpi_mom` | NBS | CPI Month-over-Month |
| `china_ppi_yoy` | NBS | PPI Year-over-Year |
| `china_gdp_yoy` | NBS | GDP Year-over-Year |
| `china_pmi_manufacturing` | NBS | Official Manufacturing PMI |
| `china_caixin_pmi_manufacturing` | Caixin | Caixin Manufacturing PMI |
| `china_caixin_pmi_services` | Caixin | Caixin Services PMI |
| `china_pmi_non_manufacturing` | NBS | Non-Manufacturing PMI |
| `china_urban_unemployment` | NBS | Urban Unemployment Rate |
| `china_m2_yoy` | PBOC | M2 Growth Year-over-Year |
| `china_social_financing` | PBOC | Total Social Financing |
| `china_bank_financing` | PBOC | Bank Financing |
| `china_central_bank_balance_sheet` | PBOC | Central Bank Balance Sheet |
| `china_bank_loan_growth` | PBOC | RMB Loan Growth |
| `china_real_estate_financing` | NBS | Real Estate Development Financing |
| `china_fixed_asset_investment_yoy` | NBS | Fixed Asset Investment YoY |
| `china_retail_sales_yoy` | NBS | Retail Sales YoY |
| `china_industrial_production_yoy` | NBS | Industrial Production YoY |
| `china_exports_yoy` | Customs | Exports YoY |
| `china_imports_yoy` | Customs | Imports YoY |
| `china_trade_balance` | Customs | Trade Balance |
| `china_fx_reserves` | SAFE | FX Reserves |

### Singapore (World Bank + FRED)

| Key | Source | Indicator |
|-----|--------|-----------|
| `sg_gdp_growth` / `sg_gdp_yoy` | World Bank | NY.GDP.MKTP.KD.ZG |
| `sg_inflation_cpi` / `sg_cpi_yoy` | World Bank | FP.CPI.TOTL.ZG |
| `sg_unemployment_rate` | World Bank | SL.UEM.TOTL.ZS |
| `sg_exports_growth` | World Bank | NE.EXP.GNFS.KD.ZG |
| `sg_imports_growth` | World Bank | NE.IMP.GNFS.KD.ZG |
| `sg_current_account_gdp` | World Bank | BN.CAB.XOKA.GD.ZS |
| `sg_real_interest_rate` | World Bank | FR.INR.RINR / FR.INR.LNDP |
| `sg_broad_money_growth` | World Bank | FM.LBL.BMNY.ZG |
| `sg_policy_rate` | World Bank | FR.INR.DPST / FR.INR.LEND |
| `sg_usd_fx` | FRED | DEXSIUS |
| `sg_government_bond_10y` | MAS SGS | 10Y Original Maturity |

### Japan / Europe Cross-Impact

| Key | FRED ID | Description |
|-----|---------|-------------|
| `jp_short_rate_3m` | IR3TIB01JPM156N | Japan 3M Interbank |
| `jp_government_bond_10y` | IRLTLT01JPM156N | Japan 10Y JGB |
| `jp_usd_fx` | DEXJPUS | USD/JPY |
| `jp_policy_rate` | AkShare macro_japan_bank_rate | BoJ Policy Rate |
| `eu_short_rate_3m` | IR3TIB01EZM156N | Eurozone 3M Interbank |
| `eu_government_bond_10y` | IRLTLT01EZM156N | Eurozone 10Y Bond |
| `eu_usd_fx` | DEXUSEU | EUR/USD |
| `eu_pmi_manufacturing` | AkShare macro_euro_manufacturing_pmi | Euro Mfg PMI |

### Global Aggregates (World Bank)

| Key | Indicator | Description |
|-----|-----------|-------------|
| `world_gdp_growth` | NY.GDP.MKTP.KD.ZG (WLD) | World GDP Growth |
| `world_inflation` | FP.CPI.TOTL.ZG (WLD) | World Inflation |

## Dimension Mapping

Each series maps to one of 5 analytical dimensions:

| Dimension | Description | Example Keys |
|-----------|-------------|-------------|
| `growth` | Economic output and activity | us_real_gdp_yoy, china_gdp_yoy, china_pmi_manufacturing |
| `inflation` | Price level and expectations | us_cpi_yoy, china_cpi_yoy, us_breakeven_10y |
| `liquidity` | Money supply, rates, FX | fed_policy_rate, sofr, china_shibor_3m, us_m2 |
| `credit` | Lending, spreads, delinquency | us_corporate_bbb_oas, china_social_financing |
| `market_feedback` | Asset prices and volatility | us_equity_sp500, us_vix, commodity_wti_crude |

## Staleness Logic

Cycle-based thresholds for refetch decisions:

| Data Frequency | Inferred `cycle_days` | Stale After | Refetch Cooldown |
|----------------|----------------------|-------------|-----------------|
| Daily (rates, VIX) | 1 | 7 days | 6 hours |
| Weekly | 7 | 21 days | 24 hours |
| Monthly (CPI, PMI) | 30 | 90 days | 3 days |
| Quarterly (GDP) | 90 | 270 days | 7 days |
| Annual (World Bank) | 365 | 1095 days | 14 days |

Staleness check priority:
1. Check `fetched_at` timestamp against cooldown - skip if recently fetched
2. If no `fetched_at`, use `payload.snapshot_at` with 7-day generous cooldown
3. Check latest observation date against `max(7, cycle_days * 3)` threshold

## Structured Output Schema

`get_macro_indicators_structured()` returns:

```python
{
    "meta": {
        "snapshot_at": "2025-06-15T12:00:00Z",
        "source_policy": "fixed_sources_v1_...",
        "series_count": 60,
        "ok_count": 55,
        "error_count": 5,
    },
    "dimensions": ["growth", "inflation", "liquidity", "credit", "market_feedback"],
    "metrics": {
        "us_cpi_yoy": {
            "dimension": "inflation",
            "country": "US",
            "source": "fred:CPIAUCSL",
            "latest_value": 3.2,
            "latest_date": "2025-05-01",
            "yoy": 0.5,
            "mom": -0.1,
            "qoq": 0.2,
            "trend_short": "down",
            "trend_medium": "flat",
            "volatility": 0.15,
            "error": None,
        },
        # ... one card per series
    },
    "chart_data": {
        "long": [
            {"key": "us_cpi_yoy", "date": "2025-05-01", "value": 3.2,
             "dimension": "inflation", "country": "US", "source": "fred:CPIAUCSL"},
            # ... flattened time series for plotting
        ]
    }
}
```
