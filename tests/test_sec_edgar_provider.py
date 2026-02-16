from __future__ import annotations

from omnifinan.data.providers.sec_edgar_provider import SECEDGARProvider


def _facts_payload(concepts: dict[str, list[dict]]) -> dict:
    return {
        "facts": {
            "us-gaap": {
                name: {
                    "units": {
                        ("shares" if "Shares" in name or "shares" in name else "USD"): rows,
                    }
                }
                for name, rows in concepts.items()
            }
        }
    }


def test_same_end_prefers_latest_filed_with_form_priority(monkeypatch):
    provider = SECEDGARProvider()
    facts = _facts_payload(
        {
            "Revenues": [
                {"end": "2024-12-31", "val": 100, "form": "10-K", "filed": "2025-02-01", "accn": "0001", "fy": 2024, "fp": "FY"},
                {"end": "2024-12-31", "val": 120, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "NetIncomeLoss": [
                {"end": "2024-12-31", "val": 20, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "Assets": [
                {"end": "2024-12-31", "val": 200, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "StockholdersEquity": [
                {"end": "2024-12-31", "val": 80, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
        }
    )
    monkeypatch.setattr(provider, "_company_facts", lambda ticker: facts)
    monkeypatch.setattr(provider, "_market_cap_with_fallback", lambda ticker, end_date, shares_outstanding: None)
    rows = provider.get_financial_metrics("TST", period="annual", limit=1)
    assert len(rows) == 1
    assert rows[0].operating_revenue == 120


def test_ttm_sum_and_fcf_formula(monkeypatch):
    provider = SECEDGARProvider()
    facts = _facts_payload(
        {
            "Revenues": [
                {"end": "2024-03-31", "val": 10, "form": "10-Q", "filed": "2024-05-01", "accn": "1", "fy": 2024, "fp": "Q1"},
                {"end": "2024-06-30", "val": 20, "form": "10-Q", "filed": "2024-08-01", "accn": "2", "fy": 2024, "fp": "Q2"},
                {"end": "2024-09-30", "val": 30, "form": "10-Q", "filed": "2024-11-01", "accn": "3", "fy": 2024, "fp": "Q3"},
                {"end": "2024-12-31", "val": 40, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
            "NetIncomeLoss": [
                {"end": "2024-03-31", "val": 5, "form": "10-Q", "filed": "2024-05-01", "accn": "1", "fy": 2024, "fp": "Q1"},
                {"end": "2024-06-30", "val": 6, "form": "10-Q", "filed": "2024-08-01", "accn": "2", "fy": 2024, "fp": "Q2"},
                {"end": "2024-09-30", "val": 7, "form": "10-Q", "filed": "2024-11-01", "accn": "3", "fy": 2024, "fp": "Q3"},
                {"end": "2024-12-31", "val": 8, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
            "NetCashProvidedByUsedInOperatingActivities": [
                {"end": "2024-03-31", "val": 8, "form": "10-Q", "filed": "2024-05-01", "accn": "1", "fy": 2024, "fp": "Q1"},
                {"end": "2024-06-30", "val": 9, "form": "10-Q", "filed": "2024-08-01", "accn": "2", "fy": 2024, "fp": "Q2"},
                {"end": "2024-09-30", "val": 10, "form": "10-Q", "filed": "2024-11-01", "accn": "3", "fy": 2024, "fp": "Q3"},
                {"end": "2024-12-31", "val": 11, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
            "PaymentsToAcquirePropertyPlantAndEquipment": [
                {"end": "2024-03-31", "val": -2, "form": "10-Q", "filed": "2024-05-01", "accn": "1", "fy": 2024, "fp": "Q1"},
                {"end": "2024-06-30", "val": -3, "form": "10-Q", "filed": "2024-08-01", "accn": "2", "fy": 2024, "fp": "Q2"},
                {"end": "2024-09-30", "val": -1, "form": "10-Q", "filed": "2024-11-01", "accn": "3", "fy": 2024, "fp": "Q3"},
                {"end": "2024-12-31", "val": -4, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
            "WeightedAverageNumberOfDilutedSharesOutstanding": [
                {"end": "2024-12-31", "val": 2, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
            "Assets": [
                {"end": "2024-12-31", "val": 200, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
            "StockholdersEquity": [
                {"end": "2024-12-31", "val": 100, "form": "10-Q", "filed": "2025-02-01", "accn": "4", "fy": 2024, "fp": "Q4"},
            ],
        }
    )
    monkeypatch.setattr(provider, "_company_facts", lambda ticker: facts)
    monkeypatch.setattr(provider, "_market_cap_with_fallback", lambda ticker, end_date, shares_outstanding: None)
    rows = provider.get_financial_metrics("TST", period="ttm", limit=1)
    assert len(rows) == 1
    assert rows[0].operating_revenue == 100
    assert rows[0].capital_expenditure == 10
    assert rows[0].free_cash_flow_per_share == 14


def test_quick_ratio_uses_inventory_and_prepaid(monkeypatch):
    provider = SECEDGARProvider()
    facts = _facts_payload(
        {
            "Revenues": [
                {"end": "2024-12-31", "val": 200, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "NetIncomeLoss": [
                {"end": "2024-12-31", "val": 40, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "AssetsCurrent": [
                {"end": "2024-12-31", "val": 100, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "LiabilitiesCurrent": [
                {"end": "2024-12-31", "val": 50, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "InventoryNet": [
                {"end": "2024-12-31", "val": 30, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "PrepaidExpenseCurrent": [
                {"end": "2024-12-31", "val": 10, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "Assets": [
                {"end": "2024-12-31", "val": 300, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
            "StockholdersEquity": [
                {"end": "2024-12-31", "val": 120, "form": "10-K", "filed": "2025-02-20", "accn": "0002", "fy": 2024, "fp": "FY"},
            ],
        }
    )
    monkeypatch.setattr(provider, "_company_facts", lambda ticker: facts)
    monkeypatch.setattr(provider, "_market_cap_with_fallback", lambda ticker, end_date, shares_outstanding: None)
    rows = provider.get_financial_metrics("TST", period="annual", limit=1)
    assert len(rows) == 1
    assert rows[0].current_ratio == 2
    assert rows[0].quick_ratio == 1.2


def test_valuation_ratios_from_market_cap(monkeypatch):
    provider = SECEDGARProvider()
    facts = _facts_payload(
        {
            "Revenues": [{"end": "2024-12-31", "val": 100, "form": "10-K", "filed": "2025-02-20", "accn": "1", "fy": 2024, "fp": "FY"}],
            "NetIncomeLoss": [{"end": "2024-12-31", "val": 20, "form": "10-K", "filed": "2025-02-20", "accn": "1", "fy": 2024, "fp": "FY"}],
            "Assets": [{"end": "2024-12-31", "val": 300, "form": "10-K", "filed": "2025-02-20", "accn": "1", "fy": 2024, "fp": "FY"}],
            "StockholdersEquity": [{"end": "2024-12-31", "val": 80, "form": "10-K", "filed": "2025-02-20", "accn": "1", "fy": 2024, "fp": "FY"}],
            "CashAndCashEquivalentsAtCarryingValue": [{"end": "2024-12-31", "val": 10, "form": "10-K", "filed": "2025-02-20", "accn": "1", "fy": 2024, "fp": "FY"}],
            "DebtLongtermAndShorttermCombinedAmount": [{"end": "2024-12-31", "val": 50, "form": "10-K", "filed": "2025-02-20", "accn": "1", "fy": 2024, "fp": "FY"}],
        }
    )
    monkeypatch.setattr(provider, "_company_facts", lambda ticker: facts)
    monkeypatch.setattr(provider, "_market_cap_with_fallback", lambda ticker, end_date, shares_outstanding: 400.0)
    rows = provider.get_financial_metrics("TST", period="annual", limit=1)
    assert len(rows) == 1
    row = rows[0]
    assert row.market_cap == 400.0
    assert row.price_to_earnings_ratio == 20.0
    assert row.price_to_book_ratio == 5.0
    assert row.price_to_sales_ratio == 4.0
    assert row.enterprise_value == 440.0
