from __future__ import annotations

from omnifinan.visualize import create_macro_figure, macro_structured_to_dataframe


def _structured_payload() -> dict:
    return {
        "meta": {"snapshot_at": "2026-02-16T00:00:00Z"},
        "metrics": {
            "us_cpi_yoy": {"dimension": "inflation", "error": None},
            "us_m2": {"dimension": "liquidity", "error": None},
        },
        "chart_data": {
            "long": [
                {
                    "key": "us_cpi_yoy",
                    "date": "2025-01-01",
                    "value": 3.1,
                    "dimension": "inflation",
                    "country": "US",
                    "source": "fred:CPIAUCSL",
                },
                {
                    "key": "us_cpi_yoy",
                    "date": "2025-02-01",
                    "value": 3.0,
                    "dimension": "inflation",
                    "country": "US",
                    "source": "fred:CPIAUCSL",
                },
                {
                    "key": "us_m2",
                    "date": "2025-01-01",
                    "value": 21000.0,
                    "dimension": "liquidity",
                    "country": "US",
                    "source": "fred:M2SL",
                },
                {
                    "key": "us_m2",
                    "date": "2025-02-01",
                    "value": 21120.0,
                    "dimension": "liquidity",
                    "country": "US",
                    "source": "fred:M2SL",
                },
            ]
        },
    }


def test_macro_structured_to_dataframe_filters():
    payload = _structured_payload()
    df_all = macro_structured_to_dataframe(payload)
    assert df_all.shape[0] == 4

    df_inf = macro_structured_to_dataframe(payload, dimension="inflation")
    assert set(df_inf["key"].unique()) == {"us_cpi_yoy"}

    df_us = macro_structured_to_dataframe(payload, country="US")
    assert df_us.shape[0] == 4


def test_create_macro_figure_has_traces():
    payload = _structured_payload()
    fig = create_macro_figure(payload, dimensions=["inflation", "liquidity"], max_series_per_dimension=2)
    assert fig is not None
    assert len(fig.data) >= 2
