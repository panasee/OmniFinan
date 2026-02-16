from __future__ import annotations

from omnifinan.data.providers import credentials


def test_get_api_key_case_insensitive_provider_name(monkeypatch) -> None:
    monkeypatch.setattr(
        credentials,
        "load_provider_credentials",
        lambda: {
            "finnhub": {"api_key": "abc"},
            "FRED": {"api_key": "fred-key-123"},
        },
    )
    assert credentials.get_api_key("fred") == "fred-key-123"
    assert credentials.get_api_key("FRED") == "fred-key-123"
