from __future__ import annotations

from omnifinan import unified_api
from omnifinan.data.providers import credentials


def test_get_api_key_case_insensitive_provider_name(monkeypatch) -> None:
    monkeypatch.setattr(
        credentials,
        "load_provider_credentials",
        lambda: {
            "tavily": {"api_key": "abc"},
            "FRED": {"api_key": "fred-key-123"},
        },
    )
    assert credentials.get_api_key("fred") == "fred-key-123"
    assert credentials.get_api_key("FRED") == "fred-key-123"


def test_fred_api_key_comes_from_config_file_only(monkeypatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "env-should-be-ignored")
    monkeypatch.setattr(
        credentials,
        "load_provider_credentials",
        lambda: {
            "FRED": {"api_key": "fred-key-from-file"},
        },
    )
    assert unified_api._get_fred_api_key() == "fred-key-from-file"
