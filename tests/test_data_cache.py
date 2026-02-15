from __future__ import annotations

from omnifinan.data.cache import DataCache


def test_data_cache_set_get(tmp_path):
    cache = DataCache(root=tmp_path / "cache", max_entries_per_namespace=5)
    params = {"ticker": "000001", "start_date": "2025-01-01", "end_date": "2025-01-31"}
    payload = {"rows": [1, 2, 3]}
    cache.set("prices", params, payload)
    loaded = cache.get("prices", params, ttl_seconds=60)
    assert loaded == payload
