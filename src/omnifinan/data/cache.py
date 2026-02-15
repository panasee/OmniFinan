"""Deterministic file cache for data API calls."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pyomnix.consts import OMNIX_PATH


class DataCache:
    def __init__(self, root: Path | None = None, max_entries_per_namespace: int = 2000):
        # Rooted under OMNIX_PATH/omnifinan for reusable local datasets.
        self.root = root or (OMNIX_PATH / "omnifinan")
        self.request_root = self.root / "request_cache"
        self.dataset_root = self.root / "datasets"
        self.request_root.mkdir(parents=True, exist_ok=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_namespace = max_entries_per_namespace

    def _request_key_path(self, namespace: str, params: dict[str, Any]) -> Path:
        key_json = json.dumps(params, ensure_ascii=False, sort_keys=True)
        key_hash = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
        ns_dir = self.request_root / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{key_hash}.json"

    def get(self, namespace: str, params: dict[str, Any], ttl_seconds: int | None = None) -> Any | None:
        path = self._request_key_path(namespace, params)
        if not path.exists():
            return None
        if ttl_seconds is not None and ttl_seconds > 0:
            age = time.time() - path.stat().st_mtime
            if age > ttl_seconds:
                return None
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("data")
        except Exception:
            return None

    def set(self, namespace: str, params: dict[str, Any], data: Any) -> None:
        path = self._request_key_path(namespace, params)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"data": data}, f, ensure_ascii=False, indent=2)
        self._cleanup_namespace(self.request_root, namespace)

    def _sanitize_key(self, key: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in key)

    def _dataset_path(self, namespace: str, key: str) -> Path:
        ns_dir = self.dataset_root / namespace
        ns_dir.mkdir(parents=True, exist_ok=True)
        return ns_dir / f"{self._sanitize_key(key)}.json"

    def get_dataset(self, namespace: str, key: str) -> Any | None:
        path = self._dataset_path(namespace, key)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("data")
        except Exception:
            return None

    def set_dataset(self, namespace: str, key: str, data: Any) -> None:
        path = self._dataset_path(namespace, key)
        payload = {
            "key": key,
            "updated_at": int(time.time()),
            "data": data,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._cleanup_namespace(self.dataset_root, namespace)

    def cleanup_expired(self, ttl_seconds: int) -> int:
        removed = 0
        if ttl_seconds <= 0:
            return removed
        cutoff = time.time() - ttl_seconds
        for ns_dir in self.request_root.iterdir():
            if not ns_dir.is_dir():
                continue
            for item in ns_dir.glob("*.json"):
                if item.stat().st_mtime < cutoff:
                    try:
                        item.unlink()
                        removed += 1
                    except Exception:
                        continue
        return removed

    def _cleanup_namespace(self, base_dir: Path, namespace: str) -> None:
        ns_dir = base_dir / namespace
        if not ns_dir.exists():
            return
        files = sorted(
            ns_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(files) <= self.max_entries_per_namespace:
            return
        for stale in files[self.max_entries_per_namespace :]:
            try:
                stale.unlink()
            except Exception:
                continue
