from __future__ import annotations

from omnifinan.utils.normalization import confidence_to_percent, confidence_to_unit


def test_confidence_to_unit_supports_legacy_formats() -> None:
    assert confidence_to_unit(0.8) == 0.8
    assert confidence_to_unit(80) == 0.8
    assert confidence_to_unit("80%") == 0.8


def test_confidence_to_percent() -> None:
    assert confidence_to_percent(0.42) == 42.0
