from __future__ import annotations

import json

from omnifinan.core.config import RuntimeConfig


def test_runtime_config_from_json_file(tmp_path):
    config_file = tmp_path / "runtime.json"
    config_file.write_text(
        json.dumps(
            {
                "enable_scratchpad": False,
                "market_type": "china",
                "enabled_analysts": ["ben_graham", "valuation"],
                "debate_rounds": 3,
                "llm": {"model_name": "deepseek-chat", "provider_api": "deepseek"},
            }
        ),
        encoding="utf-8",
    )
    config = RuntimeConfig.from_file(config_file)
    assert config.enable_scratchpad is False
    assert config.debate_rounds == 3
    assert "valuation" in config.enabled_analysts
