from __future__ import annotations

from pydantic import BaseModel

from omnifinan.llm.client import call_llm


class _SentimentSchema(BaseModel):
    signal: str


class _DummyModel:
    def with_structured_output(self, schema):  # noqa: ANN001
        return self

    def invoke(self, prompt):  # noqa: ANN001
        class _Response:
            content = "ok"

        return _Response()


class _DummyModelConfig:
    def setup_model_factory(self, provider_api):  # noqa: ANN001
        return {provider_api: lambda **_: _DummyModel()}


def test_call_llm_plain_text(monkeypatch):
    monkeypatch.setattr("omnifinan.llm.client.ModelConfig", _DummyModelConfig)
    response = call_llm("hello", "dummy-model", "dummy-provider")
    assert response == "ok"
