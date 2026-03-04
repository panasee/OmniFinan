# OmniFinan

## Runtime LLM Model Configuration Notes

These notes are for OmniFinan runtime/business code (not for `SKILL.md` prompt instructions).

### `_ConfigurableModel` usage

In some environments, `ModelConfig().setup_model_factory(provider_api).get(provider_api)`
returns a LangChain `_ConfigurableModel` object, which is not callable.

Do not call it like:

```python
model_factory(...)
```

Use `.with_config(configurable={...})` with `llm_`-prefixed keys:

```python
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.runnables import create_structured_output_chain

model_obj = ModelConfig().setup_model_factory("deepseek").get("deepseek")
if model_obj is None:
    raise ValueError("provider_api not configured")

llm = model_obj.with_config(
    configurable={
        "llm_model": "deepseek-chat",
        "llm_temperature": 0.2,
        "llm_max_retries": 3,
    }
)

chain = create_structured_output_chain(
    llm=llm,
    schema=YourSchema,
    system_prompt="You are a financial analyst.",
)
```

Known pitfall:
- `{"model": "deepseek-chat"}` may fail for `_ConfigurableModel`
- `{"llm_model": "deepseek-chat"}` is the correct key in this setup

