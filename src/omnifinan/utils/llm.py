"""Helper functions for LLM"""

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from pyomnix.model_interface.models import RawModels
from pyomnix.omnix_logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def call_llm(
    prompt: Any,
    model_name: str,
    provider_api: str,
    pydantic_model: type[T] | None = None,
    max_retries: int = 3,
    default_factory=None,
    agent_name: str | None = None,
) -> T | str:
    """
    Makes an LLM call with retry logic, handling both JSON supported and non-JSON supported models.

    Args:
        prompt: The prompt to send to the LLM
        model_name: Name of the model to use
        provider_api: Provider of the model
        pydantic_model: The Pydantic model class to structure the output
        max_retries: Maximum number of retries (default: 3)
        default_factory: Optional factory function to create default response on failure

    Returns:
        An instance of the specified Pydantic model or the response string
    """

    llm = RawModels(provider_api)
    try:
        if pydantic_model is not None:
            result = llm.chat_completion(
                provider_api=provider_api,
                model=model_name,
                messages=prompt,
                max_retries=max_retries,
                schema=pydantic_model,
            )[0]
        else:
            result = llm.chat_completion(
                provider_api=provider_api,
                model=model_name,
                messages=prompt,
                max_retries=max_retries,
            )[0].content
        return result
    except Exception as e:
        logger.error(f"Error in LLM call, returning default response: {e!s}")
        # Use default_factory if provided, otherwise create a basic default
        if default_factory:
            return default_factory()
        return create_default_response(pydantic_model)


def create_default_response(model_class: type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif (
            hasattr(field.annotation, "__origin__")
            and field.annotation.__origin__ == dict
        ):
            default_values[field_name] = {}
        # For other types (like Literal), try to use the first allowed value
        elif hasattr(field.annotation, "__args__"):
            default_values[field_name] = field.annotation.__args__[0]
        else:
            default_values[field_name] = None

    return model_class(**default_values)


def extract_json_from_response(content: str) -> dict | None:
    """Extracts JSON from markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7 :]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from response: {e}")
    return None
