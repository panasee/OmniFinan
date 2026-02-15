"""Compatibility re-exports for the unified LLM helper module."""

from ..llm.client import call_llm, create_default_response, extract_json_from_response

__all__ = ["call_llm", "create_default_response", "extract_json_from_response"]
