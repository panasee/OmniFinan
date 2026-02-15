"""Financial report parsing and synthesis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..llm.client import call_llm
from .report_parser import parse_pdf_report


def run_report_pipeline(
    file_path: str | Path,
    *,
    model_name: str,
    provider_api: str,
    language: str = "Chinese",
    trace=None,
    scratchpad=None,
) -> dict[str, Any]:
    """Extract text from PDF and synthesize key points with LLM."""
    parsed = parse_pdf_report(file_path)
    summary_prompt = (
        f"请使用{language}总结以下财报内容，输出3部分："
        "关键指标变化、风险点、投资观点。\n\n"
        f"{parsed.text[:12000]}"
    )
    synthesis = call_llm(
        prompt=summary_prompt,
        model_name=model_name,
        provider_api=provider_api,
        agent_name="report_pipeline",
        trace=trace,
        scratchpad=scratchpad,
    )
    return {
        "source": parsed.source,
        "raw_text_excerpt": parsed.text[:1000],
        "analysis": synthesis,
    }
