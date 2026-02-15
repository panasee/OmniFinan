"""Prompt registry for OmniFinan graph orchestration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OmniFinanPrompts:
    workflow_start: str = "Make trading decisions based on the provided data."
    bull_researcher: str = (
        "Summarize the most convincing bullish evidence from functional analysts."
    )
    bear_researcher: str = (
        "Summarize the most convincing bearish evidence from functional analysts."
    )
    judge: str = "Reconcile bull and bear arguments into a balanced investment decision."
    risk: str = "Assess portfolio risk constraints before execution."


PROMPTS = OmniFinanPrompts()

