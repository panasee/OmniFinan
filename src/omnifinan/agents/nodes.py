"""Node factories for OmniFinan trading graph."""

from __future__ import annotations

from collections.abc import Callable

from .state import AgentState


def create_state_node(node_fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
    """Wrap a state mutation function as a LangGraph-compatible node."""
    return node_fn


def create_named_state_node(
    _node_name: str, node_fn: Callable[[AgentState], AgentState]
) -> Callable[[AgentState], AgentState]:
    """Provide a named wrapper (kept for parity with PyOmnix style)."""
    return node_fn


def passthrough_node(state: AgentState) -> AgentState:
    """Minimal start node."""
    return state

