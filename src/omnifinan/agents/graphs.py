"""Graph builders for OmniFinan agent orchestration."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from ..utils.analysts import get_analyst_nodes, get_default_analyst_keys
from .debate_room import debate_room_agent
from .edges import should_continue_investment_debate, should_continue_risk_review
from .market_data import market_data_agent
from .nodes import create_named_state_node, create_state_node, passthrough_node
from .portfolio_manager import portfolio_management_agent
from .researcher_bear import researcher_bear_agent
from .researcher_bull import researcher_bull_agent
from .risk_manager import risk_management_agent
from .state import AgentState


def _resolve_active_analysts(selected_analysts: list[str] | None) -> list[str]:
    analyst_nodes = get_analyst_nodes()
    active_analysts = selected_analysts or get_default_analyst_keys()
    active_analysts = [key for key in active_analysts if key in analyst_nodes]
    if not active_analysts:
        active_analysts = get_default_analyst_keys()
    return active_analysts


def build_analysis_graph(selected_analysts: list[str] | None = None) -> StateGraph:
    """Build a subgraph for market-data + domain-analyst stage."""
    analyst_nodes = get_analyst_nodes()
    active_analysts = _resolve_active_analysts(selected_analysts)

    graph = StateGraph(AgentState)
    graph.add_node("analysis_start", create_state_node(passthrough_node))
    graph.add_node(
        "market_data_agent", create_named_state_node("market_data_agent", market_data_agent)
    )
    graph.add_edge("analysis_start", "market_data_agent")

    for analyst_key in active_analysts:
        node_name, node_fn = analyst_nodes[analyst_key]
        graph.add_node(node_name, create_named_state_node(node_name, node_fn))
        graph.add_edge("market_data_agent", node_name)

    graph.set_entry_point("analysis_start")
    return graph


def build_investment_debate_graph() -> StateGraph:
    """Build bull/bear debate subgraph ending at judge node."""
    graph = StateGraph(AgentState)
    graph.add_node(
        "investment_debate_start",
        create_state_node(passthrough_node),
    )
    graph.add_node(
        "researcher_bull_agent",
        create_named_state_node("researcher_bull_agent", researcher_bull_agent),
    )
    graph.add_node(
        "researcher_bear_agent",
        create_named_state_node("researcher_bear_agent", researcher_bear_agent),
    )
    graph.add_node(
        "debate_room_agent",
        create_named_state_node("debate_room_agent", debate_room_agent),
    )
    graph.add_edge("investment_debate_start", "researcher_bull_agent")
    graph.add_conditional_edges(
        "researcher_bull_agent",
        should_continue_investment_debate,
        {
            "bear": "researcher_bear_agent",
            "bull": "researcher_bull_agent",
            "judge": "debate_room_agent",
        },
    )
    graph.add_conditional_edges(
        "researcher_bear_agent",
        should_continue_investment_debate,
        {
            "bear": "researcher_bear_agent",
            "bull": "researcher_bull_agent",
            "judge": "debate_room_agent",
        },
    )
    graph.set_entry_point("investment_debate_start")
    return graph


def build_risk_graph() -> StateGraph:
    """Build risk review subgraph with optional loop by configured rounds."""
    graph = StateGraph(AgentState)
    graph.add_node("risk_start", create_state_node(passthrough_node))
    graph.add_node(
        "risk_management_agent",
        create_named_state_node("risk_management_agent", risk_management_agent),
    )
    graph.add_edge("risk_start", "risk_management_agent")
    graph.add_conditional_edges(
        "risk_management_agent",
        should_continue_risk_review,
        {
            "risk": "risk_management_agent",
            "portfolio": "execution_start",
        },
    )
    graph.set_entry_point("risk_start")
    return graph


def build_execution_graph() -> StateGraph:
    """Build execution subgraph."""
    graph = StateGraph(AgentState)
    graph.add_node("execution_start", create_state_node(passthrough_node))
    graph.add_node(
        "portfolio_management_agent",
        create_named_state_node("portfolio_management_agent", portfolio_management_agent),
    )
    graph.add_edge("execution_start", "portfolio_management_agent")
    graph.set_entry_point("execution_start")
    return graph


def _merge_stage_graph(target: StateGraph, stage: StateGraph) -> None:
    """Merge nodes and edges from a stage graph into target graph."""
    for node_name, node_spec in stage.nodes.items():
        if node_name.startswith("__"):
            continue
        if node_name not in target.nodes:
            target.add_node(node_name, node_spec.runnable)
    for source, target_node in stage.edges:
        if source.startswith("__") or target_node.startswith("__"):
            continue
        target.add_edge(source, target_node)
    for source, branch_map in stage.branches.items():
        if source.startswith("__"):
            continue
        for branch_spec in branch_map.values():
            target.add_conditional_edges(
                source,
                branch_spec.path,
                branch_spec.ends,
            )


def export_graph_snapshot(graph: StateGraph) -> dict[str, Any]:
    """Return a stable, serializable snapshot of graph structure for debugging."""
    nodes = sorted(name for name in graph.nodes.keys() if not name.startswith("__"))
    edge_pairs = sorted(
        (str(source), str(target))
        for source, target in graph.edges
        if not source.startswith("__")
    )
    edges = [{"source": source, "target": target} for source, target in edge_pairs]
    conditional_edges: dict[str, dict[str, dict[str, str]]] = {}
    for source, branch_map in graph.branches.items():
        if source.startswith("__"):
            continue
        conditional_edges[source] = {}
        for branch_name, branch_spec in branch_map.items():
            ends = branch_spec.ends or {}
            conditional_edges[source][branch_name] = {
                str(route): str(target) for route, target in sorted(ends.items())
            }
    return {
        "nodes": nodes,
        "edges": edges,
        "conditional_edges": conditional_edges,
    }


def create_trading_graph(selected_analysts: list[str] | None = None) -> StateGraph:
    """Build the full OmniFinan trading graph using modular node/edge organization."""
    analysis_graph = build_analysis_graph(selected_analysts)
    investment_graph = build_investment_debate_graph()
    risk_graph = build_risk_graph()
    execution_graph = build_execution_graph()
    analyst_nodes = get_analyst_nodes()
    active_analysts = _resolve_active_analysts(selected_analysts)
    graph = StateGraph(AgentState)
    graph.add_node("start_node", create_state_node(passthrough_node))
    graph.add_edge("start_node", "market_data_agent")

    _merge_stage_graph(graph, analysis_graph)
    _merge_stage_graph(graph, investment_graph)
    _merge_stage_graph(graph, risk_graph)
    _merge_stage_graph(graph, execution_graph)

    for analyst_key in active_analysts:
        node_name, _ = analyst_nodes[analyst_key]
        graph.add_edge(node_name, "investment_debate_start")
    graph.add_edge("debate_room_agent", "risk_start")
    graph.add_edge("portfolio_management_agent", END)
    graph.set_entry_point("start_node")
    return graph


def create_trading_graph_snapshot(selected_analysts: list[str] | None = None) -> dict[str, Any]:
    """Build the trading graph and export a debug snapshot."""
    graph = create_trading_graph(selected_analysts=selected_analysts)
    return export_graph_snapshot(graph)

