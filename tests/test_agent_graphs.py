"""Regression tests for OmniFinan graph composition."""

from omnifinan.agents.graphs import (
    build_analysis_graph,
    build_execution_graph,
    build_investment_debate_graph,
    build_risk_graph,
    create_trading_graph,
    create_trading_graph_snapshot,
)


def _visible_nodes(graph) -> set[str]:
    return {name for name in graph.nodes.keys() if not name.startswith("__")}


def test_stage_graphs_expose_expected_entry_nodes() -> None:
    assert "analysis_start" in _visible_nodes(build_analysis_graph())
    assert "investment_debate_start" in _visible_nodes(build_investment_debate_graph())
    assert "risk_start" in _visible_nodes(build_risk_graph())
    assert "execution_start" in _visible_nodes(build_execution_graph())


def test_trading_graph_contains_stage_and_decision_nodes() -> None:
    graph = create_trading_graph()
    nodes = _visible_nodes(graph)
    assert {
        "start_node",
        "analysis_start",
        "investment_debate_start",
        "risk_start",
        "execution_start",
        "researcher_bull_agent",
        "researcher_bear_agent",
        "debate_room_agent",
        "risk_management_agent",
        "portfolio_management_agent",
    }.issubset(nodes)

    # Ensure conditional routing survives stage graph merging.
    assert {
        "researcher_bull_agent",
        "researcher_bear_agent",
        "risk_management_agent",
    }.issubset(set(graph.branches.keys()))

    app = graph.compile()
    assert app is not None


def test_trading_graph_snapshot_contains_core_routes() -> None:
    snapshot = create_trading_graph_snapshot()
    assert "nodes" in snapshot
    assert "edges" in snapshot
    assert "conditional_edges" in snapshot

    assert "start_node" in snapshot["nodes"]
    assert "portfolio_management_agent" in snapshot["nodes"]
    assert {"source": "start_node", "target": "market_data_agent"} in snapshot["edges"]

    branches = snapshot["conditional_edges"]
    assert "researcher_bull_agent" in branches
    assert "risk_management_agent" in branches
    assert (
        branches["risk_management_agent"]["should_continue_risk_review"]["portfolio"]
        == "execution_start"
    )
