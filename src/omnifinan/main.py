import argparse
import json
import sys
from datetime import datetime

import questionary
from colorama import Fore, Style, init
from dateutil.relativedelta import relativedelta
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from pyomnix.consts import OMNIX_PATH

# Import all agents
from pyomnix.financial.agents.debate_room import debate_room_agent
from pyomnix.financial.agents.market_data import market_data_agent
from pyomnix.financial.agents.portfolio_manager import portfolio_management_agent
from pyomnix.financial.agents.researcher_bear import researcher_bear_agent
from pyomnix.financial.agents.researcher_bull import researcher_bull_agent
from pyomnix.financial.agents.risk_manager import risk_management_agent
from pyomnix.financial.agents.state import AgentState
from pyomnix.financial.utils.analysts import ANALYST_ORDER, get_analyst_nodes
from pyomnix.financial.utils.display import print_trading_output
from pyomnix.financial.utils.progress import progress
from pyomnix.financial.utils.visualize import save_graph_as_png
from pyomnix.omnix_logger import get_logger

logger = get_logger("main")

init(autoreset=True)


def parse_hedge_fund_response(response):
    """Parses a JSON string and returns a dictionary."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}\nResponse: {response!r}")
        return None
    except TypeError as e:
        print(
            f"Invalid response type (expected string, got {type(response).__name__}): {e}"
        )
        return None
    except Exception as e:
        print(f"Unexpected error while parsing response: {e}\nResponse: {response!r}")
        return None


##### Run the Hedge Fund #####
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict | None,
    show_reasoning: bool = False,
    selected_analysts: list[str] | None = None,
    model_name: str = "deepseek-chat",
    provider_api: str = "deepseek",
    language: str = "Chinese",
):
    # Start progress tracking
    progress.start()

    try:
        # Create a new workflow if analysts are customized
        if selected_analysts is not None:
            workflow = create_workflow(selected_analysts)
            agent = workflow.compile()
        else:
            # Use all available analysts if none specified
            workflow = create_workflow()
            agent = workflow.compile()

        final_state = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data.",
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {},
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "provider_api": provider_api,
                    "language": language,
                },
            },
        )

        json.dump(final_state, open(OMNIX_PATH / "financial" / "hedge_fund_output.json", "w"))

        return {
            "decisions": parse_hedge_fund_response(final_state["messages"][-1].content),
            "analyst_signals": final_state["data"]["analyst_signals"],
        }
    finally:
        # Stop progress tracking
        progress.stop()


def start(state: AgentState):
    """Initialize the workflow with the input message."""
    return state


def create_workflow(selected_analysts: list[str] | None = None):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    # Get analyst nodes from the configuration
    analyst_nodes = get_analyst_nodes()

    # Default to all analysts if none selected
    if selected_analysts is None:
        logger.info("No analysts selected, using all available analysts.")
        selected_analysts = list(analyst_nodes.keys())

    # Always add market data agent first
    workflow.add_node("market_data_agent", market_data_agent)
    workflow.add_edge("start_node", "market_data_agent")

    # Add selected analyst nodes
    analyst_agent_nodes = []
    for analyst_key in selected_analysts:
        # Skip special agents that are handled separately
        if analyst_key in [
            "market_data",
            "researcher_bull",
            "researcher_bear",
            "debate_room",
        ]:
            continue

        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("market_data_agent", node_name)
        analyst_agent_nodes.append(node_name)

    # Add researcher agents
    workflow.add_node("researcher_bull_agent", researcher_bull_agent)
    workflow.add_node("researcher_bear_agent", researcher_bear_agent)

    # Connect analyst nodes to researcher agents
    for node_name in analyst_agent_nodes:
        workflow.add_edge(node_name, "researcher_bull_agent")
        workflow.add_edge(node_name, "researcher_bear_agent")

    # Add debate room
    workflow.add_node("debate_room_agent", debate_room_agent)
    workflow.add_edge("researcher_bull_agent", "debate_room_agent")
    workflow.add_edge("researcher_bear_agent", "debate_room_agent")

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_management_agent", portfolio_management_agent)

    # Connect debate room to risk management
    workflow.add_edge("debate_room_agent", "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_management_agent")
    workflow.add_edge("portfolio_management_agent", END)

    workflow.set_entry_point("start_node")
    return workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the hedge fund trading system")
    parser.add_argument(
        "--initial-cash",
        type=float,
        default=100000.0,
        help="Initial cash position. Defaults to 100000.0, set to 0 to close portfolio)",
    )
    parser.add_argument(
        "--margin-requirement",
        type=float,
        default=0.0,
        help="Initial margin requirement. Defaults to 0.0",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated list of stock ticker symbols",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date (YYYY-MM-DD). Defaults to 3 months before end date",
    )
    parser.add_argument(
        "--end-date", type=str, help="End date (YYYY-MM-DD). Defaults to today"
    )
    parser.add_argument(
        "--language", type=str, help="The language to use for the LLM. Defaults to Chinese."
    )
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Show reasoning from each agent"
    )
    parser.add_argument(
        "--show-agent-graph", action="store_true", help="Show the agent graph"
    )

    args = parser.parse_args()

    # Parse tickers from comma-separated string
    tickers = [ticker.strip() for ticker in args.tickers.split(",")]

    # Select analysts
    selected_analysts = None
    choices = questionary.checkbox(
        "Select your AI analysts.",
        choices=[
            questionary.Choice(display, value=value) for display, value in ANALYST_ORDER
        ],
        instruction=(
            "\n\nInstructions: \n"
            "1. Press Space to select/unselect analysts.\n"
            "2. Press 'a' to select/unselect all.\n"
            "3. Press Enter when done to run the hedge fund.\n"
        ),
        validate=lambda x: len(x) > 0 or "You must select at least one analyst.",
        style=questionary.Style(
            [
                ("checkbox-selected", "fg:green"),
                ("selected", "fg:green noinherit"),
                ("highlighted", "noinherit"),
                ("pointer", "noinherit"),
            ]
        ),
    ).ask()

    if not choices:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        selected_analysts = choices
        print(
            f"\nSelected analysts: {', '.join(Fore.GREEN + choice.title().replace('_', ' ') + Style.RESET_ALL for choice in choices)}\n"
        )

    # Select LLM model
    llm_choice = None
    provider_api = None

    # Use the standard cloud-based LLM selection
    llm_choice = questionary.select(
        "Select your LLM model:",
        choices=[
            questionary.Choice("DeepSeek Chat", "deepseek-chat"),
            questionary.Choice("DeepSeek R1", "deepseek-reasoner"),
            questionary.Choice("DeepSeek Chat Siliconflow Free", "siliconflow-deepseek-chat-free"),
            questionary.Choice("DeepSeek Reasoner Siliconflow Free", "siliconflow-deepseek-reasoner-free"),
            questionary.Choice("DeepSeek Chat Siliconflow Pro", "siliconflow-deepseek-chat-pro"),
            questionary.Choice("DeepSeek Reasoner Siliconflow Pro", "siliconflow-deepseek-reasoner-pro"),
            questionary.Choice("DeepSeek Chat VolcEngine", "volcengine-deepseek"),
            questionary.Choice("Claude 3 Sonnet", "claude-3-sonnet-20240229"),
            questionary.Choice("GPT-4o", "gpt-4o"),
            questionary.Choice("GPT-4 Turbo", "gpt-4-turbo-2024-04-09"),
            questionary.Choice("Gemini 1.5 Pro", "gemini-1.5-pro"),
        ],
        style=questionary.Style(
            [
                ("selected", "fg:green bold"),
                ("pointer", "fg:green bold"),
                ("highlighted", "fg:green"),
                ("answer", "fg:green bold"),
            ]
        ),
    ).ask()

    if not llm_choice:
        print("\n\nInterrupt received. Exiting...")
        sys.exit(0)
    else:
        # Determine provider based on model name
        if "gpt" in llm_choice.lower():
            provider_api = "openai"
        elif "claude" in llm_choice.lower():
            provider_api = "anthropic"
        elif "gemini" in llm_choice.lower():
            provider_api = "google"
        elif "deepseek" in llm_choice.lower():
            match llm_choice:
                case "siliconflow-deepseek-chat-free":
                    provider_api = "siliconflow-deepseek"
                    llm_choice = "deepseek-ai/DeepSeek-V3"
                case "siliconflow-deepseek-reasoner-free":
                    provider_api = "siliconflow-deepseek"
                    llm_choice = "deepseek-ai/DeepSeek-R1"
                case "siliconflow-deepseek-chat-pro":
                    provider_api = "siliconflow-deepseek"
                    llm_choice = "Pro/deepseek-ai/DeepSeek-V3"
                case "siliconflow-deepseek-reasoner-pro":
                    provider_api = "siliconflow-deepseek"
                    llm_choice = "Pro/deepseek-ai/DeepSeek-R1"
                case _:
                    provider_api = "deepseek"
        else:
            provider_api = "deepseek"  # Default to deepseek
            llm_choice = "deepseek-chat"

        print(
            f"\nSelected {Fore.CYAN}{provider_api}{Style.RESET_ALL} model: {Fore.GREEN + Style.BRIGHT}{llm_choice}{Style.RESET_ALL}\n"
        )

    # Create the workflow with selected analysts
    workflow = create_workflow(selected_analysts)
    app = workflow.compile()

    if args.show_agent_graph:
        file_path = OMNIX_PATH / "financial" / "agent-graph"
        file_path.mkdir(parents=True, exist_ok=True)
        file_path /= "graph.png"
        save_graph_as_png(app, file_path)

    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError as err:
            raise ValueError("Start date must be in YYYY-MM-DD format") from err

    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError as err:
            raise ValueError("End date must be in YYYY-MM-DD format") from err

    # Set the start and end dates
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")
    if not args.start_date:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        start_date = (end_date_obj - relativedelta(months=3)).strftime("%Y-%m-%d")
    else:
        start_date = args.start_date

    # Initialize portfolio with cash amount and stock positions
    if args.initial_cash == 0:
        portfolio = {}
    else:
        portfolio = {
            "cash": args.initial_cash,  # Initial cash amount
            "margin_requirement": args.margin_requirement,  # Initial margin requirement
            "margin_used": 0.0,  # total margin usage across all short positions
            "positions": {
                ticker: {
                    "long": 0,  # Number of shares held long
                    "short": 0,  # Number of shares held short
                    "long_cost_basis": 0.0,  # Average cost basis for long positions
                    "short_cost_basis": 0.0,  # Average price at which shares were sold short
                    "short_margin_used": 0.0,  # Dollars of margin used for this ticker's short
                }
                for ticker in tickers
            },
            "realized_gains": {
                ticker: {
                    "long": 0.0,  # Realized gains from long positions
                    "short": 0.0,  # Realized gains from short positions
                }
                for ticker in tickers
            },
        }

    # Run the hedge fund
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning,
        selected_analysts=selected_analysts,
        model_name=llm_choice,
        provider_api=provider_api,
        language=args.language,
    )
    print_trading_output(result)
