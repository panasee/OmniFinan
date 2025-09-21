import json
import operator
from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage

from pyomnix.consts import SUCCESS_ICON, WAIT_ICON
from pyomnix.omnix_logger import get_logger

# 设置日志记录
logger = get_logger("agent_state")


def merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


# Define agent state


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict[str, Any], merge_dicts]
    metadata: Annotated[dict[str, Any], merge_dicts]


def show_workflow_status(agent_name: str, status: str = "processing"):
    """Display agent workflow status in a clean format.

    Args:
        agent_name: Name of the agent
        status: Status of the agent's work ("processing" or "completed")
    """
    if status == "processing":
        logger.info(f"{WAIT_ICON} {agent_name} is analyzing...")
    else:
        logger.info(f"{SUCCESS_ICON} {agent_name} analysis completed")


def show_agent_reasoning(output, agent_name):
    """Display agent's analysis results."""

    def convert_to_serializable(obj):
        if hasattr(obj, "to_dict"):  # Handle Pandas Series/DataFrame
            return obj.to_dict()
        elif hasattr(obj, "__dict__"):  # Handle custom objects
            return obj.__dict__
        elif isinstance(obj, int | float | bool | str):
            return obj
        elif isinstance(obj, list | tuple):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation

    # logger.info(f"{'='*20} {agent_name} Analysis Details {'='*20}")

    if isinstance(output, dict | list):
        # Convert the output to JSON-serializable format
        serializable_output = convert_to_serializable(output)
        logger.info(json.dumps(serializable_output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            logger.info(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            logger.info(output)

    logger.info(f"{'=' * 20} {agent_name} Analysis Details {'=' * 20}")
