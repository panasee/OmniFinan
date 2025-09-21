"""
Debate Room Agent

Facilitates debate between bull and bear researchers to reach a balanced conclusion.
"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from pyomnix.omnix_logger import get_logger

from ..utils.llm import call_llm
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning

# 获取日志记录器
logger = get_logger("debate_room")


class DebateAnalysis(BaseModel):
    """Model for the debate room output."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(description="Confidence level between 0 and 1")
    bull_confidence: float = Field(description="Bullish researcher confidence")
    bear_confidence: float = Field(description="Bearish researcher confidence")
    confidence_diff: float = Field(
        description="Difference between bull and bear confidence"
    )
    llm_score: float | None = Field(
        description="LLM score between -1 and 1", default=None
    )
    llm_analysis: str | None = Field(
        description="LLM analysis of the debate", default=None
    )
    llm_reasoning: str | None = Field(
        description="LLM reasoning for the score", default=None
    )
    mixed_confidence_diff: float = Field(description="Mixed confidence difference")
    debate_summary: list[str] = Field(description="Summary of the debate points")
    reasoning: str = Field(description="Reasoning for the final signal")


class LLMDebateOutput(BaseModel):
    """Model for the LLM debate output."""

    analysis: str = Field(description="Detailed analysis of the debate")
    score: float = Field(description="Score between -1 (bearish) and 1 (bullish)")
    reasoning: str = Field(description="Reasoning for the score")


def debate_room_agent(state: AgentState) -> AgentState:
    """
    Facilitates debate between bull and bear researchers to reach a balanced conclusion.

    Args:
        state: The current state of the agent system

    Returns:
        Updated state with debate analysis
    """
    progress.update_status("debate_room_agent", None, "Starting debate analysis")
    show_reasoning = state["metadata"].get("show_reasoning", False)

    # Get the tickers
    tickers = state["data"].get("tickers", [])

    # Initialize results container
    debate_analyses: dict[str, DebateAnalysis] = {}

    for ticker in tickers:
        progress.update_status(
            "debate_room_agent", ticker, "Collecting researcher perspectives"
        )

        # Get bullish and bearish analyses for this ticker
        bullish_analyses = state["data"].get("bullish_analyses", {})
        bearish_analyses = state["data"].get("bearish_analyses", {})

        bull_thesis = bullish_analyses.get(ticker, {})
        bear_thesis = bearish_analyses.get(ticker, {})

        if not bull_thesis or not bear_thesis:
            logger.warning(f"Missing bull or bear thesis for {ticker}, skipping debate")
            continue

        logger.info(
            f"已获取看多观点(置信度: {bull_thesis.get('confidence', 0)})和看空观点(置信度: {bear_thesis.get('confidence', 0)})"
        )

        # 比较置信度级别
        bull_confidence = bull_thesis.get("confidence", 0)
        bear_confidence = bear_thesis.get("confidence", 0)

        # 分析辩论观点
        debate_summary = []
        debate_summary.append("Bullish Arguments:")
        for point in bull_thesis.get("thesis_points", []):
            debate_summary.append(f"+ {point}")

        debate_summary.append("\nBearish Arguments:")
        for point in bear_thesis.get("thesis_points", []):
            debate_summary.append(f"- {point}")

        # 收集所有研究员的论点，准备发给 LLM
        all_perspectives = {
            "bullish": {
                "confidence": bull_confidence,
                "thesis_points": bull_thesis.get("thesis_points", []),
            },
            "bearish": {
                "confidence": bear_confidence,
                "thesis_points": bear_thesis.get("thesis_points", []),
            },
        }

        logger.info(f"准备让 LLM 分析 {len(all_perspectives)} 个研究员的观点")

        progress.update_status("debate_room_agent", ticker, "Analyzing with LLM")

        # 构建发送给 LLM 的提示
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"Always respond in {state['metadata'].get('language', 'Chinese')}. You are a professional financial analyst. "
#                    Please provide your analysis in English only, not in Chinese or any other language.",
                ),
                (
                    "user",
                    """
                    你是一位专业的金融分析师，请分析以下投资研究员的观点，并给出你的第三方分析:

                    {perspectives}

                    请提供以下格式的分析:
                    - 详细分析，评估各方观点的优劣，并指出你认为最有说服力的论点
                    - 你的评分，从 -1.0(极度看空) 到 1.0(极度看多)，0 表示中性
                    - 你给出这个评分的简要理由
                    """

#                    务必确保你的回复是有效的格式，且包含上述所有字段。回复必须使用英文，不要使用中文或其他语言。
#                    """,
                ),
            ]
        )

        # Format the perspectives for the prompt
        perspectives_text = ""
        for perspective, data in all_perspectives.items():
            perspectives_text += (
                f"\n{perspective.upper()} 观点 (置信度: {data['confidence']}):\n"
            )
            for point in data["thesis_points"]:
                perspectives_text += f"- {point}\n"

        # Prepare the prompt
        formatted_prompt = prompt_template.format(perspectives=perspectives_text)

        # Call the LLM
        llm_analysis = None
        llm_score = 0  # 默认为中性

        try:
            # Call LLM with the Pydantic model
            llm_output = call_llm(
                prompt=formatted_prompt,
                model_name=state["metadata"].get("model_name", "deepseek-chat"),
                provider_api=state["metadata"].get("provider_api", "deepseek"),
                pydantic_model=LLMDebateOutput,
                agent_name="debate_room_agent",
            )

            llm_analysis = {
                "analysis": llm_output.analysis,
                "score": llm_output.score,
                "reasoning": llm_output.reasoning,
            }
            llm_score = float(llm_output.score)
            # 确保分数在有效范围内
            llm_score = max(min(llm_score, 1.0), -1.0)
            logger.info(f"成功解析 LLM 回复，评分: {llm_score}")

        except Exception as e:
            logger.error(f"调用 LLM 失败: {e}")
            llm_analysis = {
                "analysis": "LLM API call failed",
                "score": 0,
                "reasoning": "API error",
            }

        # 计算混合置信度差异
        confidence_diff = bull_confidence - bear_confidence

        # 默认 LLM 权重为 30%
        llm_weight = 0.3

        # 将 LLM 评分（-1 到 1范围）转换为与 confidence_diff 相同的比例
        # 计算混合置信度差异
        mixed_confidence_diff = (
            1 - llm_weight
        ) * confidence_diff + llm_weight * llm_score

        logger.info(
            f"计算混合置信度差异: 原始差异={confidence_diff:.4f}, LLM评分={llm_score:.4f}, 混合差异={mixed_confidence_diff:.4f}"
        )

        progress.update_status("debate_room_agent", ticker, "Determining final signal")

        # 基于混合置信度差异确定最终建议
        if abs(mixed_confidence_diff) < 0.1:  # 接近争论
            final_signal = "neutral"
            reasoning = "Balanced debate with strong arguments on both sides"
            confidence = max(bull_confidence, bear_confidence)
        elif mixed_confidence_diff > 0:  # 看多胜出
            final_signal = "bullish"
            reasoning = "Bullish arguments more convincing"
            confidence = bull_confidence
        else:  # 看空胜出
            final_signal = "bearish"
            reasoning = "Bearish arguments more convincing"
            confidence = bear_confidence

        logger.info(f"最终投资信号: {final_signal}, 置信度: {confidence}")

        # Create the debate analysis
        debate_analysis = DebateAnalysis(
            signal=final_signal,
            confidence=confidence,
            bull_confidence=bull_confidence,
            bear_confidence=bear_confidence,
            confidence_diff=confidence_diff,
            llm_score=llm_score,
            llm_analysis=llm_analysis.get("analysis") if llm_analysis else None,
            llm_reasoning=llm_analysis.get("reasoning") if llm_analysis else None,
            mixed_confidence_diff=mixed_confidence_diff,
            debate_summary=debate_summary,
            reasoning=reasoning,
        )

        debate_analyses[ticker] = debate_analysis
        progress.update_status("debate_room_agent", ticker, "Debate analysis complete")

    # Create messages for each ticker
    messages = state["messages"].copy()
    for ticker, analysis in debate_analyses.items():
        message = HumanMessage(
            content=json.dumps(analysis.model_dump()),
            name="debate_room_agent",
        )
        messages.append(message)

        if show_reasoning:
            show_agent_reasoning(analysis.model_dump(), f"Debate Room - {ticker}")

    # Update state metadata
    if debate_analyses and show_reasoning:
        state["metadata"]["agent_reasoning"] = next(
            iter(debate_analyses.values())
        ).model_dump()

    progress.update_status("debate_room_agent", None, "Done")

    # Update state with debate analyses
    return {
        "messages": messages,
        "data": {
            **state["data"],
            "debate_analyses": {
                ticker: analysis.model_dump()
                for ticker, analysis in debate_analyses.items()
            },
        },
        "metadata": state["metadata"],
    }
