import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage
from pyomnix.agents.models_settings import ModelConfig
from pyomnix.agents.runnables import create_structured_output_chain

from pyomnix.consts import OMNIX_PATH
from pyomnix.omnix_logger import get_logger

from ..data.unified_service import UnifiedDataService
from ..data_models import SentimentAnalysis
from ..utils.normalization import confidence_to_unit
from ..utils.progress import progress
from .state import AgentState, show_agent_reasoning

# 设置日志记录
logger = get_logger("sentiment_agent")


def _news_field(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _keyword_sentiment_score(news_items: list) -> float:
    negative_keywords = [
        "fraud",
        "lawsuit",
        "probe",
        "downgrade",
        "miss",
        "bankruptcy",
        "default",
        "layoff",
        "warning",
        "decline",
    ]
    positive_keywords = [
        "beat",
        "upgrade",
        "guidance raise",
        "approval",
        "expansion",
        "record",
        "growth",
        "buyback",
        "dividend increase",
    ]
    score = 0
    for item in news_items:
        title = _news_field(item, "title", "") or ""
        content = _news_field(item, "content", "") or ""
        text = f"{title} {content}"
        low = text.lower()
        score -= sum(keyword in low for keyword in negative_keywords)
        score += sum(keyword in low for keyword in positive_keywords)
    if not news_items:
        return 0.0
    normalized = score / max(len(news_items), 1)
    return max(-1.0, min(1.0, normalized))


def _insider_activity_score(insider_trades: list) -> float:
    if not insider_trades:
        return 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    for trade in insider_trades:
        if isinstance(trade, dict):
            shares = float(trade.get("transaction_shares", 0) or 0)
        else:
            shares = float(getattr(trade, "transaction_shares", 0) or 0)
        if shares > 0:
            buy_volume += shares
        elif shares < 0:
            sell_volume += abs(shares)
    total = buy_volume + sell_volume
    if total == 0:
        return 0.0
    return (buy_volume - sell_volume) / total


def get_news_sentiment(
    news_list: list,
    num_of_news: int = 7,
    model: str = "deepseek-chat",
    provider_api: str = "deepseek",
    language: str = "Chinese",
    temperature: float | None = None,
    seed: int | None = None,
    max_retries: int = 3,
) -> float:
    """分析新闻情感得分

    Args:
        news_list (list): 新闻列表
        num_of_news (int): 用于分析的新闻数量，默认为7条

    Returns:
        float: 情感得分，范围[-1, 1]，-1最消极，1最积极
    """
    if not news_list:
        return 0.0

    cache_file = OMNIX_PATH / "omnifinan" / "request_cache" / "sentiment_cache.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # 生成新闻内容的唯一标识
    news_key = "|".join(
        [
            f"{_news_field(news, 'title', '')}|"
            f"{(_news_field(news, 'content', '') or '')[:100]}|"
            f"{_news_field(news, 'date', '')}"
            for news in news_list[:num_of_news]
        ]
    )

    # 检查缓存
    if cache_file.exists():
        logger.debug("发现情感分析缓存文件")
        try:
            with open(cache_file, encoding="utf-8") as f:
                cache = json.load(f)
                if news_key in cache:
                    logger.info("使用缓存的情感分析结果")
                    return cache[news_key]["sentiment_score"]
                logger.debug("未找到匹配的情感分析缓存")
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.error("读取情感分析缓存出错: %s", e)
            cache = {}
    else:
        logger.debug("未找到情感分析缓存文件，将创建新文件")
        cache = {}

    # 准备系统消息
    system_message = {
        "role": "system",
        "content": f"""Always respond in {language}. 你是一个专业的市场分析师，擅长解读新闻对股票走势的影响。你需要：
        1. 根据预设评分标准分析新闻情感倾向
        2. 输出结构化分析结果（含评分、置信度、分析依据）

        评分标准速查：
        - 评分范围：[-1,1]，数值越大越积极
        - 示例参照：
          1.0 → 重大利好（如超预期财报）
          0.6 → 常规利好（如新订单）
          -0.8 → 严重利空（如财务造假）

        分析时重点关注：
        1. 业绩相关：财报、业绩预告、营收利润等
        2. 政策影响：行业政策、监管政策、地方政策等
        3. 市场表现：市场份额、竞争态势、商业模式等
        4. 资本运作：并购重组、股权激励、定增配股等
        5. 风险事件：诉讼仲裁、处罚、债务等
        6. 行业地位：技术创新、专利、市占率等
        7. 舆论环境：媒体评价、社会影响等

        请确保分析：
        1. 新闻的真实性和可靠性
        2. 新闻的时效性和影响范围
        3. 对公司基本面的实际影响
        4. 市场的特殊反应规律""",
    }

    # 准备新闻内容
    news_content = "\n\n".join(
        [
            f"标题：{_news_field(news, 'title', '')}\n"
            f"来源：{_news_field(news, 'source', '')}\n"
            f"时间：{_news_field(news, 'publish_time', '')}\n"
            f"内容：{_news_field(news, 'content', '') or _news_field(news, 'title', '')}"
            for news in news_list[:num_of_news]  # 使用指定数量的新闻
        ]
    )

    user_message = {
        "role": "user",
        "content": f"请分析以下上市公司相关新闻的情感倾向：\n{news_content}",
    }

    try:
        model_factory = ModelConfig().setup_model_factory(provider_api).get(provider_api)
        if model_factory is None:
            raise ValueError(f"provider_api not configured: {provider_api}")
        model_kwargs = {"model": model, "max_retries": max_retries}
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if seed is not None:
            model_kwargs["seed"] = seed
        try:
            llm = model_factory(**model_kwargs)
        except TypeError:
            llm = model_factory(model=model, max_retries=max_retries)
        chain = create_structured_output_chain(
            llm=llm,
            schema=SentimentAnalysis,
            system_prompt=system_message["content"],
        )
        result = chain.invoke(
            {
                "messages": [HumanMessage(content=user_message["content"])],
                "summary": "No summary available",
                "user_profile": "No user profile available",
                "structured_memory": "No structured memory available",
                "retrieved_docs": "No retrieved docs available",
                "current_intent": "analyze_news_sentiment",
            }
        )

        if result is None:
            logger.error("LLM返回None，使用中性分数")
            return 0.0

        sentiment_score = result.signal  # 注意这里使用signal而不是sentiment_score

        cache_data = {
            "sentiment_score": sentiment_score,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
        }

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({news_key: cache_data}, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error("写入缓存出错: %s", e)

        return sentiment_score

    except (ValueError, RuntimeError, TypeError, OSError) as e:
        logger.error("分析新闻情感出错: %s", e)
        return 0.0  # 出错时返回中性分数


def sentiment_agent(state: AgentState):
    """Enhanced sentiment analysis combining news sentiment and insider trades"""
    # Constants for sentiment analysis
    min_news_for_llm = 3
    bullish_threshold = 0.2
    bearish_threshold = -0.2
    news_weight = 0.7
    insider_weight = 0.2
    keyword_weight = 0.1

    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")
    data_service = state["metadata"].get("data_service")
    if not isinstance(data_service, UnifiedDataService):
        raise RuntimeError("sentiment_agent requires metadata.data_service")
    num_of_news = data.get("num_of_news", 20)  # Configurable parameter, default 20

    sentiment_analysis = {}
    cutoff_date = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=7)

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching data")

        # 1. Process news sentiment (来自第二版核心算法)
        news_list = data_service.get_company_news(ticker=ticker, end_date=end_date, limit=num_of_news)
        recent_news = []
        for item in news_list:
            publish_time = _news_field(item, "publish_time")
            if not publish_time:
                continue
            try:
                if datetime.strptime(publish_time, "%Y-%m-%d %H:%M:%S") > cutoff_date:
                    recent_news.append(item)
            except (TypeError, ValueError):
                continue

        # Use integrated get_news_sentiment function for news analysis
        # If there are enough news articles, use LLM for deep analysis
        if len(recent_news) >= min_news_for_llm:
            logger.info("Using LLM to analyze %d news articles", len(recent_news))
            max_news = min(num_of_news, len(recent_news))
            news_sentiment = get_news_sentiment(
                recent_news,
                num_of_news=max_news,
                model=state["metadata"]["model_name"],
                provider_api=state["metadata"]["provider_api"],
                language=state["metadata"].get("language", "Chinese"),
                temperature=state["metadata"].get("temperature"),
                seed=state["metadata"].get("llm_seed"),
                max_retries=int(state["metadata"].get("llm_max_retries", 3)),
            )
        # Otherwise use simple sentiment analysis method
        else:
            logger.info("Not enough news (%d), using simple analysis", len(recent_news))
            sentiments = pd.Series([_news_field(n, "sentiment") for n in recent_news]).dropna()
            if len(sentiments) > 0:
                news_sentiment = np.mean(
                    np.where(
                        sentiments == "negative",
                        -1,
                        np.where(sentiments == "positive", 1, 0),
                    )
                )
            else:
                news_sentiment = 0.0

        # 2. Process insider trades (来自第一版)
        insider_trades = data_service.get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )
        insider_signal = _insider_activity_score(insider_trades)
        keyword_signal = _keyword_sentiment_score(recent_news)

        # 3. Combined signal (fusion of two algorithms)
        combined_score = (
            news_weight * news_sentiment
            + insider_weight * insider_signal
            + keyword_weight * keyword_signal
        )

        # Generate signal (using threshold method)
        if combined_score >= bullish_threshold:
            signal = "bullish"
            # Base 80% confidence + enhancement based on score
            confidence = min(0.99, 0.80 + 0.20 * combined_score)
        elif combined_score <= bearish_threshold:
            signal = "bearish"
            confidence = min(0.99, 0.80 + 0.20 * abs(combined_score))
        else:
            signal = "neutral"
            # Closer to 0, more confident in neutral assessment
            confidence = 1.0 - min(0.5 * abs(combined_score), 0.5)

        # Create reasoning text with line breaks to avoid long lines
        # Format parts of the reasoning message
        news_part = f"News: {news_sentiment:.2f} ({len(recent_news)} articles)"
        insider_part = f"Insider signal: {insider_signal:.2f}"
        keyword_part = f"Keyword signal: {keyword_signal:.2f}"
        score_part = f"Combined score: {combined_score:.2f}"

        reasoning = f"{news_part}, {insider_part}, {keyword_part}, {score_part}"

        sentiment_analysis[ticker] = {
            "signal": signal,
            "confidence": round(confidence_to_unit(confidence), 4),
            "reasoning": reasoning,
        }
        progress.update_status("sentiment_agent", ticker, "Done")

    # 保持第一版的输出格式
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Enhanced Sentiment Agent")

    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis
    return state | {"messages": [message], "data": data}
