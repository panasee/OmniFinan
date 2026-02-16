from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class MarketType(str, Enum):
    """Market type enum to distinguish between different regions' stock markets"""

    US = "us"
    CHINA = "china"
    CHINA_SZ = "sz"
    CHINA_SH = "sh"
    CHINA_BJ = "bj"
    HK = "hongkong"
    UNKNOWN = "unknown"


class GeneralAnalysis(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: dict | str | None = None


class TechnicalAnalysis(BaseModel):
    """Technical analysis signals"""

    trend: Annotated[str, Field(description="The trend of the stock")]
    mean_reversion: Annotated[str, Field(description="The mean reversion of the stock")]
    momentum: Annotated[str, Field(description="The momentum of the stock")]
    volatility: Annotated[str, Field(description="The volatility of the stock")]


class SentimentAnalysis(BaseModel):
    signal: Annotated[
        float,
        Field(
            ge=-1,
            le=1,
            description="""根据以下细则精确赋值：
        - 1表示极其积极（例如：重大利好消息、超预期业绩、行业政策支持）
        - 0.5到0.9表示积极（例如：业绩增长、新项目落地、获得订单）
        - 0.1到0.4表示轻微积极（例如：小额合同签订、日常经营正常）
        - 0表示中性（例如：日常公告、人事变动、无重大影响的新闻）
        - -0.1到-0.4表示轻微消极（例如：小额诉讼、非核心业务亏损）
        - -0.5到-0.9表示消极（例如：业绩下滑、重要客户流失、行业政策收紧）
        - -1表示极其消极（例如：重大违规、核心业务严重亏损、被监管处罚）
        """,
        ),
    ]
    confidence: Annotated[
        float,
        Field(
            ge=0,
            le=1,
            description="基于信息源可靠性、数据完整性、历史准确率计算的置信度",
        ),
    ]
    reasoning: str = Field(
        description="按此模板分析：『1) 核心事件：[摘要] 2) 影响机制：[关联到评分细则] 3) 市场预期：[对比历史同类事件]』"
    )


class Price(BaseModel):
    """Price data model for stock prices"""

    open: float
    close: float
    high: float
    low: float
    volume: int
    time: str
    # Optional fields for Chinese market
    amount: float | None = None  # 成交额
    amplitude: float | None = None  # 振幅
    pct_change: float | None = None  # 涨跌幅
    change_amount: float | None = None  # 涨跌额
    turnover: float | None = None  # 换手率
    market: MarketType | None = None  # Market type


class PriceResponse(BaseModel):
    """Response model for price data"""

    ticker: str
    prices: list[Price]
    market: MarketType | None = None


class FinancialMetrics(BaseModel):
    """Financial metrics data model for stock analysis"""

    ticker: str
    report_period: str
    period: str
    currency: str
    market: MarketType | None = None  # Market type
    market_cap: float | None = None
    enterprise_value: float | None = None
    price_to_earnings_ratio: float | None = None
    price_to_book_ratio: float | None = None
    price_to_sales_ratio: float | None = None
    enterprise_value_to_ebitda_ratio: float | None = None
    enterprise_value_to_revenue_ratio: float | None = None
    free_cash_flow_yield: float | None = None
    peg_ratio: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None
    return_on_invested_capital: float | None = None
    asset_turnover: float | None = None
    inventory_turnover: float | None = None
    receivables_turnover: float | None = None
    days_sales_outstanding: float | None = None
    operating_cycle: float | None = None
    working_capital_turnover: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    cash_ratio: float | None = None
    operating_cash_flow_ratio: float | None = None
    debt_to_equity: float | None = None
    liabilities_to_equity: float | None = None
    interest_bearing_debt: float | None = None
    debt_to_assets: float | None = None
    interest_coverage: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    book_value_growth: float | None = None
    earnings_per_share_growth: float | None = None
    free_cash_flow_growth: float | None = None
    operating_income_growth: float | None = None
    ebitda_growth: float | None = None
    payout_ratio: float | None = None
    earnings_per_share: float | None = None
    book_value_per_share: float | None = None
    free_cash_flow_per_share: float | None = None

    # Additional fields for Chinese market
    net_income: float | None = None  # 净利润
    operating_revenue: float | None = None  # 营业总收入
    operating_profit: float | None = None  # 营业利润
    working_capital: float | None = None  # 营运资金
    depreciation_and_amortization: float | None = None  # 折旧与摊销
    capital_expenditure: float | None = None  # 资本支出


class FinancialMetricsResponse(BaseModel):
    financial_metrics: list[FinancialMetrics]


class LineItem(BaseModel):
    ticker: str
    report_period: str
    period: str
    currency: str
    net_income: float | None = None
    operating_revenue: float | None = None
    operating_profit: float | None = None
    working_capital: float | None = None
    depreciation_and_amortization: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    market: MarketType | None = None
    model_config = {"extra": "allow"}


class LineItemResponse(BaseModel):
    search_results: list[LineItem]


class InsiderTrade(BaseModel):
    ticker: str
    issuer: str | None
    name: str | None
    title: str | None
    is_board_director: bool | None
    transaction_date: str | None
    transaction_shares: float | None
    transaction_price_per_share: float | None
    transaction_value: float | None
    shares_owned_before_transaction: float | None
    shares_owned_after_transaction: float | None
    security_title: str | None
    filing_date: str


class InsiderTradeResponse(BaseModel):
    insider_trades: list[InsiderTrade]


class CompanyNews(BaseModel):
    """Company news data model"""

    ticker: str
    title: str
    source: str
    date: str  # For US market, this is the publication date
    url: str
    sentiment: str | None = None
    market: MarketType | None = None  # Market type
    # Additional fields for Chinese market
    publish_time: str | None = None  # 发布时间 (format: YYYY-MM-DD HH:MM:SS)
    content: str | None = None  # 新闻内容
    keyword: str | None = None  # 关键词


class CompanyNewsResponse(BaseModel):
    """Response model for company news"""

    news: list[CompanyNews]
    market: MarketType | None = None


class CompanyFacts(BaseModel):
    """Company facts data model"""

    ticker: str
    name: str
    market: MarketType | None = None  # Market type
    cik: str | None = None
    industry: str | None = None
    sector: str | None = None
    category: str | None = None
    exchange: str | None = None
    is_active: bool | None = None
    listing_date: str | None = None
    location: str | None = None
    market_cap: float | None = None
    number_of_employees: int | None = None
    sec_filings_url: str | None = None
    sic_code: str | None = None
    sic_industry: str | None = None
    sic_sector: str | None = None
    website_url: str | None = None
    weighted_average_shares: int | None = None
    # Additional fields for Chinese market
    company_profile: str | None = None  # 公司简介
    main_business: str | None = None  # 主营业务
    business_scope: str | None = None  # 经营范围


class CompanyFactsResponse(BaseModel):
    """Response model for company facts"""

    company_facts: CompanyFacts
    market: MarketType | None = None


class Position(BaseModel):
    cash: float = 0.0
    shares: int = 0
    ticker: str


class Portfolio(BaseModel):
    positions: dict[str, Position]  # ticker -> Position mapping
    total_cash: float = 0.0


class AnalystSignal(BaseModel):
    signal: str | None = None
    confidence: float | None = None
    reasoning: dict | str | None = None
    max_position_size: float | None = None  # For risk management signals


class TickerAnalysis(BaseModel):
    ticker: str
    analyst_signals: dict[str, AnalystSignal]  # agent_name -> signal mapping


class AgentStateData(BaseModel):
    tickers: list[str]
    portfolio: Portfolio
    start_date: str
    end_date: str
    ticker_analyses: dict[str, TickerAnalysis]  # ticker -> analysis mapping


class AgentStateMetadata(BaseModel):
    show_reasoning: bool = False
    model_config = {"extra": "allow"}
