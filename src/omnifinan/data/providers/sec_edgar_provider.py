"""SEC EDGAR-backed provider for US financial statements and filings."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

from ...data_models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, MarketType, Price
from .akshare_provider import AkshareProvider
from .base import DataProvider
from .yfinance_provider import YFinanceProvider


@dataclass
class _FactPoint:
    end: str
    val: float
    form: str | None
    filed: str | None
    accn: str | None
    fy: int | None
    fp: str | None
    frame: str | None


class SECEDGARProvider(DataProvider):
    """US-only provider using SEC JSON APIs for fundamentals.

    Notes:
    - Requires compliant User-Agent per SEC guidance. Set `OMNIFINAN_SEC_USER_AGENT`
      to something like: "MyApp/1.0 your-email@example.com".
    - Prices are delegated to YFinanceProvider.
    - Macro is delegated to AkshareProvider.
    """

    _TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    _FACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    _SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, timeout: int = 20):
        self.timeout = timeout
        self._fallback_prices = YFinanceProvider()
        self._fallback_macro = AkshareProvider()
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": self._user_agent(),
                "Accept": "application/json",
            }
        )
        self._ticker_map_cache: dict[str, str] | None = None
        self._facts_cache: dict[str, dict[str, Any]] = {}
        self._submissions_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _market_for_ticker(ticker: str) -> MarketType:
        symbol = ticker.strip().upper()
        if re.fullmatch(r"\d{5}", symbol):
            return MarketType.HK
        if re.fullmatch(r"\d{6}", symbol):
            return MarketType.CHINA
        if symbol.endswith((".SH", ".SZ", ".BJ")):
            return MarketType.CHINA
        if re.fullmatch(r"[A-Z]{1,5}([._-][A-Z0-9]{1,5})?", symbol):
            return MarketType.US
        return MarketType.UNKNOWN

    @staticmethod
    def _merge_financial_metrics_prefer_primary(
        primary: list[FinancialMetrics],
        secondary: list[FinancialMetrics],
        limit: int,
    ) -> list[FinancialMetrics]:
        if not primary:
            return secondary[: max(1, limit)]
        if not secondary:
            return primary[: max(1, limit)]
        p0 = primary[0]
        s0 = secondary[0]
        p_data = p0.model_dump() if hasattr(p0, "model_dump") else dict(p0)
        s_data = s0.model_dump() if hasattr(s0, "model_dump") else dict(s0)
        merged = dict(p_data)
        for k, v in s_data.items():
            if merged.get(k) is None and v is not None:
                merged[k] = v
        return [FinancialMetrics(**merged)][: max(1, limit)]

    @staticmethod
    def _merge_line_items_prefer_primary(
        primary: list[LineItem],
        secondary: list[LineItem],
        limit: int,
    ) -> list[LineItem]:
        if not primary:
            return secondary[: max(1, limit)]
        if not secondary:
            return primary[: max(1, limit)]
        merged_by_period: dict[str, dict[str, Any]] = {}
        for row in secondary:
            payload = row.model_dump() if hasattr(row, "model_dump") else dict(row)
            period = str(payload.get("report_period", ""))
            if period:
                merged_by_period[period] = payload
        for row in primary:
            payload = row.model_dump() if hasattr(row, "model_dump") else dict(row)
            period = str(payload.get("report_period", ""))
            if not period:
                continue
            base = merged_by_period.get(period, {})
            merged = dict(base)
            merged.update(payload)
            for k, v in base.items():
                if merged.get(k) is None and v is not None:
                    merged[k] = v
            merged_by_period[period] = merged
        out = [LineItem(**merged_by_period[k]) for k in sorted(merged_by_period.keys(), reverse=True)]
        return out[: max(1, limit)]

    @staticmethod
    def _user_agent() -> str:
        env = os.getenv("OMNIFINAN_SEC_USER_AGENT", "").strip()
        if env:
            return env
        return "OmniFinan/0.1 contact@example.com"

    def _get_json(self, url: str) -> dict[str, Any]:
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data
        raise RuntimeError(f"Unexpected SEC response type from {url}: {type(data)}")

    def _ticker_to_cik(self, ticker: str) -> str:
        symbol = ticker.strip().upper()
        if self._ticker_map_cache is None:
            payload = self._get_json(self._TICKERS_URL)
            mapping: dict[str, str] = {}
            for row in payload.values():
                if not isinstance(row, dict):
                    continue
                t = str(row.get("ticker", "")).upper().strip()
                cik_num = row.get("cik_str")
                if t and isinstance(cik_num, int):
                    mapping[t] = str(cik_num).zfill(10)
            self._ticker_map_cache = mapping
        cik = self._ticker_map_cache.get(symbol)
        if not cik:
            raise ValueError(f"SEC CIK not found for ticker: {ticker}")
        return cik

    @staticmethod
    def _date_to_str(dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")

    def _latest_close_price(self, ticker: str, end_date: str | None = None) -> float | None:
        try:
            end_dt = datetime.strptime((end_date or datetime.utcnow().strftime("%Y-%m-%d"))[:10], "%Y-%m-%d")
        except ValueError:
            end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=15)
        try:
            rows = self._fallback_prices.get_prices(
                ticker=ticker,
                start_date=self._date_to_str(start_dt),
                end_date=self._date_to_str(end_dt + timedelta(days=1)),
                interval="1d",
            )
            if not rows:
                return None
            rows = sorted(rows, key=lambda x: x.time, reverse=True)
            return rows[0].close
        except Exception:
            return None

    def _market_cap_with_fallback(
        self,
        ticker: str,
        end_date: str | None,
        shares_outstanding: float | None,
    ) -> float | None:
        symbol = ticker.strip().upper()
        # 1) Try yfinance current market cap directly.
        try:
            import yfinance as yf  # type: ignore

            tk = yf.Ticker(symbol)
            fast_info = getattr(tk, "fast_info", None)
            if fast_info is not None:
                cap = fast_info.get("market_cap")
                if cap is not None:
                    return float(cap)
            info = getattr(tk, "info", None)
            if isinstance(info, dict):
                cap2 = info.get("marketCap")
                if cap2 is not None:
                    return float(cap2)
        except Exception:
            pass

        # 2) Approximate by price * report-period shares.
        px = self._latest_close_price(symbol, end_date=end_date)
        if px is not None and shares_outstanding not in (None, 0):
            return px * shares_outstanding
        return None

    def _company_facts(self, ticker: str) -> dict[str, Any]:
        cik = self._ticker_to_cik(ticker)
        if cik not in self._facts_cache:
            payload = self._get_json(self._FACTS_URL_TMPL.format(cik=cik))
            # Some cached/forwarded payloads can be wrapped as {"<cik>": {...companyfacts...}}.
            if "facts" not in payload and len(payload) == 1:
                only_val = next(iter(payload.values()))
                if isinstance(only_val, dict) and "facts" in only_val:
                    payload = only_val
            self._facts_cache[cik] = payload
        return self._facts_cache[cik]

    def _submissions(self, ticker: str) -> dict[str, Any]:
        cik = self._ticker_to_cik(ticker)
        if cik not in self._submissions_cache:
            self._submissions_cache[cik] = self._get_json(self._SUBMISSIONS_URL_TMPL.format(cik=cik))
        return self._submissions_cache[cik]

    def _extract_points(
        self,
        facts: dict[str, Any],
        concept: str,
        *,
        taxonomy: str = "us-gaap",
        units_priority: tuple[str, ...] = ("USD", "shares", "pure"),
    ) -> list[_FactPoint]:
        tax = facts.get("facts", {}).get(taxonomy, {})
        node = tax.get(concept, {})
        units = node.get("units", {})
        if not isinstance(units, dict) or not units:
            return []

        selected: list[dict[str, Any]] = []
        # Prefer configured units; if unavailable or stale, pick the unit with latest observations.
        candidate_units: list[tuple[str, list[dict[str, Any]]]] = []
        for unit_name, vals in units.items():
            if isinstance(vals, list) and vals:
                candidate_units.append((str(unit_name), vals))

        if candidate_units:
            prioritized = [
                item for item in candidate_units if item[0] in set(units_priority)
            ]
            bucket = prioritized or candidate_units

            def _unit_score(rows: list[dict[str, Any]]) -> tuple[str, int]:
                ends = [str(r.get("end", "")) for r in rows if r.get("end")]
                latest = max(ends) if ends else ""
                return latest, len(rows)

            selected = max(bucket, key=lambda x: _unit_score(x[1]))[1]

        out: list[_FactPoint] = []
        for row in selected:
            try:
                val_raw = row.get("val")
                end = str(row.get("end", ""))
                if val_raw is None or not end:
                    continue
                out.append(
                    _FactPoint(
                        end=end,
                        val=float(val_raw),
                        form=str(row.get("form")) if row.get("form") is not None else None,
                        filed=str(row.get("filed")) if row.get("filed") is not None else None,
                        accn=str(row.get("accn")) if row.get("accn") is not None else None,
                        fy=int(row["fy"]) if isinstance(row.get("fy"), int | float) else None,
                        fp=str(row["fp"]) if row.get("fp") is not None else None,
                        frame=str(row.get("frame")) if row.get("frame") is not None else None,
                    )
                )
            except Exception:
                continue
        out.sort(key=lambda x: (x.end, x.filed or ""), reverse=True)
        return out

    def _extract_points_multi(
        self,
        facts: dict[str, Any],
        concepts: list[str],
        *,
        taxonomy: str = "us-gaap",
        units_priority: tuple[str, ...] = ("USD", "shares", "pure"),
    ) -> list[_FactPoint]:
        for concept in concepts:
            pts = self._extract_points(
                facts,
                concept,
                taxonomy=taxonomy,
                units_priority=units_priority,
            )
            if pts:
                return pts
        return []

    @staticmethod
    def _form_priority(form: str | None) -> int:
        f = (form or "").upper()
        if f in {"10-K", "20-F", "40-F"}:
            return 2
        if f == "10-Q":
            return 1
        return 0

    @classmethod
    def _best_point_for_end(cls, points: list[_FactPoint], end: str) -> _FactPoint | None:
        same_end = [p for p in points if p.end == end]
        if not same_end:
            return None
        return max(
            same_end,
            key=lambda p: (cls._form_priority(p.form), p.filed or "", p.accn or ""),
        )

    @classmethod
    def _best_points_by_end(cls, points: list[_FactPoint]) -> list[_FactPoint]:
        if not points:
            return []
        out: list[_FactPoint] = []
        for end in sorted({p.end for p in points}, reverse=True):
            best = cls._best_point_for_end(points, end)
            if best is not None:
                out.append(best)
        return out

    @staticmethod
    def _infer_frequency(point: _FactPoint | None) -> str | None:
        if point is None:
            return None
        fp = (point.fp or "").upper()
        if fp in {"Q1", "Q2", "Q3", "Q4"}:
            return "quarterly"
        if fp in {"FY"}:
            return "annual"
        frame = (point.frame or "").upper()
        if "Q" in frame:
            return "quarterly"
        if "CY" in frame:
            return "annual"
        form = (point.form or "").upper()
        if form == "10-Q":
            return "quarterly"
        if form in {"10-K", "20-F", "40-F"}:
            return "annual"
        return None

    @classmethod
    def _latest_before(cls, points: list[_FactPoint], end_date: str | None) -> _FactPoint | None:
        best_points = cls._best_points_by_end(points)
        if not best_points:
            return None
        if not end_date:
            return best_points[0]
        for p in best_points:
            if p.end <= end_date:
                return p
        return best_points[0]

    @classmethod
    def _latest_form(cls, points: list[_FactPoint], forms: set[str], end_date: str | None) -> _FactPoint | None:
        filtered = [p for p in points if (p.form or "").upper() in forms]
        return cls._latest_before(filtered, end_date)

    @classmethod
    def _prev_annual(cls, points: list[_FactPoint], current_end: str) -> _FactPoint | None:
        annuals = [
            p
            for p in cls._best_points_by_end(points)
            if (p.form or "").upper() in {"10-K", "20-F", "40-F"} and p.end < current_end
        ]
        return annuals[0] if annuals else None

    @staticmethod
    def _parse_end_date(end: str | None) -> datetime | None:
        if not end:
            return None
        try:
            return datetime.strptime(end[:10], "%Y-%m-%d")
        except ValueError:
            return None

    @classmethod
    def _quarter_bucket(cls, point: _FactPoint) -> tuple[int, int] | None:
        dt = cls._parse_end_date(point.end)
        if dt is None:
            return None
        return dt.year, ((dt.month - 1) // 3) + 1

    @classmethod
    def _ttm_sum(cls, points: list[_FactPoint], end: str) -> float | None:
        best_points = [p for p in cls._best_points_by_end(points) if p.end <= end]
        quarterly = [p for p in best_points if cls._infer_frequency(p) == "quarterly"]
        if len(quarterly) >= 4:
            return sum(p.val for p in quarterly[:4])

        annuals = [p for p in best_points if cls._infer_frequency(p) == "annual"]
        if annuals:
            return annuals[0].val
        return None

    @classmethod
    def _avg_balance(cls, points: list[_FactPoint], end: str) -> float | None:
        current = cls._latest_before(points, end)
        if current is None:
            return None
        current_freq = cls._infer_frequency(current)
        prev_candidates = [p for p in cls._best_points_by_end(points) if p.end < current.end]
        if not prev_candidates:
            return current.val
        preferred = [p for p in prev_candidates if cls._infer_frequency(p) == current_freq]
        prev = preferred[0] if preferred else prev_candidates[0]
        return (current.val + prev.val) / 2.0

    @classmethod
    def _comparable_previous(cls, points: list[_FactPoint], current: _FactPoint | None) -> _FactPoint | None:
        if current is None:
            return None
        current_freq = cls._infer_frequency(current)
        all_best = cls._best_points_by_end(points)

        if current_freq == "annual":
            if current.fy is not None:
                candidates = [
                    p
                    for p in all_best
                    if cls._infer_frequency(p) == "annual" and p.fy == current.fy - 1
                ]
                if candidates:
                    return candidates[0]
            current_dt = cls._parse_end_date(current.end)
            if current_dt is not None:
                return next(
                    (
                        p
                        for p in all_best
                        if cls._infer_frequency(p) == "annual"
                        and cls._parse_end_date(p.end) is not None
                        and cls._parse_end_date(p.end).year == current_dt.year - 1
                    ),
                    None,
                )
            return None

        if current_freq == "quarterly":
            if current.fy is not None and (current.fp or "").upper() in {"Q1", "Q2", "Q3", "Q4"}:
                candidates = [
                    p
                    for p in all_best
                    if cls._infer_frequency(p) == "quarterly"
                    and p.fy == current.fy - 1
                    and (p.fp or "").upper() == (current.fp or "").upper()
                ]
                if candidates:
                    return candidates[0]
            bucket = cls._quarter_bucket(current)
            if bucket is not None:
                y, q = bucket
                return next(
                    (
                        p
                        for p in all_best
                        if cls._infer_frequency(p) == "quarterly" and cls._quarter_bucket(p) == (y - 1, q)
                    ),
                    None,
                )
            return None

        return None

    @staticmethod
    def _capex_to_fcf(ocf: float | None, capex_raw: float | None) -> tuple[float | None, float | None]:
        if capex_raw is None:
            return None, None
        capex = abs(capex_raw)
        if ocf is None:
            return capex, None
        if capex_raw < 0:
            return capex, ocf + capex_raw
        return capex, ocf - capex

    @classmethod
    def _value_for_period(
        cls,
        points: list[_FactPoint],
        report_period: str,
        period: str,
        *,
        is_balance: bool,
    ) -> float | None:
        if not points:
            return None
        if period.lower() == "ttm" and not is_balance:
            return cls._ttm_sum(points, report_period)
        point = cls._latest_before(points, report_period)
        return point.val if point is not None else None

    @staticmethod
    def _safe_div(a: float | None, b: float | None) -> float | None:
        if a is None or b in (None, 0):
            return None
        return a / b

    @staticmethod
    def _growth(curr: float | None, prev: float | None) -> float | None:
        if curr is None or prev in (None, 0):
            return None
        return curr / prev - 1.0

    def get_prices(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> list[Price]:
        return self._fallback_prices.get_prices(ticker, start_date, end_date, interval=interval)

    def get_financial_metrics(
        self, ticker: str, end_date: str | None = None, period: str = "ttm", limit: int = 1
    ) -> list[FinancialMetrics]:
        market = self._market_for_ticker(ticker)
        if market != MarketType.US:
            return self._fallback_macro.get_financial_metrics(
                ticker=ticker,
                end_date=end_date,
                period=period,
                limit=limit,
            )

        facts = self._company_facts(ticker)
        symbol = ticker.strip().upper()

        revenue_base = self._extract_points_multi(
            facts,
            [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueServicesNet",
            ],
        )
        net_income_pts = self._extract_points_multi(facts, ["NetIncomeLoss", "ProfitLoss"])
        op_income_pts = self._extract_points_multi(
            facts,
            ["OperatingIncomeLoss", "IncomeLossFromOperations"],
        )
        gross_profit_pts = self._extract_points_multi(
            facts,
            ["GrossProfit", "GrossProfitAndOther"],
        )
        assets_pts = self._extract_points_multi(facts, ["Assets"])
        equity_pts = self._extract_points_multi(
            facts,
            ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        )
        liabilities_pts = self._extract_points_multi(facts, ["Liabilities"])
        current_assets_pts = self._extract_points_multi(facts, ["AssetsCurrent"])
        current_liabilities_pts = self._extract_points_multi(facts, ["LiabilitiesCurrent"])
        cash_pts = self._extract_points_multi(
            facts,
            [
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ],
        )
        inventory_pts = self._extract_points_multi(
            facts,
            ["InventoryNet", "InventoriesNet", "InventoryFinishedGoods"],
        )
        prepaid_pts = self._extract_points_multi(
            facts,
            ["PrepaidExpenseCurrent", "PrepaidExpensesCurrent"],
        )
        ocf_pts = self._extract_points_multi(
            facts,
            ["NetCashProvidedByUsedInOperatingActivities"],
        )
        capex_cash_pts = self._extract_points_multi(
            facts,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
            ],
        )
        capex_fallback_pts = self._extract_points_multi(
            facts,
            [
                "PaymentsToAcquireProductiveAssets",
                "CapitalExpendituresIncurredButNotYetPaid",
            ],
        )
        dep_pts = self._extract_points_multi(
            facts,
            ["DepreciationDepletionAndAmortization", "Depreciation"],
        )
        eps_diluted_pts = self._extract_points_multi(
            facts,
            ["EarningsPerShareDiluted"],
            units_priority=("USD/shares", "USD", "pure"),
        )
        eps_basic_pts = self._extract_points_multi(
            facts,
            ["EarningsPerShareBasic"],
            units_priority=("USD/shares", "USD", "pure"),
        )
        weighted_shares_diluted_pts = self._extract_points_multi(
            facts,
            [
                "WeightedAverageNumberOfDilutedSharesOutstanding",
            ],
            units_priority=("shares",),
        )
        weighted_shares_basic_pts = self._extract_points_multi(
            facts,
            [
                "WeightedAverageNumberOfBasicSharesOutstanding",
                "WeightedAverageNumberOfSharesOutstandingBasic",
            ],
            units_priority=("shares",),
        )
        shares_outstanding_pts = self._extract_points_multi(
            facts,
            ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
            units_priority=("shares",),
        )
        debt_total_pts = self._extract_points_multi(
            facts,
            [
                "DebtLongtermAndShorttermCombinedAmount",
                "LongTermDebtAndFinanceLeaseObligations",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebt",
            ],
        )
        debt_short_pts = self._extract_points_multi(
            facts,
            [
                "ShortTermBorrowings",
                "CurrentPortionOfLongTermDebtAndCapitalLeaseObligations",
                "CurrentPortionOfLongTermDebt",
                "CommercialPaper",
            ],
        )
        debt_long_pts = self._extract_points_multi(
            facts,
            [
                "LongTermDebtNoncurrent",
                "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                "LongTermDebt",
            ],
        )

        if not revenue_base and not net_income_pts and not assets_pts and not equity_pts:
            return []

        flow_anchor = revenue_base + net_income_pts + ocf_pts
        balance_anchor = assets_pts + equity_pts + liabilities_pts
        anchor_pool = flow_anchor if period.lower() == "ttm" else (flow_anchor + balance_anchor)
        anchor_point = self._latest_before(anchor_pool, end_date)
        if anchor_point is None:
            return []
        report_period = anchor_point.end

        revenue = self._value_for_period(revenue_base, report_period, period, is_balance=False)
        net_income = self._value_for_period(net_income_pts, report_period, period, is_balance=False)
        operating_income = self._value_for_period(op_income_pts, report_period, period, is_balance=False)
        gross_profit = self._value_for_period(gross_profit_pts, report_period, period, is_balance=False)
        assets = self._latest_before(assets_pts, report_period).val if assets_pts else None
        equity = self._latest_before(equity_pts, report_period).val if equity_pts else None
        liabilities = self._latest_before(liabilities_pts, report_period).val if liabilities_pts else None
        current_assets = self._latest_before(current_assets_pts, report_period).val if current_assets_pts else None
        current_liabilities = (
            self._latest_before(current_liabilities_pts, report_period).val if current_liabilities_pts else None
        )
        cash = self._latest_before(cash_pts, report_period).val if cash_pts else None
        inventory = self._latest_before(inventory_pts, report_period).val if inventory_pts else 0.0
        prepaid = self._latest_before(prepaid_pts, report_period).val if prepaid_pts else 0.0
        ocf = self._value_for_period(ocf_pts, report_period, period, is_balance=False)
        capex_raw = self._value_for_period(capex_cash_pts, report_period, period, is_balance=False)
        if capex_raw is None:
            capex_raw = self._value_for_period(capex_fallback_pts, report_period, period, is_balance=False)
        dep = self._value_for_period(dep_pts, report_period, period, is_balance=False)
        eps = self._value_for_period(eps_diluted_pts, report_period, period, is_balance=False)
        if eps is None:
            eps = self._value_for_period(eps_basic_pts, report_period, period, is_balance=False)
        shares_weighted = self._value_for_period(
            weighted_shares_diluted_pts, report_period, period, is_balance=True
        )
        if shares_weighted is None:
            shares_weighted = self._value_for_period(
                weighted_shares_basic_pts, report_period, period, is_balance=True
            )
        if shares_weighted is None:
            shares_weighted = (
                self._latest_before(shares_outstanding_pts, report_period).val if shares_outstanding_pts else None
            )
        shares_outstanding = (
            self._latest_before(shares_outstanding_pts, report_period).val if shares_outstanding_pts else None
        )

        debt = self._latest_before(debt_total_pts, report_period).val if debt_total_pts else None
        if debt is None:
            debt_short = self._latest_before(debt_short_pts, report_period).val if debt_short_pts else 0.0
            debt_long = self._latest_before(debt_long_pts, report_period).val if debt_long_pts else 0.0
            if debt_short or debt_long:
                debt = debt_short + debt_long

        capex, fcf = self._capex_to_fcf(ocf, capex_raw)
        working_capital = (
            (current_assets - current_liabilities)
            if (current_assets is not None and current_liabilities is not None)
            else None
        )

        revenue_current_point = self._latest_before(revenue_base, report_period)
        net_income_current_point = self._latest_before(net_income_pts, report_period)
        prev_rev = self._comparable_previous(revenue_base, revenue_current_point)
        prev_net = self._comparable_previous(net_income_pts, net_income_current_point)
        revenue_growth = self._growth(revenue_current_point.val if revenue_current_point else None, prev_rev.val if prev_rev else None)
        earnings_growth = self._growth(
            net_income_current_point.val if net_income_current_point else None,
            prev_net.val if prev_net else None,
        )

        current_ratio = self._safe_div(current_assets, current_liabilities)
        quick_assets = (
            current_assets - (inventory or 0.0) - (prepaid or 0.0)
            if current_assets is not None
            else None
        )
        quick_ratio = self._safe_div(quick_assets, current_liabilities)
        cash_ratio = self._safe_div(cash, current_liabilities)
        debt_to_equity = self._safe_div(debt, equity)
        liabilities_to_equity = self._safe_div(liabilities, equity)
        avg_assets = self._avg_balance(assets_pts, report_period)
        avg_equity = self._avg_balance(equity_pts, report_period)
        roa = self._safe_div(net_income, avg_assets)
        roe = self._safe_div(net_income, avg_equity)
        gross_margin = self._safe_div(gross_profit, revenue)
        operating_margin = self._safe_div(operating_income, revenue)
        net_margin = self._safe_div(net_income, revenue)
        bvps = self._safe_div(equity, shares_outstanding)
        fcf_per_share = self._safe_div(fcf, shares_weighted)
        market_cap = self._market_cap_with_fallback(
            symbol,
            end_date=report_period,
            shares_outstanding=shares_outstanding,
        )
        pe = self._safe_div(market_cap, net_income)
        pb = self._safe_div(market_cap, equity)
        ps = self._safe_div(market_cap, revenue)
        enterprise_value = (
            (market_cap + debt - cash)
            if (market_cap is not None and debt is not None and cash is not None)
            else None
        )
        ev_to_rev = self._safe_div(enterprise_value, revenue)
        fcf_yield = self._safe_div(fcf, market_cap)

        sec_metrics = [
            FinancialMetrics(
                ticker=symbol,
                report_period=report_period,
                period=period,
                currency="USD",
                market=MarketType.US,
                market_cap=market_cap,
                enterprise_value=enterprise_value,
                price_to_earnings_ratio=pe,
                price_to_book_ratio=pb,
                price_to_sales_ratio=ps,
                enterprise_value_to_revenue_ratio=ev_to_rev,
                free_cash_flow_yield=fcf_yield,
                net_income=net_income,
                operating_revenue=revenue,
                operating_profit=operating_income,
                working_capital=working_capital,
                depreciation_and_amortization=dep,
                capital_expenditure=capex,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                cash_ratio=cash_ratio,
                debt_to_equity=debt_to_equity,
                liabilities_to_equity=liabilities_to_equity,
                interest_bearing_debt=debt,
                debt_to_assets=self._safe_div(debt, assets),
                revenue_growth=revenue_growth,
                earnings_growth=earnings_growth,
                earnings_per_share=eps,
                book_value_per_share=bvps,
                free_cash_flow_per_share=fcf_per_share,
            )
        ][: max(1, limit)]

        try:
            ak_metrics = self._fallback_macro.get_financial_metrics(
                ticker=ticker,
                end_date=end_date,
                period=period,
                limit=limit,
            )
        except Exception:
            ak_metrics = []
        return self._merge_financial_metrics_prefer_primary(sec_metrics, ak_metrics, limit)

    def search_line_items(self, ticker: str, period: str = "ttm", limit: int = 10) -> list[LineItem]:
        market = self._market_for_ticker(ticker)
        if market != MarketType.US:
            return self._fallback_macro.search_line_items(ticker=ticker, period=period, limit=limit)

        facts = self._company_facts(ticker)
        symbol = ticker.strip().upper()

        concepts = {
            "operating_revenue": [
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueServicesNet",
            ],
            "net_income": ["NetIncomeLoss"],
            "operating_profit": ["OperatingIncomeLoss"],
            "working_capital_assets": ["AssetsCurrent"],
            "working_capital_liabilities": ["LiabilitiesCurrent"],
            "inventory": ["InventoryNet", "InventoriesNet", "InventoryFinishedGoods"],
            "prepaid": ["PrepaidExpenseCurrent", "PrepaidExpensesCurrent"],
            "cash": [
                "CashAndCashEquivalentsAtCarryingValue",
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            ],
            "equity": [
                "StockholdersEquity",
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            ],
            "liabilities": ["Liabilities"],
            "interest_bearing_debt": [
                "DebtLongtermAndShorttermCombinedAmount",
                "LongTermDebtAndFinanceLeaseObligations",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebt",
            ],
            "interest_bearing_debt_short": [
                "ShortTermBorrowings",
                "CurrentPortionOfLongTermDebtAndCapitalLeaseObligations",
                "CurrentPortionOfLongTermDebt",
                "CommercialPaper",
            ],
            "interest_bearing_debt_long": [
                "LongTermDebtNoncurrent",
                "LongTermDebtAndCapitalLeaseObligationsNoncurrent",
                "LongTermDebt",
            ],
            "depreciation_and_amortization": ["DepreciationDepletionAndAmortization", "Depreciation"],
            "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment"],
            "capital_expenditure_fallback": ["PaymentsToAcquireProductiveAssets", "CapitalExpendituresIncurredButNotYetPaid"],
            "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
        }

        point_map: dict[str, list[_FactPoint]] = {}
        for out_key, concept_list in concepts.items():
            vals: list[_FactPoint] = []
            for concept in concept_list:
                vals = self._extract_points(facts, concept)
                if vals:
                    break
            point_map[out_key] = vals

        all_ends = sorted({p.end for pts in point_map.values() for p in self._best_points_by_end(pts)}, reverse=True)
        out: list[LineItem] = []
        for end in all_ends[: max(limit, 1)]:
            def _balance_value(key: str) -> float | None:
                pts = point_map.get(key, [])
                point = self._latest_before(pts, end)
                return point.val if point is not None else None

            def _flow_value(key: str) -> float | None:
                return self._value_for_period(point_map.get(key, []), end, period, is_balance=False)

            rev = _flow_value("operating_revenue")
            ni = _flow_value("net_income")
            op = _flow_value("operating_profit")
            ca = _balance_value("working_capital_assets")
            cl = _balance_value("working_capital_liabilities")
            dep = _flow_value("depreciation_and_amortization")
            capex_raw = _flow_value("capital_expenditure")
            if capex_raw is None:
                capex_raw = _flow_value("capital_expenditure_fallback")
            ocf = _flow_value("operating_cash_flow")
            capex, fcf = self._capex_to_fcf(ocf, capex_raw)
            wc = (ca - cl) if (ca is not None and cl is not None) else None
            inventory = _balance_value("inventory")
            prepaid = _balance_value("prepaid")
            cash = _balance_value("cash")
            equity = _balance_value("equity")
            liabilities = _balance_value("liabilities")
            debt = _balance_value("interest_bearing_debt")
            if debt is None:
                debt_short = _balance_value("interest_bearing_debt_short") or 0.0
                debt_long = _balance_value("interest_bearing_debt_long") or 0.0
                if debt_short or debt_long:
                    debt = debt_short + debt_long

            quick_assets = (ca - (inventory or 0.0) - (prepaid or 0.0)) if ca is not None else None
            out.append(
                LineItem(
                    ticker=symbol,
                    report_period=end,
                    period=period,
                    currency="USD",
                    net_income=ni,
                    operating_revenue=rev,
                    operating_profit=op,
                    working_capital=wc,
                    depreciation_and_amortization=dep,
                    capital_expenditure=capex,
                    free_cash_flow=fcf,
                    market=MarketType.US,
                    current_assets=ca,
                    current_liabilities=cl,
                    inventory=inventory,
                    prepaid_current=prepaid,
                    cash_and_equivalents=cash,
                    interest_bearing_debt=debt,
                    debt_to_equity=self._safe_div(debt, equity),
                    liabilities_to_equity=self._safe_div(liabilities, equity),
                    current_ratio=self._safe_div(ca, cl),
                    quick_ratio=self._safe_div(quick_assets, cl),
                    cash_ratio=self._safe_div(cash, cl),
                )
            )
        sec_items = out[:limit]
        try:
            ak_items = self._fallback_macro.search_line_items(ticker=ticker, period=period, limit=limit)
        except Exception:
            ak_items = []
        return self._merge_line_items_prefer_primary(sec_items, ak_items, limit)

    def get_company_news(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 10,
    ) -> list[CompanyNews]:
        payload = self._submissions(ticker)
        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            return []
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        primary_desc = recent.get("primaryDocDescription", [])
        cik_nozero = str(int(self._ticker_to_cik(ticker)))

        out: list[CompanyNews] = []
        n = min(len(forms), len(filing_dates), len(accession_numbers), len(primary_docs))
        for i in range(n):
            f_date = str(filing_dates[i])
            if start_date and f_date < start_date:
                continue
            if end_date and f_date > end_date:
                continue
            acc = str(accession_numbers[i])
            acc_no_dash = acc.replace("-", "")
            doc = str(primary_docs[i] or "")
            url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik_nozero}/{acc_no_dash}/{doc}"
                if doc
                else f"https://www.sec.gov/ixviewer/ix.html?doc=/Archives/edgar/data/{cik_nozero}/{acc_no_dash}/"
            )
            form = str(forms[i] or "")
            title = str(primary_desc[i] or "").strip() or f"SEC filing {form}"
            out.append(
                CompanyNews(
                    ticker=ticker.upper(),
                    title=title,
                    source="sec",
                    date=f_date,
                    url=url,
                    market=MarketType.US,
                    publish_time=f"{f_date} 00:00:00",
                    content=f"Form {form} filed to SEC.",
                )
            )
        out.sort(key=lambda x: x.date, reverse=True)
        return out[:limit]

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        payload = self._submissions(ticker)
        name = payload.get("name")
        recent = payload.get("filings", {}).get("recent", {})
        if not isinstance(recent, dict):
            return []
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        out: list[InsiderTrade] = []
        n = min(len(forms), len(filing_dates))
        for i in range(n):
            form = str(forms[i] or "").upper()
            if form not in {"3", "4", "5"}:
                continue
            f_date = str(filing_dates[i])
            if start_date and f_date < start_date:
                continue
            if end_date and f_date > end_date:
                continue
            out.append(
                InsiderTrade(
                    ticker=ticker.upper(),
                    issuer=str(name) if isinstance(name, str) else None,
                    name=None,
                    title=None,
                    is_board_director=None,
                    transaction_date=f_date,
                    transaction_shares=None,
                    transaction_price_per_share=None,
                    transaction_value=None,
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=None,
                    security_title=None,
                    filing_date=f_date,
                )
            )
        out.sort(key=lambda x: x.filing_date, reverse=True)
        return out[:limit]

    def get_market_cap(self, ticker: str, end_date: str | None = None) -> float | None:
        market = self._market_for_ticker(ticker)
        if market != MarketType.US:
            return self._fallback_macro.get_market_cap(ticker=ticker, end_date=end_date)

        shares: float | None = None
        try:
            facts = self._company_facts(ticker)
            shares_pts = self._extract_points_multi(
                facts,
                ["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
                units_priority=("shares",),
            )
            latest = self._latest_before(shares_pts, end_date)
            shares = latest.val if latest is not None else None
        except Exception:
            shares = None
        return self._market_cap_with_fallback(ticker=ticker, end_date=end_date, shares_outstanding=shares)

    def get_macro_indicators(self, start_date: str | None = None, end_date: str | None = None) -> dict:
        return self._fallback_macro.get_macro_indicators(start_date=start_date, end_date=end_date)
