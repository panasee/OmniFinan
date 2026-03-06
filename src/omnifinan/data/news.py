"""News discovery and deterministic cross-verification helpers."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from pyomnix.omnix_logger import get_logger

from ..data_models import CompanyNews, IntegratedNewsEvent, IntegratedNewsSource, MarketType
from .providers.credentials import get_api_key

logger = get_logger("news_integration")

_ETF_THEME_MAP: dict[str, dict[str, Any]] = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "themes": ["S&P 500", "US equities", "macro", "Fed", "rates"],
        "queries": [
            "\"S&P 500\" market news Fed inflation rates earnings",
            "\"SPY\" ETF market outlook macro news",
            "\"US stock market\" macro news S&P 500 Reuters Bloomberg",
        ],
    },
    "QQQ": {
        "name": "Invesco QQQ Trust",
        "themes": ["Nasdaq 100", "big tech", "AI", "semiconductors", "growth stocks"],
        "queries": [
            "\"Nasdaq 100\" market news big tech AI semiconductors",
            "\"QQQ\" ETF tech sector news Reuters Bloomberg",
            "\"mega cap tech\" market news Nasdaq 100 Microsoft Nvidia Apple Amazon",
        ],
    },
}

_SOCIAL_DOMAINS = {
    "x.com",
    "twitter.com",
    "reddit.com",
    "stocktwits.com",
    "facebook.com",
    "instagram.com",
    "weibo.com",
}
_WIRE_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "apnews.com",
}
_MAINSTREAM_DOMAINS = {
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "nytimes.com",
}
_PORTAL_DOMAINS = {
    "finance.yahoo.com",
    "benzinga.com",
    "seekingalpha.com",
    "investing.com",
    "fool.com",
    "msn.com",
}
_OFFICIAL_DOMAINS = {
    "sec.gov",
    "hkexnews.hk",
    "nasdaq.com",
    "nyse.com",
}
_NOISE_DOMAINS = {
    "tradingview.com",
    "tipranks.com",
}
_PUBLISHER_PATTERNS: list[tuple[tuple[str, ...], tuple[str, str, float]]] = [
    (("reuters",), ("Reuters", "wire", 0.8)),
    (("bloomberg",), ("Bloomberg", "wire", 0.8)),
    (("associated press", "ap news", "apnews"), ("Associated Press", "wire", 0.8)),
    (("sec", "u.s. securities and exchange commission"), ("SEC", "official", 1.0)),
    (("hkex", "hong kong exchanges", "hkexnews"), ("HKEX", "official", 1.0)),
    (("nasdaq",), ("Nasdaq", "official", 1.0)),
    (("nyse", "new york stock exchange"), ("NYSE", "official", 1.0)),
    (("pr newswire",), ("PR Newswire", "official", 1.0)),
    (("business wire",), ("Business Wire", "official", 1.0)),
    (("globenewswire", "globe newswire"), ("GlobeNewswire", "official", 1.0)),
    (("marketwatch",), ("MarketWatch", "mainstream_media", 0.7)),
    (("wall street journal", "wsj"), ("The Wall Street Journal", "mainstream_media", 0.7)),
    (("financial times", "ft.com"), ("Financial Times", "mainstream_media", 0.7)),
    (("cnbc",), ("CNBC", "mainstream_media", 0.7)),
    (("benzinga",), ("Benzinga", "portal", 0.4)),
    (("seeking alpha",), ("Seeking Alpha", "portal", 0.4)),
    (("yahoo finance",), ("Yahoo Finance", "portal", 0.4)),
    (("social media", "x.com", "twitter", "reddit", "stocktwits", "weibo"), ("social_media", "social_media", 0.1)),
]


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.replace(tzinfo=None)
        return parsed
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
    return None


def _infer_datetime_from_text(*values: Any) -> datetime | None:
    patterns = (
        r"\b(20\d{2})-(\d{2})-(\d{2})\b",
        r"\b(20\d{2})(\d{2})(\d{2})\b",
    )
    for value in values:
        text = str(value or "")
        if not text:
            continue
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            year, month, day = match.groups()
            try:
                return datetime(int(year), int(month), int(day))
            except ValueError:
                continue
    return None


def _format_datetime(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _extract_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
    except Exception:
        return None
    domain = parsed.netloc.lower().strip()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or None


def _canonicalize_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except Exception:
        return str(url).strip()
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in {"guccounter", "guce_referrer"}
    ]
    clean = parsed._replace(
        fragment="",
        query=urlencode(query_pairs),
        scheme=(parsed.scheme or "https").lower(),
        netloc=parsed.netloc.lower(),
    )
    return urlunparse(clean).rstrip("/")


def _normalize_text(text: str | None) -> str:
    raw = str(text or "").lower().strip()
    raw = re.sub(r"https?://\S+", " ", raw)
    raw = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _token_set(text: str | None) -> set[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return set()
    return {token for token in normalized.split(" ") if len(token) > 1}


def _publisher_from_domain(domain: str | None) -> str:
    if not domain:
        return "unknown"
    parts = domain.split(".")
    if len(parts) >= 2:
        return parts[-2]
    return domain


def classify_news_source(source: str | None, url: str | None) -> tuple[str, str, float, str | None]:
    domain = _extract_domain(url)
    source_text = str(source or "").strip()
    check_text = f"{source_text} {domain or ''}".lower()
    publisher = source_text or _publisher_from_domain(domain)

    for patterns, mapped in _PUBLISHER_PATTERNS:
        if any(token in check_text for token in patterns):
            mapped_publisher, mapped_type, mapped_weight = mapped
            return mapped_publisher, mapped_type, mapped_weight, domain

    if domain and any(domain.endswith(item) for item in _OFFICIAL_DOMAINS):
        return publisher or "official", "official", 1.0, domain
    if (
        "sec" in check_text
        or "investor relations" in check_text
        or "press release" in check_text
        or "公告" in source_text
        or "交易所" in source_text
    ):
        return publisher or "official", "official", 1.0, domain
    if domain and any(domain.endswith(item) for item in _WIRE_DOMAINS):
        return publisher or "wire", "wire", 0.8, domain
    if domain and any(domain.endswith(item) for item in _MAINSTREAM_DOMAINS):
        return publisher or "mainstream_media", "mainstream_media", 0.7, domain
    if domain and any(domain.endswith(item) for item in _PORTAL_DOMAINS):
        return publisher or "portal", "portal", 0.4, domain
    if domain and any(domain.endswith(item) for item in _SOCIAL_DOMAINS):
        return publisher or "social_media", "social_media", 0.1, domain
    if "reuters" in check_text or "bloomberg" in check_text:
        return publisher or "wire", "wire", 0.8, domain
    if "social" in check_text or "forum" in check_text or "community" in check_text:
        return publisher or "social_media", "social_media", 0.1, domain
    return publisher or "unknown", "unknown", 0.2, domain


@lru_cache(maxsize=256)
def resolve_company_name(ticker: str, market: MarketType) -> str | None:
    if market != MarketType.US:
        return None
    try:
        import yfinance as yf  # type: ignore

        tk = yf.Ticker(ticker.upper().strip())
        info = getattr(tk, "info", {}) or {}
        for key in ("longName", "shortName", "displayName", "name"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    except Exception:
        return None
    return None


@lru_cache(maxsize=256)
def resolve_asset_kind(ticker: str, market: MarketType) -> str:
    clean = ticker.upper().strip()
    if clean in _ETF_THEME_MAP:
        return "etf"
    if market != MarketType.US:
        return "equity"
    try:
        import yfinance as yf  # type: ignore

        tk = yf.Ticker(clean)
        info = getattr(tk, "info", {}) or {}
        quote_type = str(info.get("quoteType") or "").strip().lower()
        if quote_type in {"etf", "mutualfund", "fund"}:
            return "etf"
    except Exception:
        return "equity"
    return "equity"


def build_news_queries(
    ticker: str,
    market: MarketType,
    company_name: str | None = None,
    asset_kind: str = "equity",
) -> list[str]:
    base = ticker.upper().strip()
    if market == MarketType.US and asset_kind == "etf":
        if base in _ETF_THEME_MAP:
            return list(_ETF_THEME_MAP[base]["queries"])
        if company_name:
            return [
                f"\"{company_name}\" ETF market news macro outlook",
                f"\"{base}\" ETF sector news market outlook",
            ]
        return [
            f"\"{base}\" ETF market news",
            f"\"{base}\" ETF macro sector news",
        ]
    if market == MarketType.HK:
        return [
            f"{base} HK stock latest company news",
            f"{base} HKEX company announcement",
        ]
    if market == MarketType.US and company_name:
        return [
            f"\"{company_name}\" ({base}) latest company news",
            f"\"{company_name}\" {base} earnings Reuters Bloomberg",
            f"site:sec.gov \"{company_name}\" OR \"{base}\" 8-K OR 10-Q OR 10-K",
        ]
    return [
        f"{base} latest company news",
        f"{base} earnings SEC filing Reuters Bloomberg",
    ]


def _looks_like_noise(domain: str | None) -> bool:
    if not domain:
        return False
    return any(domain.endswith(item) for item in _NOISE_DOMAINS)


def _row_matches_entity(
    ticker: str,
    title: str | None,
    content: str | None,
    company_name: str | None,
    asset_kind: str = "equity",
) -> bool:
    text = _normalize_text(f"{title or ''} {content or ''}")
    if not text:
        return False
    ticker_norm = ticker.lower().strip()
    ticker_hit = bool(re.search(rf"\b{re.escape(ticker_norm)}\b", text))
    if asset_kind == "etf":
        theme = _ETF_THEME_MAP.get(ticker.upper().strip(), {})
        theme_terms = [_normalize_text(company_name)] if company_name else []
        theme_terms.extend(_normalize_text(item) for item in theme.get("themes", []))
        theme_terms = [term for term in theme_terms if term]
        theme_hit = any(term in text for term in theme_terms)
        # ETF queries often target underlying index/macro themes, not the ticker itself.
        return ticker_hit or theme_hit
    if company_name:
        company_norm = _normalize_text(company_name)
        name_tokens = [token for token in company_norm.split(" ") if len(token) >= 4]
        name_hit = company_norm in text or sum(token in text for token in name_tokens) >= min(2, len(name_tokens))
        return ticker_hit or name_hit
    return ticker_hit


def _normalize_raw_row(
    ticker: str,
    market: MarketType,
    row: dict[str, Any],
    *,
    search_provider: str | None = None,
    company_name: str | None = None,
    asset_kind: str = "equity",
) -> dict[str, Any] | None:
    title = str(row.get("title", "")).strip()
    if not title:
        return None
    raw_url = str(row.get("url", "")).strip()
    canonical_url = _canonicalize_url(raw_url)
    domain = _extract_domain(canonical_url or raw_url)
    if asset_kind == "etf" and domain == "sec.gov":
        return None
    if _looks_like_noise(domain):
        return None
    content = str(row.get("content", "")).strip() or None
    if not _row_matches_entity(ticker, title, content, company_name, asset_kind=asset_kind):
        return None
    publisher, source_type, weight, domain = classify_news_source(row.get("source"), canonical_url or raw_url)
    published_at = _parse_datetime(
        row.get("publish_time") or row.get("published_at") or row.get("date") or row.get("published_date")
    )
    if published_at is None:
        published_at = _infer_datetime_from_text(
            canonical_url,
            title,
            content,
        )
    return {
        "ticker": ticker,
        "market": market.value if market else None,
        "title": title,
        "content": content,
        "keyword": str(row.get("keyword", "")).strip() or None,
        "source": publisher,
        "source_type": source_type,
        "source_weight": weight,
        "date": _format_datetime(published_at),
        "publish_time": _format_datetime(published_at),
        "published_at": _format_datetime(published_at),
        "url": canonical_url or raw_url,
        "domain": domain,
        "search_provider": search_provider,
    }


def _coerce_company_news_rows(
    ticker: str,
    market: MarketType,
    items: list[CompanyNews] | list[dict[str, Any]],
    *,
    search_provider: str | None = None,
    company_name: str | None = None,
    asset_kind: str = "equity",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            raw = item.model_dump()
        elif isinstance(item, dict):
            raw = item
        else:
            continue
        normalized = _normalize_raw_row(
            ticker,
            market,
            raw,
            search_provider=search_provider,
            company_name=company_name,
            asset_kind=asset_kind,
        )
        if normalized:
            rows.append(normalized)
    rows.sort(key=lambda x: x.get("published_at", ""), reverse=True)
    return rows


def _tavily_search(
    ticker: str,
    market: MarketType,
    start_date: str | None,
    end_date: str | None,
    limit: int,
    timeout: int,
) -> list[dict[str, Any]]:
    key = get_api_key("tavily")
    if not key:
        return []
    company_name = resolve_company_name(ticker, market)
    asset_kind = resolve_asset_kind(ticker, market)
    payload: list[dict[str, Any]] = []
    for query in build_news_queries(ticker, market, company_name=company_name, asset_kind=asset_kind):
        body: dict[str, Any] = {
            "api_key": key,
            "query": query,
            "topic": "news",
            "search_depth": "advanced",
            "max_results": max(limit, 10),
            "include_answer": False,
            "include_raw_content": False,
        }
        if start_date:
            body["start_date"] = start_date
        if end_date:
            body["end_date"] = end_date
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json=body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Tavily search failed for %s: %s", ticker, exc)
            continue
        results = data.get("results", [])
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            publisher = item.get("source") or item.get("domain") or _publisher_from_domain(_extract_domain(item.get("url")))
            payload.append(
                {
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "url": item.get("url"),
                    "source": publisher,
                    "published_at": item.get("published_date") or item.get("published_at"),
                }
            )
    return _coerce_company_news_rows(
        ticker,
        market,
        payload,
        search_provider="tavily",
        company_name=company_name,
        asset_kind=asset_kind,
    )


def _brave_search(
    ticker: str,
    market: MarketType,
    limit: int,
    timeout: int,
) -> list[dict[str, Any]]:
    key = get_api_key("brave")
    if not key:
        return []
    company_name = resolve_company_name(ticker, market)
    asset_kind = resolve_asset_kind(ticker, market)
    rows: list[dict[str, Any]] = []
    for query in build_news_queries(ticker, market, company_name=company_name, asset_kind=asset_kind):
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={
                    "q": query,
                    "count": max(limit, 10),
                    "freshness": "pw",
                    "search_lang": "en",
                },
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": key,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Brave search failed for %s: %s", ticker, exc)
            continue
        web = data.get("web", {})
        results = web.get("results", []) if isinstance(web, dict) else []
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            profile = item.get("profile", {}) if isinstance(item.get("profile"), dict) else {}
            rows.append(
                {
                    "title": item.get("title"),
                    "content": item.get("description"),
                    "url": item.get("url"),
                    "source": profile.get("long_name") or item.get("meta_url", {}).get("hostname") or item.get("url"),
                    "published_at": item.get("page_age") or item.get("age"),
                }
            )
    return _coerce_company_news_rows(
        ticker,
        market,
        rows,
        search_provider="brave",
        company_name=company_name,
        asset_kind=asset_kind,
    )


def fetch_search_news(
    ticker: str,
    market: MarketType,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 10,
    timeout: int = 20,
) -> list[dict[str, Any]]:
    rows = _tavily_search(ticker, market, start_date, end_date, limit, timeout)
    if len(rows) < max(3, limit // 2):
        rows.extend(_brave_search(ticker, market, limit, timeout))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("url") or f"{row.get('title')}::{row.get('published_at')}"
        if not key:
            continue
        existing = deduped.get(key)
        if existing is None or float(row.get("source_weight", 0.0)) > float(existing.get("source_weight", 0.0)):
            deduped[key] = row
    out = list(deduped.values())
    out.sort(key=lambda x: (x.get("published_at", ""), float(x.get("source_weight", 0.0))), reverse=True)
    return out[: max(limit * 3, 15)]


def _event_newsiness_score(
    headline: str | None,
    summary: str | None,
    primary_source: str | None,
    source_type: str | None,
    url: str | None,
) -> float:
    text = _normalize_text(f"{headline or ''} {summary or ''}")
    score = 0.0
    domain = _extract_domain(url)

    if source_type in {"wire", "mainstream_media"}:
        score += 3.0
    elif source_type == "official":
        score += 1.5

    if primary_source in {"Reuters", "Bloomberg", "Associated Press", "MarketWatch", "CNBC"}:
        score += 2.0

    if any(token in text for token in ("earnings", "results", "announced", "launch", "partnership", "guidance", "revenue")):
        score += 1.5
    if any(token in text for token in ("press release", "news release", "investor relations")):
        score += 1.0

    sec_doc_type = _detect_sec_doc_type(headline, summary, url)
    if sec_doc_type == "earnings_release":
        score += 2.5
    elif sec_doc_type in {"8k", "10q", "10k", "press_release"}:
        score += 1.5
    elif sec_doc_type in {"filing_index", "form_144", "prospectus_424b4"}:
        score -= 3.0

    if any(token in text for token in ("edgar filing documents", "index htm", "index.htm", "form 144", "424b4")):
        score -= 4.0
    if re.fullmatch(r"[a-z0-9\-]+ sec gov", text) or re.fullmatch(r"[a-z0-9\-]+", text):
        score -= 2.0
    if domain == "sec.gov" and any(token in text for token in ("form 144", "index", "filing documents")):
        score -= 3.0

    return score


def _detect_sec_doc_type(headline: str | None, summary: str | None, url: str | None) -> str:
    text = _normalize_text(f"{headline or ''} {summary or ''} {url or ''}")
    if "form 144" in text:
        return "form_144"
    if "424b4" in text:
        return "prospectus_424b4"
    if "10 q" in text or "10-q" in text:
        return "10q"
    if "10 k" in text or "10-k" in text:
        return "10k"
    if "8 k" in text or "8-k" in text:
        return "8k"
    if "results of operations and financial conditions" in text or "financial results" in text:
        return "earnings_release"
    if "press release" in text or "news release" in text:
        return "press_release"
    if "edgar filing documents" in text or "index.htm" in text or "-index.htm" in text:
        return "filing_index"
    return "sec_filing"


def _refine_headline_and_summary(
    ticker: str,
    headline: str,
    summary: str,
    primary_source: str,
    primary_url: str,
) -> tuple[str, str]:
    if primary_source != "SEC":
        return headline, summary

    doc_type = _detect_sec_doc_type(headline, summary, primary_url)
    text = summary.strip() or headline.strip()
    low = text.lower()
    company = resolve_company_name(ticker, MarketType.US) or ticker.upper().strip()

    if doc_type == "earnings_release":
        if "fourth quarter" in text.lower():
            return f"{company} reports fourth-quarter and full-year results", text
        if "third quarter" in text.lower():
            return f"{company} reports third-quarter results", text
        return f"{company} financial results filing", text
    if doc_type == "8k":
        if "board of directors" in low or "resignation" in low:
            return f"{company} board change filing", f"{company} disclosed a board-level change in a current report filing."
        if "financial results" in low or "results of operations and financial conditions" in low:
            if "third quarter" in low:
                return f"{company} reports third-quarter results", f"{company} disclosed third-quarter financial results in an 8-K filing."
            if "fourth quarter" in low or "full year" in low:
                return f"{company} reports fourth-quarter and full-year results", f"{company} disclosed fourth-quarter and full-year financial results in an 8-K filing."
            return f"{company} financial results filing", f"{company} disclosed financial results in a current report filing."
        return f"{company} current report filing (8-K)", text
    if doc_type == "10q":
        if "developer services" in low or "blockchain infrastructure" in low:
            return f"{company} files quarterly report with business update", f"{company} filed a quarterly report describing business and product development updates."
        return f"{company} quarterly report filing (10-Q)", f"{company} filed a quarterly report with the SEC."
    if doc_type == "10k":
        return f"{company} annual report filing (10-K)", f"{company} filed an annual report with the SEC."
    if doc_type == "form_144":
        return f"{company} Form 144 filing", f"A Form 144 related to {company} securities was filed with the SEC."
    if doc_type == "prospectus_424b4":
        return f"{company} IPO prospectus filing", f"{company} filed an IPO prospectus document with the SEC."
    if doc_type == "filing_index":
        if "10-q" in low or "10 q" in low:
            return f"{company} quarterly report filing (10-Q)", f"{company} filed a quarterly report with the SEC."
        if "10-k" in low or "10 k" in low:
            return f"{company} annual report filing (10-K)", f"{company} filed an annual report with the SEC."
        return f"{company} SEC filing index", f"SEC filing index page related to {company}."
    if "board of directors" in low or "resignation" in low:
        return f"{company} board change filing", f"{company} disclosed a board-level change in an SEC filing."
    if "developer services" in low or "blockchain infrastructure" in low:
        return f"{company} business update filing", f"{company} disclosed business and product development updates in an SEC filing."
    return headline, summary


def integrate_news_rows(
    ticker: str,
    market: MarketType,
    rows: list[dict[str, Any]],
    limit: int,
) -> list[IntegratedNewsEvent]:
    events: list[dict[str, Any]] = []
    ordered = sorted(
        rows,
        key=lambda x: (x.get("published_at", ""), float(x.get("source_weight", 0.0))),
        reverse=True,
    )
    for row in ordered:
        title_norm = _normalize_text(row.get("title"))
        tokens = _token_set(row.get("title")) | _token_set(row.get("content"))
        published_at = _parse_datetime(row.get("published_at") or row.get("date"))
        canonical_url = _canonicalize_url(row.get("url"))
        matched: dict[str, Any] | None = None
        for event in events:
            if canonical_url and canonical_url in event["urls"]:
                matched = event
                break
            event_dt = event["published_dt"]
            delta_ok = (
                published_at is None
                or event_dt is None
                or abs((published_at - event_dt).total_seconds()) <= timedelta(hours=24).total_seconds()
            )
            if not delta_ok:
                continue
            title_ratio = SequenceMatcher(None, title_norm, event["title_norm"]).ratio() if title_norm else 0.0
            overlap = 0.0
            union = tokens | event["tokens"]
            if union:
                overlap = len(tokens & event["tokens"]) / len(union)
            if title_ratio >= 0.82 or overlap >= 0.6:
                matched = event
                break
        if matched is None:
            matched = {
                "title_norm": title_norm,
                "tokens": set(tokens),
                "published_dt": published_at,
                "sources": [],
                "urls": set(),
                "rows": [],
            }
            events.append(matched)
        matched["rows"].append(row)
        matched["tokens"].update(tokens)
        if canonical_url:
            matched["urls"].add(canonical_url)
        if published_at is not None:
            if matched["published_dt"] is None or published_at < matched["published_dt"]:
                matched["published_dt"] = published_at
        matched["sources"].append(
            {
                "publisher": row.get("source") or "unknown",
                "source_type": row.get("source_type") or "unknown",
                "weight": float(row.get("source_weight", 0.0) or 0.0),
                "url": row.get("url") or "",
                "domain": row.get("domain"),
                "search_provider": row.get("search_provider"),
                "published_at": row.get("published_at") or row.get("date"),
                "title": row.get("title"),
                "snippet": row.get("content"),
            }
        )

    integrated: list[IntegratedNewsEvent] = []
    for event in events:
        rows_in_event = event["rows"]
        if not rows_in_event:
            continue
        rows_sorted = sorted(
            rows_in_event,
            key=lambda x: (
                float(x.get("source_weight", 0.0) or 0.0),
                x.get("published_at", ""),
                len(str(x.get("title", ""))),
            ),
            reverse=True,
        )
        primary = rows_sorted[0]
        by_publisher: dict[str, float] = {}
        source_objs: list[IntegratedNewsSource] = []
        first_seen: datetime | None = None
        latest_seen: datetime | None = None
        official_confirmed = False
        for src in sorted(
            event["sources"],
            key=lambda x: (float(x.get("weight", 0.0)), str(x.get("published_at", ""))),
            reverse=True,
        ):
            source_objs.append(IntegratedNewsSource(**src))
            publisher = str(src.get("publisher") or "unknown")
            weight = float(src.get("weight", 0.0) or 0.0)
            by_publisher[publisher] = max(weight, by_publisher.get(publisher, 0.0))
            official_confirmed = official_confirmed or src.get("source_type") == "official"
            dt = _parse_datetime(src.get("published_at"))
            if dt is not None:
                first_seen = dt if first_seen is None else min(first_seen, dt)
                latest_seen = dt if latest_seen is None else max(latest_seen, dt)
        weighted_score = round(sum(by_publisher.values()), 3)
        high_weight_count = sum(1 for weight in by_publisher.values() if weight >= 0.7)
        source_count = len(by_publisher)
        consensus_passed = official_confirmed or (
            high_weight_count >= 2 and weighted_score >= 1.5
        ) or (source_count >= 3 and weighted_score >= 1.6)
        published_dt = event["published_dt"] or _parse_datetime(primary.get("published_at") or primary.get("date"))
        tags = sorted(
            token
            for token in event["tokens"]
            if token not in {ticker.lower(), "stock", "company", "latest", "news"} and len(token) >= 3
        )[:8]
        event_id_seed = f"{ticker}|{event['title_norm']}|{_format_datetime(published_dt)}"
        event_id = hashlib.sha1(event_id_seed.encode("utf-8")).hexdigest()[:16]
        headline = str(primary.get("title") or "").strip()
        summary = str(primary.get("content") or "").strip() or headline
        published_at_text = _format_datetime(published_dt) or str(primary.get("published_at") or primary.get("date") or "")
        primary_source = str(primary.get("source") or "unknown")
        urls = sorted(u for u in event["urls"] if u)
        primary_url = urls[0] if urls else str(primary.get("url") or "")
        headline, summary = _refine_headline_and_summary(ticker, headline, summary, primary_source, primary_url)
        newsiness = _event_newsiness_score(
            headline=headline,
            summary=summary,
            primary_source=primary_source,
            source_type=str(primary.get("source_type") or ""),
            url=primary_url,
        )
        integrated.append(
            IntegratedNewsEvent(
                event_id=event_id,
                ticker=ticker,
                market=market,
                headline=headline,
                summary=summary,
                published_at=published_at_text,
                first_seen_at=_format_datetime(first_seen) or published_at_text,
                latest_seen_at=_format_datetime(latest_seen) or published_at_text,
                primary_source=primary_source,
                source_count=source_count,
                high_weight_source_count=high_weight_count,
                weighted_source_score=weighted_score,
                consensus_passed=consensus_passed,
                official_confirmed=official_confirmed,
                sources=source_objs,
                urls=urls or ([primary_url] if primary_url else []),
                tags=tags,
                title=headline,
                source=primary_source,
                date=published_at_text,
                url=primary_url,
                publish_time=published_at_text,
                content=summary,
                keyword=", ".join(tags) or None,
            )
        )
    integrated.sort(
        key=lambda x: (
            x.consensus_passed,
            _event_newsiness_score(x.headline, x.summary, x.primary_source, x.sources[0].source_type if x.sources else "", x.url),
            x.weighted_source_score,
            x.published_at,
        ),
        reverse=True,
    )
    return integrated[:limit]
