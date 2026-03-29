"""
PITA-PM Market Crawler — Sleep Mode Intelligence

While the developer sleeps, PITA-PM doesn't. It crawls financial news,
runs sentiment analysis, and scores sectors. Because even off-shift,
this agent has a work-life balance problem.

Uses DuckDuckGo (no API key needed) for market news crawling.
"""
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .knowledge_store import (
    store_article, store_sector_score,
    get_latest_sector_scores, get_risk_flags, get_market_stats,
)


# ─── Sentiment Keywords ──────────────────────────────────────

POSITIVE_KEYWORDS = {
    'surge', 'rally', 'gain', 'rise', 'soar', 'jump', 'boom', 'bullish',
    'record high', 'outperform', 'upgrade', 'beat expectations', 'strong earnings',
    'growth', 'recovery', 'expansion', 'breakout', 'momentum', 'upside',
    'all-time high', 'profit', 'dividend increase', 'buy rating',
}

NEGATIVE_KEYWORDS = {
    'crash', 'plunge', 'drop', 'fall', 'decline', 'sell-off', 'bearish',
    'recession', 'layoff', 'bankrupt', 'default', 'downgrade', 'miss',
    'warning', 'debt', 'loss', 'risk', 'volatile', 'uncertainty',
    'overvalued', 'bubble', 'correction', 'panic', 'crisis', 'fraud',
    'investigation', 'lawsuit', 'regulatory', 'tariff', 'sanctions',
}

SECTOR_KEYWORDS = {
    'tech': {'technology', 'software', 'ai ', 'artificial intelligence', 'semiconductor',
             'chip', 'cloud', 'saas', 'cyber', 'data center', 'nvidia', 'apple',
             'microsoft', 'google', 'meta', 'amazon', 'tesla'},
    'finance': {'bank', 'financial', 'wall street', 'fed ', 'federal reserve',
                'interest rate', 'treasury', 'jpmorgan', 'goldman', 'credit',
                'mortgage', 'lending', 'fintech', 'insurance'},
    'energy': {'oil', 'gas', 'energy', 'renewable', 'solar', 'wind', 'battery',
               'ev ', 'electric vehicle', 'opec', 'petroleum', 'crude',
               'natural gas', 'pipeline', 'exxon', 'chevron'},
    'healthcare': {'health', 'pharma', 'biotech', 'drug', 'fda', 'clinical trial',
                   'vaccine', 'hospital', 'medical', 'pfizer', 'moderna',
                   'johnson & johnson', 'abbvie', 'diagnosis'},
    'consumer': {'retail', 'consumer', 'spending', 'walmart', 'target', 'amazon',
                 'e-commerce', 'luxury', 'brand', 'food', 'beverage',
                 'restaurant', 'housing', 'real estate'},
    'industrial': {'manufacturing', 'industrial', 'supply chain', 'logistics',
                   'automotive', 'aerospace', 'defense', 'construction',
                   'infrastructure', 'caterpillar', 'boeing', 'deere'},
}


def _score_sentiment(text: str) -> float:
    """
    Simple keyword-based sentiment scoring.
    Returns -1.0 (very negative) to +1.0 (very positive).
    """
    text_lower = text.lower()
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    total = pos_hits + neg_hits
    if total == 0:
        return 0.0
    return (pos_hits - neg_hits) / total


def _classify_sector(text: str) -> str:
    """Classify an article's sector based on keyword matching."""
    text_lower = text.lower()
    scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[sector] = hits
    if not scores:
        return 'general'
    return max(scores, key=scores.get)


def _fetch_ddg_news(query: str, max_results: int = 10) -> List[dict]:
    """
    Fetch news headlines via DuckDuckGo HTML search.
    Returns list of {title, url, snippet}.

    Note: This is a lightweight crawler — no API key needed.
    Respects rate limits with delays between requests.
    """
    results = []
    encoded_q = urllib.parse.quote_plus(f"{query} stock market news")
    url = f"https://html.duckduckgo.com/html/?q={encoded_q}"

    headers = {
        'User-Agent': 'PITA-PM/1.0 (Market Intelligence Agent; +https://github.com/drop-score)',
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='replace')

        # Extract result titles and snippets from DDG HTML
        # Pattern: <a class="result__a" href="...">TITLE</a>
        title_pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL
        )
        snippet_pattern = re.compile(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL
        )

        titles = title_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (href, title) in enumerate(titles[:max_results]):
            # Clean HTML tags from title
            clean_title = re.sub(r'<[^>]+>', '', title).strip()
            clean_snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            if clean_title:
                results.append({
                    'title': clean_title,
                    'url': href,
                    'snippet': clean_snippet,
                })
    except Exception:
        pass  # Network failures are expected in CI — fail silently

    return results


def crawl_market_news(sectors: Optional[List[str]] = None,
                      max_per_sector: int = 8) -> Dict[str, List[dict]]:
    """
    Crawl financial news for each sector.
    Returns {sector: [articles]}.
    """
    if sectors is None:
        sectors = list(SECTOR_KEYWORDS.keys())

    all_articles: Dict[str, List[dict]] = {}

    for sector in sectors:
        query = f"{sector} sector stock market"
        articles = _fetch_ddg_news(query, max_results=max_per_sector)

        processed = []
        for article in articles:
            full_text = f"{article['title']} {article.get('snippet', '')}"
            sentiment = _score_sentiment(full_text)
            detected_sector = _classify_sector(full_text)

            entry = {
                'title': article['title'],
                'url': article.get('url', ''),
                'sector': detected_sector if detected_sector != 'general' else sector,
                'sentiment': sentiment,
                'risk_flag': sentiment < -0.3,
            }
            processed.append(entry)

            # Store in database
            store_article(
                title=entry['title'],
                source='duckduckgo',
                sector=entry['sector'],
                sentiment_score=sentiment,
                url=entry.get('url', ''),
                risk_flag=entry['risk_flag'],
            )

        all_articles[sector] = processed
        time.sleep(2)  # Rate limiting — be a good citizen

    return all_articles


def score_sectors(articles_by_sector: Dict[str, List[dict]]) -> Dict[str, dict]:
    """
    Score each sector based on crawled articles.
    Returns {sector: {score, article_count, risk_flags, label}}.
    """
    scores = {}
    for sector, articles in articles_by_sector.items():
        if not articles:
            continue

        sentiments = [a['sentiment'] for a in articles]
        avg_sentiment = sum(sentiments) / len(sentiments)
        risk_count = sum(1 for a in articles if a.get('risk_flag'))

        # Label
        if avg_sentiment > 0.3:
            label = 'Bullish'
        elif avg_sentiment > 0.1:
            label = 'Positive'
        elif avg_sentiment > -0.1:
            label = 'Neutral'
        elif avg_sentiment > -0.3:
            label = 'Cautious'
        else:
            label = 'Bearish'

        scores[sector] = {
            'score': round(avg_sentiment, 3),
            'article_count': len(articles),
            'risk_flags': risk_count,
            'label': label,
        }

        store_sector_score(
            sector=sector,
            score=avg_sentiment,
            article_count=len(articles),
            risk_flags=risk_count,
        )

    return scores


def generate_market_report(sector_scores: Dict[str, dict],
                           risk_flags: Optional[List[dict]] = None) -> str:
    """Generate a market intelligence report in PITA-PM voice."""
    lines = []
    lines.append("=" * 70)
    lines.append("PITA-PM MARKET INTELLIGENCE REPORT")
    lines.append(f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("=" * 70)
    lines.append("")

    if not sector_scores:
        lines.append("  No market data available. Either the internet is down,")
        lines.append("  or the market finally gave up. Both seem plausible.")
        return "\n".join(lines)

    # Sector overview
    lines.append("  SECTOR SENTIMENT:")
    lines.append(f"  {'Sector':<15} {'Score':>7} {'Articles':>9} {'Flags':>6} {'Signal':<10}")
    lines.append(f"  {'-'*50}")

    for sector, info in sorted(sector_scores.items(),
                                key=lambda x: x[1]['score'], reverse=True):
        flag_marker = " !!" if info['risk_flags'] > 0 else ""
        lines.append(
            f"  {sector:<15} {info['score']:>+6.3f} {info['article_count']:>9} "
            f"{info['risk_flags']:>6} {info['label']:<10}{flag_marker}"
        )

    # Overall market mood
    all_scores = [v['score'] for v in sector_scores.values()]
    avg_market = sum(all_scores) / len(all_scores) if all_scores else 0

    lines.append("")
    if avg_market > 0.2:
        lines.append("  MARKET MOOD: Optimistic. Which, historically, is when things get interesting.")
    elif avg_market > 0:
        lines.append("  MARKET MOOD: Cautiously positive. The market is smiling but checking its phone.")
    elif avg_market > -0.2:
        lines.append("  MARKET MOOD: Mixed. The market can't decide if it wants coffee or a nap.")
    else:
        lines.append("  MARKET MOOD: Defensive. The market is wearing a helmet indoors.")

    # Risk flags
    if risk_flags:
        lines.append("")
        lines.append(f"  RISK FLAGS ({len(risk_flags)}):")
        for flag in risk_flags[:5]:
            lines.append(f"    - [{flag.get('sector', '?')}] {flag.get('title', 'Unknown')[:60]}")
        if len(risk_flags) > 5:
            lines.append(f"    ... and {len(risk_flags) - 5} more. Sleep well.")

    lines.append("")
    lines.append("=" * 70)

    # Compare to code quality (the PITA-PM special)
    lines.append("  FUN FACT: Your test coverage is probably a worse bet than the market.")
    lines.append("  At least the market has circuit breakers.")
    lines.append("=" * 70)

    return "\n".join(lines)


def run_sleep_mode(sectors: Optional[List[str]] = None) -> str:
    """
    Full sleep mode cycle: crawl, score, report.
    Returns the market report as a string.
    """
    articles = crawl_market_news(sectors)
    scores = score_sectors(articles)
    flags = get_risk_flags(hours=24)
    report = generate_market_report(scores, flags)

    # Save report to file
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(report_dir, exist_ok=True)
    filename = f"market_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
    filepath = os.path.join(report_dir, filename)
    with open(filepath, 'w') as f:
        f.write(report)

    return report
