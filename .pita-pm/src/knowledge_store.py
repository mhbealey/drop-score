"""
SQLite knowledge store for PITA-PM.

Stores:
  - Sin ledger (code issues, escalation tiers, offense counts)
  - Market intelligence (articles, sentiment scores, risk flags)
  - Scan history (diffs between scans for trend tracking)
"""
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'market-intel.db')


def _get_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Get a database connection, creating tables if needed."""
    path = db_path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)
    return conn


def _ensure_tables(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sin_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_number INTEGER,
            description TEXT,
            severity INTEGER DEFAULT 1,
            offense_count INTEGER DEFAULT 1,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            resolved INTEGER DEFAULT 0,
            resolved_at TEXT
        );

        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_time TEXT NOT NULL,
            total_issues INTEGER,
            new_issues INTEGER,
            resolved_issues INTEGER,
            vibe_score REAL,
            summary TEXT
        );

        CREATE TABLE IF NOT EXISTS market_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            title TEXT NOT NULL,
            url TEXT,
            sector TEXT,
            sentiment_score REAL,
            risk_flag INTEGER DEFAULT 0,
            crawled_at TEXT NOT NULL,
            published_at TEXT
        );

        CREATE TABLE IF NOT EXISTS sector_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sector TEXT NOT NULL,
            score REAL NOT NULL,
            article_count INTEGER,
            risk_flags INTEGER DEFAULT 0,
            scored_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_sin_type ON sin_ledger(issue_type);
        CREATE INDEX IF NOT EXISTS idx_sin_file ON sin_ledger(file_path);
        CREATE INDEX IF NOT EXISTS idx_article_sector ON market_articles(sector);
        CREATE INDEX IF NOT EXISTS idx_sector_scored ON sector_scores(scored_at);
    """)
    conn.commit()


# ─── Sin Ledger ───────────────────────────────────────────────

def record_sin(issue_type: str, file_path: str, line_number: int,
               description: str, severity: int = 1,
               db_path: Optional[str] = None) -> int:
    """Record or update a code sin. Returns the current offense count."""
    conn = _get_db(db_path)
    now = datetime.utcnow().isoformat()

    # Check if this exact sin already exists (same type + file + line)
    existing = conn.execute(
        "SELECT id, offense_count FROM sin_ledger "
        "WHERE issue_type = ? AND file_path = ? AND line_number = ? AND resolved = 0",
        (issue_type, file_path, line_number)
    ).fetchone()

    if existing:
        new_count = existing['offense_count'] + 1
        conn.execute(
            "UPDATE sin_ledger SET offense_count = ?, last_seen = ?, severity = ? "
            "WHERE id = ?",
            (new_count, now, severity, existing['id'])
        )
        conn.commit()
        conn.close()
        return new_count
    else:
        conn.execute(
            "INSERT INTO sin_ledger "
            "(issue_type, file_path, line_number, description, severity, "
            " offense_count, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (issue_type, file_path, line_number, description, severity, now, now)
        )
        conn.commit()
        conn.close()
        return 1


def resolve_sin(issue_type: str, file_path: str, line_number: int,
                db_path: Optional[str] = None) -> None:
    """Mark a sin as resolved. We remember, but we forgive."""
    conn = _get_db(db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE sin_ledger SET resolved = 1, resolved_at = ? "
        "WHERE issue_type = ? AND file_path = ? AND line_number = ? AND resolved = 0",
        (now, issue_type, file_path, line_number)
    )
    conn.commit()
    conn.close()


def get_active_sins(db_path: Optional[str] = None) -> List[dict]:
    """Get all unresolved sins, sorted by offense count descending."""
    conn = _get_db(db_path)
    rows = conn.execute(
        "SELECT * FROM sin_ledger WHERE resolved = 0 "
        "ORDER BY offense_count DESC, severity DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sin_summary(db_path: Optional[str] = None) -> Dict[str, int]:
    """Get offense counts by issue type."""
    conn = _get_db(db_path)
    rows = conn.execute(
        "SELECT issue_type, SUM(offense_count) as total, COUNT(*) as instances "
        "FROM sin_ledger WHERE resolved = 0 "
        "GROUP BY issue_type ORDER BY total DESC"
    ).fetchall()
    conn.close()
    return {r['issue_type']: {'total': r['total'], 'instances': r['instances']}
            for r in rows}


def get_chronic_offenders(min_offenses: int = 7,
                          db_path: Optional[str] = None) -> List[dict]:
    """Get sins that have reached chronic tier."""
    conn = _get_db(db_path)
    rows = conn.execute(
        "SELECT * FROM sin_ledger WHERE resolved = 0 AND offense_count >= ? "
        "ORDER BY offense_count DESC",
        (min_offenses,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Scan History ─────────────────────────────────────────────

def record_scan(total_issues: int, new_issues: int, resolved_issues: int,
                vibe_score: float, summary: str,
                db_path: Optional[str] = None) -> None:
    """Record a scan result for trend tracking."""
    conn = _get_db(db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO scan_history "
        "(scan_time, total_issues, new_issues, resolved_issues, vibe_score, summary) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (now, total_issues, new_issues, resolved_issues, vibe_score, summary)
    )
    conn.commit()
    conn.close()


def get_scan_trend(days: int = 7,
                   db_path: Optional[str] = None) -> List[dict]:
    """Get scan history for trend analysis."""
    conn = _get_db(db_path)
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM scan_history WHERE scan_time > ? ORDER BY scan_time",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─── Market Intelligence ──────────────────────────────────────

def store_article(title: str, source: str, sector: str,
                  sentiment_score: float, url: str = "",
                  risk_flag: bool = False, published_at: str = "",
                  db_path: Optional[str] = None) -> None:
    """Store a crawled market article."""
    conn = _get_db(db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO market_articles "
        "(source, title, url, sector, sentiment_score, risk_flag, "
        " crawled_at, published_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (source, title, url, sector, sentiment_score,
         1 if risk_flag else 0, now, published_at)
    )
    conn.commit()
    conn.close()


def store_sector_score(sector: str, score: float, article_count: int,
                       risk_flags: int = 0,
                       db_path: Optional[str] = None) -> None:
    """Store a sector sentiment score."""
    conn = _get_db(db_path)
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO sector_scores (sector, score, article_count, risk_flags, scored_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (sector, score, article_count, risk_flags, now)
    )
    conn.commit()
    conn.close()


def get_latest_sector_scores(db_path: Optional[str] = None) -> Dict[str, dict]:
    """Get the most recent score for each sector."""
    conn = _get_db(db_path)
    rows = conn.execute(
        "SELECT s1.* FROM sector_scores s1 "
        "INNER JOIN (SELECT sector, MAX(scored_at) as max_time "
        "            FROM sector_scores GROUP BY sector) s2 "
        "ON s1.sector = s2.sector AND s1.scored_at = s2.max_time"
    ).fetchall()
    conn.close()
    return {r['sector']: dict(r) for r in rows}


def get_risk_flags(hours: int = 24,
                   db_path: Optional[str] = None) -> List[dict]:
    """Get recent risk flags."""
    conn = _get_db(db_path)
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    rows = conn.execute(
        "SELECT * FROM market_articles "
        "WHERE risk_flag = 1 AND crawled_at > ? "
        "ORDER BY crawled_at DESC",
        (cutoff,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_market_stats(db_path: Optional[str] = None) -> dict:
    """Get overall market intelligence statistics."""
    conn = _get_db(db_path)
    stats = {}
    stats['total_articles'] = conn.execute(
        "SELECT COUNT(*) FROM market_articles"
    ).fetchone()[0]
    stats['total_risk_flags'] = conn.execute(
        "SELECT COUNT(*) FROM market_articles WHERE risk_flag = 1"
    ).fetchone()[0]
    stats['sectors_tracked'] = conn.execute(
        "SELECT COUNT(DISTINCT sector) FROM sector_scores"
    ).fetchone()[0]
    stats['total_scans'] = conn.execute(
        "SELECT COUNT(*) FROM scan_history"
    ).fetchone()[0]
    stats['total_sins_recorded'] = conn.execute(
        "SELECT COUNT(*) FROM sin_ledger"
    ).fetchone()[0]
    stats['active_sins'] = conn.execute(
        "SELECT COUNT(*) FROM sin_ledger WHERE resolved = 0"
    ).fetchone()[0]
    stats['resolved_sins'] = conn.execute(
        "SELECT COUNT(*) FROM sin_ledger WHERE resolved = 1"
    ).fetchone()[0]
    conn.close()
    return stats
