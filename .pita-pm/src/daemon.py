"""
PITA-PM Daemon — The Autonomous Loop

Runs continuously: 8 hours scanning code, 16 hours crawling markets.
Persists state across restarts. Logs everything.

Usage:
    python -m pita_pm daemon          # Foreground
    python -m pita_pm daemon --bg     # Background (writes PID file)
    python -m pita_pm daemon --stop   # Stop background daemon
"""
import json
import os
import signal
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

from .scanner import scan_codebase
from .personality import format_standup, vibe_check, generate_greeting
from .shift_runner import _load_state, _save_state, run_scan
from .market_crawler import run_sleep_mode
from .knowledge_store import (
    record_scan, get_sin_summary, get_scan_trend, get_market_stats,
)


PID_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'daemon.pid')
LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')
SCAN_INTERVAL = 30 * 60        # 30 minutes between scans
MARKET_INTERVAL = 60 * 60      # 60 minutes between market crawls
SHIFT_HOURS = 8
OFF_HOURS = 16


def _setup_logging() -> logging.Logger:
    """Set up daemon-specific logging."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger('pita-pm-daemon')
    log.setLevel(logging.INFO)

    # File handler — rotates daily
    today = datetime.utcnow().strftime('%Y%m%d')
    fh = logging.FileHandler(os.path.join(LOG_DIR, f'daemon_{today}.log'))
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    ))
    log.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s [PITA-PM] %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(ch)

    return log


def _write_pid() -> None:
    """Write PID file for background mode."""
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


def _read_pid() -> Optional[int]:
    """Read PID from file."""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return None
    return None


def _remove_pid() -> None:
    """Clean up PID file."""
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def _is_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_daemon() -> str:
    """Stop a running background daemon."""
    pid = _read_pid()
    if pid is None:
        return "No daemon PID file found. Nothing to stop."
    if not _is_running(pid):
        _remove_pid()
        return f"Daemon (PID {pid}) is not running. Cleaned up stale PID file."
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait up to 5 seconds for clean shutdown
        for _ in range(10):
            time.sleep(0.5)
            if not _is_running(pid):
                _remove_pid()
                return f"Daemon (PID {pid}) stopped."
        # Force kill
        os.kill(pid, signal.SIGKILL)
        _remove_pid()
        return f"Daemon (PID {pid}) force-killed."
    except OSError as e:
        return f"Failed to stop daemon (PID {pid}): {e}"


def run_daemon(project_dir: str = ".", background: bool = False) -> None:
    """
    Main daemon loop.

    Cycle:
      ON-SHIFT  (8 hours): Scan every 30 min, log findings, track trends.
      OFF-SHIFT (16 hours): Crawl markets every 60 min, score sectors.
    """
    log = _setup_logging()
    _write_pid()
    running = True

    def _handle_signal(signum, frame):
        nonlocal running
        log.info(f"Received signal {signum}. Shutting down gracefully...")
        running = False

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Determine initial shift state
    state = _load_state()
    shift_start = datetime.utcnow()
    state['on_shift'] = True
    state['shift_start'] = shift_start.isoformat()
    state['shift_number'] = state.get('shift_number', 0) + 1
    _save_state(state)

    log.info("=" * 60)
    log.info("PITA-PM DAEMON STARTED")
    log.info(f"  PID: {os.getpid()}")
    log.info(f"  Project: {os.path.abspath(project_dir)}")
    log.info(f"  Shift #{state['shift_number']}")
    log.info(f"  Schedule: {SHIFT_HOURS}h on / {OFF_HOURS}h off")
    log.info(f"  Scan interval: {SCAN_INTERVAL // 60} min")
    log.info(f"  Market crawl interval: {MARKET_INTERVAL // 60} min")
    log.info("=" * 60)

    # Opening greeting
    greeting = generate_greeting(
        scan_count=state['shift_number'] - 1,
        last_vibe=state.get('last_vibe_score'),
    )
    log.info(greeting)

    last_scan = 0.0
    last_market_crawl = 0.0
    on_shift = True

    while running:
        now = datetime.utcnow()
        elapsed_hours = (now - shift_start).total_seconds() / 3600

        # ── Shift transition logic ──
        if on_shift and elapsed_hours >= SHIFT_HOURS:
            # Transition to off-shift
            on_shift = False
            state['on_shift'] = False
            _save_state(state)
            log.info("")
            log.info("=" * 60)
            log.info("SHIFT ENDED — Entering sleep mode (market crawling)")
            log.info(f"  Scans completed this shift: {state.get('total_scans', 0)}")
            log.info(f"  Last vibe: {state.get('last_vibe_score', 'N/A')}")
            log.info("  I'll be back. The code will still be here.")
            log.info("  Some of it shouldn't be, but it will be.")
            log.info("=" * 60)

        elif not on_shift and elapsed_hours >= (SHIFT_HOURS + OFF_HOURS):
            # New cycle — back on shift
            shift_start = datetime.utcnow()
            on_shift = True
            state['on_shift'] = True
            state['shift_start'] = shift_start.isoformat()
            state['shift_number'] = state.get('shift_number', 0) + 1
            _save_state(state)
            log.info("")
            log.info("=" * 60)
            log.info(f"NEW SHIFT #{state['shift_number']} — Back on duty")
            log.info("  Miss me? No? That's fine. Your code missed me.")
            log.info("=" * 60)
            last_scan = 0.0  # Force immediate scan

        # ── On-shift: Code scanning ──
        if on_shift and (time.time() - last_scan) >= SCAN_INTERVAL:
            try:
                log.info("")
                log.info(f"[Scan] Starting codebase scan...")
                report = scan_codebase(root_dir=project_dir)
                sin_history = get_sin_summary()
                previous_vibe = state.get('last_vibe_score')

                standup = format_standup(report, sin_history, previous_vibe)
                for line in standup.split('\n'):
                    log.info(line)

                # Record
                record_scan(
                    total_issues=report.total_issues,
                    new_issues=report.total_issues,
                    resolved_issues=0,
                    vibe_score=report.vibe_score,
                    summary=f"{report.total_issues} issues, {report.files_scanned} files",
                )

                state['last_scan_time'] = datetime.utcnow().isoformat()
                state['last_vibe_score'] = report.vibe_score
                state['total_scans'] = state.get('total_scans', 0) + 1
                vibes = state.get('scan_history_vibes', [])
                vibes.append(round(report.vibe_score, 1))
                state['scan_history_vibes'] = vibes[-100:]
                _save_state(state)

                last_scan = time.time()
                log.info(f"[Scan] Complete. Vibe: {report.vibe_score:.0f}/100. "
                         f"Next scan in {SCAN_INTERVAL // 60} min.")
            except Exception as e:
                log.error(f"[Scan] Failed: {e}")
                last_scan = time.time()  # Don't retry immediately

        # ── Off-shift: Market crawling ──
        if not on_shift and (time.time() - last_market_crawl) >= MARKET_INTERVAL:
            try:
                log.info("")
                log.info(f"[Market] Starting market intelligence crawl...")
                report = run_sleep_mode()
                for line in report.split('\n'):
                    log.info(line)

                last_market_crawl = time.time()
                log.info(f"[Market] Complete. Next crawl in {MARKET_INTERVAL // 60} min.")
            except Exception as e:
                log.error(f"[Market] Failed: {e}")
                last_market_crawl = time.time()

        # Sleep 30 seconds between checks
        for _ in range(30):
            if not running:
                break
            time.sleep(1)

    # Clean shutdown
    _remove_pid()
    log.info("PITA-PM daemon stopped. The code is unsupervised now. Good luck.")


def daemon_status() -> str:
    """Check if daemon is running."""
    pid = _read_pid()
    if pid is None:
        return "Daemon: NOT RUNNING (no PID file)"
    if _is_running(pid):
        state = _load_state()
        shift = "ON SHIFT" if state.get('on_shift') else "OFF SHIFT (market crawling)"
        return (f"Daemon: RUNNING (PID {pid})\n"
                f"  Status: {shift}\n"
                f"  Shift #{state.get('shift_number', '?')}\n"
                f"  Last vibe: {state.get('last_vibe_score', 'N/A')}\n"
                f"  Total scans: {state.get('total_scans', 0)}")
    else:
        _remove_pid()
        return "Daemon: NOT RUNNING (stale PID file cleaned up)"
