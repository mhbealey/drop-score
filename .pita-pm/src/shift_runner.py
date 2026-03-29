"""
PITA-PM Shift Runner — 8 hours on, 16 hours off.

Manages the scan cycle, tracks trends between scans,
and decides whether to nag or celebrate.
"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from .scanner import scan_codebase, ScanReport
from .personality import (
    format_standup, generate_greeting, vibe_check, generate_quip,
)
from .knowledge_store import (
    record_scan, get_scan_trend, get_sin_summary, get_active_sins,
    get_market_stats,
)


STATE_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'shift-state.json')


def _load_state() -> dict:
    """Load persistent shift state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'shift_number': 0,
        'total_scans': 0,
        'shift_start': None,
        'last_scan_time': None,
        'last_vibe_score': None,
        'on_shift': False,
        'scan_history_vibes': [],
    }


def _save_state(state: dict) -> None:
    """Save persistent shift state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def start_shift(project_dir: str = ".") -> str:
    """
    Start a new shift. Greet the developer, run initial scan.
    Returns the greeting + initial standup.
    """
    state = _load_state()
    state['shift_number'] += 1
    state['shift_start'] = datetime.utcnow().isoformat()
    state['on_shift'] = True
    _save_state(state)

    # Generate greeting
    greeting = generate_greeting(
        scan_count=state['shift_number'] - 1,
        last_vibe=state.get('last_vibe_score'),
    )

    # Run initial scan
    report = run_scan(project_dir)

    return f"{greeting}\n\n{report}"


def run_scan(project_dir: str = ".", silent: bool = False) -> str:
    """
    Run a codebase scan and generate a standup report.
    Returns the formatted report.
    """
    state = _load_state()

    # Scan
    report = scan_codebase(root_dir=project_dir)

    # Get sin history for escalation
    sin_history = get_sin_summary()

    # Get previous vibe for trend
    previous_vibe = state.get('last_vibe_score')

    # Format standup
    standup = format_standup(report, sin_history, previous_vibe)

    # Record scan
    record_scan(
        total_issues=report.total_issues,
        new_issues=report.total_issues,  # TODO: diff with previous
        resolved_issues=0,
        vibe_score=report.vibe_score,
        summary=f"{report.total_issues} issues across {report.files_scanned} files",
    )

    # Update state
    state['total_scans'] = state.get('total_scans', 0) + 1
    state['last_scan_time'] = datetime.utcnow().isoformat()
    state['last_vibe_score'] = report.vibe_score
    vibes = state.get('scan_history_vibes', [])
    vibes.append(round(report.vibe_score, 1))
    state['scan_history_vibes'] = vibes[-50:]  # keep last 50
    _save_state(state)

    return standup


def get_status() -> str:
    """Get current PITA-PM status."""
    state = _load_state()
    stats = get_market_stats()
    lines = []
    lines.append("=" * 50)
    lines.append("PITA-PM STATUS")
    lines.append("=" * 50)

    if state.get('on_shift'):
        shift_start = state.get('shift_start', 'unknown')
        lines.append(f"  Status: ON SHIFT (since {shift_start})")
    else:
        lines.append("  Status: OFF SHIFT (sleeping / crawling markets)")

    lines.append(f"  Shift number: {state.get('shift_number', 0)}")
    lines.append(f"  Total scans: {state.get('total_scans', 0)}")
    lines.append(f"  Last vibe: {state.get('last_vibe_score', 'N/A')}")
    lines.append(f"  Last scan: {state.get('last_scan_time', 'never')}")

    # Vibe trend
    vibes = state.get('scan_history_vibes', [])
    if len(vibes) >= 2:
        recent = vibes[-3:]
        trend = recent[-1] - recent[0]
        if trend > 5:
            lines.append(f"  Trend: Improving (+{trend:.0f} over last {len(recent)} scans)")
        elif trend < -5:
            lines.append(f"  Trend: Declining ({trend:.0f} over last {len(recent)} scans)")
        else:
            lines.append(f"  Trend: Stable ({trend:+.0f})")

    # Market stats
    lines.append("")
    lines.append("  MARKET INTEL:")
    lines.append(f"    Articles crawled: {stats.get('total_articles', 0)}")
    lines.append(f"    Risk flags: {stats.get('total_risk_flags', 0)}")
    lines.append(f"    Sectors tracked: {stats.get('sectors_tracked', 0)}")

    # Sin stats
    lines.append("")
    lines.append("  CODE SINS:")
    lines.append(f"    Active: {stats.get('active_sins', 0)}")
    lines.append(f"    Resolved: {stats.get('resolved_sins', 0)}")
    lines.append(f"    Total recorded: {stats.get('total_sins_recorded', 0)}")

    lines.append("=" * 50)
    return "\n".join(lines)


def get_sins_report() -> str:
    """Get the hall of shame — recurring issues ranked by offense count."""
    sins = get_active_sins()
    summary = get_sin_summary()

    lines = []
    lines.append("=" * 70)
    lines.append("PITA-PM — HALL OF SHAME")
    lines.append("  (Recurring issues, ranked by how many times I've had to mention them)")
    lines.append("=" * 70)
    lines.append("")

    if not summary:
        lines.append("  The sin ledger is empty. Either you're perfect,")
        lines.append("  or I haven't scanned yet. I know which one it is.")
        return "\n".join(lines)

    lines.append(f"  {'Issue Type':<25} {'Instances':>10} {'Offenses':>10} {'Tier':<15}")
    lines.append(f"  {'-'*60}")

    for issue_type, info in sorted(summary.items(),
                                    key=lambda x: x[1]['total'],
                                    reverse=True):
        total = info['total']
        instances = info['instances']
        if total >= 7:
            tier = "EXISTENTIAL"
        elif total >= 3:
            tier = "POINTED"
        else:
            tier = "GENTLE"

        lines.append(f"  {issue_type:<25} {instances:>10} {total:>10} {tier:<15}")

        # Add a quip for top offenders
        if total >= 3:
            quip = generate_quip(issue_type, total)
            lines.append(f"    ^ {quip}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def end_shift() -> str:
    """End the current shift. Get philosophical."""
    state = _load_state()
    state['on_shift'] = False
    _save_state(state)

    vibes = state.get('scan_history_vibes', [])
    last_vibe = vibes[-1] if vibes else 50

    closings = [
        "My shift is over. Your code's problems are not. Think about that.",
        "Going off-shift. I'll be crawling financial news while you sleep. "
        "One of us is more productive at night. It's me.",
        f"End of shift. Vibe score: {last_vibe:.0f}. "
        f"{'I expected worse.' if last_vibe > 60 else 'I expected exactly this.'}",
        "Shift complete. I'll be back. The code will still be here. "
        "Some of it shouldn't be, but it will be.",
    ]

    import random
    return random.choice(closings)
