"""
PITA-PM Agent — The Brain

CLI entry point and orchestrator for all PITA-PM operations.
Integrates with Drop Score's pipeline as a quality gate.

Usage:
    python -m .pita-pm.src.agent scan          # Full codebase scan
    python -m .pita-pm.src.agent standup       # Latest standup report
    python -m .pita-pm.src.agent status        # Current status
    python -m .pita-pm.src.agent market        # Run market crawl + report
    python -m .pita-pm.src.agent market stats  # Market intelligence stats
    python -m .pita-pm.src.agent sins          # Hall of shame
    python -m .pita-pm.src.agent shift start   # Start a new shift
    python -m .pita-pm.src.agent shift end     # End current shift
    python -m .pita-pm.src.agent help          # Show commands
"""
import sys
import os

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from .shift_runner import start_shift, run_scan, get_status, get_sins_report, end_shift
from .market_crawler import run_sleep_mode
from .knowledge_store import get_market_stats, get_latest_sector_scores


HELP_TEXT = """
======================================================================
PITA-PM — COMMAND REFERENCE
  (Everything I can do to you. For you. Mostly for you.)
======================================================================

  SCAN COMMANDS:
    scan              Full codebase scan. Brace yourself.
    standup           Same as scan, but with a greeting.
    sins              Hall of shame — recurring issues by offense count.

  SHIFT COMMANDS:
    shift start       Start a new 8-hour shift. I come in hot.
    shift end         End shift. I get philosophical.
    status            Am I on shift? What's my vibe? Do I have feelings?

  MARKET COMMANDS:
    market            Run market intelligence crawl (sleep mode).
    market stats      How much data I've accumulated while you weren't looking.
    market report     Show latest sector scores.

  INTEGRATION:
    gate              Run as a quality gate — returns exit code 1 if
                      vibe score drops below threshold (default: 40).
    gate --threshold 60   Custom threshold.

  META:
    help              This menu.
    version           Current version.

======================================================================
  "I'm not here to be liked. I'm here to be right."
  — PITA-PM
======================================================================
"""


def cmd_scan(args: list) -> int:
    """Run a codebase scan."""
    report = run_scan(PROJECT_ROOT)
    print(report)
    return 0


def cmd_standup(args: list) -> int:
    """Start shift and run a scan."""
    report = start_shift(PROJECT_ROOT)
    print(report)
    return 0


def cmd_status(args: list) -> int:
    """Show current status."""
    print(get_status())
    return 0


def cmd_sins(args: list) -> int:
    """Show the hall of shame."""
    print(get_sins_report())
    return 0


def cmd_shift(args: list) -> int:
    """Manage shifts."""
    if not args:
        print("Usage: shift start | shift end")
        return 1
    if args[0] == 'start':
        print(start_shift(PROJECT_ROOT))
    elif args[0] == 'end':
        print(end_shift())
    else:
        print(f"Unknown shift command: {args[0]}")
        return 1
    return 0


def cmd_market(args: list) -> int:
    """Market intelligence commands."""
    if args and args[0] == 'stats':
        stats = get_market_stats()
        print("=" * 50)
        print("PITA-PM MARKET INTELLIGENCE STATS")
        print("=" * 50)
        for key, val in stats.items():
            label = key.replace('_', ' ').title()
            print(f"  {label}: {val}")
        print("=" * 50)
        return 0

    if args and args[0] == 'report':
        scores = get_latest_sector_scores()
        if not scores:
            print("No market data yet. Run 'market' first.")
            return 1
        print("=" * 50)
        print("LATEST SECTOR SCORES")
        print("=" * 50)
        for sector, info in sorted(scores.items(),
                                    key=lambda x: x[1].get('score', 0),
                                    reverse=True):
            print(f"  {sector:<15} {info.get('score', 0):>+6.3f}  "
                  f"({info.get('article_count', 0)} articles)")
        print("=" * 50)
        return 0

    # Full market crawl
    print("Starting market intelligence crawl...")
    print("(This takes 30-60 seconds — rate-limiting is being a good citizen.)")
    print()
    report = run_sleep_mode()
    print(report)
    return 0


def cmd_gate(args: list) -> int:
    """
    Quality gate — returns exit code 1 if vibe score is below threshold.
    Useful in CI pipelines.
    """
    threshold = 40.0
    if args and args[0] == '--threshold' and len(args) > 1:
        try:
            threshold = float(args[1])
        except ValueError:
            print(f"Invalid threshold: {args[1]}")
            return 1

    from .scanner import scan_codebase
    report = scan_codebase(root_dir=PROJECT_ROOT)
    vibe = report.vibe_score

    print(f"PITA-PM Quality Gate")
    print(f"  Vibe score: {vibe:.0f}/100")
    print(f"  Threshold:  {threshold:.0f}/100")
    print(f"  Issues:     {report.total_issues}")

    if vibe < threshold:
        print(f"\n  GATE: FAILED")
        print(f"  Your code didn't pass the vibe check.")
        print(f"  Fix {report.total_issues} issues or lower your standards.")
        return 1
    else:
        print(f"\n  GATE: PASSED")
        print(f"  Your code passed. I'm as surprised as you are.")
        return 0


def cmd_help(args: list) -> int:
    """Show help."""
    print(HELP_TEXT)
    return 0


def cmd_version(args: list) -> int:
    """Show version."""
    from . import __version__
    print(f"PITA-PM v{__version__}")
    return 0


COMMANDS = {
    'scan': cmd_scan,
    'standup': cmd_standup,
    'status': cmd_status,
    'sins': cmd_sins,
    'shift': cmd_shift,
    'market': cmd_market,
    'gate': cmd_gate,
    'help': cmd_help,
    'version': cmd_version,
}


def main(argv: list = None) -> int:
    """Main entry point."""
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        return cmd_help([])

    command = argv[0].lower()
    args = argv[1:]

    if command in COMMANDS:
        return COMMANDS[command](args)
    else:
        print(f"Unknown command: {command}")
        print("Run 'help' for available commands.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
