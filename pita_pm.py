"""
PITA-PM — Drop Score Integration

Thin wrapper that makes PITA-PM accessible from the project root.
Import this module or run it directly to interact with PITA-PM.

Usage:
    python pita_pm.py scan          # Full codebase scan
    python pita_pm.py standup       # Shift start + scan
    python pita_pm.py status        # Current status
    python pita_pm.py market        # Market intelligence crawl
    python pita_pm.py sins          # Hall of shame
    python pita_pm.py gate          # CI quality gate (exit 1 if bad)
    python pita_pm.py help          # All commands

Integration with Drop Score pipeline:
    from pita_pm import quality_gate, quick_scan, get_vibe_score

    # In CI: fail if code quality drops
    quality_gate(threshold=40)

    # In pipeline: log scan results
    report = quick_scan()
    print(f"Vibe: {report.vibe_score:.0f}/100")
"""
import sys
import os

# Add .pita-pm/src to path so imports work
_PITA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pita-pm', 'src')
if _PITA_DIR not in sys.path:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '.pita-pm'))

# Re-export key functions for easy integration
from src.scanner import scan_codebase, ScanReport, ScanFinding
from src.personality import vibe_check, generate_quip, format_standup
from src.shift_runner import run_scan, start_shift, get_status, get_sins_report
from src.knowledge_store import get_sin_summary, get_market_stats
from src.market_crawler import run_sleep_mode


def quick_scan(project_dir: str = ".") -> ScanReport:
    """Run a quick scan and return the report object (no formatting)."""
    return scan_codebase(root_dir=project_dir, persist_sins=False)


def get_vibe_score(project_dir: str = ".") -> float:
    """Get just the vibe score (0-100)."""
    report = scan_codebase(root_dir=project_dir, persist_sins=False)
    return report.vibe_score


def quality_gate(threshold: float = 40.0, project_dir: str = ".") -> bool:
    """
    Quality gate for CI integration.
    Returns True if code passes, False if it fails.
    Prints a report either way.
    """
    report = scan_codebase(root_dir=project_dir)
    vibe = report.vibe_score
    passed = vibe >= threshold

    print(f"PITA-PM Quality Gate: {'PASSED' if passed else 'FAILED'}")
    print(f"  Vibe: {vibe:.0f}/100 (threshold: {threshold:.0f})")
    print(f"  Issues: {report.total_issues} across {report.files_scanned} files")

    if not passed:
        # Show top 5 worst issues
        worst = sorted(report.findings, key=lambda f: f.severity, reverse=True)[:5]
        print("\n  Top issues:")
        for f in worst:
            print(f"    [{f.severity}] {f.file_path}:{f.line_number} — {f.description}")

    return passed


def pipeline_hook(stage: str = "unknown") -> None:
    """
    Hook to call from Drop Score pipeline stages.
    Logs a quick vibe check without blocking the pipeline.

    Usage in run_model.py:
        try:
            from pita_pm import pipeline_hook
            pipeline_hook("model")
        except ImportError:
            pass  # PITA-PM not installed, no worries
    """
    try:
        report = scan_codebase(root_dir=".", persist_sins=True)
        label, _ = vibe_check(report.vibe_score)
        print(f"\n  [PITA-PM] Stage '{stage}' vibe check: "
              f"{report.vibe_score:.0f}/100 ({label}) — "
              f"{report.total_issues} issues")
    except Exception:
        pass  # Never let the PM crash the pipeline. That would be too on-brand.


if __name__ == '__main__':
    # CLI mode — delegate to agent
    from src.agent import main
    sys.exit(main())
