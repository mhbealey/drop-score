"""
PITA-PM Code Scanner — The Code Archaeologist (Python Edition)

Scans the codebase for issues that future-you will regret.
Each finding becomes a sin in the ledger, escalating with repeat offenses.
"""
import ast
import os
import re
import tokenize
import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .knowledge_store import record_sin, resolve_sin, get_active_sins


@dataclass
class ScanFinding:
    """A single code issue found during scanning."""
    issue_type: str
    file_path: str
    line_number: int
    description: str
    severity: int  # 1-5
    code_snippet: str = ""


@dataclass
class ScanReport:
    """Complete scan results."""
    findings: List[ScanFinding] = field(default_factory=list)
    files_scanned: int = 0
    total_lines: int = 0
    scan_time_seconds: float = 0.0

    @property
    def total_issues(self) -> int:
        return len(self.findings)

    @property
    def by_severity(self) -> Dict[int, int]:
        counts: Dict[int, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return counts

    @property
    def by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for f in self.findings:
            counts[f.issue_type] = counts.get(f.issue_type, 0) + 1
        return counts

    @property
    def vibe_score(self) -> float:
        """0-100 vibe score. 100 = chef's kiss. 0 = dumpster fire."""
        if self.total_lines == 0:
            return 50.0  # benefit of the doubt for empty projects
        # Weight by severity: sev 5 = -5 pts, sev 1 = -0.5 pts
        penalty = sum(f.severity * 0.8 for f in self.findings)
        # Normalize: penalty per 100 lines, capped
        normalized = min(penalty / (self.total_lines / 100), 100)
        return max(0, 100 - normalized)


# ─── Ignore Patterns ──────────────────────────────────────────

DEFAULT_IGNORE = {
    '__pycache__', '.git', '.pita-pm', 'node_modules', '.tox',
    '.mypy_cache', '.pytest_cache', 'venv', '.venv', 'env',
    'htmlcov', '.eggs', '*.egg-info',
}

DEFAULT_IGNORE_FILES = {
    'setup.py', 'conftest.py',
}


def _should_ignore(path: str, ignore_dirs: Set[str]) -> bool:
    """Check if path should be ignored."""
    parts = path.split(os.sep)
    return any(p in ignore_dirs for p in parts)


# ─── Individual Scanners ──────────────────────────────────────

def _scan_print_statements(source: str, filepath: str) -> List[ScanFinding]:
    """Find print() calls in production code. Test files get a pass."""
    findings = []
    if 'test_' in os.path.basename(filepath):
        return findings

    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # Skip comments
        if stripped.startswith('#'):
            continue
        # Match print( but not # print( or """...print(
        if re.search(r'\bprint\s*\(', stripped):
            # Allow print in if __name__ == '__main__' crash handlers
            # (we'll flag them but at lower severity)
            in_crash_handler = 'CRASH REPORT' in source[max(0, source.find(stripped)-500):source.find(stripped)]
            findings.append(ScanFinding(
                issue_type='print_statement',
                file_path=filepath,
                line_number=i,
                description=f"print() in production code{' (crash handler)' if in_crash_handler else ''}",
                severity=1 if in_crash_handler else 2,
                code_snippet=stripped[:80],
            ))
    return findings


def _scan_broad_exceptions(source: str, filepath: str) -> List[ScanFinding]:
    """Find overly broad except clauses."""
    findings = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        # except Exception: or except Exception as e:
        if re.match(r'except\s+Exception\s*(as\s+\w+)?\s*:', stripped):
            findings.append(ScanFinding(
                issue_type='broad_exception',
                file_path=filepath,
                line_number=i,
                description="Broad 'except Exception' — catch specific exceptions",
                severity=3,
                code_snippet=stripped[:80],
            ))
        # bare except:
        elif re.match(r'except\s*:', stripped):
            findings.append(ScanFinding(
                issue_type='bare_exception',
                file_path=filepath,
                line_number=i,
                description="Bare 'except:' — the Bermuda Triangle of error handling",
                severity=4,
                code_snippet=stripped[:80],
            ))
    return findings


def _scan_magic_numbers(source: str, filepath: str) -> List[ScanFinding]:
    """Find magic numbers in logic (not config, not obvious 0/1/-1/100)."""
    findings = []
    boring_numbers = {0, 1, -1, 2, 100, 0.0, 1.0, 0.5, -1.0}

    # Skip config files — they're SUPPOSED to have numbers
    if 'config' in os.path.basename(filepath).lower():
        return findings

    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Find numeric literals in comparisons and assignments
        for match in re.finditer(r'(?<!=\s)(?<![<>!=])(\b\d+\.?\d*)\b', stripped):
            try:
                num = float(match.group(1))
                if num in boring_numbers:
                    continue
                # Skip line numbers, string formatting, range() calls, etc.
                if 'range(' in stripped or 'enumerate(' in stripped:
                    continue
                if 'log.info' in stripped or 'print(' in stripped:
                    continue
                # Only flag if it looks like logic (comparison, threshold, multiplier)
                context = stripped[max(0, match.start()-10):match.end()+10]
                logic_indicators = ['>', '<', '>=', '<=', '==', '!=', '*', '/',
                                    'if ', 'elif ', 'while ', 'threshold', 'cutoff']
                if any(ind in context or ind in stripped for ind in logic_indicators):
                    findings.append(ScanFinding(
                        issue_type='magic_number',
                        file_path=filepath,
                        line_number=i,
                        description=f"Magic number {match.group(1)} — give it a name",
                        severity=1,
                        code_snippet=stripped[:80],
                    ))
                    break  # one per line is enough
            except ValueError:
                continue
    return findings


def _scan_function_length(source: str, filepath: str,
                          max_lines: int = 50) -> List[ScanFinding]:
    """Find functions that have become novellas."""
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Calculate function length
            if hasattr(node, 'end_lineno') and node.end_lineno:
                length = node.end_lineno - node.lineno + 1
            else:
                # Fallback: estimate from body
                length = max((getattr(n, 'lineno', 0)
                              for n in ast.walk(node)), default=node.lineno) - node.lineno + 1

            if length > max_lines:
                sev = 2 if length < 100 else 3 if length < 200 else 4 if length < 300 else 5
                findings.append(ScanFinding(
                    issue_type='long_function',
                    file_path=filepath,
                    line_number=node.lineno,
                    description=(f"{node.name}() is {length} lines "
                                 f"({'a novella' if length > 200 else 'a short story' if length > 100 else 'a CVS receipt'})"),
                    severity=sev,
                    code_snippet=f"def {node.name}(...)  # {length} lines",
                ))
    return findings


def _scan_missing_type_hints(source: str, filepath: str) -> List[ScanFinding]:
    """Find functions missing type hints."""
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip dunder methods and private helpers
            if node.name.startswith('__') and node.name.endswith('__'):
                continue
            # Check return annotation
            missing_return = node.returns is None
            # Check argument annotations (skip self/cls)
            args = node.args
            all_args = args.args + args.posonlyargs + args.kwonlyargs
            unannotated = [a for a in all_args
                           if a.annotation is None and a.arg not in ('self', 'cls')]
            if missing_return or unannotated:
                missing_parts = []
                if missing_return:
                    missing_parts.append("return type")
                if unannotated:
                    missing_parts.append(f"{len(unannotated)} param(s)")
                findings.append(ScanFinding(
                    issue_type='missing_type_hint',
                    file_path=filepath,
                    line_number=node.lineno,
                    description=f"{node.name}() missing {', '.join(missing_parts)}",
                    severity=1,
                    code_snippet=f"def {node.name}(...)  # needs type hints",
                ))
    return findings


def _scan_missing_docstrings(source: str, filepath: str) -> List[ScanFinding]:
    """Find public functions without docstrings."""
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip private/dunder
            if node.name.startswith('_'):
                continue
            docstring = ast.get_docstring(node)
            if not docstring:
                findings.append(ScanFinding(
                    issue_type='missing_docstring',
                    file_path=filepath,
                    line_number=node.lineno,
                    description=f"{node.name}() — no docstring. Future archaeologists weep.",
                    severity=1,
                    code_snippet=f"def {node.name}(...)  # undocumented",
                ))
    return findings


def _scan_mutable_defaults(source: str, filepath: str) -> List[ScanFinding]:
    """Find the classic Python footgun: mutable default arguments."""
    findings = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return findings

    mutable_types = (ast.List, ast.Dict, ast.Set)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None and isinstance(default, mutable_types):
                    findings.append(ScanFinding(
                        issue_type='mutable_default',
                        file_path=filepath,
                        line_number=node.lineno,
                        description=(f"{node.name}() has mutable default argument — "
                                     "Python footgun hall of fame, exhibit A"),
                        severity=5,
                        code_snippet=f"def {node.name}(...)  # mutable default!",
                    ))
    return findings


def _scan_pickle_trust(source: str, filepath: str) -> List[ScanFinding]:
    """Find pickle.load() without validation — deserializing the unknown."""
    findings = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if 'pickle.load' in stripped and 'validate' not in stripped.lower():
            findings.append(ScanFinding(
                issue_type='pickle_trust',
                file_path=filepath,
                line_number=i,
                description="pickle.load() — trusting serialized data on faith",
                severity=4,
                code_snippet=stripped[:80],
            ))
    return findings


def _scan_star_imports(source: str, filepath: str) -> List[ScanFinding]:
    """Find 'from x import *' — namespace roulette."""
    findings = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if re.match(r'from\s+\S+\s+import\s+\*', stripped):
            findings.append(ScanFinding(
                issue_type='star_import',
                file_path=filepath,
                line_number=i,
                description="Star import — namespace roulette",
                severity=3,
                code_snippet=stripped[:80],
            ))
    return findings


def _scan_todos(source: str, filepath: str) -> List[ScanFinding]:
    """Find TODOs and FIXMEs without ticket references."""
    findings = []
    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        for marker in ['TODO', 'FIXME', 'HACK', 'XXX']:
            if marker in stripped.upper():
                # Check for ticket reference (common patterns: #123, JIRA-123, etc.)
                has_ticket = bool(re.search(r'#\d+|[A-Z]+-\d+', stripped))
                if not has_ticket:
                    findings.append(ScanFinding(
                        issue_type='todo_without_ticket',
                        file_path=filepath,
                        line_number=i,
                        description=f"{marker} without ticket ref — if it's not tracked, it's not happening",
                        severity=2,
                        code_snippet=stripped[:80],
                    ))
                break
    return findings


def _scan_bare_asserts(source: str, filepath: str) -> List[ScanFinding]:
    """Find assert statements in production code."""
    findings = []
    if 'test_' in os.path.basename(filepath):
        return findings  # asserts in tests are fine

    for i, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith('assert ') or stripped == 'assert':
            findings.append(ScanFinding(
                issue_type='bare_assert',
                file_path=filepath,
                line_number=i,
                description="assert in production — vanishes with python -O. Your safety net has holes.",
                severity=3,
                code_snippet=stripped[:80],
            ))
    return findings


# ─── File-Level Checks ────────────────────────────────────────

def _scan_file_length(filepath: str, line_count: int,
                      max_lines: int = 300) -> Optional[ScanFinding]:
    """Flag files that have become monoliths."""
    if line_count > max_lines:
        sev = 2 if line_count < 500 else 3 if line_count < 700 else 4
        return ScanFinding(
            issue_type='long_file',
            file_path=filepath,
            line_number=1,
            description=f"{line_count} lines — it's a monolith having an identity crisis",
            severity=sev,
        )
    return None


def _scan_missing_tests(py_files: List[str]) -> List[ScanFinding]:
    """Find source files with no corresponding test file."""
    findings = []
    test_files = {os.path.basename(f) for f in py_files if 'test_' in os.path.basename(f)}
    source_files = [f for f in py_files
                    if not os.path.basename(f).startswith('test_')
                    and os.path.basename(f) != '__init__.py'
                    and os.path.basename(f) != 'setup.py'
                    and '.pita-pm' not in f]

    for src in source_files:
        base = os.path.basename(src)
        expected_test = f"test_{base}"
        if expected_test not in test_files:
            findings.append(ScanFinding(
                issue_type='missing_test',
                file_path=src,
                line_number=0,
                description=f"No test file (expected {expected_test}) — vibes-based QA",
                severity=5,
            ))
    return findings


# ─── Main Scanner ─────────────────────────────────────────────

def scan_codebase(root_dir: str = ".",
                  max_function_lines: int = 50,
                  max_file_lines: int = 300,
                  persist_sins: bool = True) -> ScanReport:
    """
    Run a full codebase scan.

    Returns a ScanReport with all findings, plus persists sins
    to the knowledge store for escalation tracking.
    """
    import time
    start = time.time()
    report = ScanReport()
    py_files = []

    # Collect all Python files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Filter out ignored directories
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE]
        if _should_ignore(dirpath, DEFAULT_IGNORE):
            continue
        for fname in filenames:
            if fname.endswith('.py') and fname not in DEFAULT_IGNORE_FILES:
                py_files.append(os.path.join(dirpath, fname))

    # Scan each file
    for filepath in sorted(py_files):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                source = f.read()
        except (OSError, IOError):
            continue

        lines = source.splitlines()
        line_count = len(lines)
        report.files_scanned += 1
        report.total_lines += line_count

        # Run all scanners
        report.findings.extend(_scan_print_statements(source, filepath))
        report.findings.extend(_scan_broad_exceptions(source, filepath))
        report.findings.extend(_scan_magic_numbers(source, filepath))
        report.findings.extend(_scan_function_length(source, filepath, max_function_lines))
        report.findings.extend(_scan_missing_type_hints(source, filepath))
        report.findings.extend(_scan_missing_docstrings(source, filepath))
        report.findings.extend(_scan_mutable_defaults(source, filepath))
        report.findings.extend(_scan_pickle_trust(source, filepath))
        report.findings.extend(_scan_star_imports(source, filepath))
        report.findings.extend(_scan_todos(source, filepath))
        report.findings.extend(_scan_bare_asserts(source, filepath))

        # File-level checks
        file_length = _scan_file_length(filepath, line_count, max_file_lines)
        if file_length:
            report.findings.append(file_length)

    # Cross-file checks
    report.findings.extend(_scan_missing_tests(py_files))

    report.scan_time_seconds = time.time() - start

    # Persist sins to knowledge store
    if persist_sins:
        for finding in report.findings:
            record_sin(
                issue_type=finding.issue_type,
                file_path=finding.file_path,
                line_number=finding.line_number,
                description=finding.description,
                severity=finding.severity,
            )

    return report
