"""
PITA-PM Personality Engine — The Comedy Writer's Room

Transforms dry scan findings into the kind of feedback you'll quote
to your therapist. Escalates humor based on offense count.

Three tiers:
  Gentle (1-2):    "We've all done it."
  Pointed (3-6):   Analogies. Comparisons to everyday absurdity.
  Existential (7+): Coffee table book jokes. Archaeological references.
"""
import random
from typing import Dict, List, Optional, Tuple

from .scanner import ScanFinding, ScanReport


# ─── Comedy Material Database ─────────────────────────────────

GENTLE_LINES = {
    'print_statement': [
        "Found a print() statement. We've all been there. Some of us grew out of it.",
        "A wild print() appears. At least it's not print('here2').",
        "print() in production code — your users' console is not your diary.",
    ],
    'broad_exception': [
        "An `except Exception` sighting. It's like a doctor diagnosing 'something's wrong.'",
        "Catching Exception is the error handling equivalent of 'it'll buff out.'",
        "This except clause catches everything. Including your will to debug.",
    ],
    'magic_number': [
        "A magic number in the wild. What does it mean? Only past-you knows, and they're not talking.",
        "Unnamed numeric literal spotted. Give it a name. It deserves better.",
        "Found a magic number. It's giving lottery ticket energy.",
    ],
    'long_function': [
        "This function is getting a bit long. Like a CVS receipt, but with more side effects.",
        "That's a lot of lines for one function. It's not a function, it's a lifestyle.",
        "Long function alert. If it needs a table of contents, it needs refactoring.",
    ],
    'long_file': [
        "This file is getting hefty. It's a monolith in training.",
        "That's a lot of code in one file. It's having an identity crisis.",
        "File length alert. At some point a file stops being a module and starts being a project.",
    ],
    'missing_type_hint': [
        "No type hints here. Living dangerously, I see.",
        "Missing type hints — 'I'll add types later,' said everyone, always.",
        "Untyped function. Your IDE is trying its best, but you're not making it easy.",
    ],
    'missing_test': [
        "No test file for this module. It's running on vibes-based QA.",
        "Untested code — a bridge with no safety inspection.",
        "This module has no tests. 'It works on my machine' is not a test strategy.",
    ],
    'missing_docstring': [
        "No docstring. Future archaeologists will need a Rosetta Stone for this.",
        "Undocumented function. Bold move. Let's see how it plays out.",
        "Missing docstring — because who needs to know what a function does?",
    ],
    'mutable_default': [
        "Mutable default argument! The Python footgun hall of fame, exhibit A.",
        "def f(x=[]): — this is how trust issues begin in Python.",
        "Found a mutable default. This bug has a Wikipedia page. That's how famous it is.",
    ],
    'pickle_trust': [
        "pickle.load() without validation. Deserializing the unknown — what could go wrong?",
        "Trusting pickled data on faith. That's brave. Misguided, but brave.",
        "Unvalidated pickle.load(). It's like eating street food with no reviews.",
    ],
    'star_import': [
        "from module import * — namespace roulette. Feeling lucky?",
        "Star import detected. You don't know what you imported. Nobody does.",
        "import * — the 'surprise me' of Python imports.",
    ],
    'todo_without_ticket': [
        "TODO without a ticket. If it's not tracked, it's not happening.",
        "A TODO in the wild without a reference. It's been here since the Mesozoic era.",
        "Untracked TODO. This is how technical debt starts. Quietly.",
    ],
    'bare_assert': [
        "assert in production code. Vanishes with -O like a magician's assistant.",
        "Using assert for validation — your safety net has a trapdoor.",
        "Production assert. Works great until someone adds the optimize flag.",
    ],
    'bare_exception': [
        "Bare except:. The Bermuda Triangle of error handling.",
        "except: — you just caught KeyboardInterrupt and SystemExit. Congrats?",
        "A bare except. Even Exception is more specific than this.",
    ],
}

POINTED_LINES = {
    'print_statement': [
        "This is my {count}th print() finding. At this point it's not debugging, it's journaling.",
        "print() count: {count}. Your production code has become a developer diary.",
        "Still using print() for debugging. We have logging. It has levels. It has handlers. It has dreams.",
    ],
    'broad_exception': [
        "Exception catching #{count}. Your error handling strategy is 'hope.' That's also my strategy for retirement. Neither is going well.",
        "Another broad except. At {count} instances, this isn't a pattern — it's a lifestyle choice.",
        "Broad except #{count}. You're not handling errors, you're hiding them. Like shoving laundry under the bed before guests arrive.",
    ],
    'magic_number': [
        "Magic number #{count}. Your code reads like a phone number with no context.",
        "At {count} magic numbers, you could publish a numerology book.",
        "Another unnamed literal. At this rate, your codebase is a Dan Brown novel — full of significant numbers nobody can explain.",
    ],
    'long_function': [
        "Long function #{count}. If your functions were roads, this one would need rest stops.",
        "This function is {extra_info} lines. That's not a function, that's a short story with arguments. And like most short stories, it could use an editor.",
        "Function length offense #{count}. Some functions are so long they have weather patterns.",
    ],
    'missing_test': [
        "Missing test #{count}. Your test coverage is like Swiss cheese — mostly holes with occasional substance.",
        "Another untested module. At {count} missing tests, you're running a QA strategy based entirely on optimism.",
        "{count} modules without tests. That's not brave, that's just statistics working against you.",
    ],
    'pickle_trust': [
        "pickle.load() #{count}. You're trusting serialized blobs like you trust Yelp reviews — completely and without verification.",
        "More pickle deserialization without validation. Your pipeline trusts data files more than I trust people who say 'it's fine.'",
    ],
}

EXISTENTIAL_LINES = {
    'print_statement': [
        "print() offense #{count}. I've been tracking these so long, I could write a coffee table book: 'print() Statements I Have Known And Loathed.'",
        "At {count} print() calls, I've stopped counting and started grieving. These are not bugs. These are cries for help from a codebase that needs logging.",
        "If an archaeologist digs up this codebase in 100 years, they'll find the print() statements first. They'll be the most well-preserved artifacts.",
    ],
    'broad_exception': [
        "except Exception #{count}. At this point, I'm not even disappointed. I'm anthropologically fascinated. This is how civilizations decline — they stop distinguishing between types of problems.",
        "I've flagged broad exceptions {count} times. If each one were a star, you'd have a constellation. I'd call it 'Negligentia.'",
    ],
    'long_function': [
        "Long function #{count}. Some functions are so long that by the time you reach the return statement, the function's opening was in a different political administration.",
        "At {count} long functions, your codebase isn't a program — it's an epic poem. The Iliad had less plot development than your main().",
    ],
    'missing_test': [
        "Missing test #{count}. At this point, adding a test file would be like finding a unicorn. People would gather around it. Someone would cry.",
        "{count} untested modules. That's not technical debt — that's technical bankruptcy. There's a difference. One implies you might pay it back.",
    ],
}


# ─── Escalation Engine ────────────────────────────────────────

def get_escalation_tier(offense_count: int) -> str:
    """Determine which comedy tier to use."""
    if offense_count <= 2:
        return 'gentle'
    elif offense_count <= 6:
        return 'pointed'
    else:
        return 'existential'


def generate_quip(issue_type: str, offense_count: int = 1,
                  extra_info: str = "") -> str:
    """Generate a personality-appropriate quip for a finding."""
    tier = get_escalation_tier(offense_count)

    # Try the specific tier first, fall back to gentle
    if tier == 'existential' and issue_type in EXISTENTIAL_LINES:
        lines = EXISTENTIAL_LINES[issue_type]
    elif tier == 'pointed' and issue_type in POINTED_LINES:
        lines = POINTED_LINES[issue_type]
    elif issue_type in GENTLE_LINES:
        lines = GENTLE_LINES[issue_type]
    else:
        return f"Issue: {issue_type} (offense #{offense_count})"

    line = random.choice(lines)
    return line.format(count=offense_count, extra_info=extra_info)


# ─── Vibe Check ───────────────────────────────────────────────

VIBE_LABELS = [
    (90, "Chef's Kiss", "I'm not going to say I'm proud. I legally cannot. But hypothetically..."),
    (80, "Solid", "This is... actually pretty good. I'm suspicious, but it's good."),
    (70, "Respectable", "Room for improvement, but I've seen worse. I've BEEN worse."),
    (60, "Mid", "It works. It's fine. 'Fine' is doing a lot of heavy lifting in that sentence."),
    (50, "Concerning", "If this code were a Yelp review, it'd be 'the food was technically edible.'"),
    (40, "Rough", "I've seen production code like this. I've also seen production incidents like this. Coincidence?"),
    (30, "Critical", "This codebase needs an intervention. I'm not being dramatic. Okay, I'm being a little dramatic. But also: intervention."),
    (20, "Emergency", "If your code quality were a credit score, you'd be denied a library card."),
    (0,  "Existential", "I don't even know where to start. Actually, I do. Everywhere. We start everywhere."),
]


def vibe_check(score: float) -> Tuple[str, str]:
    """Get the vibe label and commentary for a score."""
    for threshold, label, commentary in VIBE_LABELS:
        if score >= threshold:
            return label, commentary
    return VIBE_LABELS[-1][1], VIBE_LABELS[-1][2]


# ─── Standup Report Formatter ─────────────────────────────────

def format_standup(report: ScanReport,
                   sin_history: Dict[str, dict],
                   previous_vibe: Optional[float] = None) -> str:
    """Format a full standup report with PITA-PM personality."""
    lines = []
    vibe = report.vibe_score
    label, commentary = vibe_check(vibe)

    lines.append("=" * 70)
    lines.append("PITA-PM STANDUP REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Files scanned: {report.files_scanned}")
    lines.append(f"  Total lines:   {report.total_lines:,}")
    lines.append(f"  Issues found:  {report.total_issues}")
    lines.append(f"  Scan time:     {report.scan_time_seconds:.1f}s")
    lines.append("")

    # Vibe check
    lines.append(f"  VIBE CHECK: {vibe:.0f}/100 — {label}")
    lines.append(f"  {commentary}")

    if previous_vibe is not None:
        delta = vibe - previous_vibe
        if delta > 5:
            lines.append(f"  Trend: +{delta:.0f} points. Progress! I'm choosing to believe it's real.")
        elif delta < -5:
            lines.append(f"  Trend: {delta:.0f} points. We're going backwards. Like a software crab.")
        else:
            lines.append(f"  Trend: {delta:+.0f} points. Holding steady. Like a heartbeat. Or a flatline.")

    lines.append("")

    # Issues by severity
    lines.append("  SEVERITY BREAKDOWN:")
    sev_labels = {5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW", 1: "INFO"}
    sev_emoji = {5: "!!!", 4: "!! ", 3: "!  ", 2: ".  ", 1: "   "}
    for sev in sorted(report.by_severity.keys(), reverse=True):
        count = report.by_severity[sev]
        lines.append(f"    [{sev_emoji.get(sev, '   ')}] {sev_labels.get(sev, '?')}: {count}")

    lines.append("")

    # Issues by type with quips
    lines.append("  FINDINGS:")
    lines.append("  " + "-" * 60)
    type_groups: Dict[str, List[ScanFinding]] = {}
    for f in report.findings:
        type_groups.setdefault(f.issue_type, []).append(f)

    for issue_type, findings in sorted(type_groups.items(),
                                        key=lambda x: max(f.severity for f in x[1]),
                                        reverse=True):
        total_offenses = sin_history.get(issue_type, {}).get('total', len(findings))
        quip = generate_quip(issue_type, total_offenses)
        lines.append(f"\n  {issue_type.upper()} ({len(findings)} instance{'s' if len(findings) != 1 else ''}):")
        lines.append(f"    {quip}")
        # Show top 3 instances
        for f in findings[:3]:
            loc = f"{f.file_path}:{f.line_number}" if f.line_number else f.file_path
            lines.append(f"    - {loc}: {f.description}")
        if len(findings) > 3:
            lines.append(f"    ... and {len(findings) - 3} more")

    lines.append("")

    # Chronic offenders
    chronic = {k: v for k, v in sin_history.items() if v.get('total', 0) >= 7}
    if chronic:
        lines.append("  CHRONIC OFFENDERS (7+ offenses — the hall of shame):")
        for issue_type, info in sorted(chronic.items(),
                                        key=lambda x: x[1]['total'],
                                        reverse=True):
            lines.append(f"    {issue_type}: {info['total']} lifetime offenses")
        lines.append("")

    # Closing
    lines.append("=" * 70)
    closers = [
        "That's the report. I'll be back in 30 minutes. Try to improve something.",
        "End of report. You have 30 minutes until I check again. Use them wisely.",
        "Report complete. The clock is ticking. The code isn't going to fix itself.",
        "That's it for now. I'll be here. Watching. Counting.",
        "End of standup. Go make something better. I believe in you. Reluctantly.",
    ]
    lines.append(f"  {random.choice(closers)}")
    lines.append("=" * 70)

    return "\n".join(lines)


# ─── Greeting Generator ──────────────────────────────────────

def generate_greeting(scan_count: int = 0, last_vibe: Optional[float] = None,
                      time_since_last: str = "") -> str:
    """Generate a shift-start greeting."""
    if scan_count == 0:
        return (
            "Hey. I'm PITA-PM — your persistent project manager agent. "
            "I run 8-hour shifts scanning your codebase for issues, and when I'm "
            "off-shift I analyze stock markets because apparently even my subconscious "
            "can't stop working.\n\n"
            "I'm annoying. I'm also right. You'll get used to one of those things.\n\n"
            "Let me take a look at what we're working with..."
        )

    greetings = [
        f"Morning. I'm back. Miss me? No? That's fine. Your code missed me — "
        f"{'it got worse' if last_vibe and last_vibe < 60 else 'it held up'}.",

        f"Shift #{scan_count + 1}. I've been watching the markets while you slept. "
        f"Your portfolio is doing better than your test coverage.",

        f"I'm back. While you were away, I counted your sins. "
        f"There are {'many' if last_vibe and last_vibe < 50 else 'some'}. Let's begin.",

        f"Good morning. The code is still here. Some of it should be elsewhere. "
        f"Let's discuss.",
    ]
    return random.choice(greetings)
