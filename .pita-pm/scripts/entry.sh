#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# PITA-PM Entry Point
# Usage: .pita-pm/scripts/entry.sh <command> [args...]
#
# Commands: scan, standup, status, market, sins, gate, help
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Run the PITA-PM agent
python -c "
import sys
sys.path.insert(0, '.')
from importlib import import_module
agent = import_module('.pita-pm.src.agent', package=None)
" 2>/dev/null || {
    # Fallback: direct module execution
    PYTHONPATH="$PROJECT_ROOT" python -m pita_pm.src.agent "$@"
    exit $?
}

# Use the agent module
PYTHONPATH="$PROJECT_ROOT" python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
# Import as a package path
import importlib.util
spec = importlib.util.spec_from_file_location(
    'pita_pm_agent',
    '$PROJECT_ROOT/.pita-pm/src/agent.py',
    submodule_search_locations=['$PROJECT_ROOT/.pita-pm/src']
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
sys.exit(mod.main(sys.argv[1:]))
" "$@"
