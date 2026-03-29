"""
DROP SCORE v18.3 — Fallback runner: runs all 3 stages sequentially.
Use this for local testing. In CI, the stages run as separate jobs.
"""
import subprocess, sys, os

# ── Install dependencies if needed ──
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["simfin", "yfinance", "xgboost", "lightgbm", "scikit-learn",
            "matplotlib", "tqdm", "optuna", "scipy"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        install(pkg)

# ── Run stages sequentially ──
stages = [
    ('run_data.py', 'STAGE 1: DATA PIPELINE'),
    ('run_validate.py', 'STAGE 2: VALIDATE (Universe A)'),
    ('run_model.py', 'STAGE 3: MODEL (Universe B)'),
]

for script, label in stages:
    print(f"\n{'='*70}")
    print(f"Running {label}...")
    print(f"{'='*70}\n")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"\n  {label} failed with exit code {result.returncode}")
        # Copy results log even on failure
        _copy_latest_log()
        sys.exit(result.returncode)

# ── Copy latest results to results/latest.txt ──
def _copy_latest_log():
    """Combine all stage logs into results/latest.txt."""
    os.makedirs('results', exist_ok=True)
    logs = sorted(
        [f for f in os.listdir('results') if f.endswith('.txt') and f != 'latest.txt'],
        key=lambda f: os.path.getmtime(os.path.join('results', f)),
    )
    if logs:
        with open('results/latest.txt', 'w') as out:
            for log in logs[-3:]:  # last 3 stage logs
                path = os.path.join('results', log)
                with open(path) as f:
                    out.write(f"{'='*70}\n{log}\n{'='*70}\n")
                    out.write(f.read())
                    out.write("\n\n")
        print(f"\nCombined results saved to results/latest.txt")

_copy_latest_log()
print("\nAll stages complete.")
