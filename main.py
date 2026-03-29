"""
DROP SCORE v18.3 — Fallback runner: runs all 3 stages sequentially.
Use this for local testing. In CI, the stages run as separate jobs.
"""
import subprocess, sys, os


def _install(pkg):
    """Install a missing package via pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _copy_latest_log():
    """Combine all stage logs into results/latest.txt."""
    os.makedirs('results', exist_ok=True)
    logs = sorted(
        [f for f in os.listdir('results') if f.endswith('.txt') and f != 'latest.txt'],
        key=lambda f: os.path.getmtime(os.path.join('results', f)),
    )
    if logs:
        with open('results/latest.txt', 'w') as out:
            for log in logs[-3:]:
                path = os.path.join('results', log)
                with open(path) as f:
                    out.write(f"{'='*70}\n{log}\n{'='*70}\n")
                    out.write(f.read())
                    out.write("\n\n")
        print(f"\nCombined results saved to results/latest.txt")


def main():
    """Run all 3 pipeline stages sequentially."""
    for pkg in ["simfin", "yfinance", "xgboost", "lightgbm", "scikit-learn",
                "matplotlib", "tqdm", "optuna", "scipy"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            _install(pkg)

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
            _copy_latest_log()
            sys.exit(result.returncode)

    _copy_latest_log()
    print("\nAll stages complete.")


if __name__ == '__main__':
    main()
