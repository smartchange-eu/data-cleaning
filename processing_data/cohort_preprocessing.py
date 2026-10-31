# run all cohort preprocessing scripts
# input data and generated results are not included in this repository

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "acd.py",
    # abcd is processed separately through a virtual machine due to restricted data access
    # "abcd.py",
    "ask.py",
    "belgian1A.py",
    "belgian1B.py",
    "belgian234.py",
    "belgian5.py",
    "belgian6.py",
    "dados.py",
    "gnn-svk.py",
    "helena.py",
    "icad.py",
    "mj_tmu.py",
    "slofit_lifelong.py",
    "slofit_students.py",
    "tmu.py",
    "uk-biobank.py",
    "uporto.py",
    "yfs.py",
]

def main():
    failed = []

    for script in SCRIPTS:
        script_path = BASE_DIR / script
        print(f"\n=== running {script} ===", flush=True)

        if not script_path.is_file():
            print(f"missing script: {script_path}")
            failed.append(script)
            continue

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            check=False,
        )

        if result.returncode != 0:
            print(f"failed: {script} (exit code {result.returncode})")
            failed.append(script)

    if failed:
        print(f"\nfinished with {len(failed)} failed or missing script(s):")
        for script in failed:
            print(f"- {script}")
        return 1

    print("\nall preprocessing scripts completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())