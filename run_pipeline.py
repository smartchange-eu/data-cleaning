# run cohort preprocessing followed by data harmonization

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

STAGES = [
    ("cohort preprocessing", "cohort_preprocessing.py"),
    ("data harmonization and final cleaning", "data_harmonization.py"),
]

def main():
    for label, filename in STAGES:
        script = BASE_DIR / filename
        if not script.is_file():
            print(f"missing {label} script: {script}", flush=True)
            return 1

        print(f"\n=== starting {label} ===", flush=True)
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=BASE_DIR,
            check=False,
        )
        if result.returncode != 0:
            print(f"\n{label} failed (exit code {result.returncode}); pipeline stopped.", flush=True)
            return result.returncode

        print(f"=== completed {label} ===", flush=True)

    print("\nall pipeline stages completed successfully", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
