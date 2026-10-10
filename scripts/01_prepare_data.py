"""Step 1 - convert the raw public downloads into analysis tables."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from hfwnpo_aml.datasets import BUILDERS  # noqa: E402


def main(only: list[str] | None = None) -> None:
    print(f"Raw data folder : {config.RAW_DIR}")
    print(f"Output folder   : {config.PROCESSED_DIR}\n")
    for name, build in BUILDERS.items():
        if only and name not in only:
            continue
        try:
            build()
        except Exception as exc:  # keep going with the other cohorts
            print(f"  [error] {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
