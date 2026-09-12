"""Fail-closed aggregate check for the completed stability experiment."""
from pathlib import Path
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "outputs"
    paths = ["FOLD_SEED_RESULTS.csv", "SUBJECT_SEED_RESULTS.csv", "DATASET_AGGREGATE_RESULTS.csv"]
    for name in paths:
        if not (root / name).is_file():
            raise SystemExit(f"missing {name}")
    frame = pd.read_csv(root / "FOLD_SEED_RESULTS.csv")
    if len(frame) != 30:
        raise SystemExit("primary table is incomplete")
    print(pd.read_csv(root / "DATASET_AGGREGATE_RESULTS.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
