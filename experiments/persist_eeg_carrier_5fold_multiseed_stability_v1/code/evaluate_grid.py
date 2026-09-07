"""Evaluation entry point for the completed stability grid.

The full grid is evaluated inside train_grid.py only after both architectures
finish each dataset/fold/seed cell. This wrapper is intentionally fail-closed
and reruns no training.
"""
from pathlib import Path
import pandas as pd


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "outputs"
    path = root / "FOLD_SEED_RESULTS.csv"
    if not path.is_file():
        raise SystemExit("FOLD_SEED_RESULTS.csv is not present; run train_grid.py")
    frame = pd.read_csv(path)
    expected = 2 * 5 * 3
    if len(frame) != expected:
        raise SystemExit(f"expected {expected} fold-seed rows, found {len(frame)}")
    print(f"validated {len(frame)} primary fold-seed rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
