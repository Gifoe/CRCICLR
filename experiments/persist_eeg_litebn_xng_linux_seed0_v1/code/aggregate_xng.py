"""Final matched comparison is intentionally unavailable until LiteBN is run."""
from pathlib import Path


def main() -> int:
    status = Path(__file__).resolve().parents[1] / "outputs" / "XNG_FIRST_STATUS.json"
    if not status.is_file():
        raise SystemExit("XNG candidate run is not complete")
    print(status.read_text(encoding="utf-8"))
    print("Matched aggregation deferred: LiteBN baseline has not been run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
