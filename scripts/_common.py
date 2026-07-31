from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-subjects", type=int)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--device", default="cpu", choices=["cpu"])
    return p

