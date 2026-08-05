from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"src"))

from hsc_tta.budgeted_risk.reporting import build_delivery_manifest


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--repo",default=str(REPO));args=parser.parse_args()
    payload=build_delivery_manifest(Path(args.repo).resolve())
    print(f"delivery files: {len(payload['files'])}")
    return 0


if __name__=="__main__":raise SystemExit(main())
