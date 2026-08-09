#!/usr/bin/env python
from __future__ import annotations

import argparse

from hsc_tta.v3.oracle_headroom import run_oracle_headroom

from _common import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); frame, summary, go = run_oracle_headroom(project_root(args.root), load_yaml(args.config), args.device, args.resume)
    print({"rows": len(frame), "summary_rows": len(summary), "oracle_gate": "GO" if go else "NO-GO"})


if __name__ == "__main__": main()
