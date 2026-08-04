#!/usr/bin/env python
from __future__ import annotations

import argparse

from hsc_tta.v3.action_search import run_action_search

from _common import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); root = project_root(args.root)
    detail, summary = run_action_search(root, load_yaml(args.config), args.device, args.resume)
    print({"subject_rows": len(detail), "summary_rows": len(summary)})


if __name__ == "__main__": main()

