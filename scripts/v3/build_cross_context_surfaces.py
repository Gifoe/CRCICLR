#!/usr/bin/env python
from __future__ import annotations

import argparse

from hsc_tta.v3.cross_context_surfaces import build_cross_context_surfaces

from _common import load_yaml, project_root


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="."); parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); diagnostics, outcomes = build_cross_context_surfaces(
        project_root(args.root), load_yaml(args.config), args.device, args.resume)
    print({"probe_rows": len(diagnostics), "future_rows": len(outcomes)})


if __name__ == "__main__": main()
