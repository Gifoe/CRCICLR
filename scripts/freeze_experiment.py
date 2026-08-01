#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hsc_tta.freeze import create_freeze_manifest
from hsc_tta.utils import require_cpu


def main() -> int:
    require_cpu("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", action="append", required=True, help="NAME=/absolute/path")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    files: dict[str, str] = {}
    for item in args.file:
        name, separator, path = item.partition("=")
        if not separator or not name or not Path(path).is_absolute():
            raise ValueError("--file entries must be NAME=/absolute/path")
        files[name] = path
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = create_freeze_manifest(
        files,
        git_commit=commit,
        metadata=json.loads(args.metadata_json),
        output_path=args.output,
    )
    print(manifest["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
