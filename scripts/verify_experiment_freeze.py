#!/usr/bin/env python
from __future__ import annotations

import argparse

from hsc_tta.freeze import verify_freeze_manifest
from hsc_tta.utils import require_cpu


def main() -> int:
    require_cpu("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    args = parser.parse_args()
    payload = verify_freeze_manifest(args.manifest)
    print(payload["manifest_hash"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
