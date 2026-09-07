#!/usr/bin/env python
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from _common import parser
from hsc_tta.utils import require_cpu


def main() -> int:
    args = parser("Audit the CPU-only runtime").parse_args()
    require_cpu(args.device)
    root = Path("/root/autodl-tmp/hsc_tta_eeg")
    usage = shutil.disk_usage(root)
    report = {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "data_root": str(root), "free_gb": usage.free / 1024**3, "writable": os.access(root, os.W_OK)}
    print(json.dumps(report, indent=2))
    return 0 if report["writable"] and report["free_gb"] >= 50 else 2


if __name__ == "__main__": raise SystemExit(main())

