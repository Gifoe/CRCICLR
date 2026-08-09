#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hsc_tta.v2.confirmation import file_sha256, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and dispatch a post-freeze confirmatory dataset adapter")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--freeze", default="/root/autodl-tmp/hsc_tta_eeg/outputs/v2_joint_certified/freeze/V2_METHOD_FREEZE.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    freeze = Path(args.freeze)
    if not freeze.exists():
        raise FileNotFoundError("method freeze must exist before any confirmatory run")
    manifest = validate_manifest(args.manifest, file_sha256(freeze))
    if args.dry_run:
        print(f"VALID: {manifest['dataset']} ({len(manifest['calibration_subjects'])} calibration, {len(manifest['test_subjects'])} test)")
        return
    raise RuntimeError("No dataset-specific adapter is registered. Implement ConfirmatoryDatasetAdapter and rerun --dry-run before opening labels.")


if __name__ == "__main__":
    main()
