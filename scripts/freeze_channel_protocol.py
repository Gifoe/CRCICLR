#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from hsc_tta.protocols import choose_common_central_channel, scan_sleep_channel_availability
from hsc_tta.utils import require_cpu


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")
REPO = ROOT / "repo"


def main() -> int:
    require_cpu("cpu")
    availability = scan_sleep_channel_availability(ROOT / "data/processed")
    protocol = choose_common_central_channel(availability)
    (REPO / "CHANNEL_PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for channel in ("C3", "C4"):
        counts = protocol["counts"][channel]
        rows.append(
            f"| {channel} | {counts['hmc_subjects']} | {counts['cap_subjects']} | "
            f"{counts['combined_subjects']} | {counts['minimum_site_subjects']} |"
        )
    report = "\n".join(
        [
            "# HMC→CAP Channel Protocol Report",
            "",
            "This report uses channel-availability metadata only. It does not read labels, model outputs, or final-test outcomes.",
            "",
            "| central channel | HMC retained | CAP retained | combined | minimum site |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            f"Selected channel: **{protocol['selected_channel']}** (`{protocol['normalized_derivation']}`).",
            "",
            "Selection rule: maximum combined availability across HMC and CAP; ties prefer C4, then C3. "
            "The selected single-channel protocol is frozen for both HMC internal and HMC→CAP experiments.",
            "",
            f"Protocol SHA256: `{protocol['protocol_hash']}`",
            "",
        ]
    )
    (REPO / "CHANNEL_PROTOCOL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"selected_channel": protocol["selected_channel"], "hash": protocol["protocol_hash"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
