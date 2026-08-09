#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from hsc_tta.v2.splits import generate_v2_splits


if __name__ == "__main__":
    print(json.dumps(generate_v2_splits(Path("/root/autodl-tmp/hsc_tta_eeg")), indent=2))
