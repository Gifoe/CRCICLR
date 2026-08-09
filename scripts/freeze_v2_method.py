#!/usr/bin/env python
from pathlib import Path

from hsc_tta.v2.method_freeze import freeze_v2_method


if __name__ == "__main__":
    path = freeze_v2_method(Path("/root/autodl-tmp/hsc_tta_eeg"))
    print(path)
