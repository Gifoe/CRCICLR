#!/usr/bin/env python
from pathlib import Path
from hsc_tta.v2.baselines import run_external_baselines

if __name__=="__main__":
    a,b=run_external_baselines(Path("/root/autodl-tmp/hsc_tta_eeg")); print(len(a),len(b))
