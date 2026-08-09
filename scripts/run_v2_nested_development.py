#!/usr/bin/env python
from pathlib import Path
from hsc_tta.v2.nested_evaluation import run_nested_development

if __name__=="__main__":
    d,m=run_nested_development(Path("/root/autodl-tmp/hsc_tta_eeg")); print("decisions",len(d),"metrics",len(m))
