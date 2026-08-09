#!/usr/bin/env python
from pathlib import Path
from hsc_tta.v2.development_surfaces import build_development_surfaces

if __name__=="__main__":
    f,o=build_development_surfaces(Path("/root/autodl-tmp/hsc_tta_eeg"),resume=True)
    print("features",len(f),"outcomes",len(o))
