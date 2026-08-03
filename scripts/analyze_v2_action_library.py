#!/usr/bin/env python
from pathlib import Path
from hsc_tta.v2.action_library import analyze_action_library

if __name__=="__main__": print(analyze_action_library(Path("/root/autodl-tmp/hsc_tta_eeg")))
