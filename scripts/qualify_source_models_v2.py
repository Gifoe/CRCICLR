#!/usr/bin/env python
from pathlib import Path
from hsc_tta.v2.source_models import qualify_source_models

if __name__ == "__main__":
    print(qualify_source_models(Path("/root/autodl-tmp/hsc_tta_eeg"),resume=True).to_string(index=False))
