#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hsc_tta.v2.token_embeddings import extract_all_token_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="hmc,cap,eegmmidb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    frame = extract_all_token_embeddings(Path("/root/autodl-tmp/hsc_tta_eeg"), datasets=args.datasets.split(","),
                                         device=args.device, batch_size=args.batch_size, resume=args.resume)
    print(frame.groupby(["dataset", "status"]).size())
