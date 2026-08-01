#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hsc_tta.gpu import TaskHeadTrainer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="hmc,eegmmidb")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    trainer = TaskHeadTrainer("/root/autodl-tmp/hsc_tta_eeg", args.device, args.batch_size)
    for dataset in args.datasets.split(","):
        for seed in map(int, args.seeds.split(",")):
            print(trainer.train(dataset, seed, resume=args.resume), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
