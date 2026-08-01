#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from hsc_tta.backbones import CBraModInputAdapter, FrozenCBraMod
from hsc_tta.gpu import extract_all_embeddings


ROOT = Path("/root/autodl-tmp/hsc_tta_eeg")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="hmc,cap,eegmmidb")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    backbone = FrozenCBraMod(ROOT / "external" / "CBraMod",
                            ROOT / "checkpoints" / "cbramod" / "pretrained_weights.pth").to(args.device).eval()
    manifest = extract_all_embeddings(ROOT, backbone, CBraModInputAdapter(),
        "0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178",
        "0ff6be918985689e7df679bc731ffb70e6c6224f", datasets=args.datasets.split(","),
        device=args.device, batch_size=args.batch_size, resume=args.resume)
    print(manifest.groupby(["dataset", "status"]).size())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
