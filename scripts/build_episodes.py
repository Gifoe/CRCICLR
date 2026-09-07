#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import h5py,pandas as pd
from _common import parser
from hsc_tta.episodes import build_sleep_episode,build_mi_episode
from hsc_tta.utils import load_yaml,require_cpu


def main() -> int:
    args=parser("Build U_s/V_s episodes from processed caches").parse_args(); require_cpu(args.device)
    cfg=load_yaml(args.config); dataset=cfg["dataset"]; root=Path("/root/autodl-tmp/hsc_tta_eeg")
    split_path=root/f"data/splits/{dataset}/seed_{args.seed}.json"
    split=json.loads(split_path.read_text(encoding="utf-8"))["roles"]
    roles={sid:role for role,ids in split.items() for sid in ids}; rows=[]
    caches=sorted((root/f"data/processed/{dataset}").glob("*.h5"))[:args.limit_subjects]
    if args.dry_run: print(f"dataset={dataset} caches={len(caches)}"); return 0
    for cache in caches:
        with h5py.File(cache,"r") as h:
            meta=json.loads(h.attrs["metadata_json"]); sid=meta["subject_id"]
            if cfg["task"]=="sleep_staging": episode=build_sleep_episode(h["window_start"][:],pd.notna(h["label"][:]),int(cfg["context_minutes"]),int(cfg["minimum_future_epochs"]))
            else: episode=build_mi_episode(h["run_id"][:],tuple(cfg["context_runs"]),tuple(cfg["future_runs"]))
            episode.update({"dataset":dataset,"seed":args.seed,"split_role":roles.get(sid,"unassigned"),"subject_id":sid,"episode_id":f"{dataset}:{args.seed}:{sid}"})
            rows.append(episode)
    out=root/f"data/episodes/{dataset}/seed_{args.seed}.parquet"; out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_parquet(out,index=False)
    print(f"episodes={len(rows)} path={out}"); return 0
if __name__=="__main__": raise SystemExit(main())

