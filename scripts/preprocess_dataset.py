#!/usr/bin/env python
from __future__ import annotations
import json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from _common import parser
from hsc_tta.preprocessing import preprocess_sleep_recording, preprocess_mi_recordings
from hsc_tta.preprocessing.storage import write_subject_hdf5
from hsc_tta.utils import config_hash,load_yaml,require_cpu


def main() -> int:
    args=parser("Preprocess audited EEG per subject into HDF5").parse_args(); require_cpu(args.device)
    cfg=load_yaml(args.config); dataset=cfg["dataset"]; root=Path("/root/autodl-tmp/hsc_tta_eeg")
    manifest=pd.read_parquet(root/"data/manifests/recordings.parquet")
    frame=manifest[(manifest.dataset==dataset)&manifest.preprocessing_eligible]
    subjects=sorted(frame.subject_id.unique())[:args.limit_subjects]
    if args.dry_run: print(f"dataset={dataset} eligible_subjects={len(subjects)}"); return 0
    failures=[]; completed=[]
    for sid in subjects:
        if shutil.disk_usage(root).free/1024**3<60: raise RuntimeError("storage guard: less than 60GB free")
        group=frame[frame.subject_id==sid]
        try:
            if cfg["task"]=="sleep_staging":
                parts=[preprocess_sleep_recording(p,dataset,float(cfg["sampling_rate"]),tuple(cfg["bandpass_hz"]),float(cfg["epoch_seconds"])) for p in group.filepath]
                arrays={key:(np.concatenate([part[key] for part in parts]) if np.asarray(parts[0][key]).ndim>0 and key not in {"channel_names","channel_mask"} else parts[0][key]) for key in parts[0]}
            else:
                arrays=preprocess_mi_recordings(group.filepath.tolist(),float(cfg["sampling_rate"]),tuple(cfg["bandpass_hz"]),float(cfg["event_seconds"]))
            status=write_subject_hdf5(root/f"data/processed/{dataset}/{sid.replace(':','_')}.h5",arrays,{"dataset":dataset,"subject_id":sid},config_hash(cfg),args.resume)
            completed.append({"subject_id":sid,"status":status,"n_windows":int(len(arrays["label"]))})
        except Exception as exc: failures.append({"subject_id":sid,"error":f"{type(exc).__name__}: {exc}"})
    state={"dataset":dataset,"completed":completed,"failed":failures,"config_hash":config_hash(cfg)}
    path=root/f"state/preprocess_{dataset}.json"; path.write_text(json.dumps(state,indent=2),encoding="utf-8")
    print(json.dumps(state,indent=2)); return 1 if failures else 0
if __name__=="__main__":
    raise SystemExit(main())
