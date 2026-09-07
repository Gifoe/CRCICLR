#!/usr/bin/env python
from __future__ import annotations
import ctypes, gc, json, os, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
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
    state_path=root/f"state/preprocess_{dataset}.json"

    def save_state() -> None:
        state={"dataset":dataset,"completed":sorted(completed,key=lambda x:x["subject_id"]),"failed":sorted(failures,key=lambda x:x["subject_id"]),"config_hash":config_hash(cfg),"updated_at":datetime.now(timezone.utc).isoformat()}
        temporary=state_path.with_suffix(state_path.suffix+".part")
        temporary.write_text(json.dumps(state,indent=2),encoding="utf-8")
        os.replace(temporary,state_path)

    def process_subject(sid: str) -> dict[str, object]:
        if shutil.disk_usage(root).free/1024**3<60: raise RuntimeError("storage guard: less than 60GB free")
        group=frame[frame.subject_id==sid]
        cache_path=root/f"data/processed/{dataset}/{sid.replace(':','_')}.h5"
        if args.resume and cache_path.exists():
            with h5py.File(cache_path,"r") as handle:
                if handle.attrs.get("complete",False) and handle.attrs.get("preprocessing_config_hash")==config_hash(cfg):
                    return {"subject_id":sid,"status":"resumed","n_windows":int(len(handle["label"]))}
        if cfg["task"]=="sleep_staging":
            parts=[preprocess_sleep_recording(p,dataset,float(cfg["sampling_rate"]),tuple(cfg["bandpass_hz"]),float(cfg["epoch_seconds"])) for p in group.filepath]
            arrays={key:(np.concatenate([part[key] for part in parts]) if np.asarray(parts[0][key]).ndim>0 and key not in {"channel_names","channel_mask"} else parts[0][key]) for key in parts[0]}
        else:
            arrays=preprocess_mi_recordings(group.filepath.tolist(),float(cfg["sampling_rate"]),tuple(cfg["bandpass_hz"]),float(cfg["event_seconds"]))
        status=write_subject_hdf5(cache_path,arrays,{"dataset":dataset,"subject_id":sid},config_hash(cfg),args.resume)
        return {"subject_id":sid,"status":status,"n_windows":int(len(arrays["label"]))}

    with ThreadPoolExecutor(max_workers=max(1,args.num_workers)) as pool:
        future_to_sid={pool.submit(process_subject,sid):sid for sid in subjects}
        for future in as_completed(future_to_sid):
            sid=future_to_sid[future]
            try: completed.append(future.result())
            except Exception as exc: failures.append({"subject_id":sid,"error":f"{type(exc).__name__}: {exc}"})
            save_state()
            gc.collect()
            # Long EDF records can leave large freed arenas in glibc. Returning
            # them after every subject keeps the 2 GiB container below its
            # cgroup limit without changing the signal computation.
            try: ctypes.CDLL(None).malloc_trim(0)
            except (AttributeError, OSError): pass
    state={"dataset":dataset,"completed":sorted(completed,key=lambda x:x["subject_id"]),"failed":sorted(failures,key=lambda x:x["subject_id"]),"config_hash":config_hash(cfg),"updated_at":datetime.now(timezone.utc).isoformat()}
    print(json.dumps(state,indent=2)); return 1 if failures else 0
if __name__=="__main__":
    raise SystemExit(main())
