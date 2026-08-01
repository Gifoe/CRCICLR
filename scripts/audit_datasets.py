#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from _common import parser
from hsc_tta.data import HMCAdapter, CAPAdapter, EEGMMIDBAdapter
from hsc_tta.utils import require_cpu


def main() -> int:
    args = parser("Audit readable EEG recordings").parse_args(); require_cpu(args.device)
    base = Path("/root/autodl-tmp/hsc_tta_eeg")
    adapters = [HMCAdapter(base / "data/raw/hmc"), CAPAdapter(base / "data/raw/cap"), EEGMMIDBAdapter(base / "data/raw/eegmmidb")]
    frames = [a.audit(args.limit_subjects) for a in adapters]
    recordings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = base / "data/manifests"; out.mkdir(parents=True, exist_ok=True)
    recordings.to_parquet(out / "recordings.parquet", index=False)
    if len(recordings):
        subjects = recordings.groupby(["dataset", "subject_id"], as_index=False).agg(n_recordings=("recording_id", "nunique"), eligible=("preprocessing_eligible", "any"), all_readable=("readable", "all"))
        subjects.to_parquet(out / "subjects.parquet", index=False)
        recordings[["dataset", "subject_id", "recording_id", "channel_names"]].explode("channel_names").rename(columns={"channel_names":"channel_name"}).to_parquet(out / "channels.parquet", index=False)
        annotation_rows=[]
        for row in recordings.itertuples():
            for raw_label in (row.raw_label_set or []):
                annotation_rows.append({"dataset":row.dataset,"subject_id":row.subject_id,"recording_id":row.recording_id,"annotation_source":row.annotation_source,"raw_label":raw_label,"recording_annotation_count":row.n_annotations})
        pd.DataFrame(annotation_rows,columns=["dataset","subject_id","recording_id","annotation_source","raw_label","recording_annotation_count"]).to_parquet(out / "annotations.parquet",index=False)
        recordings.loc[~recordings.preprocessing_eligible, ["dataset", "subject_id", "recording_id", "exclusion_reason"]].to_parquet(out / "exclusions.parquet", index=False)
    summary = {a.dataset: {"discovered": len(f), "readable": int(f.readable.sum()) if len(f) else 0, "eligible_subjects": int(f.loc[f.preprocessing_eligible, "subject_id"].nunique()) if len(f) else 0} for a, f in zip(adapters, frames)}
    (out / "dataset_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
