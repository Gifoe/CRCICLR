"""Synthetic pre-heldout check of five-fold aggregation with unequal ranks."""
import importlib.util
import json
import tempfile
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("gate_aggregate_check",HERE/"run.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

with tempfile.TemporaryDirectory(prefix="gate_v2_synthetic_") as root:
    root=Path(root)
    mod.OUT=root/"outputs"
    mod.TASKS=("OpenBMI_MI",)
    mod.FOLDS=range(5)
    mod.check_lock=lambda:None
    mod.cell=lambda task,fold,stage:root/stage/task.lower()/f"fold{fold}_seed0"
    sub=np.repeat(np.array([f"sub-{i}" for i in range(1,15)]),4)
    y=np.tile(np.array([0,0,1,1]),14)
    for fold in range(5):
        d=mod.cell("OpenBMI_MI",fold,"heldout")
        d.mkdir(parents=True)
        pred={}
        for session in (1,2):
            pred[f"S{session}_y"]=y
            pred[f"S{session}_subjects"]=sub
            for j,variant in enumerate(mod.VARIANTS):
                z=np.full((len(y),2),-0.2,dtype=np.float32)
                z[np.arange(len(y)),y]=0.2+0.01*j
                pred[f"{variant}_S{session}_z"]=z
                if variant!="BASELINE":
                    for key in ("tau_norm","intervention_norm","relative_intervention","gate_mean",
                                "gate_low_fraction","gate_high_fraction","direction_dot_pfull","pc_disagreement"):
                        pred[f"{variant}_S{session}_{key}"]=np.full(len(y),float(fold+1),dtype=np.float32)
        np.savez_compressed(d/"predictions.npz",**pred)
        rank=fold+1
        rows=[{"task":"OpenBMI_MI","fold":fold,"session":f"S{session}","variant":variant,
               "dimension":k,"mean_gate":1.0,"std_gate":0.0,"Pr_gate_lt_0p9":0.0,
               "Pr_gate_gt_1p1":0.0,"label":"POST_HOC_HELDOUT_DIAGNOSTIC"}
              for session in (1,2) for variant in mod.GATES for k in range(rank)]
        mod.B.csv_write(d/"GATE_DIMENSIONS.csv",rows)
        mod.B.json_write(d/"COMPLETE.json",{"status":"COMPLETE",
                         "predictions_sha256":mod.B.sha(d/"predictions.npz"),
                         "gate_dimensions_sha256":mod.B.sha(d/"GATE_DIMENSIONS.csv")})
    mod.aggregate()
    summary=list(mod.csv.DictReader((mod.OUT/"HELDOUT_MODEL_TASK_SUMMARY.csv").open()))
    contrasts=list(mod.csv.DictReader((mod.OUT/"PAIRED_HELDOUT_CONTRASTS.csv").open()))
    assert len(summary)==4 and len(contrasts)==9
    assert all(float(row["BA"])==1.0 for row in summary)
    assert len(list(mod.csv.DictReader((mod.OUT/"TRANSFER_GATE_AUDIT.csv").open())))==2*3*sum(range(1,6))
    print(json.dumps({"status":"PASS","fold_ranks":[1,2,3,4,5],"summary_rows":len(summary),
                      "contrast_rows":len(contrasts)}))
