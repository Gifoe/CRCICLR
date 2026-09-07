"""Write user-directed partial results without training or opening holdouts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_stage1 as r
import stage1_core as core
from model_r2eeg import R2EEG


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def load_best(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file(): raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); model.eval()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r.RUNTIME = Path("/root/rivermind-data/r2eeg_stage1_runtime")
    folds, search, split_sha = r.load_split()
    bundle = r.load_bundle("OpenBMI", search["OpenBMI"])
    rows, fold_rows = [], []
    for fold in folds["OpenBMI"][:2]:
        mean, std, norm = r.normalizer(bundle, fold["inner_train_subjects"])
        spec = (("EEGNet-ERM", core.EEGNet(bundle.channels), "eegnet_erm"),
                ("R2EEG-ERM", R2EEG(bundle.channels), "r2eeg_erm"),
                ("R2EEG-Rel", R2EEG(bundle.channels), "r2eeg_rel"))
        subject = {}
        checkpoints = {}
        for name, model, stem in spec:
            path = r.RUNTIME / "checkpoints" / "openbmi" / f"fold{fold['fold_id']}_{stem}_best.pt"
            model = model.to(device); load_best(model, path, device)
            subject[name], _ = r.evaluate(model, bundle, fold["outer_dev_subjects"], (2,), mean, std, device)
            checkpoints[name] = {"path": str(path), "sha256": sha(path)}
        for sid in fold["outer_dev_subjects"]:
            e, a, q = subject["EEGNet-ERM"][sid], subject["R2EEG-ERM"][sid], subject["R2EEG-Rel"][sid]
            rows.append({"dataset":"OpenBMI","fold":fold["fold_id"],"subject_id":sid,
                         "EEGNet_BA":e["BA"],"R2EEG_ERM_BA":a["BA"],"R2EEG_Rel_BA":q["BA"],
                         "EEGNet_macro_F1":e["macro_F1"],"R2EEG_ERM_macro_F1":a["macro_F1"],"R2EEG_Rel_macro_F1":q["macro_F1"],
                         "rel_vs_erm_delta_BA_pp":(q["BA"]-a["BA"])*100,"erm_vs_eegnet_delta_BA_pp":(a["BA"]-e["BA"])*100})
        avg = lambda name: float(np.mean([x["BA"] for x in subject[name].values()]))
        fold_rows.append({"dataset":"OpenBMI","fold":fold["fold_id"],"EEGNet_BA":avg("EEGNet-ERM"),"R2EEG_ERM_BA":avg("R2EEG-ERM"),"R2EEG_Rel_BA":avg("R2EEG-Rel"),
                          "architecture_gain_pp":(avg("R2EEG-ERM")-avg("EEGNet-ERM"))*100,"relational_gain_pp":(avg("R2EEG-Rel")-avg("R2EEG-ERM"))*100,"checkpoint_provenance":json.dumps(checkpoints,sort_keys=True)})
    frame = pd.DataFrame(rows); aggregate = {"dataset":"OpenBMI","completed_folds":[0,1],"EEGNet_BA":float(frame.EEGNet_BA.mean()),"R2EEG_ERM_BA":float(frame.R2EEG_ERM_BA.mean()),"R2EEG_Rel_BA":float(frame.R2EEG_Rel_BA.mean()),"architecture_gain_pp":float((frame.R2EEG_ERM_BA.mean()-frame.EEGNet_BA.mean())*100),"relational_gain_pp":float((frame.R2EEG_Rel_BA.mean()-frame.R2EEG_ERM_BA.mean())*100),"total_gain_pp":float((frame.R2EEG_Rel_BA.mean()-frame.EEGNet_BA.mean())*100),"status":"PARTIAL_NONFORMAL"}
    r.OUT.mkdir(parents=True, exist_ok=True); pd.DataFrame(fold_rows).to_csv(r.OUT/"PARTIAL_FOLD_RESULTS.csv",index=False); frame.to_csv(r.OUT/"PARTIAL_SUBJECT_RESULTS.csv",index=False)
    r.write_json(r.OUT/"PARTIAL_AGGREGATE.json",aggregate)
    r.write_json(r.OUT/"PARTIAL_STOP.json",{"reason":"USER_DIRECTED_STOP_AFTER_CURRENT_OPENBMI_FOLD_1","completed":{"OpenBMI":[0,1],"WBCIC":[]},"discarded_uncompleted_runtime":{"OpenBMI_fold_2":True},"no_holdout_access":True,"formal_three_fold_gate_evaluated":False,"source_split_sha256":split_sha})
    (r.OUT/"PARTIAL_DECISION.md").write_text("# User-directed partial stopping report\n\nOnly OpenBMI folds 0 and 1 completed. Fold 2 and all WBCIC folds were not evaluated and must not be treated as a formal Stage-1 gate.\n\n"+json.dumps(aggregate,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(aggregate,sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
