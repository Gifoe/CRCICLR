"""OpenBMI fold-0-only spatial identity repair for R2EEG-v1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score

REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
V1_CODE = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code"
sys.path.insert(0, str(V1_CODE))
import run_stage1 as v1  # noqa: E402
from model_r2eeg import R2EEG  # noqa: E402
from model_spatial_repair import R2EEGSpatialRepair, parameter_count  # noqa: E402

EXP = REPO / "experiments" / "persist_eeg_r2eeg_spatial_repair_fold0_v1"
CODE, PROTOCOL, OUT = EXP / "code", EXP / "protocol", EXP / "outputs"
V1_EXP = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1"
V1_RUNTIME = Path("/root/rivermind-data/r2eeg_stage1_runtime")
RUNTIME = Path("/root/rivermind-data/r2eeg_spatial_repair_fold0_runtime")
EXPECTED_EEGNET, EXPECTED_V1 = 0.7785714285714285, 0.5157142857142857


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None: v1.write_json(path, value)


def load_checkpoint(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    if not path.is_file(): raise FileNotFoundError(path)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); model.eval()


def pooled_accuracy(model: torch.nn.Module, bundle: Any, subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> float:
    idx=bundle.indices(subjects,(2,)); y=bundle.labels(idx); predicted=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(idx),128):
            logits,_=model(v1.prepare(bundle,idx[start:start+128],mean,std,device)); predicted.append(logits.argmax(1).cpu().numpy())
    return float(accuracy_score(y,np.concatenate(predicted)))


def v1_permutation(model: R2EEG, bundle: Any, fold: dict[str,Any], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str,Any]:
    candidates=bundle.indices(fold["inner_train_subjects"],(1,)); rng=np.random.default_rng(0); idx=rng.choice(candidates,size=32,replace=False)
    x=v1.prepare(bundle,np.asarray(idx,dtype=np.int64),mean,std,device); p=np.random.default_rng(0).permutation(62)
    model.eval()
    with torch.no_grad():
        logits,z=model(x); logits_p,z_p=model(x[:,torch.as_tensor(p,device=device),:])
    return {"batch_size":32,"trial_indices":list(map(int,idx)),"permutation":list(map(int,p)),
            "max_abs_logit_difference":float((logits-logits_p).abs().max().cpu()),"mean_abs_logit_difference":float((logits-logits_p).abs().mean().cpu()),
            "max_abs_feature_difference":float((z-z_p).abs().max().cpu()),"mean_abs_feature_difference":float((z-z_p).abs().mean().cpu()),
            "prediction_agreement_fraction":float((logits.argmax(1)==logits_p.argmax(1)).float().mean().cpu())}


def spatial_permutation_test(device: torch.device) -> dict[str,float]:
    v1.set_seed(0); model=R2EEGSpatialRepair(62).to(device).eval(); x=torch.randn(4,62,1000,device=device); p=torch.as_tensor(np.random.default_rng(0).permutation(62),device=device)
    with torch.no_grad(): _,z=model(x); _,zp=model(x[:,p,:])
    return {"max_abs_feature_difference":float((z-zp).abs().max().cpu()),"mean_abs_feature_difference":float((z-zp).abs().mean().cpu())}


def temporal_audit(model: R2EEGSpatialRepair) -> dict[str,Any]:
    expected={"block1.conv":{"in":1,"out":24,"kernel":[25],"stride":[2],"padding":[12],"bias":False},"dw2":{"in":24,"out":24,"kernel":[15],"dilation":[2],"padding":[14],"groups":24,"bias":False},"pw2":{"in":24,"out":32,"kernel":[1],"bias":False},"dw3":{"in":32,"out":32,"kernel":[15],"dilation":[4],"padding":[28],"groups":32,"bias":False},"pw3":{"in":32,"out":48,"kernel":[1],"bias":False},"pool2":{"kernel":4,"stride":4},"pool3":{"output":8}}
    observed={"block1.conv":{"in":model.block1[0].in_channels,"out":model.block1[0].out_channels,"kernel":list(model.block1[0].kernel_size),"stride":list(model.block1[0].stride),"padding":list(model.block1[0].padding),"bias":model.block1[0].bias is not None},"dw2":{"in":model.dw2.in_channels,"out":model.dw2.out_channels,"kernel":list(model.dw2.kernel_size),"dilation":list(model.dw2.dilation),"padding":list(model.dw2.padding),"groups":model.dw2.groups,"bias":model.dw2.bias is not None},"pw2":{"in":model.pw2.in_channels,"out":model.pw2.out_channels,"kernel":list(model.pw2.kernel_size),"bias":model.pw2.bias is not None},"dw3":{"in":model.dw3.in_channels,"out":model.dw3.out_channels,"kernel":list(model.dw3.kernel_size),"dilation":list(model.dw3.dilation),"padding":list(model.dw3.padding),"groups":model.dw3.groups,"bias":model.dw3.bias is not None},"pw3":{"in":model.pw3.in_channels,"out":model.pw3.out_channels,"kernel":list(model.pw3.kernel_size),"bias":model.pw3.bias is not None},"pool2":{"kernel":model.pool2.kernel_size,"stride":model.pool2.stride},"pool3":{"output":model.pool3.output_size}}
    return {"v1_temporal_encoder_exact_match":observed==expected,"expected":expected,"observed":observed,"allowed_new_carrier_layernorm":True}


def gate(ba: float) -> tuple[str,str]:
    if ba<.65:return "R2EEG_SPATIAL_REPAIR_CATASTROPHIC_FAIL_STOP","NO"
    if ba<.70:return "R2EEG_SPATIAL_REPAIR_PARTIAL_STOP","PARTIAL"
    if ba<.75:return "R2EEG_SPATIAL_REPAIR_SUBSTANTIAL_STOP","PARTIAL"
    if ba<.77:return "R2EEG_SPATIAL_REPAIR_CARRIER_RECOVERED_STOP","YES"
    return "R2EEG_SPATIAL_REPAIR_STRONG_RECOVERY_STOP","YES"


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--validate-only",action="store_true"); args=ap.parse_args(); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for p in (CODE,PROTOCOL,OUT,RUNTIME):p.mkdir(parents=True,exist_ok=True)
    v1.RUNTIME=V1_RUNTIME; folds,search,split_sha=v1.load_split(); fold=folds["OpenBMI"][0]; bundle=v1.load_bundle("OpenBMI",search["OpenBMI"])
    if set(folds)!={"OpenBMI","WBCIC"} or fold["fold_id"]!=0: raise RuntimeError("frozen fold0 provenance invalid")
    normal_mean,normal_std,norm=v1.normalizer(bundle,fold["inner_train_subjects"]); norm_hash=norm["mean_std_sha256"]
    manifest_path=V1_RUNTIME/"episode_manifests/openbmi_fold0.json"; validation_path=V1_EXP/"protocol/VALIDATION.json"
    if not manifest_path.is_file() or not validation_path.is_file(): raise FileNotFoundError("prior deterministic episode provenance missing")
    validation=json.loads(validation_path.read_text()); prior_fold0=validation["episode_isolation"]["OpenBMI"]["folds"][0]
    if sha(manifest_path)!=prior_fold0["episode_sha256"]: raise RuntimeError("prior episode manifest hash mismatch")
    manifest=json.loads(manifest_path.read_text())["epochs"]
    if len(manifest)!=60 or len(manifest[0])!=20: raise RuntimeError("prior episode manifest does not match OpenBMI fold0")
    partial=pd.read_csv(V1_EXP/"outputs/PARTIAL_FOLD_RESULTS.csv"); record=partial[(partial.dataset=="OpenBMI")&(partial.fold==0)].iloc[0]
    provenance=json.loads(record.checkpoint_provenance); v1_path=Path(provenance["R2EEG-ERM"]["path"]); eeg_path=Path(provenance["EEGNet-ERM"]["path"])
    if sha(v1_path)!=provenance["R2EEG-ERM"]["sha256"] or sha(eeg_path)!=provenance["EEGNet-ERM"]["sha256"]: raise RuntimeError("frozen checkpoint SHA mismatch")
    if abs(float(record.EEGNet_BA)-EXPECTED_EEGNET)>1e-12 or abs(float(record.R2EEG_ERM_BA)-EXPECTED_V1)>1e-12: raise RuntimeError("baseline result identity mismatch")
    v1_model=R2EEG(62).to(device); load_checkpoint(v1_model,v1_path,device); v1_perm=v1_permutation(v1_model,bundle,fold,normal_mean,normal_std,device)
    v1.set_seed(0); repair=R2EEGSpatialRepair(62).to(device); count=parameter_count(repair)
    if count>=150000: raise RuntimeError(f"unexpected parameter count {count}")
    spatial_perm=spatial_permutation_test(device); audit=temporal_audit(repair)
    tests={"shape":{},"parameter_under_150k":True,"spatial_sensitivity":spatial_perm,"temporal_encoder_match":audit["v1_temporal_encoder_exact_match"]}
    for c in (62,):
        x=torch.randn(4,c,1000,device=device); logits,z=repair(x); zr,zs,zf=repair.forward_latents(x); assert logits.shape==(4,2) and zr.shape==zs.shape==(4,64) and zf.shape==z.shape==(4,128); tests["shape"][str(c)]={"logits":list(logits.shape),"z_rel":list(zr.shape),"z_res":list(zs.shape),"z_full":list(zf.shape)}
    write_json(PROTOCOL/"SOURCE_PROVENANCE.json",{"source_branch":"codex/persist-eeg-r2eeg-stage1-seed0-v1","source_split":str(V1_EXP/"protocol/STAGE1_SEARCH_CV_SPLIT.json"),"source_split_sha256":split_sha,"v1_checkpoint":str(v1_path),"v1_checkpoint_sha256":sha(v1_path),"eegnet_checkpoint":str(eeg_path),"eegnet_checkpoint_sha256":sha(eeg_path)})
    write_json(PROTOCOL/"FOLD0_SPLIT.json",fold); write_json(PROTOCOL/"CACHE_PROVENANCE.json",{"root":str(v1.OPENBMI_ROOT),"task":"MI/train","shape":[100,62,1000],"subjects":search["OpenBMI"]}); write_json(PROTOCOL/"NORMALIZATION_AUDIT.json",norm); write_json(PROTOCOL/"EPISODE_MANIFEST_PROVENANCE.json",{"reused":True,"path":str(manifest_path),"sha256":sha(manifest_path),"steps_per_epoch":20,"same_CE_sample_identities_and_order":True}); write_json(PROTOCOL/"TEMPORAL_ENCODER_MATCH.json",audit); write_json(PROTOCOL/"PARAMETER_COUNT.json",{"trainable_parameters":count,"under_150k":True}); write_json(PROTOCOL/"HOLDOUT_ISOLATION_AUDIT.json",{"V8_INTERNAL_HOLDOUT_loaded":False,"V8_INTERNAL_HOLDOUT_labels_loaded":False,"WBCIC_outer_loaded":False,"WBCIC_outer_labels_loaded":False,"only_OpenBMI_SEARCH":True}); write_json(PROTOCOL/"TESTS.json",tests); write_json(OUT/"V1_CHANNEL_PERMUTATION_TEST.json",v1_perm); write_json(OUT/"SPATIAL_REPAIR_RANDOM_WEIGHT_PERMUTATION_TEST.json",spatial_perm)
    if args.validate_only: print("R2EEG_SPATIAL_REPAIR_PROTOCOL_VALID",flush=True); return 0
    # New architecture: one deterministic seed-0 initialization and CE only.
    v1.RUNTIME=RUNTIME; v1.set_seed(0); repair=R2EEGSpatialRepair(62).to(device); init=copy.deepcopy(repair.state_dict()); init_hash=v1.state_hash(init)
    info=v1.train(repair,"SpatialRepair-ERM",bundle,fold,manifest,normal_mean,normal_std,norm_hash,{"sha256":sha(manifest_path),"steps_per_epoch":20},init_hash,device)
    subject,_=v1.evaluate(repair,bundle,fold["outer_dev_subjects"],(2,),normal_mean,normal_std,device); rows=[]
    for sid in fold["outer_dev_subjects"]: rows.append({"subject_id":sid,**subject[sid]})
    pd.DataFrame(rows).to_csv(OUT/"FOLD0_SUBJECT_RESULTS.csv",index=False); ba=float(np.mean([x["BA"] for x in subject.values()])); f1=float(np.mean([x["macro_F1"] for x in subject.values()])); pooled=pooled_accuracy(repair,bundle,fold["outer_dev_subjects"],normal_mean,normal_std,device); terminal,support=gate(ba)
    result={"dataset":"OpenBMI","fold":0,"seed":0,"EEGNet_BA":EXPECTED_EEGNET,"R2EEG_v1_ERM_BA":EXPECTED_V1,"SpatialRepair_BA":ba,"SpatialRepair_macro_F1":f1,"SpatialRepair_pooled_accuracy":pooled,"SpatialRepair_minus_v1_pp":(ba-EXPECTED_V1)*100,"SpatialRepair_minus_EEGNet_pp":(ba-EXPECTED_EEGNET)*100,"selected_epoch":info["selected_epoch"],"best_inner_val_subject_BA":info["best_inner_val_subject_BA"],"checkpoint_sha256":info["checkpoint_sha256"],"terminal":terminal,"spatial_identity_primary_cause_support":support}
    write_json(OUT/"TRAINING_LOG.json",info); write_json(OUT/"FOLD0_RESULT.json",result)
    decision=("# SpatialRepair fold-0 decision\n\n"+f"Terminal: `{terminal}`\n\n"+f"R2EEG-v1 channel permutation: max logit {v1_perm['max_abs_logit_difference']:.8g}, mean logit {v1_perm['mean_abs_logit_difference']:.8g}, max latent {v1_perm['max_abs_feature_difference']:.8g}, prediction agreement {v1_perm['prediction_agreement_fraction']:.4f}.\n\n"+f"BA: EEGNet {EXPECTED_EEGNET:.4f}; v1 {EXPECTED_V1:.4f}; SpatialRepair {ba:.4f}. Recovery {result['SpatialRepair_minus_v1_pp']:+.2f} pp; EEGNet gap {result['SpatialRepair_minus_EEGNet_pp']:+.2f} pp.\n\n"+f"Interpretation: `{support}`. INNER_VAL was used only for selection; OUTER_DEV was evaluated after checkpoint freeze. No reserved holdout was accessed.\n\n"+ ("Next action: diagnose the carrier before PRD." if ba<.70 else "Next action: design a proper spatial carrier before returning to PRD.")+"\n")
    (OUT/"DECISION.md").write_text(decision,encoding="utf-8"); print(terminal,flush=True); return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except Exception as exc:
        print(f"R2EEG_SPATIAL_REPAIR_PROTOCOL_INVALID: {type(exc).__name__}: {exc}",flush=True); raise
