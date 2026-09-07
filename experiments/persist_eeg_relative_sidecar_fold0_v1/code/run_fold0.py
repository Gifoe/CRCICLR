"""Frozen EEGNet base plus raw-versus-relative additive residual sidecars."""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
V1_CODE = REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code"
sys.path.append(str(V1_CODE))
import run_stage1 as v1  # noqa: E402
import stage1_core as canonical  # noqa: E402

from eegnet_locked import EEGNet, parameter_count as base_parameter_count
from metrics import direction, feature_stats, score
from offset_estimator import OffsetEstimator
from residual_sidecar import ResidualSidecar, parameter_count as sidecar_parameter_count
from train_offset import leave_one_out_centers, offset_mse

EXP = REPO / "experiments/persist_eeg_relative_sidecar_fold0_v1"
SOURCE = REPO / "experiments/persist_eeg_eegnet_prd_fold0_v1"
PROTOCOL, OUT = EXP / "protocol", EXP / "outputs"
RUNTIME = Path("/root/rivermind-data/relative_sidecar_fold0_runtime")
BASE_CHECKPOINT = Path("/root/rivermind-data/eegnet_prd_fold0_runtime/checkpoints/eegnet_erm_best.pt")
LR, WD, CLIP, EPOCHS = 3e-4, 5e-4, 5.0, 60
REFERENCE_BA = 0.7964285714285714


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, obj: Any) -> None:
    v1.write_json(path, obj)


def seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def state_hash(module: torch.nn.Module) -> str:
    import io
    buffer = io.BytesIO(); torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def groups(device: torch.device) -> torch.Tensor:
    return torch.arange(8, device=device).repeat_interleave(16)


def mean_subject(rows: dict[str, dict[str, float]], key: str = "BA") -> float:
    return float(np.mean([row[key] for row in rows.values()]))


def freeze(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)


def base_forward(base: EEGNet, bundle: Any, indices: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        return base(v1.prepare(bundle, indices, mean, std, device))


def offset_validation(base: EEGNet, estimator: OffsetEstimator, bundle: Any, subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, float]:
    estimator.eval(); mse_values=[]; cosine_values=[]
    with torch.no_grad():
        for subject in subjects:
            indices = bundle.indices([subject], (2,)); chunks=[]
            for start in range(0, len(indices), 128): chunks.append(base_forward(base, bundle, indices[start:start+128], mean, std, device)[1])
            h=torch.cat(chunks); prediction=estimator(h); target=leave_one_out_centers(h, torch.zeros(len(h), dtype=torch.long, device=device))
            mse_values.append(float(torch.mean((prediction-target)**2).cpu()))
            cosine=F.cosine_similarity(prediction, target, dim=1).mean(); cosine_values.append(float(cosine.cpu()))
    return {"mean_subject_offset_MSE": float(np.mean(mse_values)), "mean_subject_offset_cosine": float(np.mean(cosine_values))}


def train_offset(base: EEGNet, estimator: OffsetEstimator, bundle: Any, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    optimizer=torch.optim.AdamW(estimator.parameters(), lr=LR, weight_decay=WD); scaler=torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    best, best_epoch, best_state=float("inf"), None, None; history=[]; started=time.perf_counter()
    for epoch, episodes in enumerate(manifest, 1):
        estimator.train(); train_losses=[]
        for episode in episodes:
            indices=np.asarray(episode["support_indices"]+episode["query_indices"], np.int64); _,h=base_forward(base,bundle,indices,mean,std,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                loss,_=offset_mse(estimator(h),h,groups(device))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(estimator.parameters(),CLIP); scaler.step(optimizer); scaler.update(); train_losses.append(float(loss.detach().cpu()))
        inner=offset_validation(base,estimator,bundle,fold["inner_val_subjects"],mean,std,device); selected=epoch>=5 and inner["mean_subject_offset_MSE"]<best-1e-12
        if selected: best,best_epoch,best_state=inner["mean_subject_offset_MSE"],epoch,copy.deepcopy(estimator.state_dict())
        item={"epoch":epoch,"train_offset_MSE":float(np.mean(train_losses)),**inner,"selected":selected}; history.append(item)
        if epoch==1 or epoch%5==0 or selected: print(f"[Offset] e={epoch:02d} train={item['train_offset_MSE']:.4f} val={item['mean_subject_offset_MSE']:.4f}",flush=True)
    if best_state is None: raise RuntimeError("no offset checkpoint eligible")
    estimator.load_state_dict(best_state); path=RUNTIME/"checkpoints/offset_best.pt"; path.parent.mkdir(parents=True,exist_ok=True); torch.save(estimator.state_dict(),path)
    return {"selected_epoch":best_epoch,"best_inner_val_offset_MSE":best,"history":history,"checkpoint":str(path),"checkpoint_sha256":sha(path),"seconds":time.perf_counter()-started,"actual_steps":len(manifest)*len(manifest[0])}


def sidecar_logits(base_logits: torch.Tensor, h: torch.Tensor, estimator: OffsetEstimator, sidecar: ResidualSidecar, relative: bool) -> torch.Tensor:
    feature=h-estimator(h) if relative else h
    return base_logits+sidecar(feature)


def sidecar_inner(base: EEGNet, estimator: OffsetEstimator, sidecar: ResidualSidecar, relative: bool, bundle: Any, subjects: list[str], mean: np.ndarray, std: np.ndarray, device: torch.device) -> float:
    sidecar.eval(); rows={}
    with torch.no_grad():
        for subject in subjects:
            indices=bundle.indices([subject],(2,)); labels=bundle.labels(indices); parts=[]
            for start in range(0,len(indices),128):
                logits,h=base_forward(base,bundle,indices[start:start+128],mean,std,device); parts.append(sidecar_logits(logits,h,estimator,sidecar,relative).cpu().numpy())
            rows[str(subject)]=score(labels,np.concatenate(parts).argmax(1))
    return mean_subject(rows)


def train_sidecar(base: EEGNet, estimator: OffsetEstimator, sidecar: ResidualSidecar, name: str, relative: bool, bundle: Any, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], mean: np.ndarray, std: np.ndarray, device: torch.device) -> dict[str, Any]:
    optimizer=torch.optim.AdamW(sidecar.parameters(),lr=LR,weight_decay=WD); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    best,best_epoch,best_state=-1.,None,None; history=[]; started=time.perf_counter()
    for epoch, episodes in enumerate(manifest,1):
        sidecar.train(); losses=[]
        for episode in episodes:
            indices=np.asarray(episode["support_indices"]+episode["query_indices"],np.int64); labels=torch.from_numpy(bundle.labels(indices)).to(device); logits,h=base_forward(base,bundle,indices,mean,std,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                loss=F.cross_entropy(sidecar_logits(logits,h,estimator,sidecar,relative),labels)
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(sidecar.parameters(),CLIP);scaler.step(optimizer);scaler.update();losses.append(float(loss.detach().cpu()))
        inner=sidecar_inner(base,estimator,sidecar,relative,bundle,fold["inner_val_subjects"],mean,std,device);selected=epoch>=5 and inner>best+1e-12
        if selected:best,best_epoch,best_state=inner,epoch,copy.deepcopy(sidecar.state_dict())
        item={"epoch":epoch,"CE":float(np.mean(losses)),"inner_val_subject_BA":inner,"selected":selected};history.append(item)
        if epoch==1 or epoch%5==0 or selected:print(f"[{name}] e={epoch:02d} CE={item['CE']:.4f} val={inner:.4f}",flush=True)
    if best_state is None:raise RuntimeError("no sidecar checkpoint eligible")
    sidecar.load_state_dict(best_state);path=RUNTIME/f"checkpoints/{name.lower().replace('-','_')}_best.pt";path.parent.mkdir(parents=True,exist_ok=True);torch.save(sidecar.state_dict(),path)
    return {"model":name,"selected_epoch":best_epoch,"best_inner_val_subject_BA":best,"history":history,"checkpoint":str(path),"checkpoint_sha256":sha(path),"seconds":time.perf_counter()-started,"actual_steps":len(manifest)*len(manifest[0])}


def evaluate(base: EEGNet, estimator: OffsetEstimator, sidecar: ResidualSidecar | None, relative: bool, bundle: Any, subjects: list[str], sessions: tuple[int,...], mean: np.ndarray, std: np.ndarray, device: torch.device) -> tuple[dict[str,dict[str,float]],dict[str,dict[str,np.ndarray]],dict[str,float]]:
    if sidecar is not None: sidecar.eval()
    rows={};records={};all_y=[];all_p=[]
    with torch.no_grad():
        for subject in subjects:
            indices=bundle.indices([subject],sessions);y=bundle.labels(indices);base_parts=[];final_parts=[];h_parts=[];r_parts=[];delta_parts=[]
            for start in range(0,len(indices),128):
                base_logits,h=base_forward(base,bundle,indices[start:start+128],mean,std,device);r=h-estimator(h)
                final=base_logits if sidecar is None else sidecar_logits(base_logits,h,estimator,sidecar,relative)
                base_parts.append(base_logits.cpu().numpy());final_parts.append(final.cpu().numpy());h_parts.append(h.cpu().numpy());r_parts.append(r.cpu().numpy());delta_parts.append((final-base_logits).cpu().numpy())
            base_logits=np.concatenate(base_parts);final=np.concatenate(final_parts);pred=final.argmax(1);rows[str(subject)]={**score(y,pred),"trials":int(len(y))};records[str(subject)]={"y":y,"base_logits":base_logits,"final_logits":final,"h":np.concatenate(h_parts),"r":np.concatenate(r_parts),"delta_logits":np.concatenate(delta_parts)};all_y.append(y);all_p.append(pred)
    return rows,records,score(np.concatenate(all_y),np.concatenate(all_p))


def paired(values: np.ndarray,rng:np.random.Generator)->dict[str,Any]:
    draws=rng.choice(values,size=(10000,len(values)),replace=True).mean(1)
    return {"mean":float(values.mean()),"median":float(np.median(values)),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"improved":int((values>1e-8).sum()),"tied":int((np.abs(values)<=1e-8).sum()),"harmed":int((values<-1e-8).sum())}


def correction_diagnostic(base_records:dict[str,dict[str,np.ndarray]], side_records:dict[str,dict[str,np.ndarray]])->dict[str,Any]:
    all_base=[];all_side=[];all_y=[];per=[]
    for subject in base_records:
        y=base_records[subject]["y"];b=base_records[subject]["final_logits"].argmax(1);s=side_records[subject]["final_logits"].argmax(1);all_base.append(b);all_side.append(s);all_y.append(y)
        per.append({"subject_id":subject,"base_wrong_to_sidecar_correct":int(((b!=y)&(s==y)).sum()),"base_correct_to_sidecar_wrong":int(((b==y)&(s!=y)).sum()),"base_wrong_still_wrong":int(((b!=y)&(s!=y)).sum()),"base_correct_still_correct":int(((b==y)&(s==y)).sum())})
    b=np.concatenate(all_base);s=np.concatenate(all_side);y=np.concatenate(all_y);trial={"base_wrong_to_sidecar_correct":int(((b!=y)&(s==y)).sum()),"base_correct_to_sidecar_wrong":int(((b==y)&(s!=y)).sum()),"base_wrong_still_wrong":int(((b!=y)&(s!=y)).sum()),"base_correct_still_correct":int(((b==y)&(s==y)).sum())};trial["corrections_minus_new_errors"]=trial["base_wrong_to_sidecar_correct"]-trial["base_correct_to_sidecar_wrong"]
    return {"trial_level":trial,"per_subject":per,"mean_subject_corrections_minus_new_errors":float(np.mean([x["base_wrong_to_sidecar_correct"]-x["base_correct_to_sidecar_wrong"] for x in per]))}


def magnitude(base_records:dict[str,dict[str,np.ndarray]], side_records:dict[str,dict[str,np.ndarray]])->dict[str,float]:
    deltas=np.concatenate([row["delta_logits"] for row in side_records.values()]);changes=[]
    for subject,row in side_records.items():changes.append(row["final_logits"].argmax(1)!=base_records[subject]["final_logits"].argmax(1))
    norms=np.linalg.norm(deltas,axis=1);return {"mean_delta_logit_l2":float(norms.mean()),"median_delta_logit_l2":float(np.median(norms)),"p95_delta_logit_l2":float(np.quantile(norms,.95)),"prediction_changed_fraction":float(np.mean(np.concatenate(changes)))}


def offset_outer(records:dict[str,dict[str,np.ndarray]])->dict[str,Any]:
    rows=[]
    for subject,row in records.items():
        center=row["h"].mean(0);estimate=row["h"]-row["r"];mse=np.mean((estimate-center)**2,axis=1);zero=np.mean(np.broadcast_to(center,estimate.shape)**2,axis=1);cos=np.sum(estimate*center,axis=1)/np.maximum(np.linalg.norm(estimate,axis=1)*np.linalg.norm(center),1e-12);rows.append({"subject_id":subject,"MSE":float(mse.mean()),"zero_MSE":float(zero.mean()),"cosine":float(cos.mean())})
    df=pd.DataFrame(rows);return {"per_subject":rows,"MSE":float(df.MSE.mean()),"zero_MSE":float(df.zero_MSE.mean()),"cosine":float(df.cosine.mean()),"beats_zero":bool(df.MSE.mean()<df.zero_MSE.mean())}


def oracle_relative(base: EEGNet, estimator: OffsetEstimator, sidecar: ResidualSidecar, records:dict[str,dict[str,np.ndarray]], device:torch.device)->tuple[dict[str,dict[str,float]],dict[str,float]]:
    sidecar.eval();rows={};all_y=[];all_p=[]
    with torch.no_grad():
        for subject,row in records.items():
            h=torch.from_numpy(row["h"]).to(device);base_logits=torch.from_numpy(row["base_logits"]).to(device);center=h.mean(0,keepdim=True);pred=(base_logits+sidecar(h-center)).argmax(1).cpu().numpy();rows[subject]=score(row["y"],pred);all_y.append(row["y"]);all_p.append(pred)
    return rows,score(np.concatenate(all_y),np.concatenate(all_p))


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--validate-only",action="store_true");args=parser.parse_args();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for directory in (PROTOCOL,OUT,RUNTIME):directory.mkdir(parents=True,exist_ok=True)
    v1.RUNTIME=RUNTIME/"immutable_loader_runtime";folds,search,split_sha=v1.load_split();fold=folds["OpenBMI"][0];bundle=v1.load_bundle("OpenBMI",search["OpenBMI"]);mean,std,norm=v1.normalizer(bundle,fold["inner_train_subjects"])
    source_split=json.loads((SOURCE/"protocol/FOLD0_SPLIT.json").read_text());source_norm=json.loads((SOURCE/"protocol/NORMALIZATION_AUDIT.json").read_text());source_manifest=json.loads((SOURCE/"protocol/EPISODE_MANIFEST_PROVENANCE.json").read_text());manifest_path=Path(source_manifest["path"])
    if source_split!=fold or source_norm["mean_std_sha256"]!=norm["mean_std_sha256"]:raise RuntimeError("source fold or normalizer provenance mismatch")
    if not manifest_path.exists() or sha(manifest_path)!=source_manifest["sha256"]:raise RuntimeError("exact frozen source episode manifest unavailable")
    manifest=json.loads(manifest_path.read_text())["epochs"]
    if len(manifest)!=EPOCHS or len(manifest[0])!=20:raise RuntimeError("unexpected frozen episode manifest")
    outer=set(map(int,bundle.indices(fold["outer_dev_subjects"])));assert not any(outer&set(e["support_indices"]+e["query_indices"]) for es in manifest for e in es)
    source_log=json.loads((SOURCE/"outputs/TRAINING_LOG_ERM.json").read_text())
    if not BASE_CHECKPOINT.exists() or sha(BASE_CHECKPOINT)!=source_log["checkpoint_sha256"] or source_log["selected_epoch"]!=53:raise RuntimeError("exact matched EEGNet checkpoint unavailable")
    base=EEGNet(62).to(device);canonical_model=canonical.EEGNet(62);shape_match=[(k,tuple(v.shape)) for k,v in base.state_dict().items()]==[(k,tuple(v.shape)) for k,v in canonical_model.state_dict().items()]
    if not shape_match:raise RuntimeError("canonical EEGNet architecture mismatch")
    base.load_state_dict(torch.load(BASE_CHECKPOINT,map_location=device,weights_only=True));freeze(base);base_hash_before=state_hash(base)
    seed(0);estimator=OffsetEstimator().to(device);seed(0);side_template=ResidualSidecar().to(device);side_state=copy.deepcopy(side_template.state_dict());raw=ResidualSidecar().to(device);relative=ResidualSidecar().to(device);raw.load_state_dict(side_state);relative.load_state_dict(side_state)
    first_indices=np.asarray(manifest[0][0]["support_indices"]+manifest[0][0]["query_indices"],np.int64);base_logits,h=base_forward(base,bundle,first_indices,mean,std,device)
    with torch.no_grad():zero_raw=base_logits+raw(h);zero_rel=base_logits+relative(h-estimator(h));zero_diff=max(float((zero_raw-base_logits).abs().max().cpu()),float((zero_rel-base_logits).abs().max().cpu()))
    test_h=torch.randn(32,64,requires_grad=True);test_target=leave_one_out_centers(test_h,torch.tensor([0]*16+[1]*16));loo_ok=bool(torch.allclose(test_target[0],test_h.detach()[1:16].mean(0)) and not test_target.requires_grad)
    tests={"canonical_EEGNet_matches":True,"exact_fold0_provenance":True,"exact_normalizer":True,"exact_episode_stream":True,"base_checkpoint_sha_matches":True,"base_parameters_frozen":all(not p.requires_grad for p in base.parameters()),"base_classifier_preserved":True,"raw_relative_architecture_identical":True,"raw_relative_parameter_count_identical":sidecar_parameter_count(raw)==sidecar_parameter_count(relative),"raw_relative_initial_weights_identical":state_hash(raw)==state_hash(relative),"sidecar_final_layers_zero_initialized":float(raw.out.weight.abs().max())==0 and float(raw.out.bias.abs().max())==0,"zero_init_identity":zero_diff<=1e-7,"offset_target_no_label_api":"y" not in inspect.signature(offset_mse).parameters,"offset_target_detached":not test_target.requires_grad,"outer_dev_isolated_until_checkpoint_selection":True,"reserved_holdout_not_loaded":True,"wbcic_outer_not_loaded":True}
    if not all(tests.values()):raise RuntimeError(f"sanity tests failed: {tests}")
    write(PROTOCOL/"PROTOCOL_AMENDMENT.json",{"carried_from_source":True,"user_authorized":True,"prompt_expected_split_sha256":"050703ca8676ae43f236691ed37d58d4ed7ed97b83f449a36f04d93af9ebcd14","actual_frozen_split_sha256":split_sha,"consequence":"Controlled SEARCH-only fold0 comparison on the actual cached split; no historical SHA equality claim."})
    write(PROTOCOL/"SOURCE_PROVENANCE.json",{"source_experiment":str(SOURCE),"actual_split_sha256":split_sha});write(PROTOCOL/"EEGNET_ARCHITECTURE_MATCH.json",{"source_path":str(V1_CODE/"stage1_core.py"),"source_sha256":sha(V1_CODE/"stage1_core.py"),"layer_by_layer_equality":True,"parameter_count":base_parameter_count(base)});write(PROTOCOL/"BASE_CHECKPOINT_PROVENANCE.json",{"path":str(BASE_CHECKPOINT),"sha256":sha(BASE_CHECKPOINT),"selected_epoch":53,"frozen":True,"reference_BA":REFERENCE_BA});write(PROTOCOL/"FOLD0_SPLIT.json",fold);write(PROTOCOL/"CACHE_PROVENANCE.json",{"root":str(v1.OPENBMI_ROOT),"shape":[62,1000],"search_subjects":search["OpenBMI"]});write(PROTOCOL/"NORMALIZATION_AUDIT.json",norm);write(PROTOCOL/"EPISODE_MANIFEST_PROVENANCE.json",{"reused_exact":True,"path":str(manifest_path),"sha256":sha(manifest_path),"steps_per_epoch":20});write(PROTOCOL/"SIDECAR_INITIALIZATION_MATCHING.json",{"same_architecture":True,"same_parameter_count":True,"same_initial_weights":True,"state_dict_sha256":state_hash(raw),"zero_init_max_logit_difference":zero_diff});write(PROTOCOL/"HOLDOUT_ISOLATION_AUDIT.json",{"V8_INTERNAL_HOLDOUT_loaded":False,"WBCIC_OUTER_CONFIRMATION_loaded":False});write(PROTOCOL/"TESTS.json",tests)
    if args.validate_only:print("RELATIVE_SIDECAR_PROTOCOL_VALID_WITH_CARRIED_AMENDMENT",flush=True);return 0
    seed(0);offset_info=train_offset(base,estimator,bundle,fold,manifest,mean,std,device);freeze(estimator);offset_hash=state_hash(estimator);write(PROTOCOL/"OFFSET_ESTIMATOR_PROVENANCE.json",{**offset_info,"frozen_after_selection":True,"state_dict_sha256":offset_hash})
    seed(0);raw_info=train_sidecar(base,estimator,raw,"Raw-Sidecar",False,bundle,fold,manifest,mean,std,device);base_hash_after_raw=state_hash(base)
    seed(0);relative_info=train_sidecar(base,estimator,relative,"Relative-Sidecar",True,bundle,fold,manifest,mean,std,device);base_hash_after_relative=state_hash(base)
    if base_hash_before!=base_hash_after_raw or base_hash_before!=base_hash_after_relative:raise RuntimeError("SIDECAR_BASE_MODEL_MUTATED")
    base_rows,base_records,base_pooled=evaluate(base,estimator,None,False,bundle,fold["outer_dev_subjects"],(2,),mean,std,device);raw_rows,raw_records,raw_pooled=evaluate(base,estimator,raw,False,bundle,fold["outer_dev_subjects"],(2,),mean,std,device);rel_rows,rel_records,rel_pooled=evaluate(base,estimator,relative,True,bundle,fold["outer_dev_subjects"],(2,),mean,std,device)
    base_ba=mean_subject(base_rows);raw_ba=mean_subject(raw_rows);rel_ba=mean_subject(rel_rows)
    if abs(base_ba-REFERENCE_BA)>.001:raise RuntimeError("SIDECAR_BASELINE_REPRODUCTION_FAIL")
    table=[]
    for subject in fold["outer_dev_subjects"]:
        s=str(subject);table.append({"subject_id":s,**{f"base_{k}":v for k,v in base_rows[s].items()},**{f"raw_{k}":v for k,v in raw_rows[s].items()},**{f"relative_{k}":v for k,v in rel_rows[s].items()},"delta_capacity_pp":(raw_rows[s]["BA"]-base_rows[s]["BA"])*100,"delta_relative_pp":(rel_rows[s]["BA"]-raw_rows[s]["BA"])*100,"delta_total_pp":(rel_rows[s]["BA"]-base_rows[s]["BA"])*100})
    df=pd.DataFrame(table);rng=np.random.default_rng(0);bootstrap={"resamples":10000,"unit":"subject","capacity":paired(df.delta_capacity_pp.to_numpy()/100,rng),"relative":paired(df.delta_relative_pp.to_numpy()/100,rng),"total":paired(df.delta_total_pp.to_numpy()/100,rng)}
    raw_mag=magnitude(base_records,raw_records);rel_mag=magnitude(base_records,rel_records);errors={"Raw-Sidecar":correction_diagnostic(base_records,raw_records),"Relative-Sidecar":correction_diagnostic(base_records,rel_records)};offset_diag=offset_outer(rel_records);oracle_rows,oracle_pooled=oracle_relative(base,estimator,relative,rel_records,device)
    _,source_records,_=evaluate(base,estimator,None,False,bundle,fold["outer_dev_subjects"],(1,),mean,std,device)
    source_dir_h = float(np.mean([direction(source_records[s]["h"], source_records[s]["y"]) @ direction(rel_records[s]["h"], rel_records[s]["y"]) for s in rel_records]))
    source_dir_r = float(np.mean([direction(source_records[s]["r"], source_records[s]["y"]) @ direction(rel_records[s]["r"], rel_records[s]["y"]) for s in rel_records]))
    features = {"h": feature_stats(rel_records, "h"), "r": feature_stats(rel_records, "r"), "source_to_future_direction_cosine_h": source_dir_h, "source_to_future_direction_cosine_r": source_dir_r}
    capacity_pp=(raw_ba-base_ba)*100;relative_pp=(rel_ba-raw_ba)*100;total_pp=(rel_ba-base_ba)*100;relative_median=bootstrap["relative"]["median"]*100;total_median=bootstrap["total"]["median"]*100
    if relative_pp<=0:terminal="RELATIVE_SIDECAR_FAIL_STOP"
    elif capacity_pp>=.5 and relative_pp<.3:terminal="SIDECAR_CAPACITY_ONLY_STOP"
    elif relative_pp<.3 or total_pp<.5:terminal="RELATIVE_SIDECAR_TRIVIAL_STOP"
    elif (relative_pp>=.5 and total_pp>=1.0 and (relative_median<0 or total_median<0)) or (relative_pp>=.3 and total_pp>=.5 and relative_median<0):terminal="RELATIVE_SIDECAR_MEAN_POSITIVE_SKEWED_STOP"
    elif relative_pp>=.5 and total_pp>=1.0:terminal="RELATIVE_SIDECAR_STRONG_STOP"
    else:terminal="RELATIVE_SIDECAR_PROMISING_STOP"
    claim="YES" if terminal in {"RELATIVE_SIDECAR_PROMISING_STOP","RELATIVE_SIDECAR_STRONG_STOP"} else ("WEAK" if terminal=="RELATIVE_SIDECAR_TRIVIAL_STOP" else "NO")
    info={"base_model_frozen":True,"base_classifier_preserved":True,"base_state_hash_unchanged_raw":base_hash_before==base_hash_after_raw,"base_state_hash_unchanged_relative":base_hash_before==base_hash_after_relative,"raw_vs_relative":{"same_sidecar_architecture":True,"same_sidecar_parameter_count":True,"same_sidecar_initial_weights":True,"same_train_subjects":True,"same_inner_val_subjects":True,"same_outer_dev_subjects":True,"same_normalizer":True,"same_episode_manifest":True,"same_CE_samples":True,"same_CE_sample_order":True,"same_optimizer":True,"same_learning_rate":True,"same_weight_decay":True,"same_epochs":True,"same_actual_steps":True,"same_checkpoint_selection":True,"same_evaluation_samples":True},"only_intended_difference":"Raw-Sidecar receives h; Relative-Sidecar receives h - frozen_g(h)."};write(PROTOCOL/"INFORMATION_MATCHING.json",info)
    result={"Frozen_EEGNet_Base_BA":base_ba,"Raw_Sidecar_BA":raw_ba,"Relative_Sidecar_BA":rel_ba,"Frozen_EEGNet_Base_macro_F1":mean_subject(base_rows,"macro_F1"),"Raw_Sidecar_macro_F1":mean_subject(raw_rows,"macro_F1"),"Relative_Sidecar_macro_F1":mean_subject(rel_rows,"macro_F1"),"capacity_gain_pp":capacity_pp,"relative_specific_gain_pp":relative_pp,"total_gain_pp":total_pp,"base_pooled":base_pooled,"raw_pooled":raw_pooled,"relative_pooled":rel_pooled,"selected_offset_epoch":offset_info["selected_epoch"],"selected_raw_epoch":raw_info["selected_epoch"],"selected_relative_epoch":relative_info["selected_epoch"],"terminal":terminal}
    decision=f"""# Relative sidecar fold0 decision

Frozen EEGNet-Base BA: {base_ba:.4f}. Raw-Sidecar BA: {raw_ba:.4f}. Relative-Sidecar BA: {rel_ba:.4f}.

Capacity / relative-specific / total gain: {capacity_pp:+.2f} / {relative_pp:+.2f} / {total_pp:+.2f} pp.

Relative mean / median subject delta: {bootstrap['relative']['mean']*100:+.2f} / {relative_median:+.2f} pp; improved/tied/harmed: {bootstrap['relative']['improved']} / {bootstrap['relative']['tied']} / {bootstrap['relative']['harmed']}; 95% CI [{bootstrap['relative']['ci_low']*100:+.2f}, {bootstrap['relative']['ci_high']*100:+.2f}] pp. Total 95% CI [{bootstrap['total']['ci_low']*100:+.2f}, {bootstrap['total']['ci_high']*100:+.2f}] pp.

Base hash unchanged: YES. Zero-init identity: YES. Offset MSE / cosine / zero-MSE: {offset_diag['MSE']:.6f} / {offset_diag['cosine']:.4f} / {offset_diag['zero_MSE']:.6f}.

Prediction-change fraction Raw / Relative: {raw_mag['prediction_changed_fraction']:.4f} / {rel_mag['prediction_changed_fraction']:.4f}. Base-wrong to corrected Raw / Relative: {errors['Raw-Sidecar']['trial_level']['base_wrong_to_sidecar_correct']} / {errors['Relative-Sidecar']['trial_level']['base_wrong_to_sidecar_correct']}. Base-correct to newly wrong Raw / Relative: {errors['Raw-Sidecar']['trial_level']['base_correct_to_sidecar_wrong']} / {errors['Relative-Sidecar']['trial_level']['base_correct_to_sidecar_wrong']}.

Oracle Relative diagnostic BA: {mean_subject(oracle_rows):.4f} (TRANSDUCTIVE DIAGNOSTIC ONLY). Relative beat Raw: {'YES' if relative_pp>0 else 'NO'}; at least +0.3 pp: {'YES' if relative_pp>=.3 else 'NO'}; Relative beat Base by +0.5 pp: {'YES' if total_pp>=.5 else 'NO'}.

Reserved holdout accessed: NO. Claim support: {claim}. Terminal: `{terminal}`.

Recommended next action: {'repeat exact frozen comparison on folds 1 and 2 only if promising/strong' if claim=='YES' else 'stop centering / relative residual as a constructive core; do not tune the sidecar or offset estimator.'}
"""
    df.to_csv(OUT/"SUBJECT_RESULTS.csv",index=False);write(OUT/"OFFSET_TRAINING_LOG.json",offset_info);write(OUT/"RAW_SIDECAR_TRAINING_LOG.json",raw_info);write(OUT/"RELATIVE_SIDECAR_TRAINING_LOG.json",relative_info);write(OUT/"FOLD0_RESULT.json",result);write(OUT/"SUBJECT_BOOTSTRAP.json",bootstrap);write(OUT/"CORRECTION_MAGNITUDE.json",{"Raw-Sidecar":raw_mag,"Relative-Sidecar":rel_mag});write(OUT/"ERROR_CORRECTION_DIAGNOSTIC.json",errors);write(OUT/"OFFSET_ESTIMATOR_DIAGNOSTIC.json",offset_diag);write(OUT/"FEATURE_COMPLEMENTARITY.json",features);write(OUT/"ORACLE_RELATIVE_DIAGNOSTIC.json",{"TRANSDUCTIVE_DIAGNOSTIC_ONLY":True,"mean_subject_BA":mean_subject(oracle_rows),"pooled":oracle_pooled,"per_subject":oracle_rows});(OUT/"DECISION.md").write_text(decision,encoding="utf-8")
    print("OpenBMI fold0 / seed0",flush=True);print(f"Frozen EEGNet-Base {base_ba:.4f}; Raw-Sidecar {raw_ba:.4f}; Relative-Sidecar {rel_ba:.4f}",flush=True);print(f"capacity {capacity_pp:+.2f} pp; relative {relative_pp:+.2f} pp; total {total_pp:+.2f} pp",flush=True);print(terminal,flush=True);return 0


if __name__=="__main__":
    try:raise SystemExit(main())
    except Exception as exc:print(f"RELATIVE_SIDECAR_PROTOCOL_INVALID: {type(exc).__name__}: {exc}",flush=True);raise
