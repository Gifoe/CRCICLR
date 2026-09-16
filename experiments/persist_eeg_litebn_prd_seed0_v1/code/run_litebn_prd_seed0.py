#!/usr/bin/env python3
"""Matched seed-0 LiteBN CE versus CE+PRD on OpenBMI/WBCIC MI."""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO = Path(os.environ.get("PRD_REPO", "/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_prd_seed0_v1"
OUT, PROTOCOL, RUN = EXP / "outputs", EXP / "protocol", EXP / "runtime"
BASE_RUN = REPO / "experiments/persist_eeg_litebn_ablation_v1/code/run_litebn_ablation.py"
CSGD_RUN = REPO / "experiments/persist_eeg_litebn_tfformer_csgd_v1/code/run_csgd.py"
PRD_CODE = REPO / "experiments/persist_eeg_eegnet_prd_fold0_v1/code"
sys.path.insert(0, str(PRD_CODE))
from prd_loss import prd_loss  # noqa: E402

TASKS=("OpenBMI_MI","WBCIC_MI"); CONDITIONS=("CE","CE_PLUS_PRD"); FOLDS=range(5)
LAMBDA=.25; MAX_EPOCHS=60; MIN_EPOCH=10; PATIENCE=8; LR=3e-4; WD=5e-4; CLIP=5.; BOOT=20_000


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path); obj=importlib.util.module_from_spec(spec); sys.modules[name]=obj; spec.loader.exec_module(obj); return obj
def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.part'); tmp.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n'); os.replace(tmp,path)
def atomic_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.part'); pd.DataFrame(rows).to_csv(tmp,index=False); os.replace(tmp,path)
def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.part'); tmp.write_text(text.rstrip()+'\n'); os.replace(tmp,path)
def seed(v=0):
    random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def stable(*parts): return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'little')%(2**63-1)
def hash_state(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        x=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def make_manifest(bundle,fold,task):
    subjects=list(map(str,fold["inner_train_subjects"])); source=base_global.TASKS[task]["source_sessions"]
    pools={}
    for s in subjects:
        ids=bundle.indices([s],source); y=bundle.labels(ids); pools[s]={c:np.asarray(ids)[y==c] for c in (0,1)}
        if min(map(len,pools[s].values()))<8: raise RuntimeError(f"insufficient balanced trials {task}/{s}")
    manifest=[]
    for epoch in range(1,MAX_EPOCHS+1):
        rng=np.random.default_rng(stable('litebn-prd',task,fold['fold_id'],epoch)); episodes=[]
        for step in range(20):
            order=rng.permutation(subjects); support=list(order[:4]); query=list(order[4:8])
            def draw(group):
                indices=[]; slots=[]
                for slot,s in enumerate(group):
                    chosen=np.concatenate([rng.choice(pools[s][0],8,replace=False),rng.choice(pools[s][1],8,replace=False)]);rng.shuffle(chosen)
                    indices.extend(map(int,chosen));slots.extend([slot]*16)
                return indices,slots
            si,ss=draw(support);qi,qs=draw(query);episodes.append({'support_indices':si,'support_slots':ss,'query_indices':qi,'query_slots':qs,'support_subjects':support,'query_subjects':query})
        manifest.append(episodes)
    return manifest


def eval_val(model,bundle,subjects,cache,session):
    model.eval(); values=[]
    with torch.inference_mode():
        for s in subjects:
            ids=bundle.indices([str(s)],[session]); logits=[]
            for start in range(0,len(ids),128): x,_=cache.batch(ids[start:start+128]);logits.append(model(x)[0].float().cpu().numpy())
            values.append(base_global.classification_metrics(bundle.labels(ids),np.concatenate(logits))["BA"])
    return float(np.mean(values))


def train_one(task,fold,bundle,cache,manifest,condition,initial,device):
    f=int(fold['fold_id']); directory=RUN/'checkpoints'/task/condition/f'fold{f}'; selected=directory/'selected.pt'
    if selected.is_file(): return selected
    model=base_global.build_model('LiteBN_BASELINE',task).to(device);model.load_state_dict(initial,strict=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD);scaler=torch.amp.GradScaler('cuda',enabled=True)
    best=-float('inf');best_epoch=None;best_state=None;wait=0;history=[];started=time.perf_counter()
    for epoch,episodes in enumerate(manifest,1):
        model.train(); ces=[];prds=[];totals=[]
        for ep in episodes:
            ids=np.asarray(ep['support_indices']+ep['query_indices'],np.int64);x,y=cache.batch(ids);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16):
                logits,z=model(x);ce=F.cross_entropy(logits,y);pr=torch.zeros((),device=device)
                if condition=='CE_PLUS_PRD':
                    ss=torch.tensor(ep['support_slots'],device=device);qs=torch.tensor(ep['query_slots'],device=device)
                    pr,_=prd_loss(z[:64],y[:64],ss,z[64:],y[64:],qs,num_classes=2)
                loss=ce+LAMBDA*pr
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP);scaler.step(optimizer);scaler.update()
            ces.append(float(ce.detach().cpu()));prds.append(float(pr.detach().cpu()));totals.append(float(loss.detach().cpu()))
        ba=eval_val(model,bundle,fold['inner_val_subjects'],cache,int(base_global.TASKS[task]['future_session']))
        eligible=epoch>=MIN_EPOCH; improved=eligible and ba>best+1e-12
        if improved: best=ba;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};wait=0
        elif eligible: wait+=1
        history.append({'epoch':epoch,'CE':np.mean(ces),'raw_PRD':np.mean(prds),'lambda_times_PRD':LAMBDA*np.mean(prds),'total':np.mean(totals),'inner_val_subject_BA':ba,'selected':improved})
        if epoch==1 or epoch%5==0 or improved: print(f"PRDTRAIN {task} {condition} f{f} e{epoch:02d} CE={np.mean(ces):.5f} PRD={np.mean(prds):.5f} BA={ba:.5f}",flush=True)
        if eligible and wait>=PATIENCE: break
    if best_state is None: raise RuntimeError('no eligible checkpoint')
    directory.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':best_state,'condition':condition,'selected_epoch':best_epoch,'inner_val_BA':best,'history':history,'initial_hash':hash_state(initial),'manifest_hash':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),'elapsed_seconds':time.perf_counter()-started},selected)
    del model;gc.collect();torch.cuda.empty_cache();return selected


def infer(model,bundle,raw,mean,std,subjects,sessions):
    model.eval();rows=[];emb={}
    with torch.inference_mode():
        for s in subjects:
            emb[s]={}
            for session in sessions:
                ids=bundle.indices([s],[session]); logits=[];zs=[]
                for start in range(0,len(ids),128): x,_=raw.batch(ids[start:start+128],mean,std);l,z=model(x);logits.append(l.float().cpu().numpy());zs.append(z.float().cpu().numpy())
                l=np.concatenate(logits);z=np.concatenate(zs);y=bundle.labels(ids);m=base_global.classification_metrics(y,l)
                rows.append({'subject_id':s,'session':f'S{session}','BA':m['BA'],'macro_F1':m['macro_F1'],'trials':len(ids)})
                emb[s][session]={'y':y,'z':z}
    return rows,emb
def direction(rec):
    z,y=rec['z'],rec['y'];d=z[y==1].mean(0)-z[y==0].mean(0);return d/max(np.linalg.norm(d),1e-12)


def evaluate_all(csgd,runtime,paths,device):
    rows=[];align=[]
    for task in TASKS:
        subjects,sessions=csgd.subjects_and_sessions(task);bundle=csgd.build_wbcic_outer_bundle(runtime)[0] if task=='WBCIC_MI' else base_global.build_bundle(task,subjects);raw=base_global.RawGPUCache(bundle,device)
        for fold in FOLDS:
            mean,std,_=base_global.load_tensor_pair(csgd.normalizer_path(task,fold))
            for condition in CONDITIONS:
                path=paths[(task,fold,condition)];payload=torch.load(path,map_location='cpu',weights_only=False);model=base_global.build_model('LiteBN_BASELINE',task);model.load_state_dict(payload['state_dict'],strict=True);model=model.to(device).eval()
                current,emb=infer(model,bundle,raw,mean,std,subjects,sessions)
                for r in current:r.update({'task':task,'fold':fold,'seed':0,'condition':condition,'checkpoint':str(path)})
                rows.extend(current)
                for s in subjects:
                    src=direction(emb[s][sessions[0]]) if task!='WBCIC_MI' else (direction(emb[s][0])+direction(emb[s][1]));src=src/max(np.linalg.norm(src),1e-12);future=direction(emb[s][2]);align.append({'task':task,'fold':fold,'condition':condition,'subject_id':s,'alignment':float(src@future)})
                del model;gc.collect();torch.cuda.empty_cache()
        del raw,bundle;gc.collect();torch.cuda.empty_cache()
    return rows,align
def bootstrap(v,key):
    v=np.asarray(v,float);rng=np.random.default_rng(stable('boot',key));means=np.empty(BOOT)
    for st in range(0,BOOT,2000):sp=min(BOOT,st+2000);idx=rng.integers(0,len(v),(sp-st,len(v)));means[st:sp]=v[idx].mean(1)
    lo,hi=np.quantile(means,[.025,.975]);return float(v.mean()),float(lo),float(hi)
def summarize(rows,align):
    frame=pd.DataFrame(rows);session=frame.groupby(['task','condition','subject_id','session'],as_index=False).agg(BA=('BA','mean'),macro_F1=('macro_F1','mean'),folds=('fold','nunique'))
    af=pd.DataFrame(align).groupby(['task','condition','subject_id'],as_index=False).alignment.mean();subjects=[]
    for (task,cond,sid),p in session.groupby(['task','condition','subject_id']):
        cells={r.session:r for r in p.itertuples(index=False)};required=('S0','S1','S2') if task=='WBCIC_MI' else ('S1','S2');subjects.append({'task':task,'condition':cond,'subject_id':sid,'future_BA':cells['S2'].BA,'future_macro_F1':cells['S2'].macro_F1,'WS_BA':min(cells[x].BA for x in required),'alignment':float(af[(af.task==task)&(af.condition==cond)&(af.subject_id==sid)].alignment.iloc[0])})
    sf=pd.DataFrame(subjects);summary=[]
    for task in TASKS:
        for cond in CONDITIONS:
            p=sf[(sf.task==task)&(sf.condition==cond)];summary.append({'task':task,'condition':cond,'subjects':len(p),'alignment':p.alignment.mean(),'future_BA':p.future_BA.mean(),'future_macro_F1':p.future_macro_F1.mean(),'WS_BA':p.WS_BA.mean()})
        a=sf[(sf.task==task)&(sf.condition=='CE')].set_index('subject_id');b=sf[(sf.task==task)&(sf.condition=='CE_PLUS_PRD')].set_index('subject_id');ids=sorted(set(a.index)&set(b.index));row={'task':task,'subjects':len(ids)}
        for metric in ('alignment','future_BA','future_macro_F1','WS_BA'):
            scale=1 if metric=='alignment' else 100;mean,lo,hi=bootstrap(scale*(b.loc[ids,metric].to_numpy()-a.loc[ids,metric].to_numpy()),f'{task}/{metric}');row[f'delta_{metric}']=mean;row[f'delta_{metric}_CI95_low']=lo;row[f'delta_{metric}_CI95_high']=hi
        summary.append({'task':task,'condition':'DELTA_PRD_MINUS_CE',**row})
    return session.to_dict('records'),subjects,summary


def main():
    global base_global,base_runner
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    base_runner=module('prd_base_runner',BASE_RUN);base_global=base_runner.base;csgd=module('prd_csgd',CSGD_RUN);runtime=csgd.load_runtime();device=torch.device('cuda');_,fold_map,split_sha=base_global.load_folds();paths={};audit=[]
    for task in TASKS:
        dataset=base_global.TASKS[task]['dataset']
        for fold in fold_map[dataset]:
            f=int(fold['fold_id']);bundle=base_global.build_bundle(task,fold['inner_train_subjects']+fold['inner_val_subjects']);mean,std,meta=base_global.load_tensor_pair(csgd.normalizer_path(task,f));raw=base_global.RawGPUCache(bundle,device);cache=base_runner.NormalizedCache(raw,mean,std);manifest=make_manifest(bundle,fold,task);seed(0);template=base_global.build_model('LiteBN_BASELINE',task);initial={k:v.detach().cpu().clone() for k,v in template.state_dict().items()};initial_hash=hash_state(initial);manifest_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            for condition in CONDITIONS:
                seed(0);path=train_one(task,fold,bundle,cache,manifest,condition,initial,device);paths[(task,f,condition)]=path;payload=torch.load(path,map_location='cpu',weights_only=False);audit.append({'task':task,'fold':f,'condition':condition,'initial_hash':payload['initial_hash'],'manifest_hash':payload['manifest_hash'],'selected_epoch':payload['selected_epoch'],'inner_val_BA':payload['inner_val_BA'],'checkpoint':str(path)})
            if audit[-1]['initial_hash']!=audit[-2]['initial_hash'] or audit[-1]['manifest_hash']!=audit[-2]['manifest_hash']:raise RuntimeError('matched invariant failure')
            del cache,raw,bundle;gc.collect();torch.cuda.empty_cache()
    print('PRD_ALL_CHECKPOINTS_FROZEN',flush=True);rows,align=evaluate_all(csgd,runtime,paths,device);session,subjects,summary=summarize(rows,align)
    atomic_csv(OUT/'SESSION_RESULTS.csv',session);atomic_csv(OUT/'SUBJECT_RESULTS.csv',subjects);atomic_csv(OUT/'TASK_SUMMARY.csv',summary);atomic_csv(PROTOCOL/'MATCHING_AUDIT.csv',audit)
    atomic_json(PROTOCOL/'PROTOCOL.json',{'seed':0,'tasks':TASKS,'folds':list(FOLDS),'conditions':CONDITIONS,'architecture':'exact final LiteBN','lambda':LAMBDA,'lambda_source':str(PRD_CODE/'run_fold0.py'),'prd_definition':str(PRD_CODE/'prd_loss.py'),'optimizer':'AdamW','lr':LR,'weight_decay':WD,'gradient_clip':CLIP,'max_epochs':MAX_EPOCHS,'min_checkpoint_epoch':MIN_EPOCH,'patience':PATIENCE,'checkpoint_metric':'subject-equal inner-validation BA','identical_initialization':True,'identical_CE_batches':True,'hyperparameter_search':False,'heldout_used_for_training_or_selection':False})
    lines=['# LiteBN CE versus CE+PRD seed-0','',f'PRD lambda={LAMBDA} was reused without tuning.','', '| Task | Condition | Alignment | Future BA | WS-BA |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['condition']!='DELTA_PRD_MINUS_CE':lines.append(f"| {r['task']} | {r['condition']} | {r['alignment']:.4f} | {r['future_BA']:.4f} | {r['WS_BA']:.4f} |")
    lines += ['', '| Task | Delta alignment | Delta future BA pp [95% CI] | Delta WS-BA pp [95% CI] |','|---|---:|---:|---:|']
    for r in summary:
        if r['condition']=='DELTA_PRD_MINUS_CE':lines.append(f"| {r['task']} | {r['delta_alignment']:+.4f} [{r['delta_alignment_CI95_low']:+.4f}, {r['delta_alignment_CI95_high']:+.4f}] | {r['delta_future_BA']:+.3f} [{r['delta_future_BA_CI95_low']:+.3f}, {r['delta_future_BA_CI95_high']:+.3f}] | {r['delta_WS_BA']:+.3f} [{r['delta_WS_BA_CI95_low']:+.3f}, {r['delta_WS_BA_CI95_high']:+.3f}] |")
    atomic_text(OUT/'FINAL_LITEBN_PRD_SEED0_REPORT.md','\n'.join(lines));atomic_json(OUT/'COMPLETION.json',{'status':'COMPLETE','training_cells':20,'session_rows':len(session),'subject_rows':len(subjects)})
    print('LITEBN_PRD_COMPLETE',flush=True)


if __name__=='__main__':main()
