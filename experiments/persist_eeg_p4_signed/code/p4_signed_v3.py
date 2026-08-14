from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import p4_persist_ct as base
import p4_persist_ct_v2 as v2

ROOT=base.ROOT; OUT=ROOT/'outputs'/'persist_eeg_p4_signed'; TASKS=base.TASKS; CLASSES=base.CLASSES
FOLDS=(0,1,2); SEEDS=(0,1); INNER=5; DRAWS=100; BOOT=10000; EPS=0.005

def write_json(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(base.clean(v),indent=2)+"\n",encoding='utf-8')
def risk(Xf,yf,Xe,ye,c):
    pack=base.ridge_probe(np.asarray(Xf),yf,c); pred,prob=base.probe_predict(np.asarray(Xe),pack,c); yy=np.asarray(ye,dtype=int); pp=np.asarray(prob,float)
    ce=float(-np.mean(np.log(np.clip(pp[np.arange(len(yy)),yy],1e-12,1.0))))
    ba=float(np.mean([np.mean(pred[yy==k]==k) for k in range(c) if np.any(yy==k)]))
    return ce,ba
def eval_pack(pack,Xe,ye,c):
    pred,prob=base.probe_predict(np.asarray(Xe),pack,c); yy=np.asarray(ye,dtype=int); pp=np.asarray(prob,float)
    ce=float(-np.mean(np.log(np.clip(pp[np.arange(len(yy)),yy],1e-12,1.0))))
    ba=float(np.mean([np.mean(pred[yy==k]==k) for k in range(c) if np.any(yy==k)]))
    return ce,ba
def boot(vals,seed):
    vals=np.asarray(vals,float)
    if len(vals)==0:return {'mean':None,'ci95':[None,None],'sign_probability':None,'draws':BOOT,'n_unique_subjects':0}
    rng=np.random.default_rng(seed); d=rng.choice(vals,size=(BOOT,len(vals)),replace=True).mean(1)
    return {'mean':float(vals.mean()),'ci95':[float(np.quantile(d,.025)),float(np.quantile(d,.975))],'sign_probability':float(np.mean(d>0)),'draws':BOOT,'n_unique_subjects':int(len(vals))}
def all_idx(meta,task,subs): return np.flatnonzero((meta.paradigm==task).to_numpy() & meta.subject_id.astype(str).isin(set(map(str,subs))).to_numpy())
def audit_idx(meta,task,subs,seed,per_group=32):
    frame=meta[(meta.paradigm==task)&meta.subject_id.astype(str).isin(set(map(str,subs)))]
    out=[]
    for key,g in frame.groupby(['subject_id','session_id','event_label'],sort=True):
        ix=g.index.to_numpy(dtype=np.int64); rng=np.random.default_rng(seed+int(abs(hash(str(key)))%1000003))
        if len(ix)>per_group: ix=np.sort(rng.choice(ix,size=per_group,replace=False))
        out.extend(ix.tolist())
    return np.asarray(sorted(out),dtype=np.int64)
def run(fold,seed,device):
    m=base.load_manifest(); split=next(x for x in base.load_splits() if int(x['fold'])==fold); ck,mean,std=base.historical(fold,seed); model=base.load_model(ck,m,device)
    trm,trh,try_=base.extract(model,m,split['train_subjects'],mean,std,device,190000+fold*101+seed,cap=0); vam,vah,vay=base.extract(model,m,split['validation_subjects'],mean,std,device,200000+fold*101+seed,cap=0)
    spec=v2.build_spectrum_v2(trm,trh,30000+fold*101+seed); subjects=v2.subject_sort(split['train_subjects']); rows=[]; assignments={}
    for task in TASKS:
      task_rows=[]
      for bi,block in enumerate(spec['blocks']):
        abs_by={}; spec_by={}; ba_by={}; rand_ba_by={}; split_obs=0; supported=bool(spec['audit']['persistence_support'][bi]['persistence_supported'])
        for inner in range(INNER):
          fit,ev=v2.split_subjects(subjects,inner,seed); fi=audit_idx(trm,task,fit,31000+inner+seed*100); ei=audit_idx(trm,task,ev,32000+inner+seed*100)
          yf=v2.labels(trm,task,fi); ye=v2.labels(trm,task,ei); cand=np.setdiff1d(np.arange(len(spec['rho'])),np.asarray(block)); rng=np.random.default_rng(41000+fold*1000+seed*100+inner*10+bi); choices=[rng.choice(cand if len(cand)>=len(block) else np.arange(len(spec['rho'])),size=len(block),replace=False) for _ in range(DRAWS)]
          em=trm.iloc[ei].reset_index(drop=True); base_fit=trh[fi]; base_risk,_=risk(base_fit,yf,trh[ei],ye,CLASSES[task])
          erased_fit=base.erase(base_fit,spec,block); erased_eval=base.erase(trh[ei],spec,block)
          random_packs=[base.ridge_probe(base.erase(base_fit,spec,ch),yf,CLASSES[task]) for ch in choices]
          for subj,g in em.groupby(em.subject_id.astype(str),sort=True):
            loc=g.index.to_numpy();
            if len(loc)<2: continue
            rr,bb=risk(base_fit,yf,trh[ei][loc],ye[loc],CLASSES[task]); re,be=risk(erased_fit,yf,erased_eval[loc],ye[loc],CLASSES[task]); rs=[]; rba=[]
            for pack,ch in zip(random_packs,choices):
              rv=base.erase(trh[ei][loc],spec,ch); rc,rb=eval_pack(pack,rv,ye[loc],CLASSES[task]); rs.append(rc-rr); rba.append(rb-bb)
            uabs=re-rr; uspec=uabs-float(np.mean(rs)); s=str(subj); abs_by.setdefault(s,[]).append(float(uabs)); spec_by.setdefault(s,[]).append(float(uspec)); ba_by.setdefault(s,[]).append(float(be-bb)); rand_ba_by.setdefault(s,[]).append(float(np.mean(rba))); split_obs+=1
          
        abs_u={s:float(np.mean(v)) for s,v in abs_by.items()}; sp_u={s:float(np.mean(v)) for s,v in spec_by.items()}; ba_u={s:float(np.mean(v)) for s,v in ba_by.items()}; rb_u={s:float(np.mean(v)) for s,v in rand_ba_by.items()}; ab=boot(list(abs_u.values()),70000+fold*101+seed*11+bi); sb=boot(list(sp_u.values()),71000+fold*101+seed*11+bi)
        task_rows.append({'fold':fold,'seed':seed,'task':task,'block':bi,'dimensions':len(block),'persistence_supported':supported,'n_unique_subjects':len(abs_u),'n_split_observations':split_obs,'bootstrap_hierarchy':'aggregate_unique_subject_across_inner_splits_then_subject_bootstrap','u_abs_mean':ab['mean'],'u_abs_CI95':ab['ci95'],'u_abs_sign_probability':ab['sign_probability'],'u_spec_mean':sb['mean'],'u_spec_CI95':sb['ci95'],'u_spec_sign_probability':sb['sign_probability'],'raw_BA_change':float(np.mean(list(ba_u.values()))) if ba_u else None,'same_rank_random_BA_change':float(np.mean(list(rb_u.values()))) if rb_u else None,'u_abs_bootstrap':ab,'u_spec_bootstrap':sb,'random_interventions':DRAWS})
      prot=[r['block'] for r in task_rows if r['persistence_supported'] and r['u_abs_CI95'][0] is not None and r['u_abs_CI95'][0]>0 and r['u_spec_CI95'][0]>0]
      harm=[r['block'] for r in task_rows if r['persistence_supported'] and r['u_abs_CI95'][1] is not None and r['u_abs_CI95'][1]<0 and r['u_spec_CI95'][1]<0]
      neutral=[r['block'] for r in task_rows if r['persistence_supported'] and r['u_abs_CI95'][0] is not None and r['u_abs_CI95'][0]>=-EPS and r['u_abs_CI95'][1]<=EPS]
      assignments[task]={'protected':prot,'harmful':harm,'neutral':neutral,'uncertain':[r['block'] for r in task_rows if r['block'] not in prot+harm+neutral]}; rows.extend(task_rows)
    vrows=[]
    for task in TASKS:
      ti=audit_idx(trm,task,split['train_subjects'],33000+seed); vi=audit_idx(vam,task,split['validation_subjects'],34000+seed); ytr=v2.labels(trm,task,ti); yv=v2.labels(vam,task,vi); raw_train=trh[ti]; raw_val=vah[vi]; _,raw_ba=risk(raw_train,ytr,raw_val,yv,CLASSES[task]); blocks_to_test=[('protected',b) for b in assignments[task]['protected']]+[('harmful',b) for b in assignments[task]['harmful']]+[('neutral',b) for b in assignments[task]['neutral']]
      for kind,bi in blocks_to_test:
        ids=spec['blocks'][bi]; _,b=risk(base.erase(raw_train,spec,ids),ytr,base.erase(raw_val,spec,ids),yv,CLASSES[task]); vrows.append({'fold':fold,'seed':seed,'task':task,'kind':kind,'block':bi,'validation_gain_BA':float(b-raw_ba),'raw_BA':float(raw_ba)})
      for kind,bl in [('protected_union',assignments[task]['protected']),('harmful_union',assignments[task]['harmful'])]:
        ids=sorted(set(sum((spec['blocks'][b] for b in bl),[])))
        if ids: _,b=risk(base.erase(raw_train,spec,ids),ytr,base.erase(raw_val,spec,ids),yv,CLASSES[task]); vrows.append({'fold':fold,'seed':seed,'task':task,'kind':kind,'block':'union','validation_gain_BA':float(b-raw_ba),'raw_BA':float(raw_ba)})
      for label,rank in [('same_total',len(set(sum((spec['blocks'][b] for b in assignments[task]['harmful']),[])))),('same_block',len(spec['blocks'][0]))]:
        if rank<=0: continue
        rr=[]; rng=np.random.default_rng(81000+fold*100+seed+len(label))
        for _ in range(DRAWS):
          ch=rng.choice(np.arange(len(spec['rho'])),size=min(rank,len(spec['rho'])),replace=False); _,b=risk(base.erase(raw_train,spec,ch),ytr,base.erase(raw_val,spec,ch),yv,CLASSES[task]); rr.append(b-raw_ba)
        vrows.append({'fold':fold,'seed':seed,'task':task,'kind':label,'block':'random','validation_gain_BA':float(np.mean(rr)),'raw_BA':float(raw_ba)})
    result={'fold':fold,'seed':seed,'rows':rows,'assignments':assignments,'validation':vrows,'audit_sampling':{'per_subject_session_event_cap':32,'deterministic':True},'outer_test_used':False}
    d=OUT/'audit_v3'/f'fold-{fold}'/f'seed-{seed}'; d.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(d/'SIGNED_UTILITY_V3.csv',index=False); pd.DataFrame(vrows).to_csv(d/'VALIDATION_SIGN_TRANSFER.csv',index=False); write_json(d/'SIGNED_ASSIGNMENTS.json',assignments); write_json(d/'SIGNED_AUDIT_RUN.json',result); return result

def baseline(fold,seed,device):
    m=base.load_manifest(); split=next(x for x in base.load_splits() if int(x['fold'])==fold); ck,mean,std=base.historical(fold,seed); model=base.load_model(ck,m,device); vm,vh,vy=base.extract(model,m,split['validation_subjects'],mean,std,device,220000+fold*101+seed,cap=0); out={'fold':fold,'seed':seed,'historical_checkpoint':str(ck),'metrics':{}}
    for t in TASKS:
      vi=all_idx(vm,t,split['validation_subjects']); h=vah=vh[vi]; y=v2.labels(vm,t,vi); logits=model.heads[t](torch.as_tensor(h,device=device)).detach().cpu().numpy(); pred=logits.argmax(1); out['metrics'][t]={'BA':float(np.mean([np.mean(pred[y==k]==k) for k in range(CLASSES[t]) if np.any(y==k)])),'n':int(len(y))}
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--baseline-only',action='store_true'); ap.add_argument('--fold',type=int); ap.add_argument('--seed',type=int); a=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); OUT.mkdir(parents=True,exist_ok=True); bs=[]; folds=(a.fold,) if a.fold is not None else FOLDS; seeds=(a.seed,) if a.seed is not None else SEEDS
    for f in folds:
      for s in seeds: print('[SIGNED] baseline',f,s,flush=True); bs.append(baseline(f,s,device))
    write_json(OUT/'baseline'/'BASELINE_REPRODUCTION.json',{'runs':bs,'outer_test_used':False})
    if a.baseline_only:return
    rs=[]
    for f in folds:
      for s in seeds: print('[SIGNED] audit',f,s,flush=True); rs.append(run(f,s,device))
    rows=[]
    for r in rs:
      for t in TASKS:
        vr=[x for x in r['validation'] if x['task']==t]; hu=[x['validation_gain_BA'] for x in vr if x['kind']=='harmful_union']; pr=[x['validation_gain_BA'] for x in vr if x['kind']=='protected_union']; rnd=[x['validation_gain_BA'] for x in vr if x['kind']=='same_total']; rows.append({'fold':r['fold'],'seed':r['seed'],'task':t,'harmful_blocks':len(r['assignments'][t]['harmful']),'harmful_union_gain':float(hu[0]) if hu else 0.,'protected_union_loss':float(-pr[0]) if pr else 0.,'same_total_random_gain':float(rnd[0]) if rnd else 0.})
    df=pd.DataFrame(rows); decisions={}
    for t in TASKS:
      q=df[df.task==t]; vals=q.harmful_union_gain.to_numpy(float); rng=np.random.default_rng(99000+len(t)); b=rng.choice(vals,size=(BOOT,len(vals)),replace=True).mean(1) if len(vals) else np.array([0.]); decisions[t]={'harmful_identified_ge4':int((q.harmful_blocks>0).sum())>=4,'positive_union_ge4':int((q.harmful_union_gain>0).sum())>=4,'mean_gain_ge005':float(vals.mean())>=.005 if len(vals) else False,'bootstrap_lcb_gt0':float(np.quantile(b,.025))>0,'exceeds_random':float(q.harmful_union_gain.mean())>float(q.same_total_random_gain.mean()),'protected_erasure_harmful':float(q.protected_union_loss.mean())>0,'mean_gain':float(vals.mean()) if len(vals) else 0.,'bootstrap_ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
    go={t:decisions[t]['harmful_identified_ge4'] and decisions[t]['positive_union_ge4'] and decisions[t]['mean_gain_ge005'] and decisions[t]['bootstrap_lcb_gt0'] and decisions[t]['exceeds_random'] and decisions[t]['protected_erasure_harmful'] for t in TASKS}; status='P4_SIGNED_HEADROOM_PASS' if any(go.values()) else 'P4_SIGNED_PERSISTENCE_HAS_NO_ACTIONABLE_HEADROOM'; payload={'status':status,'decisions':decisions,'task_go':go,'rows':rows,'outer_test_used':False}; pd.DataFrame(rows).to_csv(OUT/'audit_v3'/'SIGNED_UTILITY_V3_SUMMARY.csv',index=False); write_json(OUT/'audit_v3'/'SIGNED_AUDIT_REPORT.json',payload); write_json(OUT/'audit_v3'/'SIGNED_AUDIT_DECISION.json',payload); (OUT/'audit_v3'/'SIGNED_AUDIT_DECISION.md').write_text(f'# Signed Audit V3\n\nDecision: `{status}`\n\nOuter-test used: `false`\n',encoding='utf-8'); write_json(OUT/'protocol'/'P4_SIGNED_PROTOCOL.json',{'inner_splits':INNER,'random_draws':DRAWS,'bootstrap_draws':BOOT,'epsilon_neutral':EPS,'full_128D_ridge':True,'erp_sampling':{'per_subject_session_event_cap':32,'deterministic':True,'same_for_control_and_method':True},'outer_test_used':False}); print(json.dumps(base.clean(payload),indent=2))
if __name__=='__main__':main()
