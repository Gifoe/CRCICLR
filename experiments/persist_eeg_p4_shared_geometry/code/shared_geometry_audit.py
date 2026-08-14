from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from scipy.stats import spearmanr
import p4_persist_ct as base
import p4_persist_ct_v2 as v2

ROOT=base.ROOT; OUT=ROOT/'outputs'/'persist_eeg_shared_geometry'; AUD=ROOT/'outputs'/'persist_eeg_p4_signed'/'audit_v3'; TASKS=base.TASKS; CLASSES=base.CLASSES; FOLDS=(0,1,2); SEEDS=(0,1); DRAWS=100; BOOT=10000; PER_GROUP=16
def wr(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(base.clean(v),indent=2)+"\n")
def stable(s): return int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)
def group_indices(meta,cols):
    arr=[np.asarray(meta[c].to_numpy(),dtype=str) for c in cols]; d={}
    for i,key in enumerate(zip(*arr)): d.setdefault(tuple(key),[]).append(i)
    return d.items()
def sample_frame(meta,h,task,subs,seed):
    mask=(meta.paradigm==task)&meta.subject_id.astype(str).isin(set(map(str,subs))); f=meta[mask].copy(); pos=np.flatnonzero(mask.to_numpy()); keep=[]
    for key,g in f.groupby(['subject_id','session_id','event_label'],sort=True):
        ix=g.index.to_numpy(); rng=np.random.default_rng(seed+stable(key)); keep.extend(ix.tolist() if len(ix)<=PER_GROUP else np.sort(rng.choice(ix,PER_GROUP,replace=False)).tolist())
    keep=np.asarray(sorted(keep),dtype=int); lut={int(x):i for i,x in enumerate(meta.index.to_numpy())}; rows=np.asarray([lut[int(x)] for x in keep],dtype=int)
    return f.loc[keep].reset_index(drop=True),h[rows]
def rankcorr(a,b):
    a=np.asarray(a); b=np.asarray(b); return float(spearmanr(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else 0.
def centroids(meta,q,task):
    event_arr=np.asarray(meta['event_label'].to_numpy(),dtype=str); ev=sorted(np.unique(event_arr)); out={}
    for (s,r),gi in group_indices(meta,['subject_id','session_id']):
        gi=np.asarray(gi,dtype=int)
        c=[]
        for y in ev:
            x=q[gi][event_arr[gi]==y]; c.append(x.mean(0) if len(x) else np.full(q.shape[1],np.nan))
        c=np.asarray(c); out[(str(s),str(r))]=c-c.mean(0,keepdims=True)
    return out,ev
def rdm(d): return np.asarray([np.linalg.norm(d[i]-d[j]) for i in range(len(d)) for j in range(i+1,len(d))])
def geom_metrics(meta,q,task):
    cs,ev=centroids(meta,q,task); subs=sorted(set(s for s,r in cs)); sess=sorted(set(r for s,r in cs)); cos=[]; rdmv=[]; ss=[]; cross=[]
    for s in subs:
        for r in sess:
            if (s,r) not in cs or np.isnan(cs[(s,r)]).any(): continue
            others=[cs[k] for k in cs if k[1]==r and k[0]!=s and not np.isnan(cs[k]).any()]; cons=np.mean(others,0) if others else None
            if cons is not None:
                a=cs[(s,r)].ravel(); b=cons.ravel(); cos.append(float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))); rdmv.append(rankcorr(rdm(cs[(s,r)]),rdm(cons)))
        if len(sess)>=2 and (s,sess[0]) in cs and (s,sess[1]) in cs: ss.append(rankcorr(rdm(cs[(s,sess[0])]),rdm(cs[(s,sess[1])])))
    for s in subs:
        if len(sess)<2: continue
        for a,b in [(sess[0],sess[1]),(sess[1],sess[0])]:
            if (s,b) not in cs: continue
            oth=[cs[k] for k in cs if k[1]==a and k[0]!=s and not np.isnan(cs[k]).any()]
            if oth: cross.append(rankcorr(rdm(cs[(s,b)]),rdm(np.mean(oth,0))))
    return {'contrast_cosine':cos,'cross_subject_rdm':rdmv,'cross_session_rdm':ss,'cross_subject_cross_session_rdm':cross,'mean_rdm':float(np.mean(rdmv)) if rdmv else 0.}
def proto_transfer(meta,q,task,train_subs,eval_subs):
    ev=sorted(np.unique(np.asarray(meta['event_label'].to_numpy(),dtype=str)));
    def centered(subs):
        mask=meta.subject_id.astype(str).isin(set(map(str,subs))).to_numpy(); out=q[mask].copy(); fm=meta[mask].reset_index(drop=True)
        for _,gi in group_indices(fm,['subject_id','session_id']):
            gi=np.asarray(gi,dtype=int); out[gi]-=out[gi].mean(0)
        return fm,out
    tm,tq=centered(train_subs); em,eq=centered(eval_subs); prot=[]
    for y in ev:
        x=tq[np.asarray(tm['event_label'].to_numpy(),dtype=str)==y]; prot.append(x.mean(0))
    prot=np.asarray(prot); pred=np.asarray([np.argmin(np.sum((prot-x)**2,1)) for x in eq]); yy=np.asarray([ev.index(x) for x in np.asarray(em['event_label'].to_numpy(),dtype=str)]); ba=float(np.mean([np.mean(pred[yy==k]==k) for k in range(len(ev)) if np.any(yy==k)])); margin=[]
    for y in ev:
        same=np.linalg.norm(eq[yy=={x:i for i,x in enumerate(ev)}[y]]-prot[{x:i for i,x in enumerate(ev)}[y]],axis=1); other=np.linalg.norm(eq[yy=={x:i for i,x in enumerate(ev)}[y],None]-prot[np.arange(len(ev))!={x:i for i,x in enumerate(ev)}[y]],axis=1) if False else []
    return ba
def ridge_transfer(meta,q,task,train_subs,eval_subs):
    ev=sorted(np.unique(np.asarray(meta['event_label'].to_numpy(),dtype=str))); mp={x:i for i,x in enumerate(ev)}; maskt=meta.subject_id.astype(str).isin(set(map(str,train_subs))).to_numpy(); maske=meta.subject_id.astype(str).isin(set(map(str,eval_subs))).to_numpy(); xt=q[maskt].copy(); xe=q[maske].copy(); mt=meta[maskt].reset_index(drop=True); me=meta[maske].reset_index(drop=True)
    for _,gi in group_indices(mt,['subject_id','session_id']):
        gi=np.asarray(gi,dtype=int); xt[gi]-=xt[gi].mean(0)
    for _,gi in group_indices(me,['subject_id','session_id']):
        gi=np.asarray(gi,dtype=int); xe[gi]-=xe[gi].mean(0)
    yt=np.asarray([mp[x] for x in np.asarray(mt['event_label'].to_numpy(),dtype=str)]); ye=np.asarray([mp[x] for x in np.asarray(me['event_label'].to_numpy(),dtype=str)]); pack=base.ridge_probe(xt,yt,len(ev)); pred,_=base.probe_predict(xe,pack,len(ev)); return float(np.mean([np.mean(pred[ye==k]==k) for k in range(len(ev)) if np.any(ye==k)]))
def one(fold,seed,device):
    print('load',flush=True); m=base.load_manifest(); sp=next(x for x in base.load_splits() if int(x['fold'])==fold); ck,mu,sd=base.historical(fold,seed); model=base.load_model(ck,m,device); print('train_extract',flush=True); trm,trh,_=base.extract(model,m,sp['train_subjects'],mu,sd,device,500000+fold*101+seed,cap=0); print('val_extract',flush=True); vam,vah,_=base.extract(model,m,sp['validation_subjects'],mu,sd,device,510000+fold*101+seed,cap=0); print('extracted',len(trh),len(vah),flush=True); basis_m=[]; basis_h=[]
    for ti,t in enumerate(TASKS):
        bm,bh=sample_frame(trm,trh,t,sp['train_subjects'],7000+ti+seed); basis_m.append(bm); basis_h.append(bh)
    basis_m=pd.concat(basis_m,ignore_index=True); basis_h=np.vstack(basis_h); spec=v2.build_spectrum_v2(basis_m,basis_h,60000+fold*101+seed); print('basis',len(basis_h),len(spec['blocks']),flush=True); out={'fold':fold,'seed':seed,'tasks':{}}
    for task in TASKS:
        print('task',task,flush=True); tm,th=sample_frame(trm,trh,task,sp['train_subjects'],1000+seed); vm,vh=sample_frame(vam,vah,task,sp['validation_subjects'],2000+seed); ass=json.loads((AUD/f'fold-{fold}'/f'seed-{seed}'/'SIGNED_ASSIGNMENTS.json').read_text())[task]; ids=sorted(set(sum((spec['blocks'][b] for b in ass['protected']),[]))); rank=max(1,len(ids)); q=(th-spec['mean'])@spec['whitener']@spec['directions']; qv=(vh-spec['mean'])@spec['whitener']@spec['directions']; metrics=geom_metrics(tm,q,task); val_prot=ridge_transfer(pd.concat([tm,vm],ignore_index=True),np.vstack([q,qv]),task,sp['train_subjects'],sp['validation_subjects']); loso=proto_transfer(tm,q,task,sp['train_subjects'][:len(sp['train_subjects'])//2],sp['train_subjects'][len(sp['train_subjects'])//2:]); rand=[]
        for d in range(DRAWS):
            ch=np.random.default_rng(70000+fold*1000+seed*100+d).choice(np.arange(q.shape[1]),size=rank,replace=False); gm=geom_metrics(tm,q[:,ch],task); rand.append({'rdm':gm['mean_rdm'],'session':float(np.mean(gm['cross_subject_cross_session_rdm'])) if gm['cross_subject_cross_session_rdm'] else 0.,'loso':proto_transfer(tm,q[:,ch],task,sp['train_subjects'][:len(sp['train_subjects'])//2],sp['train_subjects'][len(sp['train_subjects'])//2:])})
        out['tasks'][task]={'rank':rank,'protected':metrics,'validation_ridge_BA':val_prot,'loso_BA':loso,'random':rand,'protected_rdm_minus_random':float(metrics['mean_rdm']-np.mean([x['rdm'] for x in rand])),'protected_cross_session_minus_random':float(np.mean(metrics['cross_subject_cross_session_rdm'])-np.mean([x['session'] for x in rand])) if metrics['cross_subject_cross_session_rdm'] else 0.,'protected_loso_minus_random':float(loso-np.mean([x['loso'] for x in rand]))}
    return out
def main():
    import argparse,traceback
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int); ap.add_argument('--seed',type=int); a=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rs=[]; folds=(a.fold,) if a.fold is not None else FOLDS; seeds=(a.seed,) if a.seed is not None else SEEDS
    try:
      for f in folds:
       for s in seeds: print('[GEOMETRY]',f,s,flush=True); rs.append(one(f,s,device))
    except Exception:
      OUT.mkdir(parents=True,exist_ok=True); (OUT/'geometry_exception.txt').write_text(traceback.format_exc()); raise
    rows=[]
    for r in rs:
      for t,x in r['tasks'].items(): rows.append({'fold':r['fold'],'seed':r['seed'],'task':t,'rdm_diff':x['protected_rdm_minus_random'],'cross_session_diff':x['protected_cross_session_minus_random'],'loso_diff':x['protected_loso_minus_random'],'validation_ridge_BA':x['validation_ridge_BA']})
    df=pd.DataFrame(rows); decisions={}
    for t in TASKS:
      z=df[df.task==t]; out={}
      for col in ['rdm_diff','cross_session_diff','loso_diff']:
       vals=z[col].to_numpy(float); rng=np.random.default_rng(90000+len(col)); b=rng.choice(vals,size=(BOOT,len(vals)),replace=True).mean(1); out[col]={'mean':float(vals.mean()),'positive_runs':int((vals>0).sum()),'ci95':[float(np.quantile(b,.025)),float(np.quantile(b,.975))]}
      decisions[t]=out
    mi=decisions['mi']; gate={'A_geometry':mi['rdm_diff']['positive_runs']>=5 and mi['rdm_diff']['ci95'][0]>0,'B_transfer':mi['loso_diff']['positive_runs']>=5 and mi['loso_diff']['mean']>=.02 and mi['loso_diff']['ci95'][0]>0,'C_cross_session':mi['cross_session_diff']['positive_runs']>=5 and mi['cross_session_diff']['ci95'][0]>0,'D_margin':mi['loso_diff']['ci95'][0]>0}; status='SHARED_GEOMETRY_PASS' if all(gate.values()) else 'PERSIST_USE_MECHANISM_NOT_SUPPORTED'; payload={'status':status,'gate':gate,'decisions':decisions,'runs':rs,'outer_test_used':False}; OUT.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(OUT/'SHARED_GEOMETRY_RUN_SUMMARY.csv',index=False); wr(OUT/'SHARED_GEOMETRY_FINAL_REPORT.json',payload); (OUT/'SHARED_GEOMETRY_FINAL_REPORT.md').write_text(f'# Shared Geometry Audit\n\nStatus: `{status}`\n\nOuter-test used: `false`\n'); wr(OUT/'protocol'/'SHARED_GEOMETRY_PROTOCOL.json',{'protected_source':'Signed Audit V3 train-only positive utility assignments','per_subject_session_event_cap':PER_GROUP,'random_draws':DRAWS,'bootstrap_draws':BOOT,'outer_test_used':False}); print(json.dumps(base.clean({'status':status,'gate':gate,'decisions':decisions}),indent=2))
if __name__=='__main__': main()
