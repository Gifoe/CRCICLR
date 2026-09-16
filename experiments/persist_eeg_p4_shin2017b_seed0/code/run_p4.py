"""Independent seed-0 P4 Shin2017B repeated-session replication."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, os, random, shutil, sys, time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold

ROOT = Path(os.environ.get("P4_ROOT", "/root/p4_shin2017b_seed0"))
CACHE = Path(os.environ.get("SHIN2017B_CACHE", "/root/shin2017b_eeg_cache"))
SESSIONS = ("1arithmetic", "3arithmetic", "5arithmetic")
SEED, EPOCHS, MIN_EPOCH, BATCH, LR, WD, CLIP = 0, 60, 10, 128, 3e-4, 5e-4, 5.0
MODELS = ("EEGNet", "SIRE-EEG")

class EEGNet(nn.Module):
    """Verbatim frozen EEGNet architecture; only C/T/classes are adapted."""
    def __init__(self, channels, samples, classes=2):
        super().__init__(); self.temporal=nn.Conv2d(1,8,(1,64),padding="same",bias=False); self.bn1=nn.BatchNorm2d(8)
        self.spatial=nn.Conv2d(8,16,(channels,1),groups=8,bias=False); self.bn2=nn.BatchNorm2d(16)
        self.pool1,self.drop1=nn.AvgPool2d((1,4)),nn.Dropout(.25); self.depth=nn.Conv2d(16,16,(1,16),padding="same",groups=16,bias=False)
        self.point,self.bn3=nn.Conv2d(16,16,1,bias=False),nn.BatchNorm2d(16); self.pool2,self.drop2=nn.AvgPool2d((1,8)),nn.Dropout(.25)
        self.embedding=nn.Sequential(nn.Linear(16*(samples//4//8),64),nn.ELU(),nn.LayerNorm(64)); self.head=nn.Linear(64,classes)
    def forward(self,x):
        v=x.unsqueeze(1); v=self.bn1(self.temporal(v)); v=self.drop1(self.pool1(F.elu(self.bn2(self.spatial(v)))))
        v=self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(v)))))); return self.head(self.embedding(v.flatten(1)))

class LiteBN(nn.Module):
    """Verbatim current final LiteBN implementation, reported as SIRE-EEG."""
    def __init__(self, channels, classes=2):
        super().__init__(); self.temporal=nn.ModuleList([nn.Conv2d(1,8,(1,k),padding="same",bias=False) for k in (15,63,127)])
        self.temporal_norm=nn.ModuleList([nn.BatchNorm2d(8) for _ in range(3)]); self.spatial=nn.ModuleList([nn.Conv2d(8,16,(channels,1),groups=8,bias=False) for _ in range(3)])
        self.spatial_norm=nn.ModuleList([nn.BatchNorm2d(16) for _ in range(3)]); self.depth1,self.point1,self.norm1=nn.Conv2d(48,48,(1,9),padding="same",groups=48,bias=False),nn.Conv2d(48,32,1,bias=False),nn.BatchNorm2d(32)
        self.depth2,self.point2,self.norm2=nn.Conv2d(32,32,(1,7),padding="same",groups=32,bias=False),nn.Conv2d(32,32,1,bias=False),nn.BatchNorm2d(32)
        self.pool=nn.AdaptiveAvgPool2d((1,4)); self.embedding=nn.Sequential(nn.Linear(128,64),nn.ELU(),nn.LayerNorm(64)); self.drop,self.head=nn.Dropout(.25),nn.Linear(64,classes)
    def forward(self,x):
        v=x.unsqueeze(1); branches=[]
        for temporal,tnorm,spatial,snorm in zip(self.temporal,self.temporal_norm,self.spatial,self.spatial_norm):
            b=F.elu(tnorm(temporal(v))); b=F.elu(snorm(spatial(b))); branches.append(F.dropout(F.avg_pool2d(b,(1,4)),.20,self.training))
        v=torch.cat(branches,1); v=F.dropout(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(v)))),(1,2)),.15,self.training)
        v=F.dropout(F.avg_pool2d(F.elu(self.norm2(self.point2(self.depth2(v)))),(1,2)),.15,self.training); return self.head(self.drop(self.embedding(self.pool(v).flatten(1))))

def clean(x):
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,)): return float(x)
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    return x
def atomic_json(path,x):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.part'); tmp.write_text(json.dumps(clean(x),indent=2,sort_keys=True)+'\n'); os.replace(tmp,path)
def write_csv(path,rows,fields=None):
    path.parent.mkdir(parents=True,exist_ok=True); fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    tmp=path.with_suffix(path.suffix+'.part')
    with tmp.open('w',newline='',encoding='utf-8') as f: csv.DictWriter(f,fieldnames=fields,extrasaction='ignore').writeheader(); f.seek(0,2); w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writerows([clean(r) for r in rows])
    os.replace(tmp,path)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def stable(*x): return int.from_bytes(hashlib.sha256('|'.join(map(str,x)).encode()).digest()[:8],'little')%(2**32-1)
def seed_all(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
def bootstrap(v,*parts):
    v=np.asarray(v,float); rng=np.random.default_rng(stable(*parts)); z=v[rng.integers(0,len(v),size=(20000,len(v)))].mean(1); return float(v.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))
def natural(a): return sorted(map(str,set(a)),key=lambda x:int(x.replace('sub-','')))

def reference():
    p=ROOT/'code'/'peeh_reference.py'; spec=importlib.util.spec_from_file_location('p4_peeh_reference',p); mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod
def model(name,c,t): return EEGNet(c,t) if name=='EEGNet' else LiteBN(c)
def paths(subject,session): return CACHE/f'sub-{subject:02d}'/f'ses-{session}_eeg_task-mental-arithmetic.npz'
def load_cell(subject,session):
    with np.load(paths(subject,session)) as z: return z['X'].astype(np.float32), (z['y']==3).astype(np.int64), tuple(z['channel_names'].tolist())
def assemble(subjects,sessions):
    xs=[];ys=[];ss=[];se=[]; channels=None
    for sub in subjects:
        for session in sessions:
            x,y,ch=load_cell(int(sub),session); channels=channels or ch
            if ch!=channels or x.shape!=(20,30,2000) or sorted(np.unique(y).tolist())!=[0,1]: raise RuntimeError(f'invalid cache cell {sub}/{session}')
            xs.append(x);ys.append(y);ss.extend([f'sub-{int(sub):02d}']*len(y));se.extend([session]*len(y))
    return np.concatenate(xs),np.concatenate(ys),np.asarray(ss),np.asarray(se),channels
def normalize(train,*others):
    mean=train.mean(axis=(0,2),keepdims=True); std=train.std(axis=(0,2),keepdims=True); std[std<1e-6]=1
    return [(a-mean)/std for a in (train,*others)], {'mean_sha256':hashlib.sha256(mean.tobytes()+std.tobytes()).hexdigest()}
def logits(net,x,device):
    net.eval(); out=[]
    with torch.inference_mode():
        for i in range(0,len(x),BATCH): out.append(net(torch.from_numpy(np.ascontiguousarray(x[i:i+BATCH])).to(device)).float().cpu().numpy())
    return np.concatenate(out)
def reps(net,x,device):
    values=[]; cap=[]; h=net.head.register_forward_pre_hook(lambda _m,args:cap.append(args[0].detach()))
    try:
        net.eval()
        with torch.inference_mode():
            for i in range(0,len(x),BATCH): cap.clear(); net(torch.from_numpy(np.ascontiguousarray(x[i:i+BATCH])).to(device)); assert len(cap)==1; values.append(cap[0].reshape(len(cap[0]),-1).float().cpu().numpy())
    finally: h.remove()
    return np.concatenate(values).astype(np.float32)
def eval_subjects(net,x,y,subjects,session,device,base):
    pred=logits(net,x,device).argmax(1); rows=[]
    for sub in natural(subjects):
        m=subjects==sub; rows.append({**base,'subject':sub,'session':session,'BA':float(balanced_accuracy_score(y[m],pred[m])),'Macro_F1':float(f1_score(y[m],pred[m],average='macro',zero_division=0)),'trials':int(m.sum())})
    return rows
def ridge_fit(x,y):
    mu=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1; d=np.c_[(x-mu)/sd,np.ones(len(x))]; target=np.eye(2)[y]; pen=np.eye(d.shape[1]); pen[-1,-1]=0; return np.linalg.solve(d.T@d+.01*pen,d.T@target),mu,sd
def ridge_pred(x,pack):
    w,mu,sd=pack; return (np.c_[(x-mu)/sd,np.ones(len(x))]@w).argmax(1)

def splits():
    subjects=np.arange(1,30); outer=KFold(n_splits=5,shuffle=True,random_state=SEED); result=[]
    for fold,(tr,te) in enumerate(outer.split(subjects)):
        avail=subjects[tr]; rng=np.random.default_rng(stable('P4-inner-validation',fold,SEED)); val=np.sort(rng.choice(avail,5,replace=False)); train=np.array([s for s in avail if s not in set(val)])
        result.append((fold,np.sort(train),val,np.sort(subjects[te])))
    return result
def manifest():
    rows=[]
    for fold,tr,va,te in splits():
        for s in tr: rows.append({'fold':fold,'subject':int(s),'role':'train'})
        for s in va: rows.append({'fold':fold,'subject':int(s),'role':'inner_val'})
        for s in te: rows.append({'fold':fold,'subject':int(s),'role':'outer_test'})
    write_csv(ROOT/'subject_split_manifest.csv',rows); return rows
def audit():
    rec=json.loads((CACHE/'manifest.json').read_text()); lines=['# Shin2017B P4 dataset audit','','- Dataset: Shin2017B / nm000268, EEG only.','- Subjects: 29 (sub-01 through sub-29).','- Sessions: 1arithmetic, 3arithmetic, 5arithmetic (chronological S1/S2/S3).','- Task labels: raw 3=subtraction/mental arithmetic, 4=rest; cached binary 1/0.','- Sampling: 200 Hz; no resampling; each whole task trial is 10 s = 2,000 samples.','- Tensor: every session `(20, 30, 2000)`, with 10 trials per class.','- Channels (preserved order): `'+', '.join(rec['records'][0]['channel_names'])+'`.','- Missing/invalid recordings: none after cache validation.']
    (ROOT/'dataset_audit.md').write_text('\n'.join(lines)+'\n')
def protocol():
    text='''# P4 seed-0 protocol\n\nShin2017B / nm000268 uses only the 30 EEG channels and the three mental-arithmetic sessions (S1=1arithmetic, S2=3arithmetic, S3=5arithmetic). Whole 10-second task trials are used once; no MI, NIRS, EOG, pseudo-trials, calibration, adaptation, or outer-subject information is used.\n\nOuter CV is deterministic 5-fold subject-disjoint (seed 0); each fold reserves five remaining subjects for S3 inner validation. Training uses TRAIN S1+S2. Channel-wise mean/std is fit only on TRAIN S1+S2. The fixed recipe is 60 epochs, min selection epoch 10, AdamW lr=3e-4, weight decay=5e-4, batch 128, clip=5. Models are frozen EEGNet and current final LiteBN, reported as SIRE-EEG; only C=30, T=2000, and binary head are adapted.\n\nPrimary full-model endpoint is outer S3. BA and macro-F1 are subject-equal; WS-BA=min(S1,S2,S3). PEEH/PSWA reuse the frozen PERSIST implementation: 200 permutations, active-rank/eigengap/block rules, 100 equal-rank controls, ridge alpha .01, 20,000 biological-subject bootstrap draws. Coordinates and Protected selection use TRAIN S1/S2 only; PEEH probe fits TRAIN S1 only as in the existing implementation and evaluates outer S3. PSWA probes use TRAIN S1/S2 and evaluate outer S1/S2/S3.\n'''
    (ROOT/'P4_PROTOCOL.md').write_text(text)

def train_cell(name,fold,tr,va,te,device,ref):
    cell=ROOT/'checkpoints'/name/f'fold{fold}'; ckpt=cell/'selected.pt'; recp=cell/'record.json'; cell.mkdir(parents=True,exist_ok=True)
    if recp.is_file() and ckpt.is_file():
        rec=json.loads(recp.read_text()); net=model(name,30,2000); net.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=False)['state_dict']); return net.to(device).eval(),rec
    xtr,ytr,_,_,_=assemble(tr,SESSIONS[:2]); xv,yv,sv,_,_=assemble(va,(SESSIONS[2],)); (xtr,xv),norm=normalize(xtr,xv)
    seed_all(SEED); net=model(name,30,2000).to(device); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD); tx=torch.from_numpy(xtr).to(device); ty=torch.from_numpy(ytr).to(device); vx=torch.from_numpy(xv).to(device)
    best=-1.; best_ep=0; best_state=None; hist=[]
    for ep in range(1,EPOCHS+1):
        net.train(); rng=np.random.default_rng(stable('P4-batch',name,fold,SEED,ep)); losses=[]
        for ids in [z for z in np.array_split(rng.permutation(len(ytr)),max(1,int(np.ceil(len(ytr)/BATCH))) ) if len(z)]:
            ix=torch.as_tensor(ids,device=device); opt.zero_grad(set_to_none=True); loss=F.cross_entropy(net(tx.index_select(0,ix)),ty.index_select(0,ix)); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),CLIP); opt.step(); losses.append(float(loss.detach().cpu()))
        p=logits(net,xv,device).argmax(1); scores=[balanced_accuracy_score(yv[sv==s],p[sv==s]) for s in natural(sv)]
        score=float(np.mean(scores)); sel=ep>=MIN_EPOCH and score>best+1e-12
        if sel: best,best_ep,best_state=score,ep,deepcopy({k:v.detach().cpu() for k,v in net.state_dict().items()})
        hist.append({'epoch':ep,'loss':float(np.mean(losses)),'inner_val_subject_equal_BA':score,'selected':sel})
    if best_state is None: raise RuntimeError('no checkpoint selected')
    net.load_state_dict(best_state); torch.save({'state_dict':net.state_dict(),'selected_epoch':best_ep},ckpt); rec={'model':name,'fold':fold,'seed':0,'selected_epoch':best_ep,'best_inner_validation_BA':best,'recipe':{'epochs':EPOCHS,'min_epoch':MIN_EPOCH,'batch_size':BATCH,'lr':LR,'weight_decay':WD,'gradient_clip':CLIP},'normalizer':norm,'checkpoint_sha256':sha(ckpt),'history':hist,'train_subjects':list(map(int,tr)),'inner_val_subjects':list(map(int,va)),'outer_subjects':list(map(int,te))}; atomic_json(recp,rec); return net.eval(),rec

def diagnostics(name,fold,tr,te,net,device,ref,normalizer,ckmeta):
    # All representations below are outer-free until final probe evaluation.
    x,y,sub,ses,_=assemble(tr,SESSIONS[:2]); mean=np.frombuffer(b'',dtype=np.float32) # normalize from saved source statistics is reconstructed deterministically
    x0,_,_,_,_=assemble(tr,SESSIONS[:2]); xo=[]; yo=[]; so=[]
    for session in SESSIONS:
        a,b,c,_,_=assemble(te,(session,)); xo.append(a);yo.append(b);so.append(c)
    # fit TRAIN-only normalizer, duplicated intentionally rather than loading outer data during fit
    (x,*xo)=normalize(x0,*xo)[0]; h=reps(net,x,device); hout=[reps(net,z,device) for z in xo]
    session_num=np.asarray([1 if z==SESSIONS[0] else 2 for z in ses]); spec=ref.spectrum(h,y,sub,session_num,'Shin2017B',name,fold); spec['classes']=2
    protected,assignment=ref.select_protected(h,y,sub,session_num,spec,name,'Shin2017B',fold)
    # PEEH: unchanged existing probe path: TRAIN S1 fit, outer S3 evaluation.
    fi=np.flatnonzero(ses==SESSIONS[0]); he=hout[2]; ye=yo[2]; se=so[2]; engine=ref.FixedStandardizerKernelRidge(h[fi],y[fi],he,2,spec,qfit=ref.canonical(h[fi],spec),qeval=ref.canonical(he,spec),raw_base=ref._erasure_base(spec)); p0,_=engine.predict(()); pp,_=engine.predict(protected)
    random_pred=[]; all_dims=np.arange(spec['rank'])
    for draw in range(100):
        dims=[] if not protected else np.random.default_rng(ref.stable_seed('final-random',name,'Shin2017B',fold,draw)).choice(all_dims,len(protected),replace=False).tolist(); q,_=engine.predict(dims); random_pred.append(q)
    peeh=[]
    for s in natural(se):
        m=se==s; base=balanced_accuracy_score(ye[m],p0[m]); prot=balanced_accuracy_score(ye[m],pp[m]); rnd=float(np.mean([balanced_accuracy_score(ye[m],q[m]) for q in random_pred])); peeh.append({'model':name,'fold':fold,'subject':s,'status':'VALID' if protected else 'EMPTY_PROTECTED','protected_rank':len(protected),'intact_BA':base,'protected_erased_BA':prot,'random_erased_BA':rnd,'PEEH_pp':100*(rnd-prot),'checkpoint_sha256':ckmeta['checkpoint_sha256']})
    # PSWA: retain Protected or exact equal-rank random coordinate subspaces; TRAIN S1+S2 only.
    pswa=[]
    if protected:
        qtr=ref.canonical(h,spec); qout=[ref.canonical(z,spec) for z in hout]; p_pred=[ridge_pred(z[:,protected],ridge_fit(qtr[:,protected],y)) for z in qout]
        rsets=[np.random.default_rng(ref.stable_seed('final-random',name,'Shin2017B',fold,d)).choice(all_dims,len(protected),replace=False) for d in range(100)]
        r_preds=[[ridge_pred(z[:,d],ridge_fit(qtr[:,d],y)) for z in qout] for d in rsets]
        for s in natural(so[0]):
            pbas=[]; rws=[]; row={'model':name,'fold':fold,'subject':s,'status':'VALID','protected_rank':len(protected),'checkpoint_sha256':ckmeta['checkpoint_sha256']}
            for j,lab in enumerate(('S1','S2','S3')):
                m=so[j]==s; pb=balanced_accuracy_score(yo[j][m],p_pred[j][m]); rb=float(np.mean([balanced_accuracy_score(yo[j][m],r[j][m]) for r in r_preds])); row[f'protected_BA_{lab}']=pb;row[f'random_BA_{lab}']=rb;pbas.append(pb)
            for r in r_preds: rws.append(min(balanced_accuracy_score(yo[j][so[j]==s],r[j][so[j]==s]) for j in range(3)))
            row['protected_WSBA']=min(pbas);row['random_WSBA']=float(np.mean(rws));row['PSWA_pp']=100*(row['protected_WSBA']-row['random_WSBA']);pswa.append(row)
    else:
        pswa=[{'model':name,'fold':fold,'subject':s,'status':'EMPTY_PROTECTED','protected_rank':0,'PSWA_pp':None,'checkpoint_sha256':ckmeta['checkpoint_sha256']} for s in natural(so[0])]
    return peeh,pswa,{'model':name,'fold':fold,'protected_rank':len(protected),'active_rank':spec['rank'],'assignment':assignment}

def run():
    ROOT.mkdir(parents=True,exist_ok=True); protocol(); audit(); manifest(); ref=reference(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); full=[];peeh=[];pswa=[];diag=[]
    for fold,tr,va,te in splits():
        for name in MODELS:
            net,rec=train_cell(name,fold,tr,va,te,device,ref)
            for session,short in zip(SESSIONS,('S1','S2','S3')):
                x,y,s,_,_=assemble(te,(session,)); x=normalize(assemble(tr,SESSIONS[:2])[0],x)[0][1]; full.extend(eval_subjects(net,x,y,s,short,device,{'model':name,'fold':fold,'checkpoint_sha256':rec['checkpoint_sha256'],'selected_epoch':rec['selected_epoch']}))
            a,b,c=diagnostics(name,fold,tr,te,net,device,ref,rec['normalizer'],rec);peeh.extend(a);pswa.extend(b);diag.append(c); del net; torch.cuda.empty_cache()
            write_csv(ROOT/'seed0_fullmodel_subject_metrics.csv',full);write_csv(ROOT/'seed0_peeh_subject_results.csv',peeh);write_csv(ROOT/'seed0_pswa_subject_results.csv',pswa);atomic_json(ROOT/'diagnostic_assignments.json',diag)
            print(f'P4_CELL_COMPLETE model={name} fold={fold}',flush=True)
    summary=[]; metrics={}
    for name in MODELS:
        rows=[r for r in full if r['model']==name]; by={s:{r['session']:r for r in rows if r['subject']==s} for s in natural([r['subject'] for r in rows])}; future=[v['S3']['BA'] for v in by.values()]; f1=[v['S3']['Macro_F1'] for v in by.values()]; ws=[min(v[x]['BA'] for x in ('S1','S2','S3')) for v in by.values()]; metrics[name]={'future BA':np.mean(future),'future Macro-F1':np.mean(f1),'WS-BA':np.mean(ws),'future_vec':future,'ws_vec':ws}; summary.append({'model':name,'future BA':np.mean(future),'future Macro-F1':np.mean(f1),'WS-BA':np.mean(ws),'subjects':len(by)})
    da=np.asarray(metrics['SIRE-EEG']['future_vec'])-np.asarray(metrics['EEGNet']['future_vec']); dw=np.asarray(metrics['SIRE-EEG']['ws_vec'])-np.asarray(metrics['EEGNet']['ws_vec']); _,lo,hi=bootstrap(da,'P4','future'); _,wlo,whi=bootstrap(dw,'P4','ws'); summary += [{'model':'SIRE-EEG minus EEGNet','metric':'future BA contrast','mean':da.mean(),'CI95_low':lo,'CI95_high':hi,'bootstrap_draws':20000},{'model':'SIRE-EEG minus EEGNet','metric':'WS-BA contrast','mean':dw.mean(),'CI95_low':wlo,'CI95_high':whi,'bootstrap_draws':20000}]; write_csv(ROOT/'seed0_fullmodel_summary.csv',summary)
    dsum=[]
    for name in MODELS:
        a=[r for r in peeh if r['model']==name and r['status']=='VALID'];b=[r for r in pswa if r['model']==name and r['status']=='VALID']; pm,pl,ph=bootstrap([r['PEEH_pp'] for r in a],'P4','peeh',name) if a else (np.nan,np.nan,np.nan); sm,sl,sh=bootstrap([r['PSWA_pp'] for r in b],'P4','pswa',name) if b else (np.nan,np.nan,np.nan); dsum.append({'model':name,'PEEH_pp':pm,'PEEH_CI95_low':pl,'PEEH_CI95_high':ph,'PEEH_coverage':f'{len(a)}/29','PSWA_pp':sm,'PSWA_CI95_low':sl,'PSWA_CI95_high':sh,'PSWA_coverage':f'{len(b)}/29'})
    write_csv(ROOT/'seed0_diagnostic_summary.csv',dsum)
    lines=['# P4 seed-0 summary','']
    for r in summary[:2]: lines.append(f"- {r['model']}: future BA {r['future BA']:.3f}, future Macro-F1 {r['future Macro-F1']:.3f}, WS-BA {r['WS-BA']:.3f}.")
    lines += [f"- SIRE-EEG minus EEGNet future BA: {da.mean():.3f} [{lo:.3f}, {hi:.3f}]; WS-BA: {dw.mean():.3f} [{wlo:.3f}, {whi:.3f}].",'','PEEH and PSWA are diagnostics, not model-quality rankings. See `seed0_diagnostic_summary.csv` for direction and assignment coverage.']
    (ROOT/'P4_SEED0_SUMMARY.md').write_text('\n'.join(lines)+'\n'); print('P4_SEED0_COMPLETE',flush=True)
if __name__=='__main__': run()
