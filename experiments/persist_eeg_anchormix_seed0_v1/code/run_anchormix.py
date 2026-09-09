"""AnchorMix-EEG seed-0 single-head 5-fold experiment."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

REPO = Path(os.environ.get("R2EEG_REPO", Path(__file__).resolve().parents[3])).resolve()
EXP = REPO / "experiments" / "persist_eeg_anchormix_seed0_v1"
OUT = EXP / "outputs"
RUNTIME = Path(os.environ.get("ANCHORMIX_RUNTIME", REPO.parent / "anchormix_seed0_runtime")).resolve()
SPLIT = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1" / "protocol" / "FIVEFOLD_SPLIT.json"
CARRIER_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", REPO.parent / "carrier_5fold_multiseed_stability_runtime")).resolve()
SRC = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"
STAGE = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code"
sys.path[:0] = [str(SRC), str(STAGE)]
import run_stage1 as base
import run_carrier_screen as carrier
from eegnet_locked import EEGNet

SEED=0; TRAIN_SEED=100000; EPOCHS=60; MIN_EPOCH=10; LR=3e-4; WD=5e-4; CLIP=5.0

def set_seed(s:int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True

def sha_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def state_hash(s):
    import io
    b=io.BytesIO(); torch.save(s,b); return hashlib.sha256(b.getvalue()).hexdigest()

def clean(v):
    if isinstance(v,Path): return str(v)
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,float)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    return v

def write_json(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+'.part'); q.write_text(json.dumps(clean(v),indent=2,sort_keys=True)+'\n'); os.replace(q,p)

class FastGPUCache:
    def __init__(self,bundle,mean,std,device):
        pieces=[]; n=len(bundle.search_rows)
        for st in range(0,n,256):
            idx=np.arange(st,min(n,st+256),dtype=np.int64); x=bundle.accessor.batch(idx).astype(np.float32,copy=False)
            x=(x-mean[None,:,None])/np.maximum(std[None,:,None],1e-6); pieces.append(np.ascontiguousarray(x))
        self.x=torch.from_numpy(np.concatenate(pieces,axis=0)).to(device,non_blocking=True)
        self.y=torch.from_numpy(bundle.labels(np.arange(n,dtype=np.int64))).to(device,non_blocking=True); self.device=device
    def batch(self,indices):
        ii=torch.as_tensor(np.asarray(indices,dtype=np.int64),device=self.device); return self.x.index_select(0,ii),self.y.index_select(0,ii)

class AnchorMixEEG(torch.nn.Module):
    def __init__(self,channels:int,classes:int=2):
        super().__init__(); self.channels=channels
        def stem(k):
            return torch.nn.Sequential(torch.nn.Conv2d(1,8,(1,k),padding='same',bias=False),torch.nn.BatchNorm2d(8),torch.nn.ELU(),torch.nn.Conv2d(8,16,(channels,1),groups=8,bias=False),torch.nn.BatchNorm2d(16),torch.nn.ELU(),torch.nn.AvgPool2d((1,4)),torch.nn.Dropout(.20))
        self.short=stem(15); self.anchor_stem=stem(64); self.long=stem(127)
        self.anchor_depth=torch.nn.Conv2d(16,16,(1,16),padding='same',groups=16,bias=False); self.anchor_point=torch.nn.Conv2d(16,16,1,bias=False); self.anchor_bn=torch.nn.BatchNorm2d(16); self.anchor_pool=torch.nn.AvgPool2d((1,8)); self.anchor_adapt=torch.nn.AdaptiveAvgPool2d((1,16)); self.anchor_fc=torch.nn.Linear(16*16,64); self.anchor_ln=torch.nn.LayerNorm(64); self.a_norm=torch.nn.LayerNorm(64)
        self.bottleneck=torch.nn.Sequential(torch.nn.Conv2d(48,32,1,bias=False),torch.nn.BatchNorm2d(32),torch.nn.ELU())
        self.refine1=torch.nn.Sequential(torch.nn.Conv2d(32,32,(1,15),padding='same',groups=32,bias=False),torch.nn.Conv2d(32,48,1,bias=False),torch.nn.BatchNorm2d(48),torch.nn.ELU(),torch.nn.AvgPool2d((1,2)),torch.nn.Dropout(.15))
        self.refine2=torch.nn.Sequential(torch.nn.Conv2d(48,48,(1,31),padding='same',groups=48,bias=False),torch.nn.Conv2d(48,48,1,bias=False),torch.nn.BatchNorm2d(48),torch.nn.ELU(),torch.nn.AvgPool2d((1,2)),torch.nn.Dropout(.15))
        self.context_fc=torch.nn.Linear(48*8,64); self.context_ln=torch.nn.LayerNorm(64); self.residual=torch.nn.Linear(64,64); self.residual_drop=torch.nn.Dropout(.20); self.mix_ln=torch.nn.LayerNorm(64); self.head=torch.nn.Linear(64,classes)
    def forward(self,x):
        v=x.unsqueeze(1); h15=self.short(v); h64=self.anchor_stem(v); h127=self.long(v)
        a=self.anchor_adapt(self.anchor_pool(torch.nn.functional.elu(self.anchor_bn(self.anchor_point(self.anchor_depth(h64)))))).flatten(1); a=self.anchor_ln(torch.nn.functional.elu(self.anchor_fc(a)))
        c=self.refine2(self.refine1(self.bottleneck(torch.cat([h15,h64,h127],1)))); c=torch.nn.functional.adaptive_avg_pool2d(c,(1,8)).flatten(1); m=self.context_ln(torch.nn.functional.elu(self.context_fc(c)))
        h=self.mix_ln(self.a_norm(a) + self.residual_drop(torch.nn.functional.gelu(self.residual(m))))
        return self.head(h)

def count(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
def metric(y,l):
    p=l.argmax(1); return {'BA':float(balanced_accuracy_score(y,p)),'macro_F1':float(f1_score(y,p,average='macro')),'accuracy':float(accuracy_score(y,p)),'trials':int(len(y))}
def smoke(device):
    rows=[]
    for c in (62,58):
        set_seed(0); m=AnchorMixEEG(c).to(device); x=torch.randn(4,c,1000,device=device); y=torch.tensor([0,1,0,1],device=device); z=m(x); loss=torch.nn.functional.cross_entropy(z,y); loss.backward(); assert z.shape==(4,2) and torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None); rows.append({'channels':c,'params':count(m),'shape':list(z.shape),'finite_loss_gradient':True})
    return rows

def evaluate(model,bundle,subjects,cache,device):
    model.eval(); out={}
    with torch.no_grad():
      for s in subjects:
        idx=bundle.indices([s],(2,)); y=bundle.labels(idx); ls=[]
        for st in range(0,len(idx),256):
          z=model(cache.batch(idx[st:st+256])[0])
          if isinstance(z,(tuple,list)): z=z[0]
          ls.append(z.float().cpu().numpy())
        out[str(s)]={'y':y,'logits':np.concatenate(ls)}
    return out

def train(model,dataset,fold,manifest,minfo,bundle,cache,device):
    cell=RUNTIME/dataset.lower()/f'fold{fold["fold_id"]}'; cell.mkdir(parents=True,exist_ok=True); path=cell/'checkpoint_latest.pt'; sel=cell/'selected_best.pt'; init=state_hash(copy.deepcopy(model.state_dict())); opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD); amp=device.type=='cuda'; scaler=torch.amp.GradScaler('cuda',enabled=amp); start=1; hist=[]; best=-1; be=None; bs=None
    if path.exists():
      sv=torch.load(path,map_location=device,weights_only=False)
      if sv.get('init_sha256')!=init or sv.get('manifest_sha256')!=minfo['sha256']: raise RuntimeError('resume invariant mismatch')
      model.load_state_dict(sv['current_state']); opt.load_state_dict(sv['optimizer']); scaler.load_state_dict(sv['scaler']); start=sv['epoch']+1; hist=sv['history']; best=sv['best_val_BA']; be=sv['best_epoch']; bs=sv['best_state']
    began=time.perf_counter()
    for ep in range(start,EPOCHS+1):
      model.train(); losses=[]
      for e in manifest[ep-1]:
        idx=e['support_indices']+e['query_indices']; x,y=cache.batch(idx); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp): z=model(x); loss=torch.nn.functional.cross_entropy(z,y)
        if not torch.isfinite(loss): raise RuntimeError('non-finite loss')
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
      val=evaluate(model,bundle,[str(x) for x in fold['inner_val_subjects']],cache,device); vb=float(np.mean([metric(v['y'],v['logits'])['BA'] for v in val.values()])); selected=ep>=MIN_EPOCH and vb>best+1e-12
      if selected: best,be,bs=vb,ep,copy.deepcopy(model.state_dict())
      row={'epoch':ep,'CE':float(np.mean(losses)),'inner_val_subject_BA':vb,'selected':selected}; hist.append(row); torch.save({'epoch':ep,'history':hist,'best_val_BA':best,'best_epoch':be,'best_state':bs,'current_state':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'manifest_sha256':minfo['sha256'],'init_sha256':init},path)
      if ep==1 or ep%5==0 or selected: print(f'[{dataset} fold={fold["fold_id"]}] epoch={ep:02d} loss={row["CE"]:.4f} valBA={vb:.4f}',flush=True)
    model.load_state_dict(bs); torch.save(model.state_dict(),sel); return {'selected_epoch':int(be),'best_inner_val_BA':float(best),'history':hist,'checkpoint_path':str(sel),'checkpoint_sha256':sha_file(sel),'elapsed_seconds':time.perf_counter()-began,'amp':amp}

def read_split():
    raw=json.loads(SPLIT.read_text()); return raw['folds'],raw['search_subjects'],sha_file(SPLIT)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke-test',action='store_true'); ap.add_argument('--run',action='store_true'); ap.add_argument('--device',default='auto'); a=ap.parse_args(); device=torch.device('cuda' if a.device=='auto' and torch.cuda.is_available() else a.device); set_seed(0); OUT.mkdir(parents=True,exist_ok=True)
    if a.smoke_test: write_json(OUT/'SMOKE_TEST.json',{'tests':smoke(device)}); print(json.dumps(smoke(device))); return
    folds,search,split_hash=read_split(); base.RUNTIME=RUNTIME/'manifest_runtime'; base.RUNTIME.mkdir(parents=True,exist_ok=True); bundles={d:base.load_bundle(d,[str(x) for x in search[d]]) for d in ('OpenBMI','WBCIC')}; rows=[]; trains=[]
    for d in ('OpenBMI','WBCIC'):
      b=bundles[d]
      for fraw in folds[d]:
        f=int(fraw['fold_id']); mean,std,norm=base.normalizer(b,[str(x) for x in fraw['inner_train_subjects']]); manifest,minfo=base.make_manifest(b,fraw); cache=FastGPUCache(b,mean,std,device); set_seed(0); model=AnchorMixEEG(b.channels).to(device); set_seed(TRAIN_SEED); info=train(model,d,fraw,manifest,minfo,b,cache,device); trains.append({'dataset':d,'fold':f,**info}); am=evaluate(model,b,[str(x) for x in fraw['outer_dev_subjects']],cache,device)
        refs={}
        for name,cls in [('EEGNet',EEGNet),('LiteBN',lambda c:carrier.CompactLite(c,'bn'))]:
          p=CARRIER_RUNTIME/f'{d.lower()}_fold{f}_seed0_{name.lower()}'/'selected_best.pt'; obj=cls(b.channels).to(device); obj.load_state_dict(torch.load(p,map_location=device,weights_only=True)); obj.eval(); rr=evaluate(obj,b,[str(x) for x in fraw['outer_dev_subjects']],cache,device); refs[name]=rr
        for s in [str(x) for x in fraw['outer_dev_subjects']]:
          y=am[s]['y']; z=am[s]['logits']; le=refs['EEGNet'][s]['logits']; ll=refs['LiteBN'][s]['logits']; rf=.5*le+.5*ll; mm={}
          for n,l in [('EEGNet',le),('LiteBN',ll),('REF50',rf),('AnchorMix',z)]: mm.update({f'{n}_{k}':v for k,v in metric(y,l).items()})
          rows.append({'dataset':d,'fold':f,'subject_id':s,**mm})
        print(f'[{d} fold={f}] AnchorMix done',flush=True); del cache; torch.cuda.empty_cache() if device.type=='cuda' else None
    sdf=pd.DataFrame(rows); fr=[]; summ=[]
    for d in ('OpenBMI','WBCIC'):
      ds=sdf[sdf.dataset==d]; best='EEGNet' if ds.EEGNet_BA.mean()>=ds.LiteBN_BA.mean() else 'LiteBN'; groups=[]
      for f,g in ds.groupby('fold'):
        row={'dataset':d,'fold':int(f)}
        for n in ('EEGNet','LiteBN','REF50','AnchorMix'):
          row[n]=float(g[f'{n}_BA'].mean())
          row[f'{n}_macro_F1']=float(g[f'{n}_macro_F1'].mean())
        row['Delta_vs_best_pp']=(row['AnchorMix']-max(row['EEGNet'],row['LiteBN']))*100; row['Delta_vs_REF50_pp']=(row['AnchorMix']-row['REF50'])*100; fr.append(row)
      bestv=float(ds[f'{best}_BA'].mean()); ref=float(ds.REF50_BA.mean()); anc=float(ds.AnchorMix_BA.mean())
      summary={'Dataset':d,'EEGNet':float(ds.EEGNet_BA.mean()),'LiteBN':float(ds.LiteBN_BA.mean()),'REF50':ref,'AnchorMix':anc,'DeltaBest_pp':(anc-bestv)*100,'DeltaREF50_pp':(anc-ref)*100,'BestSingle':best}
      for n in ('EEGNet','LiteBN','REF50','AnchorMix'): summary[f'{n}_macro_F1']=float(ds[f'{n}_macro_F1'].mean())
      summ.append(summary)
    fdf=pd.DataFrame(fr); sumdf=pd.DataFrame(summ); sdf.to_csv(OUT/'SUBJECT_METRICS.csv',index=False); fdf.to_csv(OUT/'FOLD_METRICS.csv',index=False); sumdf.to_csv(OUT/'DATASET_SUMMARY.csv',index=False); write_json(OUT/'MODEL_COST.json',{'params_62':count(AnchorMixEEG(62)),'params_58':count(AnchorMixEEG(58)),'smoke':smoke(device)}); write_json(OUT/'TRAINING_LOGS.json',trains)
    dsok=all((sumdf.loc[sumdf.Dataset==d,'DeltaBest_pp'].iloc[0]>=0) for d in ('OpenBMI','WBCIC')); refok=all((sumdf.loc[sumdf.Dataset==d,'DeltaREF50_pp'].iloc[0]>=-.5) for d in ('OpenBMI','WBCIC')); strong=dsok and refok; very=all((sumdf.loc[sumdf.Dataset==d,'DeltaREF50_pp'].iloc[0]>=0) for d in ('OpenBMI','WBCIC')); term='VERY_STRONG' if very else 'STRONG' if strong else 'PROMISING' if dsok else 'MIXED' if any(sumdf.loc[sumdf.Dataset==d,'DeltaBest_pp'].iloc[0]>=0 for d in ('OpenBMI','WBCIC')) else 'FAIL'
    lines=['# AnchorMix-EEG seed-0 result','',f'Final terminal: **{term}**','', '| Dataset | EEGNet | LiteBN | REF50 | AnchorMix | DeltaBest | DeltaREF50 |','|---|---:|---:|---:|---:|---:|---:|']
    for _,r in sumdf.iterrows(): lines.append(f"| {r.Dataset} | {r.EEGNet*100:.2f}% | {r.LiteBN*100:.2f}% | {r.REF50*100:.2f}% | {r.AnchorMix*100:.2f}% | {r.DeltaBest_pp:+.2f} pp | {r.DeltaREF50_pp:+.2f} pp |")
    lines += ['', 'Macro-F1 for every method is recorded in `DATASET_SUMMARY.csv` and `FOLD_METRICS.csv`.', '', '## Fold metrics', '', '| Dataset | Fold | EEGNet | LiteBN | REF50 | AnchorMix | Delta vs best single | Delta vs REF50 |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in fdf.sort_values(['dataset','fold']).iterrows(): lines.append(f"| {r.dataset} | {int(r.fold)} | {r.EEGNet*100:.2f}% | {r.LiteBN*100:.2f}% | {r.REF50*100:.2f}% | {r.AnchorMix*100:.2f}% | {r.Delta_vs_best_pp:+.2f} pp | {r.Delta_vs_REF50_pp:+.2f} pp |")
    lines += ['', 'AnchorMix is a single representation with one classifier (`self.head`); no prediction-level fusion or auxiliary heads.', f'Parameter count: {count(AnchorMixEEG(62))}.', '', 'Questions:', '1. Strongest single exceeded: '+('YES' if dsok else 'NO'), '2. Close to REF50 (within 0.5 pp): '+('YES' if refok else 'NO'), '3. Dataset direction consistent: '+('YES' if all(sumdf.loc[sumdf.Dataset==d,'DeltaBest_pp'].iloc[0]>=0 for d in ('OpenBMI','WBCIC')) else 'NO'), '4. Next step 3 seeds + ERP/SSVEP: '+('CONSIDER' if strong else 'NO; stop.')]
    (OUT/'FINAL_RESULT.md').write_text('\n'.join(lines)+'\n'); print(term,flush=True)
if __name__=='__main__': main()
