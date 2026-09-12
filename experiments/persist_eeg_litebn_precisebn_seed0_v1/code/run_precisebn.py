"""Recovered exact-historical seed0 replay gate followed by source-only Precise-BN.

Historical model source, loaders, normalization and scoring functions are loaded
from the hash-verified transferred snapshot. Signals stream from existing caches.
No optimizer, backward, checkpoint selection, or large in-GPU dataset cache.
"""
import argparse,ast,csv,gc,hashlib,importlib,json,os,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from precisebn import calibrate,tests,fingerprint

TASKS=['OpenBMI_MI','OpenBMI_ERP','OpenBMI_SSVEP','WBCIC_MI']
METRICS=['BA','macro_F1','accuracy']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.part');temp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,p)
def csvread(p):
    with Path(p).open(encoding='utf-8') as h:return list(csv.DictReader(h))
def csvwrite(p,rows):
    assert rows
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

class Runner:
    def __init__(self,a):
        self.a=a;self.out=a.repo/'experiments/persist_eeg_litebn_precisebn_seed0_v1/outputs';self.snap=a.recovered/'snapshot';self.runtime=a.runtime;self.runtime.mkdir(parents=True,exist_ok=True)
        self.receipt=read(a.recovered/'TRANSFER_RECEIPT.json')
        for r in self.receipt['files']:assert sha(a.recovered/r['relative_path'])==r['sha256'],'Transfer receipt drift'
        self.taskexp=self.snap/'persist_eeg_openbmi_task_generality_v1';self.carrier=self.snap/'persist_eeg_carrier_5fold_multiseed_stability_v1';self.final=self.snap/'persist_eeg_final_heldout_confirmation_v1'
        os.environ.update(PERSIST_OPENBMI_CACHE=str(a.cache/'openbmi/openbmi'),PERSIST_WBCIC_CACHE=str(a.cache/'wbcic/wbcic_epochs'),TASK_GENERALITY_REPO=str(a.repo),FINAL_CONFIRM_REPO=str(a.repo))
        sys.path[:0]=[str(self.taskexp/'code'),str(self.final/'code')]
        self.td=importlib.import_module('task_datasets');self.es=importlib.import_module('evaluate_search');self.lf=importlib.import_module('load_final_carriers');self.rf=importlib.import_module('run_final_confirmation')
        source=self.snap/'run_carrier_screen.py';tree=ast.parse(source.read_text())
        nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['CompactLite','norm_layer']]
        assert len(nodes)==2
        ns={'torch':torch,'nn':nn,'F':F};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns);self.CompactLite=ns['CompactLite']
        self.split_path=self.carrier/'protocol/FIVEFOLD_SPLIT.json';self.split=read(self.split_path);self.hold=read(self.final/'protocol/FINAL_HOLDOUT_MANIFEST.json')
        task_split=read(self.taskexp/'protocol/SUBJECT_SPLIT_REFERENCE.json')
        assert task_split['folds']==self.split['folds']['OpenBMI']
        self.taskprov=read(self.taskexp/'protocol/CHECKPOINT_PROVENANCE.json')['records'];self.miprov=read(self.final/'protocol/SOURCE_CHECKPOINTS.json');self.logs=read(self.carrier/'outputs/TRAINING_LOGS.json');self.minorm=read(self.carrier/'protocol/MANIFEST_HASHES.json')
        self.hist_mi=csvread(self.carrier/'outputs/SUBJECT_SEED_RESULTS.csv');self.hist_mih=csvread(self.final/'outputs/REPLICATE_SUBJECT_RESULTS.csv');self.hist_task=csvread(self.taskexp/'outputs/SEARCH_REPLICATE_RESULTS.csv');self.hist_taskh=csvread(self.taskexp/'outputs/HELDOUT_REPLICATE_RESULTS.csv')
        self.device=torch.device('cuda');assert torch.cuda.is_available()
        torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
        self.invariant=hashlib.sha256(json.dumps({'receipt_sha':sha(a.recovered/'TRANSFER_RECEIPT.json'),'code':sha(Path(__file__)),'moments':sha(Path(__file__).with_name('precisebn.py')),'batch':128,'seed':0,'tolerance':1e-10,'SD_gate':'equal_task_mean_SD_change_pp <= 0; no task increase > 1pp'},sort_keys=True).encode()).hexdigest()
        self.sources=[]
        for task in TASKS:
            d=self.dataset(task);folds=self.split['folds'][d];search=set(self.split['search_subjects'][d]);held=set(self.hold[d]['subject_ids'])
            assert not search&held and len(folds)==5
            assert set.union(*(set(f['outer_dev_subjects']) for f in folds))==search
            for f in folds:
                tr,va,od=[set(f[k]) for k in ['inner_train_subjects','inner_val_subjects','outer_dev_subjects']]
                assert not tr&va and not tr&od and not va&od and tr|va|od==search and not (tr|va|od)&held
                fid=f['fold_id'];p=a.recovered/f'checkpoints/{task}/fold{fid}_seed0/selected_best.pt'
                if self.is_mi(task):
                    r=next(x for x in self.miprov if x['dataset']==d and x['fold']==fid and x['seed']==0 and x['model']=='LiteBN');wanted=r['expected_sha256'];epoch=next(x['selected_epoch'] for x in self.logs if x['dataset']==d and x['fold']==fid and x['seed']==0 and x['model']=='LiteBN');source_commit=self.receipt['source_repository_inspection_commit'];commit_kind='repository_inspection; training_commit_not_recorded'
                else:
                    r=next(x for x in self.taskprov if x['task']==task.split('_')[1] and x['fold']==fid and x['seed']==0 and x['model']=='LiteBN');wanted=r['checkpoint_sha256'];epoch=r['selected_epoch'];source_commit=read(self.taskexp/'protocol/TASK_GENERALITY_PROTOCOL_LOCK.json')['git_commit'];commit_kind='recorded_protocol_lock_commit'
                assert sha(p)==wanted,'Historical checkpoint SHA mismatch'
                m=self.model(task,p);del m
                self.sources.append({'task':task,'fold':fid,'seed':0,'checkpoint_path':str(p),'source_checkpoint_path':r['checkpoint_path'],'checkpoint_sha256':wanted,'selected_epoch':epoch,'split_sha256':sha(self.split_path),'source_commit':source_commit,'source_commit_kind':commit_kind,'canonical_model_source_sha256':sha(source),'strict_state_load':'PASS','normalization_preprocessing_provenance':str(self.taskexp/'protocol/TASK_EPOCH_PROTOCOL.json') if not self.is_mi(task) else str(self.final/'protocol/PREPROCESSING_PROVENANCE.json')})
        csvwrite(self.out/'LITEBN_PRECISEBN_SOURCE_MANIFEST.csv',self.sources)
        dump(self.out/'PRECISE_BN_UNIT_TESTS.json',tests())

    @staticmethod
    def is_mi(t):return t.endswith('_MI')
    @staticmethod
    def dataset(t):return 'WBCIC' if t=='WBCIC_MI' else 'OpenBMI'
    def fold(self,t,f):return self.split['folds'][self.dataset(t)][f]
    def path(self,t,f):return self.a.recovered/f'checkpoints/{t}/fold{f}_seed0/selected_best.pt'
    def model(self,t,p):
        m=self.CompactLite(58 if t=='WBCIC_MI' else 62,'bn')
        if t=='OpenBMI_SSVEP':m.head=nn.Linear(64,4)
        m.load_state_dict(torch.load(p,map_location='cpu',weights_only=True),strict=True);m.to(self.device).eval()
        for q in m.parameters():q.requires_grad_(False)
        return m
    def normalizer(self,t,f):
        tr=self.fold(t,f)['inner_train_subjects'];d=self.dataset(t)
        if self.is_mi(t):
            mean,std,norm=self.lf.source_normalizer(d,tr);wanted=next(x['normalizer'] for x in self.minorm if x['dataset']==d and x['fold']==f)
        else:
            bundle=self.td.load_bundle(t.split('_')[1],tr,sessions=(1,));mean,std,norm=self.td.normalizer(bundle,tr);wanted=next(x['normalizer'] for x in self.taskprov if x['task']==t.split('_')[1] and x['fold']==f and x['seed']==0 and x['model']=='LiteBN');del bundle
        assert norm==wanted,f'Normalizer differs from history {t} f{f}: {norm} != {wanted}'
        return mean,std,norm
    def evaluate(self,t,model,subjects,mean,std):
        model.eval();rows=[]
        for s in subjects:
            if self.is_mi(t):
                x,y=self.lf.load_eval_subject(self.dataset(t),s);z=self.rf._logits(model,x,mean,std,self.device);metric=self.rf._metric(y,z.argmax(1))
            else:
                b=self.td.load_bundle(t.split('_')[1],[s],sessions=(2,));idx=b.indices([s],(2,));y=b.labels(idx)
                class StreamCache:
                    def batch(_,indices,mu,sigma):
                        raw=torch.from_numpy(b.signal_batch(indices)).to(self.device);m=torch.as_tensor(mu,device=self.device)[None,:,None];st=torch.as_tensor(sigma,device=self.device)[None,:,None]
                        return (raw-m)/torch.clamp(st,min=1e-6),None
                z=self.es.logits(model,StreamCache(),idx,mean,std);metric=self.es.score(t.split('_')[1],y,z)
            rows.append({'subject_id':str(s),'trials':int(len(y)),**{k:metric[k] for k in METRICS}})
        return rows
    def historical(self,t,f,pop):
        if self.is_mi(t):
            source=self.hist_mi if pop=='outer' else self.hist_mih
            rows=[r for r in source if r['dataset']==self.dataset(t) and int(r['fold'])==f and int(r['seed'])==0 and (pop=='outer' or r['method']=='LiteBN')]
            return {r['subject_id']:{k:float(r[('LiteBN_' if pop=='outer' else '')+k]) for k in METRICS} for r in rows}
        source=self.hist_task if pop=='outer' else self.hist_taskh
        rows=[r for r in source if r['task']==t.split('_')[1] and int(r['fold'])==f and int(r['seed'])==0 and r['method']=='LiteBN']
        return {r['subject_id']:{k:float(r[k]) for k in METRICS} for r in rows}
    def cellfile(self,t,f):return self.runtime/f'{t}_fold{f}_replay.json'
    def replay(self):
        # All outer original replays, then all already-open internal heldout replays.
        for pop in ['outer','heldout']:
            for t in TASKS:
                for f in range(5):
                    path=self.cellfile(t,f);saved=read(path) if path.exists() else {'invariant':self.invariant}
                    assert saved['invariant']==self.invariant,'Stale replay resume'
                    if pop in saved:
                        assert saved[pop]['pass'],'Failed replay cannot be resumed as passed'
                        continue
                    mean,std,norm=self.normalizer(t,f) if 'normalizer' not in saved else (np.asarray(saved['mean'],np.float32),np.asarray(saved['std'],np.float32),saved['normalizer'])
                    subjects=self.fold(t,f)['outer_dev_subjects'] if pop=='outer' else self.hold[self.dataset(t)]['subject_ids'];hist=self.historical(t,f,pop);assert set(map(str,subjects))==set(hist)
                    model=self.model(t,self.path(t,f));before=fingerprint(model.state_dict().items());rows=self.evaluate(t,model,subjects,mean,std);assert before==fingerprint(model.state_dict().items())
                    errors=[abs(r[k]-hist[r['subject_id']][k]) for r in rows for k in METRICS];maxerr=max(errors)
                    saved.update(mean=mean.tolist(),std=std.tolist(),normalizer=norm);saved[pop]={'rows':rows,'historical':hist,'max_abs_metric_error':maxerr,'pass':maxerr<=1e-10};dump(path,saved)
                    print('REPLAY',t,f,pop,'maxerr',maxerr,flush=True)
                    del model;gc.collect();torch.cuda.empty_cache()
                    assert maxerr<=1e-10,f'BASELINE_REPLAY_MISMATCH {t} f{f} {pop}'
        rows=[]
        for t in TASKS:
            for f in range(5):
                s=read(self.cellfile(t,f));assert s['invariant']==self.invariant
                for pop in ['outer','heldout']:
                    assert s[pop]['pass']
                    for r in s[pop]['rows']:
                        h=s[pop]['historical'][r['subject_id']]
                        rows.append({'task':t,'fold':f,'seed':0,'population':pop,'subject_id':r['subject_id'],'historical_BA':h['BA'],'replayed_BA':r['BA'],'delta_BA':r['BA']-h['BA'],'historical_macro_F1':h['macro_F1'],'replayed_macro_F1':r['macro_F1'],'historical_accuracy':h['accuracy'],'replayed_accuracy':r['accuracy'],'status':'PASS'})
        csvwrite(self.out/'LITEBN_PRECISEBN_BASELINE_REPLAY.csv',rows);dump(self.runtime/'REPLAY_GATE.json',{'invariant':self.invariant,'pass':True,'rows':len(rows),'checkpoints':20})
    def batches(self,t,f,mean,std):
        subjects=self.fold(t,f)['inner_train_subjects'];seen=[];batch=[]
        # Stream source signals only, labels are not loaded in this generator.
        for s in sorted(subjects,key=lambda x:int(x.replace('sub-',''))):
            sessions=self.lf.SOURCE_SESSIONS[self.dataset(t)] if self.is_mi(t) else (1,)
            for se in sessions:
                p=self.lf.signal_path(self.dataset(t),s,se) if self.is_mi(t) else self.td.task_path(t.split('_')[1],s,se,'signal');x=np.load(p,mmap_mode='r',allow_pickle=False)
                for i in range(len(x)):
                    batch.append(np.asarray(x[i],np.float32));seen.append((s,se,i))
                    if len(batch)==128:
                        yield self.normalize(t,np.stack(batch),mean,std);batch=[]
        if batch:yield self.normalize(t,np.stack(batch),mean,std)
        assert len(seen)==len(set(seen)),'Repeated calibration example'
    def normalize(self,t,x,mean,std):
        if self.is_mi(t):return self.lf.normalize_to_device(x,mean,std,self.device)
        raw=torch.from_numpy(x).to(self.device);return (raw-torch.as_tensor(mean,device=self.device)[None,:,None])/torch.as_tensor(std,device=self.device)[None,:,None].clamp_min(1e-6)
    def run(self):
        gate=read(self.runtime/'REPLAY_GATE.json');assert gate['pass'] and gate['checkpoints']==20 and gate['invariant']==self.invariant
        # Calibrate each original checkpoint and evaluate all outer folds first.
        for t in TASKS:
            for f in range(5):
                p=self.runtime/f'{t}_fold{f}_precise.json'
                if p.exists():assert read(p)['invariant']==self.invariant;continue
                s=read(self.cellfile(t,f));mean,std=np.asarray(s['mean'],np.float32),np.asarray(s['std'],np.float32);model=self.model(t,self.path(t,f))
                stats,audit,bn=calibrate(model,self.batches(t,f,mean,std));assert audit['calibration_trials']==s['normalizer']['trials']
                outer=self.evaluate(t,model,self.fold(t,f)['outer_dev_subjects'],mean,std);bp=self.runtime/f'{t}_fold{f}_BN_ONLY.pt';torch.save(bn,bp)
                dump(p,{'invariant':self.invariant,'layer_stats':stats,'state_audit':audit,'calibration_subjects':self.fold(t,f)['inner_train_subjects'],'calibration_sessions':s['normalizer']['sessions'],'batch_size':128,'bn_path':str(bp),'bn_sha256':sha(bp),'outer':outer})
                print('CALIBRATED_OUTER',t,f,'trials',audit['calibration_trials'],flush=True);del model;gc.collect();torch.cuda.empty_cache()
        for t in TASKS:
            for f in range(5):
                p=self.runtime/f'{t}_fold{f}_precise.json';result=read(p);assert result['invariant']==self.invariant
                if 'heldout' in result:continue
                s=read(self.cellfile(t,f));mean,std=np.asarray(s['mean'],np.float32),np.asarray(s['std'],np.float32);model=self.model(t,self.path(t,f));assert sha(result['bn_path'])==result['bn_sha256']
                state=model.state_dict();before=fingerprint(model.named_parameters());bn=torch.load(result['bn_path'],map_location='cpu',weights_only=True)
                assert set(bn)=={f'{n}.{k}' for n,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm) for k in ['running_mean','running_var','num_batches_tracked']}
                state.update(bn);model.load_state_dict(state,strict=True);assert before==fingerprint(model.named_parameters())
                result['heldout']=self.evaluate(t,model,self.hold[self.dataset(t)]['subject_ids'],mean,std);dump(p,result);print('PRECISE_HELDOUT',t,f,flush=True);del model;gc.collect();torch.cuda.empty_cache()
        self.summarize()
    def summarize(self):
        import pandas as pd
        layer=[];audit=[];pairs=[];budgets=[]
        for t in TASKS:
            for f in range(5):
                b=read(self.cellfile(t,f));r=read(self.runtime/f'{t}_fold{f}_precise.json');assert r['invariant']==self.invariant
                layer += [dict(task=t,fold=f,**x) for x in r['layer_stats']];audit.append(dict(task=t,fold=f,**r['state_audit']));budgets.append({'task':t,'fold':f,**{k:r[k] for k in ['calibration_subjects','calibration_sessions','batch_size']},'trials':r['state_audit']['calibration_trials'],'batches':r['state_audit']['calibration_batches']})
                for pop in ['outer','heldout']:
                    base={x['subject_id']:x for x in b[pop]['rows']}
                    for x in r[pop]:
                        old=base[x['subject_id']];row={'task':t,'fold':f,'seed':0,'population':pop,'subject_id':x['subject_id'],'trials':x['trials']}
                        for k in METRICS:row.update({f'original_{k}':old[k],f'preciseBN_{k}':x[k],f'delta_{k}_pp':100*(x[k]-old[k])})
                        pairs.append(row)
        csvwrite(self.out/'LITEBN_PRECISEBN_LAYER_STATS.csv',layer);csvwrite(self.out/'LITEBN_PRECISEBN_STATE_AUDIT.csv',audit);dump(self.out/'CALIBRATION_BUDGETS.json',budgets)
        df=pd.DataFrame(pairs);df.to_csv(self.out/'LITEBN_PRECISEBN_PAIRED_SUBJECT_REPLICATES.csv',index=False)
        cols=[c for c in df if c.startswith(('original_','preciseBN_','delta_'))]
        outer=df[df.population=='outer'];held=df[df.population=='heldout'];of=outer.groupby(['task','fold'],as_index=False)[cols].mean();hs=held.groupby(['task','subject_id'],as_index=False)[cols].mean()
        of.to_csv(self.out/'LITEBN_PRECISEBN_OUTER_FOLD_RESULTS.csv',index=False);hs.to_csv(self.out/'LITEBN_PRECISEBN_HELDOUT_SUBJECT_RESULTS.csv',index=False)
        ot=[];ht=[]
        for t in TASKS:
            z=of[of.task==t];h=hs[hs.task==t];o={'task':t,**{c:float(z[c].mean()) for c in cols},'original_fold_SD':float(z.original_BA.std(ddof=0)),'preciseBN_fold_SD':float(z.preciseBN_BA.std(ddof=0))};o['SD_change_pp']=100*(o['preciseBN_fold_SD']-o['original_fold_SD'])
            delta=z.delta_BA_pp.to_numpy();o.update(positive_folds=int((delta>1e-8).sum()),negative_folds=int((delta<-1e-8).sum()),tied_folds=int((abs(delta)<=1e-8).sum()));ot.append(o)
            delta=h.delta_BA_pp.to_numpy();draw=np.random.default_rng(0).choice(delta,size=(10000,len(delta)),replace=True).mean(1)
            ht.append({'task':t,**{c:float(h[c].mean()) for c in cols},'positive_subjects':int((delta>1e-8).sum()),'negative_subjects':int((delta<-1e-8).sum()),'tied_subjects':int((abs(delta)<=1e-8).sum()),'paired_bootstrap_95CI_low_pp':float(np.quantile(draw,.025)),'paired_bootstrap_95CI_high_pp':float(np.quantile(draw,.975)),'bootstrap_unit':'subject after five seed0 checkpoint metric mean','bootstrap_seed':0,'bootstrap_resamples':10000})
        csvwrite(self.out/'LITEBN_PRECISEBN_OUTER_TASK_SUMMARY.csv',ot);csvwrite(self.out/'LITEBN_PRECISEBN_HELDOUT_TASK_SUMMARY.csv',ht)
        od=np.array([r['delta_BA_pp'] for r in ot]);hd=np.array([r['delta_BA_pp'] for r in ht]);sd=np.array([r['SD_change_pp'] for r in ot]);signal=bool((od>1e-8).sum()>=3 and (hd>1e-8).sum()>=3 and od.mean()>0 and hd.mean()>0 and sd.mean()<=0 and sd.max()<=1.)
        decision='PRECISE_BN_SIGNAL_FOUND' if signal else 'PRECISE_BN_NO_USEFUL_SIGNAL'
        meta={'status':'COMPLETE','decision':decision,'SEED':0,'LEARNABLE_WEIGHTS_UPDATED':'NO','CALIBRATION_DATA':'INNER_TRAIN_ONLY','INNER_VAL_USED_FOR_CALIBRATION':'NO','OUTER_USED_FOR_CALIBRATION':'NO','CURRENT_HELDOUT_USED_FOR_CALIBRATION':'NO','NEW_SEALED_FINAL_DATA_ACCESSED':'NO','checkpoints':20,'baseline_replay_pass':True,'invariant':self.invariant,'outer_equal_task_delta_pp':float(od.mean()),'heldout_equal_task_delta_pp':float(hd.mean()),'outer_positive_tasks':int((od>1e-8).sum()),'heldout_positive_tasks':int((hd>1e-8).sum()),'outer_median_task_delta_pp':float(np.median(od)),'heldout_median_task_delta_pp':float(np.median(hd)),'outer_worst_task_delta_pp':float(od.min()),'heldout_worst_task_delta_pp':float(hd.min()),'folds_improved':sum(r['positive_folds'] for r in ot),'equal_task_SD_change_pp':float(sd.mean()),'fold_SD_ddof':0,'batch_size':128,'calibration_passes':1,'torch':torch.__version__,'device':torch.cuda.get_device_name(),'heldout_status':'current internal heldout diagnostic','aggregation':'mean metrics across five seed0 checkpoints per subject, then equal subjects; no logit ensemble; historical rule restricted to requested seed0','source_commit_limitation':'MI checkpoint creation commit not recorded; source inspection commit and original checkpoint/result hashes recorded','outer':ot,'heldout':ht}
        dump(self.out/'PRECISE_BN_METADATA.json',meta)
        lines=['# Source-only Precise-BN seed0 final diagnostic','',decision,'','Recovered checkpoints supersede the initial source-gate blocker. All 20 original checkpoints replayed against historical per-subject BA, macro-F1 and accuracy before any recalibration.','', '| Task | Original outer BA | Precise outer BA | Delta pp | Original heldout BA | Precise heldout BA | Delta pp | Fold SD change pp |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for o,h in zip(ot,ht):lines.append(f"| {o['task']} | {o['original_BA']:.6f} | {o['preciseBN_BA']:.6f} | {o['delta_BA_pp']:+.3f} | {h['original_BA']:.6f} | {h['preciseBN_BA']:.6f} | {h['delta_BA_pp']:+.3f} | {o['SD_change_pp']:+.3f} |")
        lines += ['',f"Outer equal-task delta: {od.mean():+.3f} pp; positive tasks: {(od>1e-8).sum()}/4; median {np.median(od):+.3f}; worst {od.min():+.3f}.",f"Internal heldout equal-task delta: {hd.mean():+.3f} pp; positive tasks: {(hd>1e-8).sum()}/4; median {np.median(hd):+.3f}; worst {hd.min():+.3f}.",f"Improved folds: {meta['folds_improved']}/20; mean fold-SD change: {sd.mean():+.3f} pp (population SD, ddof=0).",'',f"BN statistics: {len(layer)} layer/checkpoint records; median mean shift {np.median([r['mean_shift'] for r in layer]):.6g}, median variance shift {np.median([r['variance_shift'] for r in layer]):.6g}. Layerwise absolute and relative changes are in LITEBN_PRECISEBN_LAYER_STATS.csv.",'','All learnable tensors and non-BN buffers were bitwise identical; maximum parameter change is zero. BN moments used all legal inner-train source examples exactly once; batch size 128, deterministic subject/session/trial order, float64 accumulation, dropout disabled.','', '## Interpretation','', 'The predefined four-task improvement/variability criteria are met; BN statistics warrant a separately authorized follow-up, but this diagnostic does not establish causality uniquely or independent final-test improvement.' if signal else 'The predefined four-task improvement/variability criteria are not met. This uniform Precise-BN procedure does not justify promotion to the next LiteBN method on this diagnostic. Do not rescue individual tasks by selective application or tuning.', '', 'Current heldout is an already-open internal diagnostic, not untouched final confirmation. Historical averaging is restricted to the requested seed0 (five checkpoints), not all fifteen historical seed/fold replicates. Bootstrap resamples subjects after checkpoint averaging, 10,000 draws, seed0; no multiple-task adjustment.','', 'Source provenance caveat: MI original training commit was not recorded; source inspection commit is labeled as such. Original selected binary hashes, source normalizer metadata and historical per-subject replay provide the recovery verification.','', '```text']
        lines += [f'{k} = {meta[k]}' for k in ['SEED','LEARNABLE_WEIGHTS_UPDATED','CALIBRATION_DATA','INNER_VAL_USED_FOR_CALIBRATION','OUTER_USED_FOR_CALIBRATION','CURRENT_HELDOUT_USED_FOR_CALIBRATION','NEW_SEALED_FINAL_DATA_ACCESSED']]+['```','','No new training, seed1/2 evaluation, or light-tail-risk experiment was run. Existing server tasks were not terminated.']
        (self.out/'FINAL_PRECISE_BN_DECISION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(decision,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--recovered',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--phase',choices=['replay','run','all','summary'],required=True);a=p.parse_args();r=Runner(a)
    if a.phase in ['replay','all']:r.replay()
    if a.phase in ['run','all']:r.run()
    if a.phase=='summary':r.summarize()
if __name__=='__main__':main()
