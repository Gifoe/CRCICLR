"""Exact historical training with a single supplementary LocalStats readout."""
import argparse,ast,copy,csv,gc,hashlib,importlib,inspect,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from localstats import attach

TASKS=['OpenBMI_MI','OpenBMI_ERP','OpenBMI_SSVEP','WBCIC_MI']
METRICS=['BA','macro_F1','accuracy']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def csvread(p):
    with Path(p).open(encoding='utf-8') as h:return list(csv.DictReader(h))
def csvwrite(p,rows):
    assert rows
    with Path(p).open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def tensors_equal(a,b):return set(a)==set(b) and all(torch.equal(a[k],b[k]) for k in a)
def rng_equal(a,b):return torch.equal(a['torch'],b['torch']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda'])) and a['python']==b['python'] and all(np.array_equal(x,y) for x,y in zip(a['numpy'],b['numpy']))

class Experiment:
    def __init__(self,a):
        self.a=a;self.snap=a.recovered/'snapshot';self.exp=a.repo/'experiments/persist_eeg_litebn_localstats_seed0_v1';self.out=self.exp/'outputs';self.out.mkdir(parents=True,exist_ok=True);self.rt=a.runtime;self.rt.mkdir(parents=True,exist_ok=True)
        os.environ.update(R2EEG_REPO=str(a.repo),TASK_GENERALITY_REPO=str(a.repo),FINAL_CONFIRM_REPO=str(a.repo),PERSIST_OPENBMI_CACHE=str(a.cache/'openbmi/openbmi'),PERSIST_WBCIC_CACHE=str(a.cache/'wbcic/wbcic_epochs'))
        self.carrier=self.snap/'persist_eeg_carrier_5fold_multiseed_stability_v1';self.taskexp=self.snap/'persist_eeg_openbmi_task_generality_v1';self.final=self.snap/'persist_eeg_final_heldout_confirmation_v1'
        sys.path[:0]=[str(self.carrier/'code'),str(self.taskexp/'code'),str(self.final/'code'),str(a.repo/'experiments/persist_eeg_litebn_lighttail_seed0_v1/code')]
        self.h=importlib.import_module('train_grid');self.v=self.h.v1;self.c=self.h.carrier;self.td=importlib.import_module('task_datasets');self.tt=importlib.import_module('train_task_carriers');self.lf=importlib.import_module('load_final_carriers');self.rf=importlib.import_module('run_final_confirmation');self.es=importlib.import_module('evaluate_search');self.legacy=importlib.import_module('legacy_init')
        assert inspect.getsource(self.c.CompactLite).replace('\r','') in (self.snap/'run_carrier_screen.py').read_text().replace('\r','')
        self.split=read(self.carrier/'protocol/FIVEFOLD_SPLIT.json');self.hold=read(self.final/'protocol/FINAL_HOLDOUT_MANIFEST.json');assert read(self.taskexp/'protocol/SUBJECT_SPLIT_REFERENCE.json')['folds']==self.split['folds']['OpenBMI']
        self.h.validate_split(self.split['search_subjects'],self.split['folds'])
        for d in ['OpenBMI','WBCIC']:assert not set(self.split['search_subjects'][d])&set(self.hold[d]['subject_ids'])
        self.logs=read(self.carrier/'outputs/TRAINING_LOGS.json');self.manmeta=read(self.carrier/'protocol/MANIFEST_HASHES.json');self.taskprov=read(self.taskexp/'protocol/CHECKPOINT_PROVENANCE.json')['records']
        self.device=torch.device('cuda');assert torch.cuda.is_available();torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;self.h.set_seed(0)
        files=[Path(__file__),Path(__file__).with_name('localstats.py'),Path(self.legacy.__file__),Path(self.h.__file__),Path(self.v.__file__),Path(self.c.__file__),Path(self.v.core.__file__),Path(self.td.__file__),Path(self.tt.__file__),Path(self.lf.__file__),Path(self.rf.__file__),Path(self.es.__file__),self.carrier/'protocol/FIVEFOLD_SPLIT.json',self.taskexp/'protocol/CHECKPOINT_PROVENANCE.json']
        self.hashes={str(p):sha(p) for p in files};self.invariant=hashlib.sha256(json.dumps(self.hashes,sort_keys=True).encode()).hexdigest()
    def dataset(self,t):return 'WBCIC' if t=='WBCIC_MI' else 'OpenBMI'
    def mi(self,t):return t.endswith('_MI')
    def fold(self,t,f):return self.split['folds'][self.dataset(t)][f]
    def cell(self,t,f):return self.rt/f'{t}_fold{f}'
    def meta(self,t,f):return next(r for r in (self.logs if self.mi(t) else self.taskprov) if r['model']=='LiteBN' and r['fold']==f and r['seed']==0 and (r['dataset']==self.dataset(t) if self.mi(t) else r['task']==t.split('_')[1]))
    def reference(self,t,f):return self.a.recovered/f'checkpoints/{t}/fold{f}_seed0/selected_best.pt'
    def initial(self,t,f):
        record=self.meta(t,f);ref=self.reference(t,f);assert sha(ref)==record['checkpoint_sha256']
        def constructor(name,unused):return self.h.constructor(name,58 if t=='WBCIC_MI' else 62) if self.mi(t) else self.td.build_model(name,t.split('_')[1])
        return self.legacy.construct_exact(constructor,self.h.set_seed,ref,record['init_sha256'],self.device)[0]
    def bundle(self,t):return self.v.load_bundle(self.dataset(t),self.split['search_subjects'][self.dataset(t)]) if self.mi(t) else self.td.load_bundle(t.split('_')[1],self.split['search_subjects']['OpenBMI'])
    def normalizer(self,t,f,b):
        tr=self.fold(t,f)['inner_train_subjects'];mean,std,norm=(self.v.normalizer(b,tr) if self.mi(t) else self.td.normalizer(b,tr));wanted=next(r['normalizer'] for r in self.manmeta if r['dataset']==self.dataset(t) and r['fold']==f) if self.mi(t) else self.meta(t,f)['normalizer'];assert norm==wanted,'NORMALIZER_MISMATCH';return mean,std,norm
    def manifest(self,t,f,b):
        fold=self.fold(t,f);record=self.meta(t,f)
        if self.mi(t):
            d=self.dataset(t).lower();p=self.a.historical_runtime/f'{d}_fold{f}/manifest_runtime/episode_manifests/{d}_fold{f}.json'
            assert sha(p)==record['manifest_sha256'];original=read(p)
            self.v.RUNTIME=self.cell(t,f)/'reconstructed';m,info=self.v.make_manifest(b,fold);assert read(info['path'])==original;assert hashlib.sha256(Path(info['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==sha(p)
            batches=[[np.asarray(e['support_indices']+e['query_indices'],np.int64) for e in epoch] for epoch in m]
            return m,batches,sha(p)
        idx=b.indices(fold['inner_train_subjects'],(1,));short=t.split('_')[1];mh=hashlib.sha256(idx.tobytes()+f'{short}:{f}:full_permutation_batch64'.encode()).hexdigest();assert mh==record['manifest_sha256']
        batches=[self.td.epoch_batches(idx,f,short,e) for e in range(1,61)]
        for epoch in batches:assert np.array_equal(np.sort(np.concatenate(epoch)),np.sort(idx)) and all(len(x)<=64 for x in epoch)
        return None,batches,mh
    def prepare(self,t):
        b=self.bundle(t);audits=[];budget=[];manifests=[]
        for f in range(5):
            cell=self.cell(t,f);cell.mkdir(parents=True,exist_ok=True);mean,std,norm=self.normalizer(t,f,b);_,batches,mh=self.manifest(t,f,b)
            for e,epoch in enumerate(batches,1):
                digest=hashlib.sha256()
                for idx in epoch:digest.update(len(idx).to_bytes(8,'little'));digest.update(idx.tobytes())
                manifests.append({'task':t,'fold':f,'epoch':e,'batches':len(epoch),'trial_exposure':sum(map(len,epoch)),'ordered_batches_sha256':digest.hexdigest(),'historical_manifest_sha256':mh,'match':True})
            baseline=self.initial(t,f);shared=copy.deepcopy(baseline.state_dict());rng=self.h.rng_state();candidate=attach(copy.deepcopy(baseline));assert rng_equal(rng,self.h.rng_state());assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in shared.items());assert torch.count_nonzero(candidate.local_U.weight)==0 and torch.count_nonzero(candidate.local_V.weight)>0
            idx=batches[0][0]
            if self.mi(t):x=self.v.prepare(b,idx,mean,std,self.device)
            else:x=(torch.from_numpy(b.signal_batch(idx)).to(self.device)-torch.as_tensor(mean,device=self.device)[None,:,None])/torch.as_tensor(std,device=self.device)[None,:,None].clamp_min(1e-6)
            candstate=copy.deepcopy(candidate.state_dict());maxdiff=0;mismatch=0
            for training in [False,True]:
                for amp in [False,True]:
                    baseline.load_state_dict(shared);candidate.load_state_dict(candstate);baseline.train(training);candidate.train(training)
                    self.h.set_seed(100000)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):zb=baseline(x)[0]
                    after=self.h.rng_state();self.h.set_seed(100000)
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):zc=candidate(x)[0]
                    assert rng_equal(after,self.h.rng_state()),'FORWARD_RNG_DRIFT'
                    diff=float((zb-zc).abs().max());mm=int((zb.argmax(1)!=zc.argmax(1)).sum());maxdiff=max(maxdiff,diff);mismatch+=mm
                    assert diff<=1e-6 and mm==0 and all(torch.equal(v,candidate.state_dict()[k]) for k,v in baseline.state_dict().items()),'INITIAL_FUNCTION_MISMATCH'
            baseline.load_state_dict(shared);candidate.load_state_dict(candstate)
            # Backprop at zero-U must give a finite nonzero U gradient.
            candidate.eval();candidate.zero_grad();self.h.set_seed(100000);z=candidate(x)[0];y=torch.as_tensor(b.labels(idx),device=self.device);torch.nn.functional.cross_entropy(z,y).backward();ug=float(candidate.local_U.weight.grad.norm());assert np.isfinite(ug) and ug>0
            audits.append({'task':t,'fold':f,'seed':0,'shared_parameters_identical':True,'U_zero':True,'V_nonzero':True,'rng_restored':True,'modes_checked':'eval/train x FP32/AMP','max_abs_logit_diff':maxdiff,'prediction_mismatch':mismatch,'U_initial_gradient_norm':ug,'historical_init_sha256':self.meta(t,f)['init_sha256'],'status':'PASS'})
            n=sum(p.numel() for p in baseline.parameters());nc=sum(p.numel() for p in candidate.parameters());assert nc-n==candidate.shallow_channels*16*8+512
            budget.append({'task':t,'fold':f,'shallow_channels':candidate.shallow_channels,'q_dim':candidate.shallow_channels*16,'rank':8,'bins':8,'baseline_parameters':n,'LS_parameters':nc,'extra_parameters':nc-n,'increase_percent':100*(nc-n)/n})
            wi=None
            if not self.mi(t):
                _,wi=self.td.class_weights(b,self.fold(t,f)['inner_train_subjects']);assert wi==self.meta(t,f)['class_weight_info']
            dump(cell/'preflight.json',{'invariant':self.invariant,'pass':True,'mean':mean.tolist(),'std':std.tolist(),'normalizer':norm,'class_weight_info':wi,'manifest_sha256':mh,'baseline_checkpoint':str(self.reference(t,f)),'baseline_checkpoint_sha256':sha(self.reference(t,f))})
            (self.out/f'FORWARD_{t}.py').write_text(candidate.transformed_forward+'\n',encoding='utf-8');print('INITIAL_FUNCTION_PASS',t,f,flush=True)
            del baseline,candidate,shared,candstate,x,z,zb,zc;gc.collect();torch.cuda.empty_cache()
        dump(self.out/f'PREFLIGHT_{t}.json',{'invariant':self.invariant,'audits':audits,'parameters':budget,'manifests':manifests})
    def train(self,t,f):
        gate=read(self.out/'PREFLIGHT_GATE.json');assert gate['pass'] and gate['invariant']==self.invariant
        cell=self.cell(t,f);saved=read(cell/'preflight.json');assert saved['invariant']==self.invariant
        if (cell/'result.json').exists():assert read(cell/'result.json')['invariant']==self.invariant;return
        b=self.bundle(t);mean,std,norm=self.normalizer(t,f,b);manifest,batches,mh=self.manifest(t,f,b);assert mh==saved['manifest_sha256']
        model=attach(self.initial(t,f));self.h.set_seed(100000);fold=dict(self.fold(t,f));fold['_seed']=0
        historyfile=cell/'update_audit.json';updates=read(historyfile) if historyfile.exists() else []
        def observe(epoch,optimizer,scaler):
            steps={int(v['step']) for v in optimizer.state.values()};assert len(steps)==1
            steps=steps.pop();r={'task':t,'fold':f,'epoch':epoch,'attempts':len(batches[epoch-1]),'exposure':sum(map(len,batches[epoch-1])),'successful_cumulative':steps,'AMP_scale':float(scaler.get_scale())}
            updates[:]=[x for x in updates if x['epoch']<epoch]+[r];dump(historyfile,updates)
            if epoch==1 or epoch%5==0:print('TRAIN',t,f,'epoch',epoch,'steps',steps,flush=True)
        function=self.h.train_model if self.mi(t) else self.tt.train_one
        tree=ast.parse(inspect.getsource(function));inserted=0
        class Observe(ast.NodeTransformer):
            def visit_Expr(self,node):
                nonlocal inserted
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='history.append':
                    inserted+=1;return [node,ast.parse('observe(epoch, optimizer, scaler)').body[0]]
                return node
        tree=Observe().visit(tree);ast.fix_missing_locations(tree);assert inserted==1
        ns=dict(function.__globals__);ns['observe']=observe
        def restore(state):
            state=dict(state);state['torch']=state['torch'].cpu();state['cuda']=[x.cpu() for x in state['cuda']];self.h.restore_rng(state)
        ns['restore_rng']=restore
        if not self.mi(t):ns['RUNTIME']=cell
        exec(compile(tree,str(inspect.getfile(function))+'[logging only]','exec'),ns)
        (self.out/f'TRAINING_LOOP_{t}.py').write_text(ast.unparse(tree)+'\n',encoding='utf-8')
        if self.mi(t):
            cache=self.c.GPUCache(b,mean,std,self.device);result=ns[function.__name__](model,'LiteBN-LS',b,fold,manifest,cache,mh,cell,self.device)
        else:
            cache=self.td.RawGPUCache(b,self.device);weights,wi=self.td.class_weights(b,fold['inner_train_subjects']);wi['_normalizer']=norm
            result=ns[function.__name__](model,'LiteBN-LS',t.split('_')[1],fold,b,cache,mean,std,weights,wi,self.device,self.invariant)
        assert len(result['history'])==60 and len(updates)==60
        result.update(invariant=self.invariant,normalizer=norm,task=t,fold=f,seed=0);dump(cell/'result.json',result);print('CELL_COMPLETE',t,f,flush=True)
    def historical(self,t,f,pop):
        if self.mi(t):
            path=self.carrier/'outputs/SUBJECT_SEED_RESULTS.csv' if pop=='outer' else self.final/'outputs/REPLICATE_SUBJECT_RESULTS.csv'
            rows=[r for r in csvread(path) if r['dataset']==self.dataset(t) and int(r['fold'])==f and int(r['seed'])==0 and (pop=='outer' or r['method']=='LiteBN')];prefix='LiteBN_' if pop=='outer' else ''
        else:
            path=self.taskexp/('outputs/SEARCH_REPLICATE_RESULTS.csv' if pop=='outer' else 'outputs/HELDOUT_REPLICATE_RESULTS.csv');rows=[r for r in csvread(path) if r['task']==t.split('_')[1] and int(r['fold'])==f and int(r['seed'])==0 and r['method']=='LiteBN'];prefix=''
        return {r['subject_id']:{k:float(r[prefix+k]) for k in METRICS} for r in rows},path
    def evaluate(self,t,f,pop):
        cell=self.cell(t,f);target=cell/f'{pop}.json'
        if target.exists():assert read(target)['invariant']==self.invariant;return
        saved=read(cell/'preflight.json');result=read(cell/'result.json');assert saved['invariant']==result['invariant']==self.invariant
        mean,std=np.asarray(saved['mean'],np.float32),np.asarray(saved['std'],np.float32)
        baseline=self.initial(t,f);baseline.load_state_dict(torch.load(self.reference(t,f),map_location=self.device,weights_only=True));baseline.eval()
        candidate=attach(self.initial(t,f));assert sha(result['checkpoint_path'])==result['checkpoint_sha256'];candidate.load_state_dict(torch.load(result['checkpoint_path'],map_location=self.device,weights_only=True));candidate.eval();candidate.collect_diagnostics=True
        subjects=self.fold(t,f)['outer_dev_subjects'] if pop=='outer' else self.hold[self.dataset(t)]['subject_ids'];hist,hpath=self.historical(t,f,pop);assert set(subjects)==set(hist);rows=[]
        for s in subjects:
            if self.mi(t):
                x,y=self.lf.load_eval_subject(self.dataset(t),s);logits=[self.rf._logits(m,x,mean,std,self.device) for m in [baseline,candidate]];scores=[self.rf._metric(y,z.argmax(1)) for z in logits]
            else:
                b=self.td.load_bundle(t.split('_')[1],[s],sessions=(2,));idx=b.indices([s],(2,));y=b.labels(idx)
                class Cache:
                    def batch(_,ii,mu,sigma):
                        x=torch.from_numpy(b.signal_batch(ii)).to(self.device);return (x-torch.as_tensor(mu,device=self.device)[None,:,None])/torch.as_tensor(sigma,device=self.device)[None,:,None].clamp_min(1e-6),None
                logits=[self.es.logits(m,Cache(),idx,mean,std) for m in [baseline,candidate]];scores=[self.es.score(t.split('_')[1],y,z) for z in logits]
            row={'task':t,'fold':f,'seed':0,'population':pop,'subject_id':s,'trials':len(y)}
            for k in METRICS:
                assert abs(scores[0][k]-hist[s][k])<=1e-10,'BASELINE_REPLAY_MISMATCH'
                row.update({f'LiteBN_{k}':scores[0][k],f'LS_{k}':scores[1][k],f'delta_{k}_pp':100*(scores[1][k]-scores[0][k])})
            rows.append(row)
        dr=candidate.diagnostic_rows;total=sum(r['trials'] for r in dr);diag={k:sum(r[k]*r['trials'] for r in dr)/total for k in dr[0] if k!='trials'};diag['q_std']=float(np.sqrt(max(0,diag['q_second_moment']-diag['q_mean']**2)))
        dump(target,{'invariant':self.invariant,'rows':rows,'diagnostics':dict(task=t,fold=f,population=pop,trials=total,**diag),'baseline_result_path':str(hpath),'baseline_result_sha256':sha(hpath),'baseline_checkpoint_sha256':sha(self.reference(t,f)),'baseline_metric_replay_pass':True});print('EVALUATED',t,f,pop,flush=True)

def summarize(e):
    import pandas as pd
    pairs=[];diags=[];provenance=[];training=[];updates=[]
    for t in TASKS:
        for f in range(5):
            training.append(read(e.cell(t,f)/'result.json'));updates+=read(e.cell(t,f)/'update_audit.json')
            for pop in ['outer','heldout']:
                r=read(e.cell(t,f)/f'{pop}.json');assert r['invariant']==e.invariant and r['baseline_metric_replay_pass'];pairs+=r['rows'];diags.append(r['diagnostics']);provenance.append(dict(task=t,fold=f,population=pop,**{k:v for k,v in r.items() if k.startswith('baseline_')}))
    df=pd.DataFrame(pairs);cols=[c for c in df if c.startswith(('LiteBN_','LS_','delta_'))];outer=df[df.population=='outer'];fold=outer.groupby(['task','fold'],as_index=False)[cols].mean();held=df[df.population=='heldout'].groupby(['task','subject_id'],as_index=False)[cols].mean();summaries={}
    for pop,data in [('outer',fold),('heldout',held)]:
        summary=[]
        for t in TASKS:
            z=data[data.task==t];subjects=outer[outer.task==t] if pop=='outer' else z;d=subjects.delta_BA_pp.to_numpy();draw=np.random.default_rng(0).choice(d,(10000,len(d)),replace=True).mean(1)
            row={'task':t,**{c:float(z[c].mean()) for c in cols},'positive_units':int((z.delta_BA_pp>1e-10).sum()),'negative_units':int((z.delta_BA_pp<-1e-10).sum()),'tied_units':int((abs(z.delta_BA_pp)<=1e-10).sum()),'positive_subjects':int((d>1e-10).sum()),'negative_subjects':int((d<-1e-10).sum()),'tied_subjects':int((abs(d)<=1e-10).sum()),'median_subject_delta_pp':float(np.median(d)),'subject_bootstrap_low_pp':float(np.quantile(draw,.025)),'subject_bootstrap_high_pp':float(np.quantile(draw,.975))}
            summary.append(row)
        summaries[pop]=summary
    ds=np.asarray([r['delta_BA_pp'] for r in summaries['outer']]);npos=int((ds>0).sum());terminal='LOCALSTATS_SEED0_4OF4_SIGNAL' if npos==4 else 'LOCALSTATS_SEED0_PARTIAL_SIGNAL' if npos==3 and ds.min()>=-.5 and ds.mean()>0 else 'LOCALSTATS_SEED0_NO_BROAD_SIGNAL'
    flags={'SEED':0,'ARCHITECTURE':'EXACT_LITEBN_PLUS_SHALLOW_LOCAL_STATS','LOCAL_STATS_LOCATION':'BEFORE_FIRST_TEMPORAL_POOLING','LOCAL_STATS':'8BIN_MEAN_PLUS_LOGVAR','LOW_RANK':8,'AUXILIARY_LOSS':'NONE','NEW_BN':'NONE','NEW_CLASSIFIER':'NO','ENSEMBLE':'NO','ROUTING':'NO','TEST_TIME_ADAPTATION':'NO','NEW_SEALED_FINAL_DATA_ACCESSED':'NO'}
    meta={'terminal':terminal,**flags,'invariant':e.invariant,'source_hashes':e.hashes,'tasks':TASKS,'training_cells':20,'initial_equivalence_passed':20,'equal_task_mean_delta_pp':float(ds.mean()),'median_task_delta_pp':float(np.median(ds)),'worst_task_delta_pp':float(ds.min()),'positive_tasks':npos,'positive_folds':int((fold.delta_BA_pp>1e-10).sum()),'outer':summaries['outer'],'heldout':summaries['heldout'],'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},'scope_limit':'seed0 diagnostic; Windows runtime not identical to historical Linux; no new sealed population'}
    csvwrite(e.out/'PAIRED_SUBJECT_RESULTS.csv',pairs);csvwrite(e.out/'LOCALSTATS_DIAGNOSTICS.csv',diags);fold.to_csv(e.out/'OUTER_FOLD_RESULTS.csv',index=False);held.to_csv(e.out/'INTERNAL_HELDOUT_SUBJECT_RESULTS.csv',index=False);csvwrite(e.out/'OUTER_TASK_SUMMARY.csv',summaries['outer']);csvwrite(e.out/'INTERNAL_HELDOUT_TASK_SUMMARY.csv',summaries['heldout']);csvwrite(e.out/'UPDATE_AUDIT.csv',updates);dump(e.out/'TRAINING_LOGS.json',training);dump(e.out/'BASELINE_PROVENANCE.json',provenance);dump(e.out/'RUN_METADATA.json',meta)
    params=csvread(e.out/'PARAMETER_COUNT.csv');lines=['# LocalStats seed0 decision','',terminal,'','1. Historical LiteBN: shared architecture/initialization, exact manifests or original deterministic task batches, normalizer/loss/optimizer/selection retained. Original baseline metrics replayed within 1e-10. Windows runtime is not the historical Linux build.','2. Parameter counts (baseline -> LS):']
    for r in params:
        if r['fold']=='0':lines.append(f"   {r['task']}: {r['baseline_parameters']} -> {r['LS_parameters']}; +{r['extra_parameters']} ({float(r['increase_percent']):.3f}%).")
    lines+=['3. Initial function equivalence: 20/20 PASS, including eval/train and FP32/AMP; zero prediction mismatches.','','| Task | LiteBN BA | LS BA | Outer delta pp | Delta F1 pp | Positive folds | Internal heldout delta pp |','|---|---:|---:|---:|---:|---:|---:|']
    for o,h in zip(summaries['outer'],summaries['heldout']):lines.append(f"| {o['task']} | {o['LiteBN_BA']:.6f} | {o['LS_BA']:.6f} | {o['delta_BA_pp']:+.3f} | {o['delta_macro_F1_pp']:+.3f} | {o['positive_units']}/5 | {h['delta_BA_pp']:+.3f} |")
    lines+=['',f"4-5. Outer and current internal heldout deltas: table above. Heldout is previously exposed diagnostic, never untouched final test.",f"6. Positive tasks: {npos}/4; positive folds {meta['positive_folds']}/20. Four-of-four: {npos==4}.",f"7. Worst task {ds.min():+.3f} pp; equal-task mean {ds.mean():+.3f} pp; median {np.median(ds):+.3f} pp.",'8. Readout contribution (trial-weighted outer, diagnostics not used for selection):']
    for t in TASKS:
        z=[r for r in diags if r['task']==t and r['population']=='outer'];n=sum(r['trials'] for r in z);lines.append(t+': '+', '.join(f'{k}={sum(r[k]*r["trials"] for r in z)/n:.6f}' for k in ['base_norm','stats_norm','stats_base_ratio','mu_contribution_norm','logvar_contribution_norm']))
    lines+=['9. '+('A separately authorized frozen multiseed follow-up may be considered, but this is not established broad upgrade.' if npos>=3 and terminal!='LOCALSTATS_SEED0_NO_BROAD_SIGNAL' else 'No evidence for broad-upgrade continuation; do not expand seeds/tasks.'),'10. No new sealed test accessed. No seed1/2 or ablation was run.','','```text']+[f'{k} = {v}' for k,v in flags.items()]+['```'];(e.out/'FINAL_LOCALSTATS_SEED0_DECISION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(terminal,flush=True)

def main():
    p=argparse.ArgumentParser()
    for name in ['repo','recovered','cache','historical-runtime','runtime']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--phase',choices=['all','preflight','train','outer','heldout','summarize'],required=True);p.add_argument('--task',choices=TASKS);p.add_argument('--fold',type=int,choices=range(5));a=p.parse_args();e=Experiment(a)
    if a.phase=='all':
        def child(phase,t,f=None):
            cmd=[sys.executable,'-u',__file__]
            for name in ['repo','recovered','cache','historical_runtime','runtime']:cmd+=['--'+name.replace('_','-'),str(getattr(a,name))]
            cmd+=['--phase',phase,'--task',t]
            if f is not None:cmd+=['--fold',str(f)]
            subprocess.run(cmd,check=True)
        for t in TASKS:child('preflight',t)
        audits=[];params=[];man=[]
        for t in TASKS:
            r=read(e.out/f'PREFLIGHT_{t}.json');assert r['invariant']==e.invariant;audits+=r['audits'];params+=r['parameters'];man+=r['manifests']
        assert len(audits)==20 and all(r['status']=='PASS' for r in audits);csvwrite(e.out/'INITIAL_FUNCTION_EQUIVALENCE.csv',audits);csvwrite(e.out/'PARAMETER_COUNT.csv',params);csvwrite(e.out/'EXACT_BATCH_AUDIT.csv',man);dump(e.out/'PREFLIGHT_GATE.json',{'pass':True,'invariant':e.invariant,'cells':20})
        for t in TASKS:
            for f in range(5):child('train',t,f)
        for phase in ['outer','heldout']:
            for t in TASKS:
                for f in range(5):child(phase,t,f)
        summarize(e)
    elif a.phase=='preflight':e.prepare(a.task)
    elif a.phase=='train':e.train(a.task,a.fold)
    elif a.phase in ['outer','heldout']:e.evaluate(a.task,a.fold,a.phase)
    else:summarize(e)
if __name__=='__main__':main()
