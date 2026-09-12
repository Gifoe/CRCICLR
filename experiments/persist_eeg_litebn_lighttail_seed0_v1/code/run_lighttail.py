"""Single-loss intervention in the recovered historical LiteBN training loop."""
import argparse,ast,copy,csv,gc,hashlib,importlib,inspect,json,os,sys
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from legacy_init import construct_exact

TASK='OpenBMI_MI';SEED=0;TAIL_WEIGHT=.1
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.part');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,p)
def csvwrite(p,rows):
    assert rows;p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def subject_loss(logits,y,groups,weight):
    ce=F.cross_entropy(logits,y,reduction='none')
    names,counts=torch.unique(groups,sorted=True,return_counts=True)
    assert len(names)==8 and bool((counts==16).all()),'Historical subject-balanced episode required'
    risks=torch.stack([ce[groups==s].mean() for s in names]);mean=risks.mean();tail=risks.topk(4).values.mean()
    return (1-weight)*mean+weight*tail,ce.mean(),mean,tail,risks,names

class Experiment:
    def __init__(self,a):
        self.a=a;self.exp=a.repo/'experiments/persist_eeg_litebn_lighttail_seed0_v1';self.out=self.exp/'outputs';self.rt=a.runtime;self.rt.mkdir(parents=True,exist_ok=True)
        os.environ.update(R2EEG_REPO=str(a.repo),PERSIST_OPENBMI_CACHE=str(a.cache/'openbmi/openbmi'),PERSIST_WBCIC_CACHE=str(a.cache/'wbcic/wbcic_epochs'))
        self.snap=a.recovered/'snapshot';carrier=self.snap/'persist_eeg_carrier_5fold_multiseed_stability_v1'
        sys.path.insert(0,str(carrier/'code'));self.h=importlib.import_module('train_grid');self.v=self.h.v1;self.c=self.h.carrier
        assert Path(self.h.__file__).resolve()==(carrier/'code/train_grid.py').resolve()
        assert inspect.getsource(self.c.CompactLite).replace('\r','') in (self.snap/'run_carrier_screen.py').read_text().replace('\r','')
        self.splitpath=carrier/'protocol/FIVEFOLD_SPLIT.json';self.split=read(self.splitpath);self.folds=self.split['folds']['OpenBMI'];self.search=self.split['search_subjects']['OpenBMI']
        self.h.validate_split({'OpenBMI':self.search},{'OpenBMI':self.folds})
        self.manmeta=read(carrier/'protocol/MANIFEST_HASHES.json');self.oldlogs=read(carrier/'outputs/TRAINING_LOGS.json');self.device=torch.device('cuda');assert torch.cuda.is_available()
        torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
        self.bundle=self.v.load_bundle('OpenBMI',self.search)
        self.invariant=hashlib.sha256(json.dumps({'code':sha(__file__),'init_recovery':sha(Path(__file__).with_name('legacy_init.py')),'historical_training':sha(self.h.__file__),'historical_model':sha(self.c.__file__),'historical_sampler':sha(self.v.__file__),'core':sha(self.v.core.__file__),'split':sha(self.splitpath),'tail_weight':TAIL_WEIGHT},sort_keys=True).encode()).hexdigest()
        self.manifests={};self.normalizers={};self.audits=[];self.eq=[]
        dump(self.out/'SOURCE_CODE_PROVENANCE.json',{'invariant':self.invariant,'files':{str(p):sha(p) for p in [Path(__file__),Path(self.h.__file__),Path(self.v.__file__),Path(self.v.core.__file__),Path(self.c.__file__)]},'source_inspection_commit':read(a.recovered/'TRANSFER_RECEIPT.json')['source_repository_inspection_commit'],'checkpoint_creation_commit':'not recorded; original init, manifest, normalizer and selected binary hashes verified'})
    def old(self,f):return next(r for r in self.oldlogs if r['dataset']=='OpenBMI' and r['model']=='LiteBN' and r['seed']==0 and r['fold']==f)
    def init(self,f):
        reference=self.a.recovered/f'checkpoints/OpenBMI_MI/fold{f}_seed0/selected_best.pt'
        assert sha(reference)==self.old(f)['checkpoint_sha256']
        m,actual=construct_exact(self.h.constructor,self.h.set_seed,reference,self.old(f)['init_sha256'],self.device)
        dump(self.out/f'INITIALIZATION_FOLD{f}.json',{'historical_init_sha256':self.old(f)['init_sha256'],'reconstructed_historical_layout_sha256':actual,'reference_checkpoint_sha256':sha(reference),'tensor_preserving_container_translation':True,'pass':True})
        return m
    def prepare(self):
        # Audit all five folds and every episode before any candidate training.
        for f in self.folds:
            fid=f['fold_id'];meta=next(r for r in self.manmeta if r['dataset']=='OpenBMI' and r['fold']==fid)
            p=self.a.historical_runtime/f'openbmi_fold{fid}/manifest_runtime/episode_manifests/openbmi_fold{fid}.json'
            assert p.is_file(),'EXACT_MANIFEST_NOT_RECOVERABLE'
            original=read(p);assert sha(p)==meta['sha256']==self.old(fid)['manifest_sha256'],'Historical manifest SHA mismatch'
            self.v.RUNTIME=self.rt/f'reconstructed/fold{fid}';manifest,info=self.v.make_manifest(self.bundle,f);reconstructed=read(info['path'])
            # Windows newline bytes can differ: compare full payload plus LF hash.
            lfhash=hashlib.sha256(Path(info['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            assert reconstructed==original and lfhash==sha(p),'EXACT_MANIFEST_NOT_RECOVERABLE'
            assert len(manifest)==60 and original['steps_per_epoch']==21
            for epoch,(old,new) in enumerate(zip(original['epochs'],manifest),1):
                assert len(old)==len(new)==21
                for ei,(o,n) in enumerate(zip(old,new)):
                    indices=n['support_indices']+n['query_indices'];origi=o['support_indices']+o['query_indices'];rr=[self.bundle.search_rows[i] for i in indices]
                    ss=[r.subject for r in rr];se=[r.session for r in rr];trial=[r.cache_index for r in rr]
                    assert n==o and indices==origi and len(indices)==128 and len(set(ss))==8 and set(Counter(ss).values())=={16}
                    assert set(ss)<=set(f['inner_train_subjects']) and se[:64]==[1]*64 and se[64:]==[2]*64
                    self.audits.append({'fold':fid,'epoch':epoch,'episode':ei,'historical_episodes_per_epoch':len(old),'candidate_episodes_per_epoch':len(new),'historical_trials':len(origi),'candidate_trials':len(indices),'subject_ids':json.dumps(ss),'session_ids':json.dumps(se),'bundle_trial_indices':json.dumps(indices),'within_file_trial_indices':json.dumps(trial),'subject_composition_equal':True,'session_composition_equal':True,'trial_indices_equal':True,'sample_order_equal':True,'episode_order_equal':True,'historical_manifest_sha256':sha(p),'reconstructed_LF_sha256':lfhash})
            mean,std,norm=self.v.normalizer(self.bundle,f['inner_train_subjects']);assert norm==meta['normalizer'],'Historical normalizer mismatch'
            self.manifests[fid]=original['epochs'];self.normalizers[fid]=(mean,std,norm)
            model=self.init(fid);del model
            print('EXACT_MANIFEST_PASS',fid,'60x21x128',flush=True)
        csvwrite(self.out/'EXACT_MANIFEST_AUDIT.csv',self.audits)
        for f in self.folds:self.equivalence(f)
        csvwrite(self.out/'TAIL0_EQUIVALENCE_AUDIT.csv',self.eq)
        dump(self.out/'PREFLIGHT_GATE.json',{'status':'PASS','invariant':self.invariant,'manifest_episodes':len(self.audits),'equivalence_tests':len(self.eq),'all_initialization_hashes_match':True})
    def groups(self,ep):
        idx=ep['support_indices']+ep['query_indices'];return torch.tensor([int(self.bundle.search_rows[i].subject) for i in idx],device=self.device)
    def equivalence(self,f):
        fid=f['fold_id'];mean,std,_=self.normalizers[fid];ep=self.manifests[fid][0][0];indices=ep['support_indices']+ep['query_indices'];x=self.v.prepare(self.bundle,np.asarray(indices),mean,std,self.device);y=torch.as_tensor(self.bundle.labels(np.asarray(indices)),device=self.device);groups=self.groups(ep)
        for amp in [False,True]:
            m1=self.init(fid);m2=copy.deepcopy(m1);o1=torch.optim.AdamW(m1.parameters(),lr=3e-4,weight_decay=5e-4);o2=torch.optim.AdamW(m2.parameters(),lr=3e-4,weight_decay=5e-4);g1=torch.amp.GradScaler('cuda',enabled=amp);g2=torch.amp.GradScaler('cuda',enabled=amp)
            def step(m,opt,scaler,tail0):
                self.h.set_seed(100000);m.train();opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
                    z,_=m(x);loss=subject_loss(z,y,groups,0.)[0] if tail0 else F.cross_entropy(z,y)
                scaler.scale(loss).backward();scaler.unscale_(opt);grads=[p.grad.detach().clone() for p in m.parameters()];torch.nn.utils.clip_grad_norm_(m.parameters(),5.);scaler.step(opt);scaler.update();return float(loss.detach()),grads
            l1,d1=step(m1,o1,g1,False);l2,d2=step(m2,o2,g2,True)
            assert all(torch.isfinite(g).all() for g in d1+d2),'TAIL0_EQUIVALENCE_NONFINITE_GRADIENT'
            ld=abs(l1-l2);gd=max((x-y).abs().max().item() for x,y in zip(d1,d2));pd=max((x-y).abs().max().item() for x,y in zip(m1.parameters(),m2.parameters()))
            assert ld<=1e-6 and all(torch.allclose(x,y,atol=1e-6,rtol=1e-5) for x,y in zip(d1,d2)) and pd<=1e-6,'TAIL0_EQUIVALENCE_FAILED'
            assert o1.state and o2.state,'One-step test must perform an optimizer update'
            self.eq.append({'fold':fid,'amp':amp,'episode_epoch':1,'episode_index':0,'ordinary_CE':l1,'tail0_loss':l2,'scalar_abs_diff':ld,'max_abs_gradient_diff':gd,'max_abs_parameter_update_diff':pd,'scalar_atol':1e-6,'gradient_atol':1e-6,'gradient_rtol':1e-5,'parameter_atol':1e-6,'status':'PASS'})
            del m1,m2,o1,o2,g1,g2,d1,d2;gc.collect();torch.cuda.empty_cache()
        print('TAIL0_EQUIVALENCE_PASS',fid,flush=True)
    def train(self):
        gate=read(self.out/'PREFLIGHT_GATE.json');assert gate['status']=='PASS' and gate['invariant']==self.invariant
        for f in self.folds:
            fid=f['fold_id'];cell=self.rt/f'fold{fid}';resultpath=cell/'result.json'
            if resultpath.exists():assert read(resultpath)['invariant']==self.invariant;continue
            self.active_fold=fid;self.episode_records=[];self.trajectory=[];self.epoch_success_previous=0
            mean,std,_=self.normalizers[fid];cache=self.c.GPUCache(self.bundle,mean,std,self.device);model=self.init(fid)
            if (cell/'candidate_invariant.json').exists():assert read(cell/'candidate_invariant.json')['invariant']==self.invariant
            else:dump(cell/'candidate_invariant.json',{'invariant':self.invariant})
            # Compile the historical function with exactly one mathematical edit:
            # the loss expression. Add observational logging after its history row.
            tree=ast.parse(inspect.getsource(self.h.train_model));changed=0;logged=0
            class Edit(ast.NodeTransformer):
                def visit_Assign(_,node):
                    nonlocal changed
                    if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='loss':
                        changed+=1;node.value=ast.parse('instrumented_loss(logits, y, episode, epoch)',mode='eval').body
                    return node
                def visit_Expr(_,node):
                    nonlocal logged
                    if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='history' and node.value.func.attr=='append':
                        logged+=1;return [node,ast.parse('observe_epoch(epoch, val_rows, val_ba, selected, optimizer, scaler)').body[0]]
                    return node
            tree=Edit().visit(tree);assert changed==logged==1;ast.fix_missing_locations(tree)
            scope=dict(vars(self.h));scope.update(instrumented_loss=self.loss,observe_epoch=self.epoch_end)
            # Engineering-only RNG restore: CPU RNG tensors must stay on CPU.
            def restore(state):
                fixed=dict(state);fixed['torch']=state['torch'].cpu()
                if 'cuda' in fixed:fixed['cuda']=[v.cpu() for v in fixed['cuda']]
                self.h.restore_rng(fixed)
            scope['restore_rng']=restore;exec(compile(tree,str(self.h.__file__)+'[loss-only instrumentation]','exec'),scope)
            # Record transformed code for a human-verifiable loss-only audit.
            (self.out/f'HISTORICAL_LOOP_INSTRUMENTED_FOLD{fid}.py').write_text(ast.unparse(tree)+'\n',encoding='utf-8')
            if (cell/'trajectory.json').exists():
                saved=torch.load(cell/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
                self.trajectory=[r for r in read(cell/'trajectory.json') if r['epoch']<=saved['epoch']]
                stepcounts={int(v['step']) for v in saved['optimizer']['state'].values()}
                assert len(stepcounts)==1;self.epoch_success_previous=stepcounts.pop()
            result=scope['train_model'](model,'LiteBN-LightTail',self.bundle,f,self.manifests[fid],cache,self.old(fid)['manifest_sha256'],cell,self.device)
            assert len(result['history'])==60
            dump(resultpath,{'invariant':self.invariant,**result});print('LIGHTTAIL_FOLD_COMPLETE',fid,flush=True)
            del model,cache;gc.collect();torch.cuda.empty_cache()
        allrows=[]
        for f in range(5):allrows+=read(self.rt/f'fold{f}/trajectory.json')
        assert len(allrows)==300;csvwrite(self.out/'LIGHTTAIL_TRAINING_TRAJECTORY.csv',allrows)
    def loss(self,z,y,ep,epoch):
        groups=self.groups(ep);loss,ce,mean,tail,risks,names=subject_loss(z,y,groups,TAIL_WEIGHT)
        assert abs(float(ce.detach())-float(mean.detach()))<=1e-5,'Subject mean must equal batch CE'
        rr=risks.detach().float().cpu().tolist();nn=names.cpu().tolist();hard=risks.detach().topk(4).indices.cpu().tolist()
        self.episode_records.append({'fold':self.active_fold,'epoch':epoch,'episode':len(self.episode_records),'historical_mean_CE':float(ce.detach()),'mean_subject_risk':float(mean.detach()),'tail_risk':float(tail.detach()),'total_loss':float(loss.detach()),'hardest_subjects':[nn[i] for i in hard],'per_subject_risk':dict(zip(map(str,nn),rr))})
        return loss
    def epoch_end(self,epoch,val_rows,val_ba,selected,optimizer,scaler):
        fid=self.active_fold;records=self.episode_records;assert len(records)==21
        subject_exposure=Counter();session_exposure=Counter()
        for ep in self.manifests[fid][epoch-1]:
            for i in ep['support_indices']+ep['query_indices']:
                row=self.bundle.search_rows[i];subject_exposure[row.subject]+=1;session_exposure[str(row.session)]+=1
        steps={int(s['step'].item()) for s in optimizer.state.values()};assert len(steps)==1;total=steps.pop();successful=total-self.epoch_success_previous;self.epoch_success_previous=total
        row={'fold':fid,'epoch':epoch,**{k:float(np.mean([r[k] for r in records])) for k in ['historical_mean_CE','mean_subject_risk','tail_risk','total_loss']},'hardest_subjects':json.dumps([r['hardest_subjects'] for r in records]),'per_subject_risk':json.dumps([r['per_subject_risk'] for r in records]),'validation_BA':val_ba,'validation_macro_F1':float(np.mean([r['macro_F1'] for r in val_rows.values()])),'selected':bool(selected),'optimizer_update_attempts':21,'successful_optimizer_updates':successful,'cumulative_successful_optimizer_updates':total,'AMP_skipped_updates':21-successful,'AMP_scale':float(scaler.get_scale()),'trial_exposure':sum(subject_exposure.values()),'subject_exposure':json.dumps(dict(subject_exposure),sort_keys=True),'session_exposure':json.dumps(dict(session_exposure),sort_keys=True)}
        assert row['trial_exposure']==2688 and session_exposure==Counter({'1':1344,'2':1344})
        self.trajectory=[r for r in self.trajectory if r['epoch']<epoch]+[row]
        dump(self.rt/f'fold{fid}/trajectory.json',self.trajectory);dump(self.rt/f'fold{fid}/episodes/epoch{epoch:02d}.json',records);self.episode_records=[]
    def evaluate(self):
        # Original heldout pipeline, loaded only after all five candidate folds.
        final=self.snap/'persist_eeg_final_heldout_confirmation_v1';sys.path.insert(0,str(final/'code'));os.environ['PERSIST_OPENBMI_CACHE']=str(self.a.cache/'openbmi/openbmi')
        lf=importlib.import_module('load_final_carriers');rf=importlib.import_module('run_final_confirmation');held=read(final/'protocol/FINAL_HOLDOUT_MANIFEST.json')['OpenBMI']['subject_ids'];assert not set(held)&set(self.search)
        pairs=[];replay=[]
        for pop in ['outer','heldout']:
            for f in self.folds:
                fid=f['fold_id'];result=read(self.rt/f'fold{fid}/result.json');assert result['epochs_completed']==60 and result['invariant']==self.invariant
                mean,std,_=self.normalizers[fid];subjects=f['outer_dev_subjects'] if pop=='outer' else held
                models={}
                for name,path in [('original',self.a.recovered/f'checkpoints/OpenBMI_MI/fold{fid}_seed0/selected_best.pt'),('lighttail',Path(result['checkpoint_path']))]:
                    m=self.h.constructor('LiteBN',62).to(self.device);m.load_state_dict(torch.load(path,map_location=self.device,weights_only=True),strict=True);m.eval();models[name]=m
                historyfile=self.snap/('persist_eeg_carrier_5fold_multiseed_stability_v1/outputs/SUBJECT_SEED_RESULTS.csv' if pop=='outer' else 'persist_eeg_final_heldout_confirmation_v1/outputs/REPLICATE_SUBJECT_RESULTS.csv')
                with historyfile.open() as h:hist={r['subject_id']:r for r in csv.DictReader(h) if r['dataset']=='OpenBMI' and int(r['fold'])==fid and int(r['seed'])==0 and (pop=='outer' or r['method']=='LiteBN')}
                for subject in subjects:
                    x,y=lf.load_eval_subject('OpenBMI',subject);scores={name:rf._metric(y,rf._logits(m,x,mean,std,self.device).argmax(1)) for name,m in models.items()}
                    row={'fold':fid,'population':pop,'subject_id':subject}
                    for k in ['BA','macro_F1','accuracy']:
                        expected=float(hist[subject][('LiteBN_' if pop=='outer' else '')+k]);assert abs(scores['original'][k]-expected)<=1e-10,'Original baseline replay mismatch'
                        row.update({f'original_{k}':scores['original'][k],f'lighttail_{k}':scores['lighttail'][k],f'delta_{k}_pp':100*(scores['lighttail'][k]-scores['original'][k])})
                    pairs.append(row)
                print('EVALUATED',fid,pop,flush=True);del models;gc.collect();torch.cuda.empty_cache()
        csvwrite(self.out/'LIGHTTAIL_PAIRED_REPLICATES.csv',pairs);self.summarize(pairs)
    def summarize(self,pairs):
        import pandas as pd
        df=pd.DataFrame(pairs);cols=[c for c in df if c.startswith(('original_','lighttail_','delta_'))];outer=df[df.population=='outer'];held=df[df.population=='heldout'].groupby('subject_id',as_index=False)[cols].mean();fold=outer.groupby('fold',as_index=False)[cols].mean()
        fold.to_csv(self.out/'LIGHTTAIL_OUTER_FOLD_RESULTS.csv',index=False);held.to_csv(self.out/'LIGHTTAIL_HELDOUT_SUBJECT_RESULTS.csv',index=False)
        osummary={c:float(fold[c].mean()) for c in cols};osummary.update(original_fold_SD=float(fold.original_BA.std(ddof=0)),lighttail_fold_SD=float(fold.lighttail_BA.std(ddof=0)),positive_folds=int((fold.delta_BA_pp>1e-8).sum()),negative_folds=int((fold.delta_BA_pp<-1e-8).sum()),tied_folds=int((abs(fold.delta_BA_pp)<=1e-8).sum()))
        hsummary={c:float(held[c].mean()) for c in cols};delta=held.delta_BA_pp.to_numpy();draw=np.random.default_rng(0).choice(delta,(10000,len(delta)),replace=True).mean(1);hsummary.update(CI_low_pp=float(np.quantile(draw,.025)),CI_high_pp=float(np.quantile(draw,.975)),positive_subjects=int((delta>1e-8).sum()),negative_subjects=int((delta<-1e-8).sum()),tied_subjects=int((abs(delta)<=1e-8).sum()))
        csvwrite(self.out/'LIGHTTAIL_OUTER_SUMMARY.csv',[osummary]);csvwrite(self.out/'LIGHTTAIL_HELDOUT_SUMMARY.csv',[hsummary])
        lower=[]
        for pop,z in [('outer',outer),('heldout',held)]:
            d=z.delta_BA_pp.to_numpy();hard=z.nsmallest(max(1,int(np.ceil(len(z)*.25))),'original_BA')
            lower.append({'population':pop,'subjects':len(z),'mean_delta_BA_pp':float(d.mean()),'median_delta_BA_pp':float(np.median(d)),'q25_delta_BA_pp':float(np.quantile(d,.25)),'q10_delta_BA_pp':float(np.quantile(d,.1)),'worst_subject_delta_BA_pp':float(d.min()),'positive_subject_ratio':float((d>1e-8).mean()),'baseline_hardest_quartile_mean_delta_pp':float(hard.delta_BA_pp.mean()),'baseline_hardest_quartile_subjects':json.dumps(hard.subject_id.tolist())})
        csvwrite(self.out/'LIGHTTAIL_LOWER_TAIL_ANALYSIS.csv',lower)
        # Frozen operational thresholds: >=0.5pp both, no fold <=-5pp,
        # no SD increase >1pp; hardest quartile improves in both populations.
        useful=(osummary['delta_BA_pp']>=.5 and hsummary['delta_BA_pp']>=.5 and fold.delta_BA_pp.min()>-5 and 100*(osummary['lighttail_fold_SD']-osummary['original_fold_SD'])<=1 and all(r['baseline_hardest_quartile_mean_delta_pp']>0 for r in lower))
        decision='LIGHTTAIL_SIGNAL_FOUND' if useful else 'LIGHTTAIL_NO_USEFUL_SIGNAL'
        trajectories=[r for f in range(5) for r in read(self.rt/f'fold{f}/trajectory.json')]
        assert len(trajectories)==300 and all(r['optimizer_update_attempts']==21 and r['trial_exposure']==2688 for r in trajectories)
        metadata={'decision':decision,'TASK':TASK,'SEED':0,'FOLDS':5,'EXACT_HISTORICAL_MANIFEST':'YES','TRAINING_EXPOSURE_MATCHED':'YES','TAIL_WEIGHT':.1,'SAMPLER_CHANGED':'NO','BATCH_SIZE_CHANGED':'NO','CHECKPOINT_SELECTION_CHANGED':'NO','NORMALIZATION_CHANGED':'NO','OUTER_DEVELOPMENT_EVALUATED':'YES','CURRENT_INTERNAL_HELDOUT_EVALUATED':'YES','NEW_SEALED_FINAL_DATA_ACCESSED':'NO','invariant':self.invariant,'epochs_per_fold':60,'episodes_per_epoch':21,'batch_size':128,'updates_attempted':6300,'actual_successful_updates':sum(r['successful_optimizer_updates'] for r in trajectories),'AMP_skipped_updates':sum(r['AMP_skipped_updates'] for r in trajectories),'outer':osummary,'heldout':hsummary,'lower_tail':lower,'heldout_aggregation':'mean five seed0 fold metrics per subject, then equal subject mean; bootstrap 10000 subject resamples seed0','source_commit_note':'creation commit absent; source code, initialization serialization hash, normalizer and exact manifest matched'}
        dump(self.out/'LIGHTTAIL_METADATA.json',metadata)
        lines=['# LiteBN-LightTail seed0 final decision','',decision,'',f"Exact manifest: all 6,300 episodes matched; exposure 21 x 128 trials per epoch, 60 epochs per fold. Tail0 tests passed for all folds in full precision and historical AMP.",'',f"Outer BA: {osummary['original_BA']:.6f} -> {osummary['lighttail_BA']:.6f}, delta {osummary['delta_BA_pp']:+.3f} pp; positive/negative/tied folds {osummary['positive_folds']}/{osummary['negative_folds']}/{osummary['tied_folds']}.",f"Current internal heldout BA: {hsummary['original_BA']:.6f} -> {hsummary['lighttail_BA']:.6f}, delta {hsummary['delta_BA_pp']:+.3f} pp; paired bootstrap 95% CI [{hsummary['CI_low_pp']:+.3f}, {hsummary['CI_high_pp']:+.3f}] pp.",f"Fold SD: {100*osummary['original_fold_SD']:.3f} -> {100*osummary['lighttail_fold_SD']:.3f} pp (ddof=0).",'','Lower-tail subjects (delta BA pp):']
        for r in lower:lines.append(f"- {r['population']}: median {r['median_delta_BA_pp']:+.3f}; q25 {r['q25_delta_BA_pp']:+.3f}; q10 {r['q10_delta_BA_pp']:+.3f}; worst {r['worst_subject_delta_BA_pp']:+.3f}; positive ratio {r['positive_subject_ratio']:.3f}; baseline-hardest-quartile mean {r['baseline_hardest_quartile_mean_delta_pp']:+.3f}.")
        lines += ['', 'Worth a separately authorized multiseed/task follow-up under the frozen diagnostic thresholds; no generalization claim yet.' if useful else 'Does not meet the frozen continuation criteria. Do not expand seeds/tasks or change tail weight based on this run.','',f"AMP successful updates {metadata['actual_successful_updates']}/6300 attempts; skipped updates {metadata['AMP_skipped_updates']}. Both attempts and complete sample exposure match the historical manifest. AMP scaler behavior is unchanged; any overflow is logged.",'','This is the already-open current internal heldout diagnostic, not untouched final data. No new population or split was created.','', '```text']+[f'{k} = {v}' for k,v in metadata.items() if k.isupper()]+['```']
        (self.out/'FINAL_LIGHTTAIL_DECISION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(decision,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--recovered',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--historical-runtime',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--phase',choices=['preflight','all'],required=True);a=p.parse_args();e=Experiment(a);e.prepare()
    if a.phase=='all':e.train();e.evaluate()
if __name__=='__main__':main()
