"""Run one complete task, then fail closed before the next task if outer TSW <= baseline."""
import argparse,ast,copy,csv,gc,hashlib,inspect,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

REPO_HINT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO_HINT/'experiments/persist_eeg_litebn_localstats_seed0_v1/code'))
import run_localstats as foundation
from scale_models import attach

ARCHS=['StaticScale','TSW']; METRICS=['BA','macro_F1','accuracy']
read,sha,dump,csvread,csvwrite=foundation.read,foundation.sha,foundation.dump,foundation.csvread,foundation.csvwrite

def rng_equal(a,b):
    return a['python']==b['python'] and all(np.array_equal(x,y) for x,y in zip(a['numpy'],b['numpy'])) and torch.equal(a['torch'],b['torch']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))

class Experiment(foundation.Experiment):
    def __init__(self,a):
        super().__init__(a);self.exp=a.repo/'experiments/persist_eeg_litebn_tsw_seed0_v1';self.out=self.exp/'outputs';self.out.mkdir(parents=True,exist_ok=True);self.rt=a.runtime;self.rt.mkdir(parents=True,exist_ok=True)
        extra=[Path(__file__),Path(__file__).with_name('scale_models.py'),self.exp/'protocol/TSW_PROTOCOL.md']
        self.hashes.update({str(p):sha(p) for p in extra});self.invariant=hashlib.sha256(json.dumps(self.hashes,sort_keys=True).encode()).hexdigest()
    def cell(self,t,f):return self.rt/f'{t}_fold{f}_preflight'
    def archcell(self,t,f,a):return self.rt/f'{t}_fold{f}_{a.lower()}'
    def input_batch(self,t,b,idx,mean,std):
        if self.mi(t):return self.v.prepare(b,idx,mean,std,self.device)
        raw=torch.from_numpy(b.signal_batch(idx)).to(self.device);return (raw-torch.as_tensor(mean,device=self.device)[None,:,None])/torch.as_tensor(std,device=self.device)[None,:,None].clamp_min(1e-6)
    def preflight(self,t):
        b=self.bundle(t);audits=[];params=[];batchrows=[]
        for f in range(5):
            mean,std,norm=self.normalizer(t,f,b);manifest,batches,mh=self.manifest(t,f,b);base=self.initial(t,f);common=copy.deepcopy(base.state_dict());rng=self.h.rng_state();idx=batches[0][0];x=self.input_batch(t,b,idx,mean,std);y=torch.as_tensor(b.labels(idx),device=self.device)
            for kind in ARCHS:
                # The preceding architecture's train-mode equivalence forward
                # updates BN buffers. Restore the frozen common state before
                # constructing/auditing the next independent architecture.
                base.load_state_dict(common)
                self.h.restore_rng(rng)
                cand=attach(copy.deepcopy(base),kind);assert rng_equal(rng,self.h.rng_state());extra=set(cand.state_dict())-set(common);expected={'beta'} if kind=='StaticScale' else {'lambda_scale','scale_mlp.0.weight','scale_mlp.0.bias','scale_mlp.2.weight','scale_mlp.2.bias'};assert extra==expected
                assert all(torch.equal(v,cand.state_dict()[k]) for k,v in common.items());assert (torch.count_nonzero(cand.beta)==0 if kind=='StaticScale' else torch.count_nonzero(cand.lambda_scale)==0)
                maxdiff=0;mismatch=0
                initial=copy.deepcopy(cand.state_dict())
                for training in [False,True]:
                    for amp in [False,True]:
                        base.load_state_dict(common);cand.load_state_dict(initial);base.train(training);cand.train(training);self.h.set_seed(100000)
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):z0=base(x)[0]
                        after=self.h.rng_state();self.h.set_seed(100000)
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):z1=cand(x)[0]
                        assert rng_equal(after,self.h.rng_state());diff=float((z0-z1).abs().max());mm=int((z0.argmax(1)!=z1.argmax(1)).sum());maxdiff=max(maxdiff,diff);mismatch+=mm
                        assert diff<=1e-6 and mm==0 and all(torch.equal(v,cand.state_dict()[k]) for k,v in base.state_dict().items())
                cand.load_state_dict(initial);cand.eval();cand.zero_grad();F.cross_entropy(cand(x)[0],y).backward();gategrad=float((cand.beta.grad if kind=='StaticScale' else cand.lambda_scale.grad).norm());assert np.isfinite(gategrad) and gategrad>0
                n=sum(p.numel() for p in base.parameters());nc=sum(p.numel() for p in cand.parameters());expected_extra=3 if kind=='StaticScale' else 1252;assert nc-n==expected_extra
                params.append({'task':t,'fold':f,'architecture':kind,'baseline_parameters':n,'model_parameters':nc,'increase':nc-n,'increase_percent':100*(nc-n)/n})
                audits.append({'task':t,'fold':f,'architecture':kind,'common_state_bitwise_identical':True,'extra_state_keys':json.dumps(sorted(extra)),'RNG_restored':True,'modes':'eval/train x FP32/AMP','max_abs_logits_diff':maxdiff,'prediction_mismatch':mismatch,'gate_initial_gradient_norm':gategrad,'status':'PASS'})
                del cand,initial,z0,z1
            for e,epoch in enumerate(batches,1):
                h=hashlib.sha256()
                for ii in epoch:h.update(len(ii).to_bytes(8,'little'));h.update(ii.tobytes())
                batchrows.append({'task':t,'fold':f,'epoch':e,'batches':len(epoch),'trials':sum(map(len,epoch)),'ordered_batches_sha256':h.hexdigest(),'historical_manifest_sha256':mh,'match':True})
            dump(self.cell(t,f)/'gate.json',{'invariant':self.invariant,'pass':True,'mean':mean.tolist(),'std':std.tolist(),'normalizer':norm,'manifest_sha256':mh});print('PREFLIGHT_PASS',t,f,flush=True);del base,common,x,y;gc.collect();torch.cuda.empty_cache()
        csvwrite(self.out/f'{t}_INITIALIZATION_AUDIT.csv',audits);csvwrite(self.out/f'{t}_PARAMETER_COUNT.csv',params);csvwrite(self.out/f'{t}_BATCH_AUDIT.csv',batchrows);return audits,params
    def train_arch(self,t,f,kind):
        cell=self.archcell(t,f,kind);cell.mkdir(parents=True,exist_ok=True);resultpath=cell/'result.json'
        if resultpath.exists():assert read(resultpath)['invariant']==self.invariant;return
        gate=read(self.cell(t,f)/'gate.json');assert gate['invariant']==self.invariant and gate['pass'];b=self.bundle(t);mean,std,norm=self.normalizer(t,f,b);manifest,batches,mh=self.manifest(t,f,b);assert mh==gate['manifest_sha256'];model=attach(self.initial(t,f),kind);self.h.set_seed(100000);fold=dict(self.fold(t,f));fold['_seed']=0;updates=[]
        def observe(epoch,optimizer,scaler):
            steps={int(v['step']) for v in optimizer.state.values()};assert len(steps)==1;updates.append({'task':t,'fold':f,'architecture':kind,'epoch':epoch,'attempts':len(batches[epoch-1]),'trials':sum(map(len,batches[epoch-1])),'successful_cumulative':steps.pop(),'AMP_scale':float(scaler.get_scale())});dump(cell/'updates.json',updates)
        fn=self.h.train_model if self.mi(t) else self.tt.train_one;tree=ast.parse(inspect.getsource(fn));inserted=0
        class Edit(ast.NodeTransformer):
            def visit_Expr(_,node):
                nonlocal inserted
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='history.append':inserted+=1;return [node,ast.parse('observe(epoch, optimizer, scaler)').body[0]]
                return node
        tree=Edit().visit(tree);ast.fix_missing_locations(tree);assert inserted==1;ns=dict(fn.__globals__);ns['observe']=observe
        def restore(state):
            state=dict(state);state['torch']=state['torch'].cpu();state['cuda']=[v.cpu() for v in state['cuda']];self.h.restore_rng(state)
        ns['restore_rng']=restore;exec(compile(tree,str(inspect.getfile(fn))+'[logging only]','exec'),ns)
        if self.mi(t):cache=self.c.GPUCache(b,mean,std,self.device);result=ns[fn.__name__](model,'LiteBN-'+kind,b,fold,manifest,cache,mh,cell,self.device)
        else:
            cache=self.td.RawGPUCache(b,self.device);weights,wi=self.td.class_weights(b,fold['inner_train_subjects']);wi['_normalizer']=norm;ns['RUNTIME']=self.rt/f'{t}_{kind}_runtime';result=ns[fn.__name__](model,'LiteBN-'+kind,t.split('_')[1],fold,b,cache,mean,std,weights,wi,self.device,self.tt.lock_sha())
        assert len(result['history'])==60 and len(updates)==60;result.update(invariant=self.invariant,task=t,fold=f,architecture=kind,normalizer=norm);dump(resultpath,result);print('TRAIN_COMPLETE',t,f,kind,flush=True)
    def evaluate(self,t,f,kind,pop):
        target=self.archcell(t,f,kind)/f'{pop}.json'
        if target.exists():assert read(target)['invariant']==self.invariant;return
        gate=read(self.cell(t,f)/'gate.json');result=read(self.archcell(t,f,kind)/'result.json');mean,std=np.asarray(gate['mean'],np.float32),np.asarray(gate['std'],np.float32);baseline=self.initial(t,f);baseline.load_state_dict(torch.load(self.reference(t,f),map_location=self.device,weights_only=True));baseline.eval();candidate=attach(self.initial(t,f),kind);assert sha(result['checkpoint_path'])==result['checkpoint_sha256'];candidate.load_state_dict(torch.load(result['checkpoint_path'],map_location=self.device,weights_only=True));candidate.eval();candidate.collect_scale_diagnostics=kind=='TSW';subjects=self.fold(t,f)['outer_dev_subjects'] if pop=='outer' else self.hold[self.dataset(t)]['subject_ids'];hist,hpath=self.historical(t,f,pop);assert set(subjects)==set(hist);rows=[];diag=[]
        for s in subjects:
            candidate.scale_diagnostic_chunks=[]
            if self.mi(t):x,y=self.lf.load_eval_subject(self.dataset(t),s);zs=[self.rf._logits(m,x,mean,std,self.device) for m in [baseline,candidate]];scores=[self.rf._metric(y,z.argmax(1)) for z in zs]
            else:
                b=self.td.load_bundle(t.split('_')[1],[s],sessions=(2,));idx=b.indices([s],(2,));y=b.labels(idx)
                class Cache:
                    def batch(_,ii,mu,sd):
                        x=torch.from_numpy(b.signal_batch(ii)).to(self.device);return (x-torch.as_tensor(mu,device=self.device)[None,:,None])/torch.as_tensor(sd,device=self.device)[None,:,None].clamp_min(1e-6),None
                zs=[self.es.logits(m,Cache(),idx,mean,std) for m in [baseline,candidate]];scores=[self.es.score(t.split('_')[1],y,z) for z in zs]
            row={'task':t,'fold':f,'architecture':kind,'population':pop,'subject_id':s,'trials':len(y)}
            for k in METRICS:assert abs(scores[0][k]-hist[s][k])<=1e-10;row.update({f'LiteBN_{k}':scores[0][k],f'model_{k}':scores[1][k],f'delta_{k}_pp':100*(scores[1][k]-scores[0][k])})
            rows.append(row)
            if kind=='TSW':
                aa=torch.cat([q[0] for q in candidate.scale_diagnostic_chunks]).numpy();ss=torch.cat([q[1] for q in candidate.scale_diagnostic_chunks]).numpy();assert len(aa)==len(y)
                for group,label,mask in [('subject',str(s),np.ones(len(y),bool))]+[('class',str(c),y==c) for c in np.unique(y)]:
                    a0=aa[mask];s0=ss[mask];dominant=a0.argmax(1);diag.append({'task':t,'fold':f,'population':pop,'group_type':group,'group_id':label,'trials':len(a0),**{f'alpha{k}_{stat}':float(getattr(np,kern)(a0[:,i])) for i,k in enumerate([15,63,127]) for stat,kern in [('mean','mean'),('std','std')]},**{f'scale{k}_{stat}':float(getattr(np,kern)(s0[:,i])) for i,k in enumerate([15,63,127]) for stat,kern in [('mean','mean'),('std','std')]},'scale_min':float(s0.min()),'scale_max':float(s0.max()),**{f'dominant{k}_percent':100*float((dominant==i).mean()) for i,k in enumerate([15,63,127])}})
        extra={'lambda_scale':float(candidate.lambda_scale.detach()) if kind=='TSW' else None,'tanh_lambda':float(torch.tanh(candidate.lambda_scale).detach()) if kind=='TSW' else None,'diagnostics':diag,'baseline_path':str(hpath),'baseline_sha256':sha(hpath),'baseline_checkpoint_sha256':sha(self.reference(t,f)),'baseline_replay_pass':True};dump(target,{'invariant':self.invariant,'rows':rows,**extra});print('EVAL_COMPLETE',t,f,kind,pop,flush=True)
    def task_summary(self,t):
        import pandas as pd
        pairs=[];diags=[];prov=[]
        for f in range(5):
            for kind in ARCHS:
                for pop in ['outer','heldout']:
                    r=read(self.archcell(t,f,kind)/f'{pop}.json');pairs+=r['rows'];diags+=r['diagnostics'];prov.append({'task':t,'fold':f,'architecture':kind,'population':pop,'baseline_checkpoint_sha256':r['baseline_checkpoint_sha256'],'baseline_result_sha256':r['baseline_sha256'],'baseline_replay_pass':r['baseline_replay_pass']})
        df=pd.DataFrame(pairs);outer=df[df.population=='outer'];held=df[df.population=='heldout'];fold=outer.groupby(['architecture','fold'],as_index=False)[['LiteBN_BA','model_BA','delta_BA_pp','LiteBN_macro_F1','model_macro_F1','delta_macro_F1_pp','LiteBN_accuracy','model_accuracy','delta_accuracy_pp']].mean();rows=[]
        for kind in ARCHS:
            z=fold[fold.architecture==kind];rows.append({'task':t,'architecture':kind,'LiteBN_BA':float(z.LiteBN_BA.mean()),'model_BA':float(z.model_BA.mean()),'delta_BA_pp':float(z.delta_BA_pp.mean()),'positive_folds':int((z.delta_BA_pp>1e-10).sum()),'negative_folds':int((z.delta_BA_pp<-1e-10).sum()),'tied_folds':int((abs(z.delta_BA_pp)<=1e-10).sum()),'fold_SD_pp':100*float(z.model_BA.std(ddof=0))})
        static,dynamic=rows;static['delta_vs_StaticScale_pp']=0.0;dynamic['delta_vs_StaticScale_pp']=100*(dynamic['model_BA']-static['model_BA']);status='PASS' if dynamic['delta_BA_pp']>0 else 'FAIL';gate={'task':t,'status':status,'criterion':'five-fold mean outer TSW delta vs LiteBN > 0','TSW_delta_BA_pp':dynamic['delta_BA_pp'],'Static_delta_BA_pp':static['delta_BA_pp'],'TSW_vs_Static_pp':dynamic['delta_vs_StaticScale_pp'],'next_task_authorized':status=='PASS'}
        held_subject=held.groupby(['architecture','subject_id'],as_index=False)[['LiteBN_BA','model_BA','delta_BA_pp','LiteBN_macro_F1','model_macro_F1','delta_macro_F1_pp','LiteBN_accuracy','model_accuracy','delta_accuracy_pp']].mean()
        held_summary=[]
        for kind in ARCHS:
            z=held_subject[held_subject.architecture==kind];held_summary.append({'task':t,'architecture':kind,**{c:float(z[c].mean()) for c in ['LiteBN_BA','model_BA','delta_BA_pp','LiteBN_macro_F1','model_macro_F1','delta_macro_F1_pp','LiteBN_accuracy','model_accuracy','delta_accuracy_pp']},'positive_subjects':int((z.delta_BA_pp>1e-10).sum()),'negative_subjects':int((z.delta_BA_pp<-1e-10).sum()),'tied_subjects':int((abs(z.delta_BA_pp)<=1e-10).sum()),'median_subject_delta_pp':float(z.delta_BA_pp.median())})
        training=[]
        for f in range(5):
            for kind in ARCHS:
                r=read(self.archcell(t,f,kind)/'result.json');training.append({k:r[k] for k in ['task','fold','seed','architecture','selected_epoch','best_inner_val_BA','checkpoint_sha256','manifest_sha256','normalizer','amp','epochs_completed'] if k in r})
        fold.to_csv(self.out/f'{t}_OUTER_FOLD_RESULTS.csv',index=False);csvwrite(self.out/f'{t}_TASK_SUMMARY.csv',rows);csvwrite(self.out/f'{t}_BASELINE_PROVENANCE.csv',prov);csvwrite(self.out/f'{t}_TSW_SCALE_DIAGNOSTICS.csv',diags);held_subject.to_csv(self.out/f'{t}_INTERNAL_HELDOUT_SUBJECT_RESULTS.csv',index=False);csvwrite(self.out/f'{t}_INTERNAL_HELDOUT_TASK_SUMMARY.csv',held_summary);df.to_csv(self.out/f'{t}_PAIRED_SUBJECT_RESULTS.csv',index=False);dump(self.out/f'{t}_TRAINING_METADATA.json',{'task':t,'seed':0,'invariant_at_training':read(self.archcell(t,0,'TSW')/'result.json')['invariant'],'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},'cells':training});dump(self.out/f'{t}_TASK_GATE.json',gate);print('TASK_GATE',json.dumps(gate),flush=True);return gate

def main():
    p=argparse.ArgumentParser()
    for n in ['repo','recovered','cache','historical-runtime','runtime']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--task',choices=['OpenBMI_MI','WBCIC_MI','OpenBMI_ERP','OpenBMI_SSVEP'],required=True);p.add_argument('--summarize-only',action='store_true');a=p.parse_args();e=Experiment(a)
    if a.summarize_only:
        gate=e.task_summary(a.task);raise SystemExit(0 if gate['status']=='PASS' else 20)
    aud,par=e.preflight(a.task);assert len(aud)==10 and all(r['status']=='PASS' for r in aud)
    for kind in ARCHS:
        for f in range(5):e.train_arch(a.task,f,kind)
    for kind in ARCHS:
        for pop in ['outer','heldout']:
            for f in range(5):e.evaluate(a.task,f,kind,pop)
    gate=e.task_summary(a.task);raise SystemExit(0 if gate['status']=='PASS' else 20)
if __name__=='__main__':main()
