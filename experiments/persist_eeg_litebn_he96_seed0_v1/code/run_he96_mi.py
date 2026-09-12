import sys,json,copy,hashlib
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'experiments/persist_eeg_litebn_tsw_seed0_v1/code'),str(Path(__file__).parent)]
import run_tsw_task as t
from he96 import attach
t.ARCHS=['HE96'];t.attach=attach
class E(t.Experiment):
 def __init__(self,a):
  super().__init__(a);self.exp=a.repo/'experiments/persist_eeg_litebn_he96_seed0_v1';self.out=self.exp/'outputs';self.out.mkdir(parents=True,exist_ok=True);self.rt=a.runtime;owned=[Path(__file__),Path(__file__).with_name('he96.py'),self.exp/'protocol/HE96_PROTOCOL.md'];self.invariant=hashlib.sha256((self.invariant+''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in owned)).encode()).hexdigest()
 def preflight(self,task):
  b=self.bundle(task);rows=[];pars=[]
  for f in range(5):
   mean,std,norm=self.normalizer(task,f,b);_,batches,mh=self.manifest(task,f,b);base=self.initial(task,f);common=copy.deepcopy(base.state_dict());rng=self.h.rng_state();x=self.input_batch(task,b,batches[0][0],mean,std)
   cand=attach(copy.deepcopy(base));assert t.rng_equal(rng,self.h.rng_state());initial=copy.deepcopy(cand.state_dict());zero=bool(torch.count_nonzero(cand.point2.weight[:,64:])==0);md=me=0;mm=0;mode_audit=[]
   for train in [False,True]:
    for amp in [False,True]:
     base.load_state_dict(common);cand.load_state_dict(initial);base.train(train);cand.train(train);self.h.set_seed(100000)
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):z0,e0=base(x)
     after=self.h.rng_state();self.h.set_seed(100000)
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):z1,e1=cand(x)
     if not t.rng_equal(after,self.h.rng_state()):raise RuntimeError(f'HE96_RNG_AUDIT_FAIL fold={f} train={train} amp={amp}')
     dz=float((z0-z1).abs().max());de=float((e0-e1).abs().max());dm=int((z0.argmax(1)!=z1.argmax(1)).sum());md=max(md,dz);me=max(me,de);mm+=dm;mode_audit.append({'train':train,'amp':amp,'logit':dz,'embedding':de,'mismatch':dm})
   if not (md<=1e-6 and me<=1e-6 and mm==0 and zero): raise RuntimeError(f'HE96_INITIALIZATION_AUDIT_FAIL fold={f} modes={mode_audit} zero={zero}')
   rows.append({'task':task,'fold':f,'modes':'eval/train x FP32/AMP','max_abs_logit_diff':md,'max_abs_embedding_diff':me,'prediction_mismatch':mm,'point2_extra_columns_zero':zero,'rng_preserved':True,'status':'PASS'});n=sum(p.numel() for p in base.parameters());pars.append({'task':task,'fold':f,'LiteBN_parameters':n,'HE96_parameters':sum(p.numel() for p in cand.parameters()),'increase':sum(p.numel() for p in cand.parameters())-n});t.dump(self.cell(task,f)/'gate.json',{'invariant':self.invariant,'pass':True,'mean':mean.tolist(),'std':std.tolist(),'normalizer':norm,'manifest_sha256':mh})
  t.csvwrite(self.out/'HE96_INITIALIZATION_AUDIT.csv',rows);t.csvwrite(self.out/'HE96_PARAMETER_COUNTS.csv',pars);return rows,pars
 def task_summary(self,task):
  import pandas as pd
  pairs=[]
  for f in range(5):
   for pop in ['outer','heldout']:pairs+=t.read(self.archcell(task,f,'HE96')/f'{pop}.json')['rows']
  d=pd.DataFrame(pairs);o=d[d.population=='outer'];fold=o.groupby('fold',as_index=False)[['LiteBN_BA','model_BA','delta_BA_pp']].mean();delta=float(fold.delta_BA_pp.mean());s={'task':task,'LiteBN_outer_BA':float(fold.LiteBN_BA.mean()),'HE96_outer_BA':float(fold.model_BA.mean()),'delta_BA_pp':delta,'positive_folds':int((fold.delta_BA_pp>0).sum()),'worst_fold_delta_pp':float(fold.delta_BA_pp.min())};fold.to_csv(self.out/'HE96_FOLD_RESULTS.csv',index=False);t.csvwrite(self.out/'HE96_TASK_SUMMARY.csv',[s]);d.to_csv(self.out/'HE96_OUTER_SUBJECT_RESULTS.csv',index=False);t.dump(self.out/'TASK_GATE.json',{'status':'PASS' if delta>0 else 'FAIL','delta_BA_pp':delta});print(json.dumps(s),flush=True);return {'status':'PASS' if delta>0 else 'FAIL'}
def main():
 p=t.argparse.ArgumentParser();[p.add_argument('--'+n,type=Path,required=True) for n in ['repo','recovered','cache','historical-runtime','runtime']];a=p.parse_args();e=E(a);e.preflight('OpenBMI_MI');[e.train_arch('OpenBMI_MI',f,'HE96') for f in range(5)];[e.evaluate('OpenBMI_MI',f,'HE96',pop) for pop in ['outer','heldout'] for f in range(5)];g=e.task_summary('OpenBMI_MI');raise SystemExit(0 if g['status']=='PASS' else 20)
if __name__=='__main__':main()
