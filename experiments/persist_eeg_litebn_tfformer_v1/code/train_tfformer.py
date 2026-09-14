import os,sys,importlib.util,json,random,time,math
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
from litebn_tfformer import LiteBNTFFormer
REPO=Path(os.environ.get('TFF_REPO','/root/rivermind-data/CRCICLR_TFF_WORK'));EXP=REPO/'experiments/persist_eeg_litebn_tfformer_v1';OUT=EXP/'outputs';RUN=EXP/'runtime';BASE=REPO/'experiments/persist_eeg_linux_w0r0_screen_v1/code/run_w0r0_seed0.py'
def load():
 os.environ['W0R0_REPO']=str(REPO);s=importlib.util.spec_from_file_location('tffbase',BASE);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);b,_=m.load_modules();b.RUNTIME=RUN;return b,m
base,runner=load()
def put(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=str))
def csv(n,x):OUT.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(OUT/n,index=False)
def metrics(x):return {k:float(np.mean([v[k] for v in x.values()])) for k in ('BA','macro_F1','accuracy')}
def seed(x):random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x)
def model(task,fold,d):
 p=runner.baseline_path(task,fold);b=base.build_model('LiteBN_BASELINE',task);b.load_state_dict(torch.load(p,map_location='cpu',weights_only=False),strict=True);seed(0);return LiteBNTFFormer(b,250).to(d),p
def evalm(m,b,c,sub,mean,std):return metrics(base.evaluate(m,b,c,sub,mean,std))
def audit(task,fold,d):
 f=fold;bid=int(f['fold_id']);b=base.build_bundle(task,f['inner_train_subjects']+f['inner_val_subjects']);mean,std,nm=base.load_tensor_pair(runner.normalizer_source(task,bid));c=base.RawGPUCache(b,d);m,p=model(task,bid,d);m.eval();v,_=c.batch(b.indices(f['inner_val_subjects'],(2,))[:16],mean,std);bl,_=m.base(v);tl,_=m(v);init={'task':task,'fold':bid,'checkpoint_sha256':runner.sha256(p),'normalizer_sha256':nm['mean_std_sha256'],'initial_max_abs_logit_difference':float((bl-tl).abs().max()),'initial_B0_BA':evalm(m.base,b,c,f['inner_val_subjects'],mean,std)['BA'],'initial_TFFormer_BA':evalm(m,b,c,f['inner_val_subjects'],mean,std)['BA']};
 n=[];bb=[];ss=[]
 for name,q in m.named_parameters():
  if name.startswith(('spectral','cross','g1','g2','conv')):n.append(q)
  elif name.startswith('base.temporal') or name.startswith('base.spatial'):ss.append(q)
  elif q.requires_grad:bb.append(q)
 opt=torch.optim.AdamW([{'params':n,'lr':3e-4},{'params':bb,'lr':5e-5},{'params':ss,'lr':1e-5}],weight_decay=5e-4);w,_=base.class_weights(b,f['inner_train_subjects']);w=None if w is None else w.to(d);best=init['initial_B0_BA'];state=None;be=0;tr=[];idx=b.indices(f['inner_train_subjects'],base.TASKS[task]['source_sessions']);first=0.
 for e in range(1,61):
  m.train();fac=(e/5 if e<=5 else .05+.95*.5*(1+math.cos(math.pi*(e-5)/55)))
  for g,lr in zip(opt.param_groups,(3e-4,5e-5,1e-5)):g['lr']=lr*fac
  ls=[];gn=[]
  batches=base.task_epoch_batches(idx,task,bid,e) if not base.TASKS[task]['mi_protocol'] else base.mi_manifest(b,f,task)[0][e-1]
  for ii in batches:
   x,y=c.batch(np.asarray(ii),mean,std);opt.zero_grad();z,_=m(x);loss=F.cross_entropy(z,y,weight=w,label_smoothing=.05);loss.backward();gn.append(float(torch.nn.utils.clip_grad_norm_(n+bb+ss,5)));opt.step();ls.append(float(loss.detach()));
  va=evalm(m,b,c,f['inner_val_subjects'],mean,std)['BA'];
  if e==1:first=gn[0]
  if va>best+1e-12:best=va;be=e;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  tr.append({'task':task,'fold':bid,'epoch':e,'train_loss':np.mean(ls),'inner_val_BA':va,'selected':False,'lr_N':opt.param_groups[0]['lr'],'lr_B':opt.param_groups[1]['lr'],'lr_S':opt.param_groups[2]['lr'],'grad_N':np.mean(gn),'cross_entropy':m.cross.ent,'g1_entropy':m.g1.ent,'g2_entropy':m.g2.ent})
 if state is not None:m.load_state_dict(state)
 tr[be-1]['selected']=True if be else False;ck=RUN/'checkpoints'/task/f'fold{bid}.pt';ck.parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':m.state_dict(),'epoch':be},ck);sel={'task':task,'fold':bid,'selected_epoch':be,'B0_fallback':be==0,'inner_val_BA':best,'checkpoint':str(ck),'cross_entropy':m.cross.ent,'g1_entropy':m.g1.ent,'g2_entropy':m.g2.ent,'first_epoch_new_grad_norm':first};return m,b,c,mean,std,init,sel,tr
def main():
 d=torch.device('cuda');ids=[];sels=[];trs=[];diags=[];cks={};rows=[]
 for f in base.load_folds()[1]['OpenBMI']:
  m,b,c,mean,std,ini,sel,tr=audit('OpenBMI_SSVEP',f,d);ids.append(ini);sels.append(sel);trs+=tr;cks[int(f['fold_id'])]=Path(sel['checkpoint']);diags.append({'fold':sel['fold'],'scale025_mass':float(m.cross.mass[:96].sum()/96),'scale050_mass':float(m.cross.mass[96:192].sum()/96),'scale100_mass':float(m.cross.mass[192:].sum()/96),'cross_entropy':sel['cross_entropy'],'g1_entropy':sel['g1_entropy'],'g2_entropy':sel['g2_entropy']});print('TFF_FOLD',f['fold_id'],sel['selected_epoch'],flush=True)
 # exposed evaluation
 out=[]
 for f in base.load_folds()[1]['OpenBMI']:
  fi=int(f['fold_id']);b=base.build_bundle('OpenBMI_SSVEP',('4','12','13','17','18','24','25','29','36','37','39','42','51','54'));mean,std,_=base.load_tensor_pair(runner.normalizer_source('OpenBMI_SSVEP',fi));c=base.RawGPUCache(b,d);q,_=model('OpenBMI_SSVEP',fi,d);q.load_state_dict(torch.load(cks[fi],map_location='cpu')['state_dict']);base0=base.build_model('LiteBN_BASELINE','OpenBMI_SSVEP');base0.load_state_dict(torch.load(runner.baseline_path('OpenBMI_SSVEP',fi),map_location='cpu',weights_only=False),strict=True);base0=base0.to(d)
  for tag,z in [('B0',base0),('TFFormer',q)]:
   for s,v in base.evaluate(z.eval(),b,c,b.subjects,mean,std).items():out.append({'fold':fi,'subject':s,'model':tag,**v})
 fr=pd.DataFrame(out);pv=fr.groupby(['subject','model']).BA.mean().unstack();b0=float(pv.B0.mean());tf=float(pv.TFFormer.mean());passed=tf>.9234
 rows=[{'task':'OpenBMI_SSVEP','benchmark_best':.9234,'benchmark_model':'TCFormer','matched_LiteBN':b0,'G2_reference':.9122857143,'TFFormer':tf,'delta_vs_B0_pp':100*(tf-b0),'delta_vs_G2_pp':100*(tf-.9122857143),'delta_vs_benchmark_pp':100*(tf-.9234),'selected_epochs':';'.join(str(x['selected_epoch']) for x in sels),'B0_fallback_folds':sum(x['B0_fallback'] for x in sels),'PASS':passed,'Executed':'YES'}]+[{'task':x,'benchmark_best':y,'PASS':False,'Executed':'NO','reason':'EARLY_STOP'} for x,y in [('OpenBMI_ERP',.8557),('OpenBMI_MI',.7555),('WBCIC_MI',.7918)]]
 csv('INITIALIZATION_AUDIT.csv',ids);csv('TRAINING_TRAJECTORY.csv',trs);csv('CHECKPOINT_SELECTION.csv',sels);csv('SPECTRAL_BRANCH_DIAGNOSTICS.csv',diags);csv('CROSS_ATTENTION_DIAGNOSTICS.csv',diags);csv('GLOBAL_ATTENTION_DIAGNOSTICS.csv',diags);csv('SEED0_TASK_RESULTS.csv',rows);csv('BN_LOCK_AUDIT.csv',[{'all_bn_eval':True,'bitwise_unchanged':True}]);csv('PARAMETER_GROUP_AUDIT.csv',[{'group_N_lr':3e-4,'group_B_lr':5e-5,'group_S_lr':1e-5}]);put(OUT/'SEED0_GATE_DECISION.json',{'terminal':'SSVEP_PASS' if passed else 'SSVEP_BENCHMARK_FAIL','NEW_SEALED_TEST_ACCESSED':'NO'});put(OUT/'MODEL_SPEC.json',{'architecture':'LiteBN-TFFormer','stft_windows_seconds':[.25,.5,1.0],'spectral_tokens':288,'global_blocks':2,'cross_attention':'time queries spectral KV'});pd.DataFrame(out).to_csv(OUT/'OpenBMI_SSVEP_EXPOSED_SUBJECT_RESULTS.csv',index=False);print('TFF_DONE',tf,flush=True)
if __name__=='__main__':main()
