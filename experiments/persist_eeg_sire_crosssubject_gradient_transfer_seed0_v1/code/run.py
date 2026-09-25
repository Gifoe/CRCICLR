"""One-step, subject-grouped gradient-transfer diagnostic for frozen SIRE."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

EXP=Path(__file__).resolve().parents[1];REPO=EXP.parents[1]
PRIOR=REPO/'experiments/persist_eeg_sire_stagewise_pu_shared_refinement_seed0_v1'
sp=importlib.util.spec_from_file_location('sire_stagewise_frozen',PRIOR/'code/run.py')
S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
L=S.L;P=S.P;A=S.A;DEVICE=S.DEVICE;TASK='OpenBMI_MI';FOLDS=tuple(range(5))
OUT=EXP/'outputs';PROTOCOL=EXP/'protocol'
RUNTIME=Path(os.environ.get('SIRE_GRADIENT_TRANSFER_RUNTIME',str(REPO.parent/'sire_crosssubject_gradient_transfer_seed0_runtime'))).resolve()
ALLOWED=tuple(sorted(S.ALLOWED));BASE=('FULL_CE','P_UTILITY','SHARED1_PERSISTENCE')
DIRECTIONS=BASE+('PU_PER_RAW','BALANCED','CONFLICT_AWARE')
STEPS=(1e-4,3e-4,1e-3);PRIMARY=3e-4;EPS=1e-12;BOOT=20_000;BATCH=128
torch.set_num_threads(min(int(os.environ.get('SIRE_GRADIENT_CPU_THREADS','8')),os.cpu_count() or 1))
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def cell(f):return RUNTIME/'cells'/f'fold{f}_seed0'
def jwrite(p,x):P.jwrite(p,x)
def cwrite(p,x):P.cwrite(p,x)
def cread(p):return P.cread(p)
def sha(p):return P.sha(p)
def seed(*parts):return P.seed('crosssubject_gradient_transfer',*parts)
def tensor(a):return S.tensor(a)
def scalar(x):return float(x.detach().cpu()) if torch.is_tensor(x) else float(x)

def splits(lock):
    result=[]
    for f,c in enumerate(lock['cells']):
        subjects=list(c['train_subjects']);rng=np.random.default_rng(seed('meta_subjects',f));perm=list(rng.permutation(subjects))
        chunks=np.array_split(np.asarray(perm),5)
        for m,part in enumerate(chunks):
            val=sorted(map(str,part),key=int);train=sorted(set(subjects)-set(val),key=int)
            if not val or set(val)&set(train):raise RuntimeError('invalid subject split')
            result.append({'fold':f,'meta_fold':m,'meta_train_subjects':train,'meta_val_subjects':val,
                'meta_train_n_subjects':len(train),'meta_val_n_subjects':len(val)})
        if sorted([s for r in result if r['fold']==f for s in r['meta_val_subjects']],key=int)!=sorted(subjects,key=int):
            raise RuntimeError('meta val coverage')
    return result

def preflight():
    old=S.preflight();cells=[]
    for f,c in enumerate(old['cells']):
        path=Path(c['geometry_archive'])
        if sha(path)!=c['geometry_archive_sha256']:raise RuntimeError('prior geometry archive drift')
        cells.append(c)
    lock={'schema':'SIRE_CROSSSUBJECT_GRADIENT_TRANSFER_SEED0_V1','task':TASK,'seed':0,'folds':list(FOLDS),
        'cells':cells,'source_stagewise_protocol_sha256':sha(PRIOR/'protocol/PROTOCOL_LOCK.json'),
        'source_stagewise_code_sha256':sha(PRIOR/'code/run.py'),
        'trainable_scope':list(ALLOWED),'directions':list(DIRECTIONS),'relative_steps':list(STEPS),
        'primary_relative_step':PRIMARY,'meta_split_seed_rule':'stable SHA256 seed; five disjoint subject chunks per outer fold',
        'persistence_objective':'negative mean explicit centered Pearson across matched subject-class Shared1 P centroids; epsilon 1e-12',
        'gradient_definition':'exact full-role mean CE/PU gradients; exact centroid-mean Pearson gradient via per-trial VJP',
        'virtual_step':'unit full-scope gradient times delta times frozen full-scope parameter L2 norm',
        'step_comparison':'all six directions matched parameter-space L2 displacement',
        'bootstrap_draws':BOOT,'outer_dev_eeg_reads':0,'final_heldout_eeg_reads':0,
        'no_persistent_updates':True,'discovery_after_all_meta_frozen':True,
        'coordinate_system_caveat':'fixed Q/mu/pathway fit on all inner-train subjects; meta-val gradients and effects do not fit geometry',
        'historical_provenance_caveat':old['historical_provenance_caveat'],
        'decision_rule':{'transfer_sign_fraction_min':.70,'no_systematic_task_deterioration':'mean held CE change <= 0',
            'combined_directions_require_both_PU_and_persistence':True,
            'conflict_label':'PU_PER cosine negative in >=70% of splits; RAW fails at least one P metric; conflict-aware exceeds RAW on both P metrics by >20% of absolute RAW effect plus 1e-8',
            'subject_specific_label':'source intended metric improves, train-val cosine <=0.05, held intended metric does not improve',
            'discovery_consistency':'same intended metric beneficial mean sign, not direction selection'}}
    path=PROTOCOL/'PROTOCOL_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise RuntimeError('protocol lock drift')
    jwrite(path,lock)
    manifest={'schema':'FIVE_FOLD_INNER_TRAIN_SUBJECT_META_SPLITS','splits':splits(lock),'discovery_subjects_excluded':{str(f):c['discovery_subjects'] for f,c in enumerate(cells)}}
    mp=OUT/'META_SUBJECT_SPLITS.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise RuntimeError('meta splits drift')
    jwrite(mp,manifest)
    return lock,manifest['splits']

def role(data,subjects):
    ix=np.flatnonzero(np.isin(data['subject'],np.asarray(subjects)))
    if not len(ix):raise RuntimeError('empty role')
    keys=('y','subject','session','trial','hc','h1','h2','emb','z0','p10','c10','p20','c20')
    return {k:data[k][ix] for k in keys}

def model_and_params(ref):
    m=S.student(ref);named=dict(m.named_parameters());params=[named[n] for n in ALLOWED]
    if any(n not in named for n in ALLOWED):raise RuntimeError('scope missing')
    return m,params

def vec(grads,params):
    return torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1) for g,p in zip(grads,params)]).detach()

def cos(a,b):
    na=torch.linalg.vector_norm(a);nb=torch.linalg.vector_norm(b)
    return float(torch.dot(a,b)/(na*nb)) if na>EPS and nb>EPS else None

def norm(a):return float(torch.linalg.vector_norm(a))
def unit(a):return a/(torch.linalg.vector_norm(a)+EPS)

def pearson_components(p,rr):
    """Differentiable per-coordinate Pearson from full subject-class centroids."""
    subjects=np.asarray(rr['subject']);labels=np.asarray(rr['y']);sessions=np.asarray(rr['session'])
    ss=sorted(set(map(int,sessions)))
    if len(ss)!=2:raise RuntimeError('two sessions required')
    units=sorted(set((str(s),int(y)) for s,y in zip(subjects,labels)),key=lambda x:(int(x[0]),x[1]))
    groups=[];aa=[];bb=[]
    for sub,y in units:
        ix1=np.flatnonzero((subjects==sub)&(labels==y)&(sessions==ss[0]))
        ix2=np.flatnonzero((subjects==sub)&(labels==y)&(sessions==ss[1]))
        if not len(ix1) or not len(ix2):raise RuntimeError('missing matched subject-class session')
        groups.append((ix1,ix2));aa.append(p[ix1].mean(0));bb.append(p[ix2].mean(0))
    a=torch.stack(aa);b=torch.stack(bb);ac=a-a.mean(0);bc=b-b.mean(0)
    denom=torch.sqrt(ac.square().sum(0)+EPS)*torch.sqrt(bc.square().sum(0)+EPS)
    rk=(ac*bc).sum(0)/denom
    contrib=(ac*bc/denom).mean(1)
    if not torch.isfinite(rk).all():raise RuntimeError('Pearson nonfinite')
    return rk.mean(),contrib,units,groups

def gradients(m,ref,rr,geo):
    """Compute three exact mean-loss gradients without holding full-role graphs."""
    names=dict(m.named_parameters());params=[names[n] for n in ALLOWED];n=len(rr['y'])
    q1,mu1,q2,mu2=[tensor(geo[st][k]) for st,k in [('H_SHARED1','q'),('H_SHARED1','mu'),('H_SHARED2','q'),('H_SHARED2','mu')]]
    gce=torch.zeros(sum(p.numel() for p in params),device=DEVICE);gpu=gce.clone();gper=gce.clone()
    for start in range(0,n,BATCH):
        ix=slice(start,min(start+BATCH,n));out=S.forward_batch(m,tensor(rr['hc'][ix]),q1,mu1,q2,mu2,tensor(rr['c20'][ix]),ref)
        yy=torch.as_tensor(rr['y'][ix],device=DEVICE,dtype=torch.long)
        lce=F.cross_entropy(out[6],yy,reduction='sum')/n;lpu=F.cross_entropy(out[7],yy,reduction='sum')/n
        gce+=vec(torch.autograd.grad(lce,params,retain_graph=True,allow_unused=True),params)
        gpu+=vec(torch.autograd.grad(lpu,params,allow_unused=True),params)
    # Exact frozen-checkpoint centroid gradient: dL/dm is spread equally over each trial in that centroid.
    p=torch.as_tensor(rr['p10'],device=DEVICE,dtype=torch.float32)
    p_leaf=p.detach().clone().requires_grad_(True)
    rho,_,_,_=pearson_components(p_leaf,rr)
    weights=torch.autograd.grad(-rho,p_leaf)[0].detach()
    for start in range(0,n,BATCH):
        ix=slice(start,min(start+BATCH,n))
        h1=S.L.next_stage(m,'H_CONCAT',tensor(rr['hc'][ix]));p1=(h1-mu1)@q1
        loss=(p1*weights[ix]).sum()
        gper+=vec(torch.autograd.grad(loss,params,allow_unused=True),params)
    if not all(torch.isfinite(g).all() and norm(g)>EPS for g in (gce,gpu,gper)):raise RuntimeError('nonfinite or zero gradient')
    return dict(zip(BASE,(gce,gpu,gper))),float(rho.detach())

def directions(base):
    ce=base['FULL_CE'];pu=base['P_UTILITY'];per=base['SHARED1_PERSISTENCE']
    raw=pu+per;bal=unit(unit(pu)+unit(per));c=cos(pu,per)
    if c is None:raise RuntimeError('zero base gradient')
    if c<0:
        pp=pu-torch.dot(pu,per)/(torch.dot(per,per)+EPS)*per
        qq=per-torch.dot(per,pu)/(torch.dot(pu,pu)+EPS)*pu
        pc=unit(unit(pp)+unit(qq))
    else:pc=bal.clone()
    out={**base,'PU_PER_RAW':raw,'BALANCED':bal,'CONFLICT_AWARE':pc}
    if not all(torch.isfinite(v).all() and norm(v)>EPS for v in out.values()):raise RuntimeError('invalid combined direction')
    return out

def eval_role(m,ref,rr,geo,fold,with_subject=False):
    inf=S.inference(m,ref,rr,geo);y=rr['y'];z=inf['z'];zp=inf['z_p']
    fit=geo['H_SHARED2'];dims=fold['protected_coordinates']
    target=A.canonical_targets(rr['emb'],geo['spectrum'],dims)
    pred=L.predict_path(inf['h2'],fit);den=float(np.square(target-target.mean(0)).sum())
    rec=float(1-np.square(target-pred).sum()/max(den,EPS))
    p=tensor(inf['p1']);rho,contrib,units,_=pearson_components(p,rr)
    d={'full_CE':S.P.metric(y,z)['NLL'],'full_BA':S.P.metric(y,z)['BA'],'full_macro_F1':S.P.metric(y,z)['macro_F1'],
       'Ponly_CE':S.P.metric(y,zp)['NLL'],'Ponly_BA':S.P.metric(y,zp)['BA'],'rho1':float(rho),
       'shared2_recoverability_R2':rec}
    for k,key in [('P1','p1'),('C1','c1'),('P2','p2'),('C2','c2')]:
        d[k+'_drift_RMS']=float(np.sqrt(np.mean(np.square(inf[key]-rr[key.lower()+'0']))))
    if with_subject:
        # Equal numbers of class units per subject: average subject contribution equals pooled rho.
        sub=defaultdict(float);N=len(set(x[0] for x in units))
        for (s,_),v in zip(units,contrib.detach().cpu().numpy()):sub[s]+=float(v)*N
        d['subject_rho_contributions']=dict(sub)
        subject_metrics={}
        for s in sorted(set(rr['subject']),key=int):
            ix=rr['subject']==s
            y_s=y[ix]
            subject_metrics[s]={'full_CE':S.P.metric(y_s,z[ix])['NLL'],
                'full_BA':S.P.metric(y_s,z[ix])['BA'],
                'Ponly_CE':S.P.metric(y_s,zp[ix])['NLL'],
                'Ponly_BA':S.P.metric(y_s,zp[ix])['BA']}
        d['subject_metrics']=subject_metrics
    return d

def effect(before,after):
    return {k:float(after[k]-before[k]) for k in ('full_CE','full_BA','full_macro_F1','Ponly_CE','Ponly_BA','rho1','shared2_recoverability_R2',
        'P1_drift_RMS','C1_drift_RMS','P2_drift_RMS','C2_drift_RMS')}

def perturb(m,ref,rr,geo,fold,u,delta,restore):
    named=dict(m.named_parameters());pars=[named[n] for n in ALLOWED];theta=torch.cat([p.detach().reshape(-1) for p in pars]);scope=norm(theta)
    original=[p.detach().clone() for p in pars];before=A.model_state_sha(m)
    if before!=restore:raise RuntimeError('pre-step state hash differs from canonical')
    step=float(delta*scope);off=0
    try:
        with torch.no_grad():
            for p in pars:
                z=u[off:off+p.numel()].reshape_as(p);p.sub_(step*z);off+=p.numel()
        if abs(float(torch.linalg.vector_norm(torch.cat([(p.detach()-v).reshape(-1) for p,v in zip(pars,original)])))-step)>max(1e-6,step*1e-3):
            raise RuntimeError('virtual displacement mismatch')
        out=eval_role(m,ref,rr,geo,fold,True)
    finally:
        with torch.no_grad():
            for p,v in zip(pars,original):p.copy_(v)
    if A.model_state_sha(m)!=restore:raise RuntimeError('canonical state restoration failure')
    return out,step

def gradient_rows(f,mm,role_name,gg):
    rows=[];sizes={n:p.numel() for n,p in dict(mm.named_parameters()).items() if n in ALLOWED}
    for name,g in gg.items():
        row={'fold':f,'meta_fold':role_name[0],'role':role_name[1],'objective':name,'gradient_L2':norm(g)};o=0
        for layer in ALLOWED:row[layer+'_gradient_L2']=norm(g[o:o+sizes[layer]]);o+=sizes[layer]
        rows.append(row)
    for a in DIRECTIONS:
        for b in DIRECTIONS:
            if a>=b:continue
            rows.append({'fold':f,'meta_fold':role_name[0],'role':role_name[1],'objective':a,'other_objective':b,'pairwise_cosine':cos(gg[a],gg[b])})
    return rows

def cross_rows(f,meta,m,gt,gv):
    rows=[];sizes={n:p.numel() for n,p in dict(m.named_parameters()).items() if n in ALLOWED}
    for name in BASE:
        rows.append({'fold':f,'meta_fold':meta,'objective':name,'layer':'TOTAL','train_val_cosine':cos(gt[name],gv[name])})
        o=0
        for layer in ALLOWED:
            n=sizes[layer];rows.append({'fold':f,'meta_fold':meta,'objective':name,'layer':layer,'train_val_cosine':cos(gt[name][o:o+n],gv[name][o:o+n])});o+=n
    return rows

def role_pieces(data,sub):return role(data,sub)

def meta_fold(f,meta,split,ref,data,geo,lock):
    m,params=model_and_params(ref);original=A.model_state_sha(m)
    tr=role_pieces(data,split['meta_train_subjects']);va=role_pieces(data,split['meta_val_subjects'])
    if set(tr['subject'])&set(va['subject']):raise RuntimeError('meta overlap')
    gt,rhot=gradients(m,ref,tr,geo);gv,rhov=gradients(m,ref,va,geo)
    dirs=directions(gt);gval=directions(gv)
    align=gradient_rows(f,m,(meta,'META_TRAIN'),dirs)+gradient_rows(f,m,(meta,'META_VAL'),gval)
    cross=cross_rows(f,meta,m,gt,gv)
    before_tr=eval_role(m,ref,tr,geo,lock['cells'][f],True);before_va=eval_role(m,ref,va,geo,lock['cells'][f],True)
    if abs(before_tr['rho1']-rhot)>1e-5 or abs(before_va['rho1']-rhov)>1e-5:raise RuntimeError('Pearson gradient/metric mismatch')
    effects=[];source=[];first=[];subject=[];restores=0
    scope=norm(torch.cat([p.detach().reshape(-1) for p in params]));
    for name in DIRECTIONS:
        u=unit(dirs[name])
        for delta in STEPS:
            new_tr,step=perturb(m,ref,tr,geo,lock['cells'][f],u,delta,original);restores+=1
            new_va,step2=perturb(m,ref,va,geo,lock['cells'][f],u,delta,original);restores+=1
            if abs(step-step2)>1e-12:raise RuntimeError('step mismatch')
            et=effect(before_tr,new_tr);ev=effect(before_va,new_va)
            record={'fold':f,'meta_fold':meta,'direction':name,'relative_step':delta,'parameter_displacement_L2':step,
                'meta_val_subjects':'|'.join(split['meta_val_subjects'])}
            for k in ev:record['before_'+k]=before_va[k];record['after_'+k]=new_va[k];record['delta_'+k]=ev[k]
            record['T_task']=-ev['full_CE'];record['T_PU']=-ev['Ponly_CE'];record['T_PER']=ev['rho1']
            effects.append(record)
            sr={'fold':f,'meta_fold':meta,'direction':name,'relative_step':delta}
            for k in ('full_CE','full_BA','Ponly_CE','Ponly_BA','rho1'):
                benefit=lambda x:(-x if k in ('full_CE','Ponly_CE') else x)
                st=benefit(et[k]);sv=benefit(ev[k]);sr['source_'+k+'_benefit']=st;sr['held_'+k+'_benefit']=sv
                sr[k+'_transfer_ratio']=sv/st if st>1e-9 else None
                sr[k+'_source_improves_held_worsens']=bool(st>0 and sv<0)
            source.append(sr)
            if delta==PRIMARY:
                for k,obj in [('full_CE','FULL_CE'),('Ponly_CE','P_UTILITY'),('rho1','SHARED1_PERSISTENCE')]:
                    predicted=(-step*float(torch.dot(gv[obj],u))) if k!='rho1' else step*float(torch.dot(gv[obj],u))
                    observed=ev[k];first.append({'fold':f,'meta_fold':meta,'direction':name,'metric':k,
                        'predicted_first_order_delta':predicted,'observed_virtual_delta':observed,
                        'sign_agreement':bool(np.sign(predicted)==np.sign(observed))})
                for sub in split['meta_val_subjects']:
                    ix=va['subject']==sub
                    if not ix.any():raise RuntimeError('subject missing from role')
                    for k in ('full_CE','full_BA','Ponly_CE','Ponly_BA'):
                        record['subject_'+sub+'_delta_'+k]=new_va['subject_metrics'][sub][k]-before_va['subject_metrics'][sub][k]
                    subject.append({'fold':f,'meta_fold':meta,'direction':name,'subject':sub,
                        'delta_rho1_contribution':new_va['subject_rho_contributions'][sub]-before_va['subject_rho_contributions'][sub],
                        'subject_meta_val_size':int(ix.sum())})
        print('META_DIRECTION',f,meta,name,flush=True)
    if A.model_state_sha(m)!=original:raise RuntimeError('meta fold model not restored')
    return align,cross,effects,source,first,subject,{'fold':f,'meta_fold':meta,'virtual_evaluations_restored':restores,
        'canonical_state_sha256':original,'final_state_sha256':A.model_state_sha(m)}

def run_meta(f,lock,all_splits):
    ref,data,geo=S.prepare_train(f,lock);geo['path']=lock['cells'][f]['geometry_archive']
    c=cell(f);c.mkdir(parents=True,exist_ok=True)
    buckets=[[] for _ in range(6)];audits=[]
    for split in [x for x in all_splits if x['fold']==f]:
        result=meta_fold(f,split['meta_fold'],split,ref,data,geo,lock)
        for i in range(6):buckets[i].extend(result[i])
        audits.append(result[6]);print('META_COMPLETE',f,split['meta_fold'],flush=True)
    for name,rows in zip(('GRADIENT_ALIGNMENT','CROSS_SUBJECT_GRADIENT_ALIGNMENT','META_VAL_VIRTUAL_STEP_EFFECTS',
                          'SOURCE_VS_HELD_TRANSFER','FIRST_ORDER_TRANSFER_CHECK','SUBJECT_RHO_CONTRIBUTIONS'),buckets):cwrite(c/(name+'.csv'),rows)
    jwrite(c/'PARAMETER_RESTORE_AUDIT.json',{'fold':f,'meta_splits':audits,
        'all_virtual_perturbations_restored_exactly':True,'no_persistent_update':True,
        'total_virtual_evaluations':sum(x['virtual_evaluations_restored'] for x in audits)})
    jwrite(c/'META_COMPLETE.json',{'fold':f,'meta_folds':5,'all_results_written':True})
    print('FOLD_META_COMPLETE',f,flush=True)

def correlation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or np.std(x[ok])<1e-12 or np.std(y[ok])<1e-12:return (None,None)
    return (float(pearsonr(x[ok],y[ok]).statistic),float(spearmanr(x[ok],y[ok]).statistic))

def aggregate_meta(lock):
    names=('GRADIENT_ALIGNMENT','CROSS_SUBJECT_GRADIENT_ALIGNMENT','META_VAL_VIRTUAL_STEP_EFFECTS',
           'SOURCE_VS_HELD_TRANSFER','FIRST_ORDER_TRANSFER_CHECK','SUBJECT_RHO_CONTRIBUTIONS')
    bags={n:[] for n in names};audits=[]
    for f in FOLDS:
        if not (cell(f)/'META_COMPLETE.json').exists():raise RuntimeError('all five meta folds required before freeze')
        for n in names:bags[n].extend(cread(cell(f)/(n+'.csv')))
        audits.append(json.loads((cell(f)/'PARAMETER_RESTORE_AUDIT.json').read_text()))
    for n in names:
        if n!='SUBJECT_RHO_CONTRIBUTIONS':cwrite(OUT/(n+'.csv'),bags[n])
    cross=bags['CROSS_SUBJECT_GRADIENT_ALIGNMENT'];ev=bags['META_VAL_VIRTUAL_STEP_EFFECTS'];src=bags['SOURCE_VS_HELD_TRANSFER']
    primary=[r for r in ev if abs(float(r['relative_step'])-PRIMARY)<1e-12]
    signs=[]
    for name in DIRECTIONS:
        rr=[r for r in primary if r['direction']==name]
        if len(rr)!=25:raise RuntimeError('expected 25 primary meta effects')
        positive=lambda k:sum(float(x[k])>0 for x in rr)/len(rr)
        signs.append({'direction':name,'meta_splits':len(rr),'task_improve_fraction':positive('T_task'),
           'Ponly_improve_fraction':positive('T_PU'),'persistence_improve_fraction':positive('T_PER'),
           'Ponly_and_persistence_fraction':sum(float(x['T_PU'])>0 and float(x['T_PER'])>0 for x in rr)/len(rr),
           'task_and_persistence_fraction':sum(float(x['T_task'])>0 and float(x['T_PER'])>0 for x in rr)/len(rr),
           'all_three_improve_fraction':sum(all(float(x[k])>0 for k in ('T_task','T_PU','T_PER')) for x in rr)/len(rr)})
    cwrite(OUT/'TRANSFER_SIGN_CONSISTENCY.csv',signs)
    # Subject-paired resampling: collapse repeated outer-fold appearances before bootstrap.
    subject_data=defaultdict(lambda:defaultdict(list))
    # Subject-level CE/BA needs per-subject predictions. These are written in the primary-effect rows below.
    for r in primary:
        for k in ('full_CE','full_BA','Ponly_CE','Ponly_BA'):
            for sub in r['meta_val_subjects'].split('|'):
                field='subject_'+sub+'_delta_'+k
                if field in r and r[field]!='':subject_data[(r['direction'],k)][sub].append(float(r[field]))
    for r in bags['SUBJECT_RHO_CONTRIBUTIONS']:
        subject_data[(r['direction'],'rho1')][r['subject']].append(float(r['delta_rho1_contribution']))
    summary=[]
    for name in DIRECTIONS:
        rr=[r for r in primary if r['direction']==name]
        for k in ('full_CE','full_BA','Ponly_CE','Ponly_BA','rho1'):
            by=subject_data[(name,k)]
            if len(by)<3:raise RuntimeError(f'missing biological subject effects {name} {k}')
            values=np.asarray([np.mean(by[s]) for s in sorted(by,key=int)],float)
            rng=np.random.default_rng(seed('bootstrap',name,k));draw=rng.integers(0,len(values),size=(BOOT,len(values)))
            boot=values[draw].mean(1)
            summary.append({'direction':name,'metric':k,'biological_subjects':len(values),'mean_delta':float(values.mean()),
                'ci95_lower':float(np.quantile(boot,.025)),'ci95_upper':float(np.quantile(boot,.975)),
                'bootstrap_draws':BOOT,'meta_split_mean_delta':float(np.mean([float(x['delta_'+k]) for x in rr]))})
    cwrite(OUT/'TRANSFER_SUMMARY.csv',summary)
    conflict=[]
    # Descriptive associations only; 25 split observations, no causal reading.
    indexed={(int(x['fold']),int(x['meta_fold']),x['objective'],x['layer']):float(x['train_val_cosine']) if x['train_val_cosine'] else np.nan for x in cross}
    for name,obj,target in [('P_UTILITY','P_UTILITY','T_PU'),('SHARED1_PERSISTENCE','SHARED1_PERSISTENCE','T_PER'),
                            ('FULL_CE','FULL_CE','T_task')]:
        rr=[r for r in primary if r['direction']==name]
        x=[indexed[(int(r['fold']),int(r['meta_fold']),obj,'TOTAL')] for r in rr];y=[float(r[target]) for r in rr]
        pc,sc=correlation(x,y);conflict.append({'analysis':'train_val_gradient_cosine_vs_held_transfer','direction':name,
            'metric':target,'n':len(rr),'Pearson':pc,'Spearman':sc})
    ar=bags['GRADIENT_ALIGNMENT'];puper=[float(r['pairwise_cosine']) for r in ar if r['role']=='META_TRAIN' and
        {r['objective'],r['other_objective']}=={'P_UTILITY','SHARED1_PERSISTENCE'} and r['pairwise_cosine']]
    conflict.append({'analysis':'PU_PER_train_conflict','n':len(puper),'mean_cosine':float(np.mean(puper)),
        'negative_fraction':float(np.mean(np.asarray(puper)<0))})
    for name in ('PU_PER_RAW','BALANCED','CONFLICT_AWARE'):
        rr=[r for r in primary if r['direction']==name]
        conflict.append({'analysis':'combined_direction_held_transfer','direction':name,'n':len(rr),
            'mean_T_task':float(np.mean([float(x['T_task']) for x in rr])),
            'mean_T_PU':float(np.mean([float(x['T_PU']) for x in rr])),
            'mean_T_PER':float(np.mean([float(x['T_PER']) for x in rr]))})
    cwrite(OUT/'GRADIENT_CONFLICT_AUDIT.csv',conflict)
    jwrite(OUT/'PARAMETER_RESTORE_AUDIT.json',{'status':'PASS','fold_audits':audits,
        'all_virtual_updates_restored_to_original_state_sha256':True,'no_persistent_parameter_updates':True,
        'BN_and_classifier_unchanged':True,'geometry_hashes_unchanged':True})
    files=[OUT/(n+'.csv') for n in names if n!='SUBJECT_RHO_CONTRIBUTIONS']+[OUT/'TRANSFER_SIGN_CONSISTENCY.csv',OUT/'TRANSFER_SUMMARY.csv',
        OUT/'GRADIENT_CONFLICT_AUDIT.csv',OUT/'META_SUBJECT_SPLITS.json',PROTOCOL/'PROTOCOL_LOCK.json']
    freeze={'schema':'META_ANALYSIS_FREEZE_BEFORE_DISCOVERY','code_sha256':sha(Path(__file__)),
        'primary_relative_step':PRIMARY,'directions':list(DIRECTIONS),'decision_rule':lock['decision_rule'],
        'locked_file_sha256':{p.name:sha(p) for p in files},'all_five_outer_folds_meta_complete':True,
        'discovery_eeg_reads_at_freeze':0}
    fp=PROTOCOL/'META_ANALYSIS_FREEZE.json'
    if fp.exists() and json.loads(fp.read_text())!=freeze:raise RuntimeError('meta analysis freeze drift')
    jwrite(fp,freeze)
    print('META_ANALYSIS_FROZEN',sha(fp),flush=True)

def verify_freeze():
    fp=PROTOCOL/'META_ANALYSIS_FREEZE.json'
    if not fp.exists():raise RuntimeError('discovery forbidden before complete inner-meta freeze')
    z=json.loads(fp.read_text())
    if z['code_sha256']!=sha(Path(__file__)):raise RuntimeError('analysis code changed after freeze')
    for name,h in z['locked_file_sha256'].items():
        path=(PROTOCOL if name=='PROTOCOL_LOCK.json' else OUT)/name
        if sha(path)!=h:raise RuntimeError(f'frozen meta result changed: {name}')
    return z

def run_discovery(f,lock):
    verify_freeze();ref,data,geo=S.prepare_train(f,lock);geo['path']=lock['cells'][f]['geometry_archive']
    m,params=model_and_params(ref);original=A.model_state_sha(m);g,_=gradients(m,ref,data,geo);dirs=directions(g)
    d=S.discover(f,lock,data)
    # Discovery extraction contains native stages but no P/C cache. Derive the same
    # frozen-reference coordinates used by inner-meta evaluation before any step.
    for suffix,stage in [('1','H_SHARED1'),('2','H_SHARED2')]:
        d['p'+suffix+'0'],d['c'+suffix+'0']=S.decompose_np(d['h'+suffix],geo[stage])
    before=eval_role(m,ref,d,geo,lock['cells'][f],True)
    rows=[];count=0
    for name in DIRECTIONS:
        after,step=perturb(m,ref,d,geo,lock['cells'][f],unit(dirs[name]),PRIMARY,original);count+=1
        ev=effect(before,after);row={'fold':f,'direction':name,'relative_step':PRIMARY,'parameter_displacement_L2':step}
        for k in ev:row['before_'+k]=before[k];row['after_'+k]=after[k];row['delta_'+k]=ev[k]
        row['T_task']=-ev['full_CE'];row['T_PU']=-ev['Ponly_CE'];row['T_PER']=ev['rho1'];rows.append(row)
    c=cell(f);cwrite(c/'DISCOVERY_VIRTUAL_STEP_EFFECTS.csv',rows)
    jwrite(c/'DISCOVERY_RESTORE_AUDIT.json',{'fold':f,'virtual_evaluations_restored':count,
        'canonical_state_sha256':original,'final_state_sha256':A.model_state_sha(m),
        'all_parameters_restored':True,'meta_freeze_sha256':sha(PROTOCOL/'META_ANALYSIS_FREEZE.json')})
    jwrite(c/'DISCOVERY_COMPLETE.json',{'fold':f,'directions':list(DIRECTIONS)})
    print('DISCOVERY_COMPLETE',f,flush=True)

def aggregate_final(lock):
    verify_freeze();rows=[];audits=[]
    for f in FOLDS:
        if not (cell(f)/'DISCOVERY_COMPLETE.json').exists():raise RuntimeError('all discovery folds required')
        rows.extend(cread(cell(f)/'DISCOVERY_VIRTUAL_STEP_EFFECTS.csv'))
        audits.append(json.loads((cell(f)/'DISCOVERY_RESTORE_AUDIT.json').read_text()))
    cwrite(OUT/'DISCOVERY_VIRTUAL_STEP_EFFECTS.csv',rows)
    restore=json.loads((OUT/'PARAMETER_RESTORE_AUDIT.json').read_text());restore['discovery_fold_audits']=audits
    restore['total_virtual_evaluations_restored']=sum(x['total_virtual_evaluations'] for x in restore['fold_audits'])+sum(x['virtual_evaluations_restored'] for x in audits)
    jwrite(OUT/'PARAMETER_RESTORE_AUDIT.json',restore)
    jwrite(OUT/'FINAL_HELDOUT_EXCLUSION_AUDIT.json',{'outer_dev_eeg_reads':0,'final_heldout_eeg_reads':0,
        'meta_objectives':'inner_train_subjects_only','discovery_access_after_meta_freeze':True,
        'discovery_used_to_choose_direction_or_step':False,'geometry_fitted_on_inner_train_only':True,
        'historical_provenance_caveat':lock['historical_provenance_caveat']})
    signs={r['direction']:r for r in cread(OUT/'TRANSFER_SIGN_CONSISTENCY.csv')}
    cross=cread(OUT/'CROSS_SUBJECT_GRADIENT_ALIGNMENT.csv');effects=cread(OUT/'META_VAL_VIRTUAL_STEP_EFFECTS.csv')
    primary=[r for r in effects if abs(float(r['relative_step'])-PRIMARY)<1e-12]
    mean=lambda rs,k:float(np.mean([float(x[k]) for x in rs]))
    discovery={name:{k:mean([r for r in rows if r['direction']==name],k) for k in ('T_task','T_PU','T_PER','delta_full_BA','delta_Ponly_BA')} for name in DIRECTIONS}
    meta={name:{k:mean([r for r in primary if r['direction']==name],k) for k in ('T_task','T_PU','T_PER','delta_full_BA','delta_Ponly_BA')} for name in DIRECTIONS}
    cosines={name:mean([r for r in cross if r['objective']==name and r['layer']=='TOTAL'],'train_val_cosine') for name in BASE}
    # Each P-informed direction must pass the predeclared intended metric, no task harm, and same discovery sign.
    intended={'P_UTILITY':('T_PU',),'SHARED1_PERSISTENCE':('T_PER',),
              'BALANCED':('T_PU','T_PER'),'CONFLICT_AWARE':('T_PU','T_PER')}
    winners=[]
    for name,targets in intended.items():
        if (all(float(signs[name]['Ponly_improve_fraction' if k=='T_PU' else 'persistence_improve_fraction'])>=.70
                and meta[name][k]>0 and discovery[name][k]>0 for k in targets)
                and meta[name]['T_task']>=0):winners.append(name)
    pu_cos=cosines['P_UTILITY'];per_cos=cosines['SHARED1_PERSISTENCE']
    conflict=cread(OUT/'GRADIENT_CONFLICT_AUDIT.csv');pair=next(x for x in conflict if x['analysis']=='PU_PER_train_conflict')
    source=cread(OUT/'SOURCE_VS_HELD_TRANSFER.csv')
    sm=lambda name,k:mean([r for r in source if r['direction']==name and abs(float(r['relative_step'])-PRIMARY)<1e-12],k)
    raw=meta['PU_PER_RAW'];aware=meta['CONFLICT_AWARE']
    raw_poor=raw['T_PU']<=0 or raw['T_PER']<=0
    aware_material=all(aware[k]-raw[k]>.2*abs(raw[k])+1e-8 for k in ('T_PU','T_PER'))
    if winners:label='CROSS_SUBJECT_TRANSFERABLE_P_GRADIENT_FOUND'
    elif float(pair['negative_fraction'])>=.7 and raw_poor and aware_material:
        label='PU_PERSIST_GRADIENT_CONFLICT'
    elif pu_cos<=.05 and sm('P_UTILITY','source_Ponly_CE_benefit')>0 and meta['P_UTILITY']['T_PU']<=0:
        label='P_UTILITY_GRADIENT_SUBJECT_SPECIFIC'
    elif per_cos<=.05 and sm('SHARED1_PERSISTENCE','source_rho1_benefit')>0 and meta['SHARED1_PERSISTENCE']['T_PER']<=0:
        label='PERSISTENCE_GRADIENT_SUBJECT_SPECIFIC'
    else:label='NO_TRANSFERABLE_P_GRADIENT'
    recommendation=('Test the eligible direction(s) '+', '.join(winners)+' in a future subject-meta training experiment.' if winners else
        'Do not wrap the same P objectives in subject-level meta optimization yet; change representation or objective definition first.')
    decision={'terminal_label':label,'eligible_directions':winners,'mean_train_val_gradient_cosine':cosines,
        'primary_meta_held_transfer':meta,'primary_sign_consistency':signs,'discovery_one_step_transfer':discovery,
        'recommended_next_action':recommendation,'outer_dev_eeg_reads':0,'final_heldout_eeg_reads':0,
        'canonical_model_weights_fully_restored':True}
    jwrite(OUT/'DECISION_SUMMARY.json',decision)
    report=f'''# Cross-subject SIRE gradient transfer — seed 0\n\nDiagnostic only: 5 outer folds × 5 disjoint subject meta folds, six predeclared directions, three matched parameter-space virtual steps. No training checkpoint was produced. Geometry is the exact frozen layerwise PathFit geometry. It was originally fitted on all inner-train subjects, including those serving as meta-val here; no geometry is refitted in a meta split. Discovery was read after the meta results, code, primary step and decision rules were frozen.\n\n## Q1. Ordinary CE gradient\nMean train/held cosine {cosines['FULL_CE']:.6f}. Primary-step held task transfer {meta['FULL_CE']['T_task']:+.6f}; beneficial sign in {float(signs['FULL_CE']['task_improve_fraction']):.1%} of meta splits.\n\n## Q2. P-utility gradient\nMean train/held cosine {pu_cos:.6f}. Source P-only CE benefit {sm('P_UTILITY','source_Ponly_CE_benefit'):+.6f}; held benefit {meta['P_UTILITY']['T_PU']:+.6f}; beneficial sign in {float(signs['P_UTILITY']['Ponly_improve_fraction']):.1%}.\n\n## Q3. Persistence gradient\nMean train/held cosine {per_cos:.6f}. Source persistence gain {sm('SHARED1_PERSISTENCE','source_rho1_benefit'):+.6f}; held gain {meta['SHARED1_PERSISTENCE']['T_PER']:+.6f}; beneficial sign in {float(signs['SHARED1_PERSISTENCE']['persistence_improve_fraction']):.1%}. Persistence uses explicit centered Pearson, matching the evaluation quantity.\n\n## Q4. Gradient conflict\nMean PU/PER source cosine {float(pair['mean_cosine']):+.6f}; negative in {float(pair['negative_fraction']):.1%} of splits. Layerwise cosines and descriptive associations are in the CSV audits.\n\n## Q5 and Q6. Balancing and conflict projection\nAt the primary step, raw combined held transfer (task, PU, persistence) = ({meta['PU_PER_RAW']['T_task']:+.6f}, {meta['PU_PER_RAW']['T_PU']:+.6f}, {meta['PU_PER_RAW']['T_PER']:+.6f}); balanced = ({meta['BALANCED']['T_task']:+.6f}, {meta['BALANCED']['T_PU']:+.6f}, {meta['BALANCED']['T_PER']:+.6f}); conflict-aware = ({meta['CONFLICT_AWARE']['T_task']:+.6f}, {meta['CONFLICT_AWARE']['T_PU']:+.6f}, {meta['CONFLICT_AWARE']['T_PER']:+.6f}). All are equal-displacement directions, not trained models.\n\n## Q7 and Q8. Transferable direction and discovery\nEligible P-informed directions under the locked rule: {winners}. Discovery primary-step effects are tabulated in `DISCOVERY_VIRTUAL_STEP_EFFECTS.csv`; the corresponding mean (task, PU, persistence) for each direction is {discovery}. Discovery did not choose a direction or step.\n\n## Decision and integrity\n**{label}**. {recommendation} All {restore['total_virtual_evaluations_restored']} virtual candidate evaluations restored the exact canonical model state. No persistent parameter update, BN change, or classifier change occurred. Outer-dev EEG reads 0; final-heldout EEG reads 0. The original canonical checkpoints retain the historical final-heldout diagnostic provenance caveat. Subject-level bootstrap CIs and sign fractions are in `TRANSFER_SUMMARY.csv` and `TRANSFER_SIGN_CONSISTENCY.csv`.\n'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(decision,indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--meta-fold',type=int);ap.add_argument('--freeze-meta',action='store_true')
    ap.add_argument('--discovery-fold',type=int);ap.add_argument('--aggregate',action='store_true');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    lock,ss=preflight()
    if args.smoke:
        ref,data,geo=S.prepare_train(0,lock);m,_=model_and_params(ref);original=A.model_state_sha(m)
        split=next(x for x in ss if x['fold']==0 and x['meta_fold']==0)
        tr=role(data,split['meta_train_subjects']);va=role(data,split['meta_val_subjects'])
        gt,rt=gradients(m,ref,tr,geo);gv,rv=gradients(m,ref,va,geo)
        base=eval_role(m,ref,va,geo,lock['cells'][0],True)
        if abs(base['rho1']-rv)>1e-5:raise RuntimeError('smoke Pearson mismatch')
        after,step=perturb(m,ref,va,geo,lock['cells'][0],unit(gt['P_UTILITY']),PRIMARY,original)
        if A.model_state_sha(m)!=original:raise RuntimeError('smoke restore mismatch')
        jwrite(PROTOCOL/'SMOKE_AUDIT.json',{'status':'PASS','fold':0,'meta_fold':0,
            'canonical_reference_logits_replayed':True,'geometry_hashes_verified':True,
            'meta_train_subjects':split['meta_train_subjects'],'meta_val_subjects':split['meta_val_subjects'],
            'gradient_norms':{k:norm(v) for k,v in gt.items()},'meta_val_gradient_norms':{k:norm(v) for k,v in gv.items()},
            'Pearson_gradient_metric_identity_error':abs(base['rho1']-rv),'virtual_step_L2':step,
            'canonical_state_restored_exactly':True,'discovery_not_accessed':True})
        print('SMOKE_PASS',flush=True)
    elif args.meta_fold is not None:
        if args.meta_fold not in FOLDS:raise ValueError(args.meta_fold)
        if (PROTOCOL/'META_ANALYSIS_FREEZE.json').exists():raise RuntimeError('meta frozen; no rerun after discovery gate')
        run_meta(args.meta_fold,lock,ss)
    elif args.freeze_meta:aggregate_meta(lock)
    elif args.discovery_fold is not None:
        if args.discovery_fold not in FOLDS:raise ValueError(args.discovery_fold)
        run_discovery(args.discovery_fold,lock)
    elif args.aggregate:aggregate_final(lock)
    else:ap.error('select --meta-fold F, --freeze-meta, --discovery-fold F, or --aggregate')
if __name__=='__main__':main()
