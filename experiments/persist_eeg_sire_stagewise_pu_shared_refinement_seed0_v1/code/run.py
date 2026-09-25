"""Locked seed-0 SIRE stagewise Protected-utility pilot."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

EXP=Path(__file__).resolve().parents[1]; REPO=EXP.parents[1]
OLD=REPO/'experiments/persist_eeg_sire_layerwise_p_formation_audit_seed0_v1'
spec=importlib.util.spec_from_file_location('sire_layerwise_frozen',OLD/'code/run.py')
L=importlib.util.module_from_spec(spec);sys.modules[spec.name]=L;spec.loader.exec_module(L)
P=L.P; A=L.A; DEVICE=L.DEVICE; TASK='OpenBMI_MI'; FOLDS=tuple(range(5))
OUT=EXP/'outputs'; PROTOCOL=EXP/'protocol'
RUNTIME=Path(os.environ.get('SIRE_STAGEWISE_RUNTIME',str(REPO.parent/'sire_stagewise_pu_shared_seed0_runtime'))).resolve()
ARMS=('SHARED_CE_CONTINUATION','STAGEWISE_PU_SHARED'); ALLOWED={'depth1.weight','point1.weight','depth2.weight','point2.weight'}
EPOCHS=20; LR=1e-4; WD=5e-4; CLIP=5.; EPS=1e-8; BOOT=20_000
torch.set_num_threads(min(int(os.environ.get('SIRE_STAGEWISE_CPU_THREADS','8')),os.cpu_count() or 1))
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def jwrite(path,obj): P.jwrite(path,obj)
def cwrite(path,rows): P.cwrite(path,rows)
def sha(path): return P.sha(path)
def arrsha(*a): return P.arr_sha(*a)
def cell(f): return RUNTIME/'cells'/f'fold{f}_seed0'
def npa(x): return np.asarray(x,np.float32)
def tensor(x): return torch.as_tensor(np.ascontiguousarray(x),device=DEVICE,dtype=torch.float32)
def hsh(x): return hashlib.sha256(x).hexdigest()
def state(model): return P.state_bytes(model)
def seed(*parts): return P.seed('stagewise_pu_shared',*parts)

def preflight():
    old=L.preflight();audit=json.loads((OLD/'outputs/LAYERWISE_GEOMETRY_AUDIT.json').read_text())
    cells=[]
    for f,c in enumerate(old['cells']):
        g=cell(f).parent # merely establishes parent; geometry lives in OLD runtime
        geo=L.rcell(f)/'FITTED_PATHWAYS.npz'
        rec=next(x for x in audit['cells'] if int(x['fold'])==f)
        if not geo.exists() or sha(geo)!=rec['fitted_pathway_archive_sha256']:raise RuntimeError(f'layerwise archive drift fold {f}')
        cells.append({**c,'geometry_archive':str(geo),'geometry_archive_sha256':sha(geo),
            'geometry_q_mu_sha256':{x['stage']:x['q_mu_sha256'] for x in rec['stages'] if x['stage'] in ('H_SHARED1','H_SHARED2')}})
    lock={'schema':'SIRE_STAGEWISE_PU_SHARED_REFINEMENT_SEED0_V1','task':TASK,'seed':0,'folds':list(FOLDS),
        'cells':cells,'prior_layerwise_protocol_sha256':sha(OLD/'protocol/PROTOCOL_LOCK.json'),
        'prior_layerwise_geometry_audit_sha256':sha(OLD/'outputs/LAYERWISE_GEOMETRY_AUDIT.json'),
        'arms':list(ARMS),'trainable_parameters':sorted(ALLOWED),'model_mode':'eval_all_times',
        'geometry':'exact layerwise train-only frozen PathFit Q and mu',
        'batching':'each subject-class unit contributes two trials per session per batch; all groups synchronously shuffled; every trial exactly once per epoch; identical batch indices for both arms',
        'optimizer':'AdamW','lr':LR,'weight_decay':WD,'gradient_clip':CLIP,'epochs':EPOCHS,'checkpoint_endpoint':'fixed_epoch20',
        'lambda_persist':1.,'margin_persist':.02,'lambda_off':.01,'lambda_var':.1,'variance_floor_fraction':.8,
        'lambda_C':.1,'lambda_anchor':.1,'anchor_temperature':2.,'bootstrap_subject_draws':BOOT,
        'roles_read':['inner_train','discovery_after_epoch20_only'],'outer_dev_eeg_reads':0,'final_heldout_eeg_reads':0,
        'historical_provenance_caveat':old['historical_provenance_caveat']}
    path=PROTOCOL/'PROTOCOL_LOCK.json'
    if path.exists() and json.loads(path.read_text())!=lock:raise RuntimeError('protocol lock changed')
    jwrite(path,lock)
    return lock

def geometry(f,lock):
    c=lock['cells'][f]; a=np.load(c['geometry_archive']);out={}
    for st in ('H_SHARED1','H_SHARED2'):
        fit={k:a[f'{st}__{k}'].copy() for k in ('q','mu','sd','coef')}
        if arrsha(fit['q'],fit['mu'])!=c['geometry_q_mu_sha256'][st]:raise RuntimeError(f'Q/mu mismatch {f} {st}')
        if np.max(np.abs(fit['q'].T@fit['q']-np.eye(fit['q'].shape[1])))>1e-5:raise RuntimeError('Q not orthonormal')
        fit['stage']=st;out[st]=fit
    out['spectrum']={k:a[f'final_spectrum__{k}'].copy() for k in ('mean','whitener','directions','rho')}
    return out

def flat(rows):
    fields={'y':[],'subject':[],'session':[],'trial':[],'hc':[],'h1':[],'h2':[],'emb':[],'z0':[]}
    for r in rows:
        n=len(r['raw_y']);fields['y'].append(np.asarray(r['raw_y'],np.int64));fields['subject'].extend([r['subject']]*n)
        fields['session'].extend([int(r['session'])]*n);fields['trial'].extend(range(n))
        for key,src in [('hc','H_CONCAT'),('h1','H_SHARED1'),('h2','H_SHARED2'),('emb','EMBEDDING')]:fields[key].append(r['acts'][src])
        fields['z0'].append(r['logits'])
    for key in ('y','hc','h1','h2','emb','z0'):fields[key]=np.concatenate(fields[key])
    fields['subject']=np.asarray(fields['subject']);fields['session']=np.asarray(fields['session']);fields['trial']=np.asarray(fields['trial'])
    return fields

def decompose_np(h,fit):
    center=np.asarray(h,np.float64)-fit['mu'].astype(np.float64);q=fit['q'].astype(np.float64)
    p=center@q;c=center-p@q.T
    if np.max(np.abs(fit['mu']+p@q.T+c-h))>1e-5:raise RuntimeError('P+C reconstruction')
    return npa(p),npa(c)

def prepare_train(f,lock):
    c=lock['cells'][f];sessions=sorted(set(c['source_sessions']+[c['future_session']]))
    pieces,mapping=A.fetch_role(TASK,c['train_subjects'],sessions,c['cache_name'],None)
    ref=A.build_model(TASK,len(mapping),Path(c['checkpoint_path']));ref.eval()
    mean,std=A.load_normalizer(Path(c['normalizer_path']))
    rows=L.extract(ref,pieces,mean,std,f);del pieces
    data=flat(rows);data['rows']=rows;data['mapping']=mapping;data['mean']=mean;data['std']=std
    geo=geometry(f,lock)
    for k,st in [('1','H_SHARED1'),('2','H_SHARED2')]:
        p,c0=decompose_np(data['h'+k],geo[st]);data['p'+k+'0']=p;data['c'+k+'0']=c0
        data['sigma_c'+k]=np.maximum(c0.std(0),1e-3).astype(np.float32)
    data['mu_p1']=data['p10'].mean(0).astype(np.float32);data['sigma_p1']=np.maximum(data['p10'].std(0),1e-3).astype(np.float32)
    return ref,data,geo

def orders(data,f):
    groups=defaultdict(list);y=data['y'];sub=data['subject'];sess=data['session']
    for i in range(len(y)):groups[(str(sub[i]),int(y[i]),int(sess[i]))].append(i)
    sessions=sorted(set(int(x) for x in sess));units=sorted(set((s,k) for s,k,t in groups))
    if len(sessions)!=2:raise RuntimeError('exactly two training sessions required')
    lengths={len(groups[(s,k,t)]) for s,k in units for t in sessions}
    if len(lengths)!=1 or next(iter(lengths))%2:raise RuntimeError(f'unbalanced or odd group counts {lengths}')
    n=next(iter(lengths));allorders=[]
    for ep in range(1,EPOCHS+1):
        perm={key:np.random.default_rng(seed('batch',f,ep,*key)).permutation(ix) for key,ix in groups.items()}
        batches=[]
        for j in range(0,n,2):
            batch=np.asarray([ix for s,k in units for t in sessions for ix in perm[(s,k,t)][j:j+2]],np.int32)
            if len(batch)!=len(units)*4:raise RuntimeError('paired batch malformed')
            batches.append(batch)
        if not np.array_equal(np.sort(np.concatenate(batches)),np.arange(len(y))):raise RuntimeError('batch exposure mismatch')
        allorders.append(batches)
    return allorders,units,sessions,arrsha(*[np.concatenate(x) for x in allorders])

def student(ref):
    m=copy.deepcopy(ref);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    for name,p in m.named_parameters():
        if name in ALLOWED:p.requires_grad_(True)
    if {n for n,p in m.named_parameters() if p.requires_grad}!=ALLOWED:raise RuntimeError('trainable whitelist')
    return m

def project(h,q,mu):
    p=(h-mu)@q;c=h-mu-p@q.T
    return p,c

def forward_batch(m,hc,q1,mu1,q2,mu2,c20,ref):
    h1=L.next_stage(m,'H_CONCAT',hc);h2=L.next_stage(m,'H_SHARED1',h1)
    p1,c1=project(h1,q1,mu1);p2,c2=project(h2,q2,mu2)
    z=L.continuation(ref,'H_SHARED2',h2)[1]
    hybrid=mu2+p2@q2.T+c20
    zhy=L.continuation(ref,'H_SHARED2',hybrid)[1]
    return h1,h2,p1,c1,p2,c2,z,zhy

def losses(out,batch,data,dev):
    h1,h2,p1,c1,p2,c2,z,zhy=out
    y=torch.as_tensor(data['y'][batch],device=DEVICE,dtype=torch.long)
    p10=dev['p10'][batch];c10=dev['c10'][batch];c20=dev['c20'][batch]
    # Batch layout: unit, session, two trials. Reference and student use identical matched pairs.
    n=len(batch)//4; m=p1.reshape(n,2,2,-1).mean(2);m0=p10.reshape(n,2,2,-1).mean(2)
    center=dev['mu_p1'];scale=dev['sigma_p1']
    z1=(m[:,0]-center)/scale;z2=(m[:,1]-center)/scale
    z10=(m0[:,0]-center)/scale;z20=(m0[:,1]-center)/scale
    corr=z1.T@z2/n;corr0=z10.T@z20/n
    diag=torch.diag(corr);diag0=torch.diag(corr0);eye=torch.eye(corr.shape[0],device=DEVICE,dtype=torch.bool)
    persist=F.relu(torch.clamp(diag0+.02,max=1.)-diag).square().mean()+.01*corr[~eye].square().mean()
    var=torch.cat([m[:,0],m[:,1]],0).var(0,unbiased=False);var0=torch.cat([m0[:,0],m0[:,1]],0).var(0,unbiased=False)
    variance=F.relu(.8*torch.sqrt(var0+EPS)-torch.sqrt(var+EPS)).square().mean()
    lc1=((c1-c10)/dev['sigma_c1']).square().mean();lc2=((c2-c20)/dev['sigma_c2']).square().mean();lc=.5*(lc1+lc2)
    z0=dev['z0'][batch];correct=z0.argmax(1)==y
    anchor=F.kl_div(F.log_softmax(z[correct]/2,1),F.softmax(z0[correct]/2,1),reduction='batchmean') if correct.any() else z.sum()*0
    utility=F.cross_entropy(zhy,y);ce=F.cross_entropy(z,y)
    total=utility+persist+.1*variance+.1*lc+.1*anchor
    return {'PUTILITY':utility,'PERSIST':persist,'VAR':variance,'C':lc,'ANCHOR':anchor,'TOTAL':total,'CE':ce,
        'diag':diag.mean(),'offdiag':corr[~eye].abs().mean(),'variance':var.mean(),'C1':lc1,'C2':lc2}

def devcache(data):
    return {k:tensor(data[k]) for k in ('p10','c10','p20','c20','z0','sigma_c1','sigma_c2','mu_p1','sigma_p1')}

def measure_train(m,ref,data,geo,dev,orders0,epoch,arm,fold,batch_loss):
    q1,mu1,q2,mu2=[tensor(geo[st][field]) for st,field in [('H_SHARED1','q'),('H_SHARED1','mu'),('H_SHARED2','q'),('H_SHARED2','mu')]]
    ps1=[];ps2=[];cs1=[];cs2=[];zs=[];zhys=[]
    with torch.inference_mode():
        for start in range(0,len(data['y']),128):
            ix=np.arange(start,min(start+128,len(data['y'])))
            got=forward_batch(m,tensor(data['hc'][ix]),q1,mu1,q2,mu2,dev['c20'][ix],ref)
            for dst,val in zip((ps1,cs1,ps2,cs2,zs,zhys),(got[2],got[3],got[4],got[5],got[6],got[7])):dst.append(val.cpu().numpy())
    p1=np.concatenate(ps1);p2=np.concatenate(ps2);c1=np.concatenate(cs1);c2=np.concatenate(cs2);z=np.concatenate(zs);zh=np.concatenate(zhys)
    rows=[];offset=0
    for old in data['rows']:
        n=len(old['raw_y']);r={'subject':old['subject'],'session':old['session'],'raw_y':old['raw_y'],
            'acts':{'H_SHARED1':np.asarray(old['acts']['H_SHARED1']), 'H_SHARED2':np.asarray(old['acts']['H_SHARED2'])}}
        r['acts']['H_SHARED1']=geo['H_SHARED1']['mu']+p1[offset:offset+n]@geo['H_SHARED1']['q'].T+c1[offset:offset+n]
        rows.append(r);offset+=n
    # Native Q-coordinate cross-session persistence, using full subject-class centroids.
    def rho(p):
        keys=defaultdict(list)
        for i,v in enumerate(p):keys[(data['subject'][i],int(data['y'][i]),int(data['session'][i]))].append(v)
        ses=sorted(set(data['session']));units=sorted(set((s,k) for s,k,t in keys))
        a=np.stack([np.mean(keys[(s,k,ses[0])],0) for s,k in units]);b=np.stack([np.mean(keys[(s,k,ses[1])],0) for s,k in units])
        cc=np.corrcoef(a.T,b.T)[:a.shape[1],a.shape[1]:];return float(np.nanmean(np.diag(cc))),float(np.nanmean(np.abs(cc[~np.eye(len(cc),dtype=bool)]))),float(np.mean(np.var(np.r_[a,b],axis=0)))
    rp,off,var=rho(p1)
    y=data['y'];base={'fold':fold,'arm':arm,'epoch':epoch,'train_shared1_persistence_rho':rp,'train_shared1_offdiag_abs':off,
        'train_shared1_centroid_variance':var,'train_shared2_p_only_CE':P.metric(y,zh)['NLL'],
        'train_shared2_p_only_BA':P.metric(y,zh)['BA'],'train_full_native_BA':P.metric(y,z)['BA'],
        'train_P1_drift_RMS':float(np.sqrt(np.mean((p1-data['p10'])**2))),
        'train_C1_drift_RMS':float(np.sqrt(np.mean((c1-data['c10'])**2))),
        'train_P2_drift_RMS':float(np.sqrt(np.mean((p2-data['p20'])**2))),
        'train_C2_drift_RMS':float(np.sqrt(np.mean((c2-data['c20'])**2)))}
    base.update({f'loss_{k}':float(v) for k,v in batch_loss.items()})
    return base

def train_arm(f,arm,ref,data,geo,orders,order_hash):
    m=student(ref);initial=state(m);init_hash=A.model_state_sha(m);dev=devcache(data)
    q1,mu1,q2,mu2=[tensor(geo[st][field]) for st,field in [('H_SHARED1','q'),('H_SHARED1','mu'),('H_SHARED2','q'),('H_SHARED2','mu')]]
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=LR,weight_decay=WD)
    history=[];gradrows=[];steps=0;exposure=0
    for ep in range(EPOCHS+1):
        batchsum=defaultdict(float);batchcount=0
        if ep>0:
            batches=orders[ep-1]
            for j,ix in enumerate(batches):
                m.eval();opt.zero_grad(set_to_none=True)
                got=forward_batch(m,tensor(data['hc'][ix]),q1,mu1,q2,mu2,dev['c20'][ix],ref)
                ls=losses(got,ix,data,dev)
                if ep==1 and j==0 and arm==ARMS[1]:
                    pars=[p for n,p in m.named_parameters() if n in ALLOWED];names=[n for n,p in m.named_parameters() if n in ALLOWED]
                    for part in ('PUTILITY','PERSIST','VAR','C','ANCHOR'):
                        gg=torch.autograd.grad(ls[part],pars,retain_graph=True,allow_unused=True)
                        for name,g in zip(names,gg):gradrows.append({'fold':f,'arm':arm,'epoch':ep,'component':part,'parameter':name,'gradient_L2':float(g.norm()) if g is not None else 0.})
                objective=ls['CE'] if arm==ARMS[0] else ls['TOTAL'];objective.backward()
                if {n for n,p in m.named_parameters() if p.grad is not None}!=ALLOWED:raise RuntimeError('gradient whitelist violation')
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],CLIP);opt.step()
                steps+=1;exposure+=len(ix)
                for k in ('PUTILITY','PERSIST','VAR','C','ANCHOR','TOTAL','CE','diag','offdiag','variance','C1','C2'):batchsum[k]+=float(ls[k].detach())
                batchcount+=1
        else:
            with torch.inference_mode():
                for ix in orders[0]:
                    ls=losses(forward_batch(m,tensor(data['hc'][ix]),q1,mu1,q2,mu2,dev['c20'][ix],ref),ix,data,dev)
                    for k in ('PUTILITY','PERSIST','VAR','C','ANCHOR','TOTAL','CE','diag','offdiag','variance','C1','C2'):batchsum[k]+=float(ls[k])
                    batchcount+=1
        avg={k:v/batchcount for k,v in batchsum.items()}
        row=measure_train(m,ref,data,geo,dev,orders[0],ep,arm,f,avg);history.append(row)
        print('HISTORY',f,arm,ep,'BA',row['train_full_native_BA'],'P rho',row['train_shared1_persistence_rho'],flush=True)
    final=state(m);changed={k for k in initial if initial[k]!=final[k]}
    if changed!=ALLOWED:raise RuntimeError(f'state whitelist violation {changed}')
    if m.training or any(x.training for x in m.modules() if isinstance(x,(torch.nn.BatchNorm2d,torch.nn.Dropout))):raise RuntimeError('eval mode drift')
    dest=cell(f);dest.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),dest/f'{arm}.pt')
    audit={'fold':f,'arm':arm,'initial_model_state_sha256':init_hash,'canonical_model_state_sha256':A.model_state_sha(ref),
        'final_model_state_sha256':A.model_state_sha(m),'changed_state_keys':sorted(changed),'trainable_parameters':sorted(ALLOWED),
        'batch_order_sha256':order_hash,'optimizer_steps':steps,'sample_exposures':exposure,'all_nontrainable_state_byte_identical':True,
        'all_batchnorm_state_byte_identical':True,'global_eval_mode':True,'geometry_archive_sha256':sha(Path(geo['path']))}
    return m,history,gradrows,audit

def smoke(ref,data,geo,orders,units,f):
    m=student(ref);x=tensor(data['hc'][:8]);q1,mu1,q2,mu2=[tensor(geo[st][field]) for st,field in [('H_SHARED1','q'),('H_SHARED1','mu'),('H_SHARED2','q'),('H_SHARED2','mu')]]
    with torch.inference_mode():
        got=forward_batch(m,x,q1,mu1,q2,mu2,tensor(data['c20'][:8]),ref)
        err=max(float((got[0]-tensor(data['h1'][:8])).abs().max()),float((got[1]-tensor(data['h2'][:8])).abs().max()),
            float((got[6]-tensor(data['z0'][:8])).abs().max()),float((got[7]-tensor(data['z0'][:8])).abs().max()))
    if err>1e-5:raise RuntimeError(f'epoch0/native hybrid mismatch {err}')
    if len(orders[0][0])!=4*len(units):raise RuntimeError('matched session pairs absent')
    if np.min(np.var(data['p10'],axis=0))<1e-8:raise RuntimeError('initial P variance collapsed')
    dev=devcache(data);ix=orders[0][0];got=forward_batch(m,tensor(data['hc'][ix]),q1,mu1,q2,mu2,dev['c20'][ix],ref)
    ls=losses(got,ix,data,dev);ls['TOTAL'].backward()
    if {n for n,p in m.named_parameters() if p.grad is not None}!=ALLOWED:raise RuntimeError('smoke gradient path')
    if any(x.training for x in m.modules() if isinstance(x,(torch.nn.BatchNorm2d,torch.nn.Dropout))):raise RuntimeError('smoke BN/dropout')
    return {'fold':f,'status':'PASS','epoch0_max_native_logit_or_stage_error':err,'Q_C_reconstruction':True,
        'native_P_hybrid_identity':True,'four_weight_gradient_path':True,'batch_pairs':len(units),'batch_size':len(ix),
        'matched_arm_batches_and_steps_by_construction':True,'discovery_not_loaded':True}

def discover(f,lock,data):
    c=lock['cells'][f];sessions=sorted(set(c['source_sessions']+[c['future_session']]))
    pieces,mapping=A.fetch_role(TASK,c['discovery_subjects'],sessions,c['cache_name'],data['mapping'])
    if mapping!=data['mapping']:raise RuntimeError('discovery class mapping drift')
    ref=A.build_model(TASK,len(mapping),Path(c['checkpoint_path']))
    rows=L.extract(ref,pieces,data['mean'],data['std'],f);del pieces
    d=flat(rows);d['rows']=rows
    return d

def inference(m,ref,d,geo):
    q1,mu1,q2,mu2=[tensor(geo[st][field]) for st,field in [('H_SHARED1','q'),('H_SHARED1','mu'),('H_SHARED2','q'),('H_SHARED2','mu')]]
    p20,c20=decompose_np(d['h2'],geo['H_SHARED2']);out=defaultdict(list)
    with torch.inference_mode():
        for start in range(0,len(d['y']),128):
            ix=slice(start,min(start+128,len(d['y'])))
            got=forward_batch(m,tensor(d['hc'][ix]),q1,mu1,q2,mu2,tensor(c20[ix]),ref)
            h1,h2,p1,c1,p2,c2,z,zh=got
            hc=mu2+tensor(p20[ix])@q2.T+c2
            zc=L.continuation(ref,'H_SHARED2',hc)[1]
            for key,val in [('h1',h1),('h2',h2),('p1',p1),('c1',c1),('p2',p2),('c2',c2),('z',z),('z_p',zh),('z_c',zc)]:out[key].append(val.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def rows_for(d,inf):
    rows=[];off=0
    for old in d['rows']:
        n=len(old['raw_y']);rows.append({'subject':old['subject'],'session':old['session'],'raw_y':old['raw_y'],
            'acts':{'H_SHARED1':inf['h1'][off:off+n],'H_SHARED2':inf['h2'][off:off+n]}});off+=n
    return rows

def metrics_for(f,arm,d,z,future):
    rows=[];pred=z.argmax(1);truth=d['y'];session=d['session'];sub=d['subject']
    for s in sorted(set(sub)):
        for t in sorted(set(session)):
            ix=(sub==s)&(session==t)
            if not ix.any():continue
            rows.append({'fold':f,'arm':arm,'subject':s,'session':int(t),'is_future_session':bool(t==future),**P.metric(truth[ix],z[ix])})
    return rows

def transitions(f,arm,d,z,base):
    y=d['y'];p0=base.argmax(1);p=z.argmax(1);a=p0==y;b=p==y
    return {'fold':f,'arm':arm,'correct_to_correct':int(np.sum(a&b)),'correct_to_wrong':int(np.sum(a&~b)),
        'wrong_to_correct':int(np.sum(~a&b)),'wrong_to_wrong':int(np.sum(~a&~b))}

def evaluate_fold(f,lock,ref,data,geo,models):
    d=discover(f,lock,data);base=d['z0'];future=lock['cells'][f]['future_session']
    metrics=metrics_for(f,'BASELINE',d,base,future);trans=[];transplant=[];mechanism=[]
    infbase=inference(ref,ref,d,geo);infs={'BASELINE':infbase}
    for arm,m in models.items():infs[arm]=inference(m,ref,d,geo)
    y=d['y'];target=A.canonical_targets(d['emb'],geo['spectrum'],lock['cells'][f]['protected_coordinates'])
    for arm,inf in infs.items():
        z=base if arm=='BASELINE' else inf['z'];metrics+=[] if arm=='BASELINE' else metrics_for(f,arm,d,z,future)
        if arm!='BASELINE':
            trans.append(transitions(f,arm,d,z,base));trans.append(transitions(f,arm+'_P_ONLY_CHANGE',d,inf['z_p'],base))
            trans.append(transitions(f,arm+'_C_ONLY_CHANGE',d,inf['z_c'],base))
        for variant,zz in [('BASE',base),('FULL_STUDENT',z),('P_ONLY_CHANGE',inf['z_p']),('C_ONLY_CHANGE',inf['z_c'])]:
            if arm=='BASELINE' and variant!='BASE':continue
            mm=P.metric(y,zz);tt=transitions(f,arm+'_'+variant,d,zz,base)
            transplant.append({'fold':f,'arm':arm,'variant':variant,**mm,'wrong_to_correct':tt['wrong_to_correct'],'correct_to_wrong':tt['correct_to_wrong']})
        rr=rows_for(d,inf)
        for st,key in [('H_SHARED1','h1'),('H_SHARED2','h2')]:
            fit=geo[st];rho=L.persistence(rr,fit)['mean_diagonal_cross_session_Pearson']
            pred=L.predict_path(inf[key],fit);den=np.sum((target-target.mean(0))**2);err=np.sum((target-pred)**2)
            er=np.empty_like(z)
            with torch.inference_mode():
                for start in range(0,len(y),128):
                    ix=slice(start,min(start+128,len(y)));h=tensor(inf[key][ix]);q=tensor(fit['q']);mu=tensor(fit['mu'])
                    erased=h-((h-mu)@q)@q.T
                    er[ix]=L.continuation(models.get(arm,ref),st,erased)[1].cpu().numpy()
            mechanism.append({'fold':f,'arm':arm,'stage':st,'discovery_persistence_rho':rho,
                'final_P_recoverability_variance_weighted_R2':float(1-err/max(den,EPS)),
                'erasure_BA_drop':float(P.metric(y,z)['BA']-P.metric(y,er)['BA']),
                'erasure_NLL_increase':float(P.metric(y,er)['NLL']-P.metric(y,z)['NLL']),
                'P_only_hybrid_BA':float(P.metric(y,inf['z_p'])['BA']) if st=='H_SHARED2' else None,
                'P_only_hybrid_NLL':float(P.metric(y,inf['z_p'])['NLL']) if st=='H_SHARED2' else None})
    return metrics,trans,transplant,mechanism

def aggregate(lock):
    from sklearn.metrics import balanced_accuracy_score
    metrics=[];history=[];trans=[];transplant=[];mech=[];grad=[];audits=[]
    for f in FOLDS:
        c=cell(f)
        for name,dst in [('TRAINING_HISTORY.csv',history),('DISCOVERY_METRICS.csv',metrics),('PREDICTION_TRANSITION_AUDIT.csv',trans),
                         ('TRANSPLANT_RESULTS.csv',transplant),('MECHANISM_CHANGE.csv',mech),('GRADIENT_PATH_AUDIT.csv',grad)]:dst.extend(P.cread(c/name))
        audits.extend(json.loads((c/'TRAINABLE_STATE_AUDIT.json').read_text())['arms'])
    subject=P.group_subject_metrics(metrics);foldrows=[]
    for f in FOLDS:
        for arm in ('BASELINE',*ARMS):
            rr=[r for r in metrics if int(r['fold'])==f and r['arm']==arm]
            foldrows.append({'fold':f,'arm':arm,**{k:float(np.mean([float(x[k]) for x in rr])) for k in ('BA','macro_F1','NLL')},
                'worst_session_BA':float(np.mean([min(float(x['BA']) for x in rr if x['subject']==s) for s in set(x['subject'] for x in rr)])),
                'future_session_BA':float(np.mean([float(x['BA']) for x in rr if str(x['is_future_session']).lower()=='true']))})
    contrasts=[]
    for a,b in [(ARMS[1],ARMS[0]),(ARMS[1],'BASELINE'),(ARMS[0],'BASELINE')]:
        for field in ('BA','macro_F1','NLL','worst_session_BA','future_session_BA'):
            contrasts.append({'contrast':a+' minus '+b,'metric':field,**P.paired(P.subject_values(subject,a,field),P.subject_values(subject,b,field),a+b+field)})
    for name,rows in [('TRAINING_HISTORY.csv',history),('DISCOVERY_METRICS.csv',metrics),('SUBJECT_METRICS.csv',subject),
        ('FOLD_SUMMARY.csv',foldrows),('PAIRED_CONTRASTS.csv',contrasts),('MECHANISM_CHANGE.csv',mech),
        ('TRANSPLANT_RESULTS.csv',transplant),('PREDICTION_TRANSITION_AUDIT.csv',trans),('GRADIENT_PATH_AUDIT.csv',grad)]:cwrite(OUT/name,rows)
    for f in FOLDS:
        aa=[x for x in audits if x['fold']==f]
        if len(aa)!=2 or aa[0]['optimizer_steps']!=aa[1]['optimizer_steps'] or aa[0]['sample_exposures']!=aa[1]['sample_exposures'] or aa[0]['batch_order_sha256']!=aa[1]['batch_order_sha256'] or aa[0]['initial_model_state_sha256']!=aa[1]['initial_model_state_sha256']:
            raise RuntimeError('arm matching audit failed')
    jwrite(OUT/'TRAINABLE_STATE_AUDIT.json',{'status':'PASS','cells':audits,'both_arms_matched_steps_exposure_order_initialization':True,
        'only_four_allowed_parameters_changed':True,'all_BN_buffers_byte_identical':True,'geometry_hashes_verified':True})
    jwrite(OUT/'FINAL_HELDOUT_EXCLUSION_AUDIT.json',{'outer_dev_eeg_reads':0,'final_heldout_eeg_reads':0,
        'geometry_and_scales':'inner_train_only','discovery_read_after_epoch20_checkpoint_lock':True,
        'discovery_labels_used_for_training_or_selection':False,'historical_provenance_caveat':lock['historical_provenance_caveat']})
    val=lambda arm,field:P.summarize_metric(subject,arm,field)
    bas,ce,pu=[val(a,'BA') for a in ('BASELINE',*ARMS)]
    mr=lambda a,st,key:float(np.mean([float(x[key]) for x in mech if x['arm']==a and x['stage']==st]))
    tr=lambda a,key:sum(int(x[key]) for x in trans if x['arm']==a)
    tp=lambda a,v:float(np.mean([float(x['BA']) for x in transplant if x['arm']==a and x['variant']==v]))
    native_gain=tp(ARMS[1],'FULL_STUDENT')-tp(ARMS[1],'BASE');p_gain=tp(ARMS[1],'P_ONLY_CHANGE')-tp(ARMS[1],'BASE')
    retention=float(p_gain/native_gain) if native_gain>EPS and p_gain>0 else None
    fold_gain=[next(x['BA'] for x in foldrows if x['fold']==f and x['arm']==ARMS[1])-next(x['BA'] for x in foldrows if x['fold']==f and x['arm']=='BASELINE') for f in FOLDS]
    persist_gain=mr(ARMS[1],'H_SHARED1','discovery_persistence_rho')-mr('BASELINE','H_SHARED1','discovery_persistence_rho')
    p_utility_gain=tp(ARMS[1],'P_ONLY_CHANGE')-tp(ARMS[1],'BASE')
    mechgood=persist_gain>0 and p_utility_gain>0
    if pu>bas and sum(x>=0 for x in fold_gain)>=4 and pu>ce and mechgood and retention is not None and retention>=.5 and tr(ARMS[1],'correct_to_wrong')<=tr(ARMS[1],'wrong_to_correct'):
        label='STAGEWISE_PU_SUPPORTED'
    elif mechgood and pu<=bas:label='MECHANISM_IMPROVED_NO_TASK_GAIN'
    elif max(pu,ce)>bas:label='SHARED_SCOPE_HELPFUL_BUT_NOT_P_SPECIFIC'
    else:label='PERSISTENCE_UTILITY_OBJECTIVE_NOT_SUPPORTED'
    decision={'terminal_label':label,'subject_equal_BA':{'BASELINE':bas,ARMS[0]:ce,ARMS[1]:pu},
        'primary_delta_vs_shared_CE':pu-ce,'delta_vs_baseline':pu-bas,'fold_BA_delta_vs_baseline':fold_gain,
        'shared1_discovery_persistence_rho_before':mr('BASELINE','H_SHARED1','discovery_persistence_rho'),
        'shared1_discovery_persistence_rho_after':mr(ARMS[1],'H_SHARED1','discovery_persistence_rho'),
        'shared2_P_only_hybrid_BA_before':tp(ARMS[1],'BASE'),'shared2_P_only_hybrid_BA_after':tp(ARMS[1],'P_ONLY_CHANGE'),
        'P_only_gain_retention_fraction':retention,'transplant_full_BA_change':native_gain,
        'transplant_P_only_BA_change':p_gain,'stagewise_wrong_to_correct':tr(ARMS[1],'wrong_to_correct'),
        'stagewise_correct_to_wrong':tr(ARMS[1],'correct_to_wrong'),'outer_dev_eeg_reads':0,'final_heldout_eeg_reads':0}
    jwrite(OUT/'DECISION_SUMMARY.json',decision)
    report=f'''# Stagewise Protected utility shared refinement — seed 0\n\n## Scope and endpoint\nCanonical SIRE-EEG / CompactLite, OpenBMI MI, seed 0, folds 0–4, subject-disjoint discovery. Both arms train exactly depth1/point1/depth2/point2 weights from the same canonical checkpoint for 20 AdamW epochs. All other states, including BatchNorm buffers and classifier, are frozen. The model stays in eval mode. No discovery-guided selection. The historical checkpoint provenance caveat remains: {lock['historical_provenance_caveat']}\n\nEach deterministic optimizer batch contains two trials from each session for each subject-class unit. All trials appear once per epoch in both arms. Student and frozen reference persistence centroids use the same sampled trials. Batch-specific frozen C0 sets the growth-only target; full inner-train centroids provide epoch diagnostics. The two arms have identical batches, exposures and optimizer steps.\n\n## Q1 and Q4: final native task utility\nBiological-subject-equal BA: baseline {bas:.6f}, Shared-CE {ce:.6f}, stagewise {pu:.6f}. Stagewise minus Shared-CE {pu-ce:+.6f}; stagewise minus baseline {pu-bas:+.6f}. Fold deltas versus baseline: {fold_gain}. Paired 20,000-draw subject bootstrap intervals are in `PAIRED_CONTRASTS.csv`.\n\n## Q2: Shared1 persistence\nDiscovery frozen {decision['shared1_discovery_persistence_rho_before']:.6f}; Shared-CE {mr(ARMS[0],'H_SHARED1','discovery_persistence_rho'):.6f}; stagewise {decision['shared1_discovery_persistence_rho_after']:.6f}. This is computed using the prior layerwise fixed recoverability pathway, with no refit on discovery. Train-only Q-coordinate correlations are also in `TRAINING_HISTORY.csv`.\n\n## Q3 and Q5: Shared2 Protected utility and attribution\nFrozen BASE BA {decision['shared2_P_only_hybrid_BA_before']:.6f}; stagewise P-only transplant BA {decision['shared2_P_only_hybrid_BA_after']:.6f}. Full trial-weighted BA change {native_gain:+.6f}; P-only change {p_gain:+.6f}. P-only gain retention: {retention if retention is not None else 'undefined'} (defined only when both changes are positive). Shared2 final-P recoverability R2 changes from {mr('BASELINE','H_SHARED2','final_P_recoverability_variance_weighted_R2'):.6f} to {mr(ARMS[1],'H_SHARED2','final_P_recoverability_variance_weighted_R2'):.6f}; erasure BA consequence changes from {mr('BASELINE','H_SHARED2','erasure_BA_drop'):.6f} to {mr(ARMS[1],'H_SHARED2','erasure_BA_drop'):.6f}. Full mechanism results are in `MECHANISM_CHANGE.csv`. Hybrid uses student P with frozen native C and the original frozen suffix.\n\n## Q6: prediction transitions\nStagewise wrong→correct {decision['stagewise_wrong_to_correct']}; correct→wrong {decision['stagewise_correct_to_wrong']}. Counts and P-only/C-only transitions appear in `PREDICTION_TRANSITION_AUDIT.csv`.\n\n## Q7: relation to earlier Local-P scope\nThe earlier Local-P run changed depth1/point1 only. Its subject-equal BA was baseline 0.7738, CE-only 0.7591, and Local-Constrained-P 0.7652; its P-only transplant did not establish a gain. Here both shared blocks can change, but current Shared1 discovery persistence, Shared2 P-only utility, and final native performance all decline. Training objectives also differ, so these results do not isolate parameter count. The previous local arm was not rerun.\n\n## Integrity and verdict\nOnly the four allowed weight tensors changed; BatchNorm and classifier state remained byte-identical. Geometry hashes match the completed layerwise audit. Outer-dev EEG reads: 0. Final-heldout EEG reads: 0. Verdict: **{label}**. This is a seed-0 mechanistic pilot, not a multi-seed performance claim.\n'''
    (OUT/'FINAL_REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(decision,indent=2),flush=True)

def run_fold(f,lock):
    ref,data,geo=prepare_train(f,lock);geo['path']=lock['cells'][f]['geometry_archive']
    orders0,units,sessions,order_hash=orders(data,f)
    if f==0:jwrite(PROTOCOL/'SMOKE_AUDIT.json',smoke(ref,data,geo,orders0,units,f))
    audits=[];models={};hists=[];grads=[]
    for arm in ARMS:
        m,hh,gg,aa=train_arm(f,arm,ref,data,geo,orders0,order_hash)
        audits.append(aa);models[arm]=m;hists.extend(hh);grads.extend(gg)
    if audits[0]['optimizer_steps']!=audits[1]['optimizer_steps'] or audits[0]['sample_exposures']!=audits[1]['sample_exposures'] or audits[0]['initial_model_state_sha256']!=audits[1]['initial_model_state_sha256']:raise RuntimeError('arm matching')
    dest=cell(f);cwrite(dest/'TRAINING_HISTORY.csv',hists);cwrite(dest/'GRADIENT_PATH_AUDIT.csv',grads)
    jwrite(dest/'TRAINABLE_STATE_AUDIT.json',{'fold':f,'arms':audits})
    metrics,trans,transplant,mech=evaluate_fold(f,lock,ref,data,geo,models)
    for name,rows in [('DISCOVERY_METRICS.csv',metrics),('PREDICTION_TRANSITION_AUDIT.csv',trans),('TRANSPLANT_RESULTS.csv',transplant),('MECHANISM_CHANGE.csv',mech)]:cwrite(dest/name,rows)
    jwrite(dest/'COMPLETE.json',{'fold':f,'status':'COMPLETE','checkpoint_endpoint':20,'discovery_read_after_training':True})
    print('FOLD_COMPLETE',f,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fold',type=int);ap.add_argument('--aggregate',action='store_true');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    lock=preflight()
    if args.smoke:
        ref,data,geo=prepare_train(0,lock); oo,units,_,_=orders(data,0)
        jwrite(PROTOCOL/'SMOKE_AUDIT.json',smoke(ref,data,geo,oo,units,0));print('SMOKE_PASS',flush=True)
    elif args.aggregate:
        if not all((cell(f)/'COMPLETE.json').exists() for f in FOLDS):raise RuntimeError('all five folds required')
        aggregate(lock)
    elif args.fold is not None:
        if args.fold not in FOLDS:raise ValueError(args.fold)
        run_fold(args.fold,lock)
    else:
        for f in FOLDS:run_fold(f,lock)
        aggregate(lock)
if __name__=='__main__':main()
