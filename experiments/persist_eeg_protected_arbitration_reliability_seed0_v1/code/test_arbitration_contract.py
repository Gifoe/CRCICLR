"""Data-free protocol contract tests; no EEG or final-heldout data are loaded."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import ast
import hashlib
import json
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,f1_score

HERE=Path(__file__).parent
def definitions(path,names=None):
    return ast.Module(body=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)],type_ignores=[])
def natural(s):return sorted(set(np.asarray(s).astype(str)))
brscope=dict(globals(),Any=object,SUBJECT_FOLDS=5,EPS=1e-12,MAPPING_FEATURE_CHUNK=16384,RIDGE_ALPHAS=(0.,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.))
upstream=HERE.parents[1]/'persist_eeg_protected_emergence_routing_seed0_v1'/'code'/'run_emergence_routing.py'
exec(compile(definitions(upstream,{'standardised_tensor','kernel_predictions','r2_matrix','subject_groups'}),'upstream','exec'),brscope)
br=SimpleNamespace(**{k:brscope[k] for k in ('natural','subject_groups','standardised_tensor','kernel_predictions','r2_matrix','RIDGE_ALPHAS')})
ns=dict(globals(),Any=object,BR=br,EPS=1e-12,LAM=(.001,.01,.1,1.,10.),CS=(.001,.01,.1,1.,10.),DELTAS=(0.,.01,.02,.05,.1),ALPHA=np.array([0.,.25,.5,.75,1.],np.float32))
exec(compile(definitions(HERE/'run_arbitration.py'),'runner','exec'),ns)
rng=np.random.default_rng(12);n=100
x=rng.normal(size=(n,6)).astype(np.float32);xo=x[:12].copy();s=np.repeat(np.array([str(i) for i in range(10)]),10);y=rng.normal(size=(n,3)).astype(np.float32)
o,p,l,e=ns['nested_gain_predictions'](x,y,s,xo)
held=ns['groups'](s)[0];changed=y.copy();changed[held]+=10
o2,_,_,_=ns['nested_gain_predictions'](x,changed,s,xo)
np.testing.assert_array_equal(o[held],o2[held])
print('PASS nested held-subject gain predictions exclude held labels')
zp=np.array([[1.,0.],[1.,0.]],np.float32);zc=np.array([[0.,1.],[0.,1.]],np.float32);z0=np.array([[0.,3.]],np.float32)
f=ns['features'](z0,zp,zc,z0+zp+zc,np.ones((2,2)))
assert (f[:,1]==1).all() and (f[:,3]==1).all()
print('PASS P-only and C-only features include z0')
z=rng.normal(size=(n,3)).astype(np.float32);zp=rng.normal(size=(n,3)).astype(np.float32);z0=np.array([[.2,-.2,.4]],np.float32);zc=z-z0-zp;labels=rng.integers(0,3,n)
out,rows,meta=ns['reliability_arbitration'](x,x,zp,zc,zp,zc,z0,z0,labels,labels,s,('test','A'))
agree=(z0+zp).argmax(1)==(z0+zc).argmax(1)
np.testing.assert_allclose(out[agree],z[agree],atol=5e-7)
assert meta['selected_alpha'] in (.25,.5,.75)
print('PASS reliability routing preserves agreement and fixed suppression grid')
af=rng.normal(size=(30,20)).astype(np.float32);at=rng.normal(size=(7,20)).astype(np.float32);target=rng.normal(size=(30,3,4)).astype(np.float32)
joint=ns['shared_pathway_predictions'](af,target,at)
for j in range(3):np.testing.assert_array_equal(joint[:,:,j],br.kernel_predictions(af,target[:,j],at))
print('PASS shared GPU dual kernels match independently solved targets')
grid=np.array([(b,a) for b in (.5,.75,1.,1.25,1.5) for a in (.5,.75,1.,1.25,1.5)],np.float32)
orows,surface=ns['oracle'](z0,zp,zc,labels,s,grid)
assert surface['true_margin'].shape==(n,25)
assert sum(sum(row.get(k,0) for k in ('C_DOWN_ONLY','P_UP_ONLY','P_DOWN_ONLY','C_UP_ONLY','JOINT_REWEIGHT','NOT_REWEIGHT_RECOVERABLE')) for row in ns['taxonomy'](surface,labels,s,grid))==int((z.argmax(1)!=labels).sum())
print('PASS surface shapes and complete error taxonomy')
with tempfile.TemporaryDirectory(prefix='arbitration_contract_') as directory:
    ns['RUNTIME']=Path(directory)
    aq=rng.normal(size=(n,5)).astype(np.float32);oq=aq[:12]
    acts={'early':x,'late':np.c_[x,x[:,:2]]};outer_acts={key:value[:12] for key,value in acts.items()}
    raw=np.eye(5,dtype=np.float32);w=rng.normal(size=(3,5)).astype(np.float32)
    bank=ns['pathway_bank'](acts,outer_acts,aq,oq,s,[np.array([0,1]),np.array([2,3])],raw,w,('synthetic','MI',0))
    assert len(bank)==2 and bank[0][0].shape==(n,14) and bank[0][1].shape==(12,14)
    assert len(list(Path(directory).rglob('*.npz')))==2
    print('PASS pathway feature bank, nested mapping and runtime checkpoints')
assert np.isfinite(out).all() and np.isfinite(p).all()
print('ALL CONTRACT TESTS PASSED')
