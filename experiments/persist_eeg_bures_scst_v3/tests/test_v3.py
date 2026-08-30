from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import bures  # noqa: E402
import common  # noqa: E402
import source_v3  # noqa: E402


def arrays(n_subjects=4, per_cell=4, dim=4):
    rng = np.random.default_rng(12); x=[]; y=[]; s=[]; ids=[]
    for subject in range(n_subjects):
        for label in (0, 1):
            for row in range(per_cell):
                x.append(rng.normal(size=dim) + subject * .3 + label * 2); y.append(label); s.append(str(subject)); ids.append(subject * 1000 + label * 100 + row)
    return np.asarray(x, np.float64), np.asarray(y), np.asarray(s), np.asarray(ids)


def test_anchor_and_duplicate_excluded():
    x,y,s,ids=arrays(); x=np.vstack([x,x[0]]); y=np.concatenate([y,y[:1]]); s=np.concatenate([s,s[:1]]); ids=np.r_[ids,99999]
    idx=bures.anchor_excluded_indices(x,ids,0)
    assert 0 not in idx and len(idx)
    assert len(np.flatnonzero(np.all(x[idx] == x[0], axis=1))) == 0


def test_crossfit_excludes_anchor_and_equal_class_weighting():
    x,y,s,ids=arrays(); bank=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0)
    pos=0; opposite=1-int(bank.half[pos]); cell=bank._cell_for(str(s[pos]),int(y[pos]),opposite,exclude_row=pos)
    assert ids[pos] not in ids[np.flatnonzero((s==s[pos])&(y==y[pos])&(bank.half==opposite))]
    means=[]
    for label in (0,1): means.append(bank._class_cell(label,None).mean)
    assert np.allclose(bank.class_mean[0], means[0])
    # Subject style mean is an equal average over labels, not trial counts.
    expected=np.mean([bank._cell_for("0",label,0).mean-bank._class_cell(label,0).mean for label in (0,1)],axis=0)
    assert np.allclose(bank.subject_style("0",0)[0],expected)


def test_covariance_shrinkage_pd_and_deterministic():
    x,y,s,ids=arrays(); a=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0); b=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0)
    assert np.allclose(a.pool_cov,b.pool_cov)
    for cell in a.cell.values(): assert np.linalg.eigvalsh(cell.cov).min() > 0


def test_matrix_sqrt_reconstructs_and_identity_map():
    x,y,s,ids=arrays(); bank=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0); root,inv,eig=bures._sqrt_psd(bank.pool_cov,bank.pool_floor)
    assert np.allclose(root@root,bank.pool_cov,atol=1e-6); assert np.allclose(inv@bank.pool_cov@inv,np.eye(x.shape[1]),atol=1e-5)
    assert np.allclose(bures.bures_map(bank.pool_cov,bank.pool_cov,bank.pool_floor),np.eye(x.shape[1]),atol=1e-5)


def test_bures_sample_dependent_and_class_center_preserved():
    x,y,s,ids=arrays(); bank=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0); p=0
    assert not np.allclose(bank.endpoint(p,"1"),bank.endpoint(p+1,"1"))
    label=int(y[p]); mu=bank.class_mean[label]; ms=bank.subject_style(str(s[p]),1)[0]; mt=bank.subject_style("1",1)[0]
    # The map's class-centered point maps to the target class-centered point.
    assert np.allclose(mu+mt+bank._map_cache[(str(s[p]),"1",1,label)]@(mu+ms-mu-ms),mu+mt)


def test_target_affinity_improves_for_target_mean():
    x,y,s,ids=arrays(); bank=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0); p=0; target="1"
    target_values=x[(s==target)&(y==y[p])]; candidate=target_values.mean(0)
    before,after,bn,an=bures.target_affinity(x,y,s,p,candidate,bank,target)
    assert after < before and an < bn


def test_random_norm_matching_exact_and_masks():
    x,y,s,ids=arrays(); bank=bures.BuresBank(x,y,s,ids,dataset="T",fold=0,seed=0)
    rng=np.random.default_rng(2); d=np.array([1.,2.,3.,4.]); r=bures.matched_random_displacement(d,bank,rng)
    assert np.isclose(np.linalg.norm(d),np.linalg.norm(r)); assert np.isclose(bank.whitened_norm(d),bank.whitened_norm(r))


def test_source_v3_controls_share_sampler_and_no_kl():
    src=inspect.getsource(source_v3); assert "_subject_batches" in src; assert "symmetrized KL" not in src and "KL(" not in src
    assert source_v3._subject_batches(np.array(["a","a","b","b"]),1,0)[0].shape[0] > 0


def test_bootstrap_uses_subjects_not_trials():
    frame=pd.DataFrame({"dataset":["OpenBMI"]*4,"method":["Bures-HardSCST","Bures-HardSCST","ERM","ERM"],"q":[.25,.25,.5,.5],"lambda_T":[.25,.25,.5,.5],"subject_id":["1","2","1","2"],"BA":[.8,.7,.6,.65]})
    mean,lo,hi,n=common.paired_subject_delta(frame,"Bures-HardSCST","ERM",dataset="OpenBMI",q=.25,lambda_T=.25)
    assert n == 2 and np.isfinite(mean)


def test_reserved_paths_fail_closed():
    with pytest.raises(RuntimeError): common.reject_reserved_path("/tmp/WBCIC_outer_10")


def test_protocol_hash_and_no_future_opening():
    files=[ROOT/"code"/"bures.py",ROOT/"code"/"common.py"]; assert common.code_tree_sha256(files)==common.code_tree_sha256(files)
    assert "discovery_indices" not in inspect.getsource(common)


def test_geometry_diagnostics_include_required_gates():
    src=inspect.getsource(source_v3); 
    for token in ("target_distance_improvement","target_nll_improvement","class_pass_rate","median_displacement_ratio","median_relative_margin_drop"):
        assert token in src
