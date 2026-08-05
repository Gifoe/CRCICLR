from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hsc_tta.budgeted_risk.access import BudgetedAccessController
from hsc_tta.budgeted_risk.acquisition import acquisition_order,temporal_stratified_order
from hsc_tta.budgeted_risk.inclusion_index import critical_index_from_kappa,inclusion_indices,risk_curve_from_kappa
from hsc_tta.budgeted_risk.query_oracle import QueryOracle
from hsc_tta.budgeted_risk.run_state import REQUIRED_HASHES,RunState
from hsc_tta.budgeted_risk.budget import _local_index
from hsc_tta.budgeted_risk.stage0 import _fit,_predict,_select_model


def test_inclusion_index_and_higher_quantile_are_consistent():
    p=np.asarray([[.8,.2],[.55,.45],[.1,.9]])
    k=inclusion_indices(p,np.asarray([0,1,1]));j=critical_index_from_kappa(k,.10)
    assert 0<=j<=20
    assert risk_curve_from_kappa(k)[j] <= .10


def test_query_oracle_enforces_budget_repeat_and_freeze():
    oracle=QueryOracle("hmc","hmc:1",0,np.asarray([4,8]),np.asarray([1,2]),budget=1,strategy="first")
    assert oracle.query(4,kappa_by_label=np.asarray([3,5,7]))==1
    assert oracle.query(4)==1 and oracle.queried_count==1
    with pytest.raises(RuntimeError,match="exhausted"):oracle.query(8)
    frozen=oracle.freeze();assert oracle.verify(oracle.transcript) and isinstance(frozen,str)
    with pytest.raises(RuntimeError,match="QUERY_FROZEN"):oracle.query(4)


def test_temporal_order_is_nested_and_deterministic():
    order=temporal_stratified_order(11);assert sorted(order.tolist())==list(range(11));assert order[0]==5
    positions=(np.arange(11)+.5)/11
    for prefix in range(2,11):
        previous=order[:prefix-1];remaining=set(range(11))-set(previous)
        expected=min(remaining,key=lambda i:(-min(abs(positions[i]-positions[j]) for j in previous),i))
        assert order[prefix-1]==expected


def test_every_acquisition_is_a_permutation():
    p=np.asarray([[.7,.2,.1],[.4,.35,.25],[.2,.7,.1],[.34,.33,.33]])
    x=np.arange(20,dtype=float).reshape(4,5)
    for strategy in ("random","first","temporal","predictive_entropy","class_balanced","diversity","risk_entropy","active"):
        order=acquisition_order(strategy,p,x,dataset="hmc",seed=0,subject_id="hmc:1")
        assert sorted(order.tolist())==list(range(4))


def test_future_opens_only_after_hashed_decision(tmp_path):
    controller=BudgetedAccessController("hmc","hmc:1",0,"evaluation")
    with pytest.raises(RuntimeError):controller.open_future([1],tmp_path/"decision.json")
    controller.begin_queries();controller.freeze_queries("q")
    payload={"dataset":"hmc","subject_id":"hmc:1","seed":0,"role":"evaluation","budget":1,"strategy":"first","alpha":.1,"delta":.1,"query_hash":"q","source_model_hash":"m","episode_hash":"e","certified_index":2}
    path=tmp_path/"decision.json";controller.freeze_decision(payload,path)
    assert controller.open_future([7],path)==[7]
    data=json.loads(path.read_text());data["certified_index"]=1;path.write_text(json.dumps(data))
    controller.phase="RISK_DECISION_FROZEN"
    with pytest.raises(RuntimeError,match="hash"):controller.open_future([7],path)


def test_run_state_rejects_skips_and_allows_hard_stop(tmp_path):
    state=RunState(tmp_path/"RUN_STATE.json");meta={key:"x" for key in REQUIRED_HASHES}
    state.advance("INITIALIZED",**meta)
    with pytest.raises(RuntimeError,match="illegal"):state.advance("COHORTS_VERIFIED",**meta)
    state.advance("AUDIT_COMPLETE",**meta);state.advance("COHORTS_VERIFIED",**meta);state.advance("SOURCE_MODELS_VERIFIED",**meta)
    state.advance("STOPPED_NO_GO",**meta)
    assert state.read()["state"]=="STOPPED_NO_GO"


def test_hierarchical_cdf_local_index_moves_with_queries():
    prior=np.linspace(0,1,21)
    baseline=_local_index(prior,np.asarray([],int),tau=5,alpha=.1)
    improved=_local_index(prior,np.zeros(10,int),tau=5,alpha=.1)
    assert baseline==18
    assert improved<baseline


def test_ordinal_candidate_preserves_ordered_class_scale():
    x=np.arange(24,dtype=float).reshape(12,2);y=np.repeat([2,7,15],4);j=np.arange(12)
    model=_fit("ordinal_1",x,y,j);prediction=_predict(model,x,j)
    assert prediction.shape==(12,)
    assert np.all((prediction>=2)&(prediction<=15))


def test_model_selection_accepts_budget_local_index_column():
    frame=pd.DataFrame({"screening_fold":np.tile([0,1,2],4),"j_local":np.arange(12),"feature":np.arange(12),"j_future":np.arange(12)})
    selected,scores=_select_model(frame,["feature","j_local"],("direct","ridge_1"),index_column="j_local")
    assert selected in {"direct","ridge_1"}
    assert len(scores)==2
