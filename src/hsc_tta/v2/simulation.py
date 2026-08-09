from __future__ import annotations

import math

import numpy as np
import pandas as pd


def simulate_joint(*, repetitions:int=5000,calibration_size:int=20,action_count:int=3,
                   scenario:str="one_beneficial",site_shift:float=0.0,heteroskedastic:bool=False,
                   selector:str="max_lower",seed:int=0)->dict[str,object]:
    rng=np.random.default_rng(seed); r=repetitions; m=calibration_size; a=action_count
    # Action zero is no_tta. U is observable; residuals remain episode-level and may be heteroskedastic in U.
    u=rng.normal(size=(r,m+1,a)); scale=1+.5*np.abs(u) if heteroskedastic else 1
    risk_noise=rng.normal(size=(r,m+1,a))*scale
    benefit_noise=rng.normal(size=(r,m+1,a))*(.02*scale)
    predicted_j=np.clip(8+1.5*u,0,20); true_j=np.clip(np.ceil(predicted_j+risk_noise),0,20)
    true_j[:,-1]=np.clip(true_j[:,-1]+site_shift,0,20)
    mean=np.full(a,-.03); mean[0]=0
    if scenario=="one_beneficial" and a>1: mean[1]=.04
    if scenario=="mixed": mean[1:]=np.linspace(-.03,.04,a-1)
    predicted_benefit=np.broadcast_to(mean,(r,m+1,a)).copy()
    if scenario!="misspecified": predicted_benefit += .015*u
    true_benefit=predicted_benefit+benefit_noise; predicted_benefit[:,:,0]=0; true_benefit[:,:,0]=0
    # Fixed scales emulate a separate meta-OOF sample and are not estimated on calibration episodes.
    c_j=max(float(np.quantile(np.abs(risk_noise[:,:m]),.9)),1.0)
    c_d=max(float(np.quantile(np.abs(benefit_noise[:,:m,1:]),.9)),.01)
    risk_score=(true_j[:,:m]-predicted_j[:,:m])/c_j
    benefit_score=(predicted_benefit[:,:m]-true_benefit[:,:m])/c_d; benefit_score[:,:,0]=-np.inf
    subject_score=np.maximum(risk_score,benefit_score).max(2)
    k=math.ceil((m+1)*.9); q=np.maximum(0,np.sort(subject_score,axis=1)[:,min(k,m)-1])
    upper=np.clip(np.ceil(predicted_j[:,-1]+q[:,None]*c_j),0,20)
    lower=predicted_benefit[:,-1]-q[:,None]*c_d
    eligible=(upper<20)&(lower>0); eligible[:,0]=False
    if selector=="adversarial_u_only":
        utility=u[:,-1].copy(); utility[~eligible]=-np.inf; chosen=utility.argmax(1)
    else:
        utility=lower.copy(); utility[~eligible]=-np.inf; chosen=utility.argmax(1)
    none=~eligible.any(1); chosen[none]=0
    idx=np.arange(r); safe=true_j[:,-1][idx,chosen]<=upper[idx,chosen]
    nonharm=(chosen==0)|(true_benefit[:,-1][idx,chosen]>=0)
    simultaneous=((true_j[:,-1]<=upper)&((np.arange(a)[None,:]==0)|(true_benefit[:,-1]>=lower))).all(1)
    return {"repetitions":r,"calibration_size":m,"action_count":a,"scenario":scenario,
        "site_shift":site_shift,"heteroskedastic":heteroskedastic,"selector":selector,
        "joint_simultaneous_validity":float(simultaneous.mean()),"selected_risk_validity":float(safe.mean()),
        "selected_nonharm_validity":float(nonharm.mean()),"tta_selection_rate":float((chosen!=0).mean()),
        "positive_selection_precision":float(true_benefit[:,-1][idx[chosen!=0],chosen[chosen!=0]].__ge__(0).mean()) if (chosen!=0).any() else np.nan,
        "mean_q":float(q.mean()),"full_set_fallback_rate":float((upper[idx,chosen]>=20).mean())}


def simulation_grid(repetitions:int=5000)->pd.DataFrame:
    configurations=[]
    for m in (10,15,20,30,50,100):
        for a in (2,3,5,10):
            configurations.append((m,a,"one_beneficial",0.0,False,"max_lower"))
    configurations += [(20,3,"all_harmful",0,False,"max_lower"),(20,3,"mixed",0,False,"max_lower"),
        (20,3,"misspecified",0,False,"max_lower"),(20,3,"one_beneficial",.5,False,"max_lower"),
        (20,3,"one_beneficial",0,True,"max_lower"),(20,5,"one_beneficial",0,True,"adversarial_u_only")]
    return pd.DataFrame([simulate_joint(repetitions=repetitions,calibration_size=m,action_count=a,scenario=s,
        site_shift=shift,heteroskedastic=hetero,selector=selector,seed=71000+i)
        for i,(m,a,s,shift,hetero,selector) in enumerate(configurations)])
