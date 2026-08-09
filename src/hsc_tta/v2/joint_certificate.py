from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class JointScales:
    c_j: float
    c_delta: float


def estimate_scales(meta_oof: pd.DataFrame) -> JointScales:
    if "residual_source" not in meta_oof or not (meta_oof.residual_source == "meta_oof").all():
        raise ValueError("joint scales must use meta-fit OOF residuals only")
    risk=np.abs(meta_oof.true_critical_index-meta_oof.predicted_critical_index)
    tta=meta_oof[meta_oof.action!="no_tta"]
    benefit=np.abs(tta.true_benefit-tta.predicted_benefit)
    return JointScales(max(float(risk.quantile(.9)),1.0),max(float(benefit.quantile(.9)),.01))


def subject_joint_scores(calibration: pd.DataFrame, scales: JointScales) -> pd.DataFrame:
    required={"subject_id","action","true_critical_index","predicted_critical_index","true_benefit","predicted_benefit"}
    if not required.issubset(calibration): raise ValueError(f"missing calibration fields: {required-set(calibration)}")
    frame=calibration.copy()
    frame["risk_score"]=(frame.true_critical_index-frame.predicted_critical_index)/scales.c_j
    frame["benefit_score"]=np.where(frame.action=="no_tta",-np.inf,
        (frame.predicted_benefit-frame.true_benefit)/scales.c_delta)
    frame["joint_row_score"]=frame[["risk_score","benefit_score"]].max(axis=1)
    return frame.groupby("subject_id",as_index=False).joint_row_score.max().rename(columns={"joint_row_score":"joint_score"})


def finite_sample_quantile(scores: np.ndarray|pd.Series, delta: float=.1) -> tuple[float,int]:
    values=np.sort(np.asarray(scores,float)); m=len(values)
    if m==0 or not 0<delta<1: raise ValueError("invalid calibration input")
    k=math.ceil((m+1)*(1-delta))
    raw=float(values[min(k,m)-1])
    return max(0.0,raw),k


def joint_bounds(predicted_j: np.ndarray, predicted_benefit: np.ndarray, q: float,
                 scales: JointScales, sentinel_index: int) -> tuple[np.ndarray,np.ndarray]:
    upper=np.clip(np.ceil(np.asarray(predicted_j,float)+q*scales.c_j),0,sentinel_index).astype(int)
    lower=np.asarray(predicted_benefit,float)-q*scales.c_delta
    return upper,lower
