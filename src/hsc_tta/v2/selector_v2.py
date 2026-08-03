from __future__ import annotations

import pandas as pd


REQUIRED={"action","available","certified_critical_index","benefit_lower","context_average_set_size","adaptation_cost"}


def select_joint_action(candidates: pd.DataFrame, *, sentinel_index: int) -> dict[str, object]:
    missing=REQUIRED-set(candidates)
    if missing: raise ValueError(f"missing selector fields: {sorted(missing)}")
    if candidates.action.duplicated().any(): raise ValueError("duplicate action candidates")
    no=candidates[candidates.action=="no_tta"]
    if len(no)!=1: raise ValueError("exactly one no_tta action is required")
    tta=candidates[(candidates.action!="no_tta") & candidates.available.astype(bool)
                   & (candidates.certified_critical_index<sentinel_index) & (candidates.benefit_lower>0)].copy()
    if len(tta):
        row=tta.sort_values(["benefit_lower","context_average_set_size","adaptation_cost","action"],
                            ascending=[False,True,True,True]).iloc[0]
        reason="joint_safe_positive_tta"
    else:
        row=no.iloc[0]; reason="no_positive_joint_certified_tta"
    full=bool(row.certified_critical_index>=sentinel_index)
    return {"selected_action":row.action,"certified_critical_index":int(row.certified_critical_index),
            "benefit_lower":float(row.benefit_lower),"full_set_fallback":full,
            "nontrivial_certified":not full,"selection_reason":reason}
