import pandas as pd

from hsc_tta.v3.policy_search import evaluate_thresholds, grouped_oof_policy_search, select_thresholds


def fixtures():
    diagnostics=[];outcomes=[]
    for subject in ["s1","s2","s3","s4","s5","s6"]:
        for cost,action in enumerate(["official_t3a","robust_residual_adapter"],1):
            diagnostics.append({"dataset":"hmc","seed":0,"subject_id":subject,"action":action,"action_available":True,
                "r_class":1.0,"normalized_update_magnitude":.01,"g_set":.02 if cost==2 else .01,"g_aug":0.0,
                "positive_probe_block_fraction":1.0,"temporal_mad":.01,"d_src":.01*cost,"action_cost":cost})
            outcomes.append({"dataset":"hmc","seed":0,"subject_id":subject,"action":action,"source_safe_size":3.0,
                "safe_size":2.8 if cost==2 else 2.9,"oracle_gain":.2 if cost==2 else .1,"source_argmax_error":.2,
                "argmax_error":.2,"classification_degradation":0.0,"action_available":True})
    return pd.DataFrame(diagnostics),pd.DataFrame(outcomes)


def grid():
    return {"tau_set":[0.0],"tau_aug_margin":[0.0],"tau_positive_blocks":[2/3],"tau_time_mad_quantile":[.9],
            "tau_drift_quantile":[.9],"tau_class":[.7],"tau_update_quantile":[.9],
            "max_harmful_intervention_rate":.2,"minimum_intervention_rate":.01}


def test_vectorized_threshold_search_and_grouped_oof():
    diagnostics,outcomes=fixtures();thresholds,search=select_thresholds(diagnostics,outcomes,grid(),.01)
    decisions,metrics=evaluate_thresholds(diagnostics,outcomes,thresholds,.01)
    assert len(search)==1 and decisions.intervention.all() and metrics["mean_set_size_gain"]>0
    oof,searches=grouped_oof_policy_search(diagnostics,outcomes,grid(),.01,folds=3)
    assert set(oof.subject_id)==set(diagnostics.subject_id) and len(searches)==3
