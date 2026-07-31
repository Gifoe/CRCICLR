from hsc_tta.schemas import ActionSurfaceRow


def test_action_surface_schema():
    row=ActionSurfaceRow.model_validate({"dataset":"hmc","seed":0,"subject_id":"s","split_role":"test","episode_id":"e","action":"no_tta","lambda":.8,"predicted_risk":.1,"within_subject_empirical_risk":.1,"within_subject_margin":.1,"within_subject_upper_risk":.2,"certified_upper_bound":.3,"future_risk":.1,"argmax_error":.2,"macro_f1":.8,"balanced_accuracy":.7,"average_set_size":2.,"singleton_rate":.4,"n_context":10,"n_future":20,"n_future_blocks":3,"status":"ok"})
    assert row.lambda_ == .8

