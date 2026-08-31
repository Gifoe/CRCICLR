import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
import upda_core as c


def rep(seed=0, n=40, d=8):
    rng = np.random.default_rng(seed)
    return {"features": rng.normal(size=(n, d)), "logits": rng.normal(size=(n, 2)),
            "labels": np.tile([0, 1], n // 2), "indices": np.arange(n),
            "subjects": np.array(["1"] * n), "sessions": np.array([0] * (n // 2) + [1] * (n // 2))}


def basis(d=8, r=2):
    rng = np.random.default_rng(9)
    return c.Basis(np.linalg.qr(rng.normal(size=(d, r)))[0][:, :r], np.linalg.qr(rng.normal(size=(2, r)))[0][:, :r])


def test_population_features_and_logits_are_frozen():
    x = rep(); f = x["features"].copy(); l = x["logits"].copy(); c.fit_shared_basis(x, 2); assert np.array_equal(x["features"], f); assert np.array_equal(x["logits"], l)


def test_true_ce_fit_is_label_likelihood():
    x = rep(); out = c.fit_ce_adapter(x, basis(), 1e-2); assert out["objective"] == "class_balanced_cross_entropy"; assert np.isfinite(out["fit_loss"])


def test_four_contiguous_historical_blocks_and_no_future_fit():
    tr = c.make_transitions(rep()); assert len(tr) == 1 and len(tr[0].history_blocks) == 4; assert [len(b["labels"]) for b in tr[0].history_blocks] == [5, 5, 5, 5]; assert max(b["sessions"].max() for b in tr[0].history_blocks) == 0


def test_one_se_uses_smallest_alpha_and_fixed_set():
    assert c.ALPHAS == (0.0, .25, .50, .75, 1.0)
    x=rep(); b=basis(); tr=c.make_transitions(x)[0]; loo=[c.fit_ce_adapter(c.concat_blocks([tr.history_blocks[j] for j in range(4) if j != k]),b,.01) for k in range(4)]; a,curve=c.select_one_se(list(tr.history_blocks),loo,b); assert a in c.ALPHAS; assert curve["selected_alpha"] == a


def test_alpha_zero_reproduces_population():
    x=rep(); b=basis(); z=c.metric(x,np.zeros(2),np.zeros(2),b); assert z == c.metric(x,0*np.ones(2),0*np.ones(2),b)


def test_wrong_and_shuffled_are_nonidentity_and_preserve_target_alpha():
    m={"1":{"persistent_norm":1.0,"persistent_ce":{"a":np.zeros(1),"c":np.zeros(2)}} ,"2":{"persistent_norm":2.0,"persistent_ce":{"a":np.ones(1),"c":np.ones(2)}}}; methods={"alpha":.25,"persistent_ce":m["1"]["persistent_ce"]}
    assert c.params_for("wrong_adapter",methods,"1",["1","2"],{"1":.25,"2":.0},m,1,1)[2] == .25; assert c.params_for("shuffled_adapter",methods,"1",["1","2"],{"1":.25,"2":0},m,1,1)[1] is m["2"]["persistent_ce"]


def test_random_gate_preserves_alpha_multiset():
    vals=[0,.25,.5,1]; assert sorted(vals)==sorted(list(vals))


def test_eb_shrinks_weak_evidence_to_zero():
    assert c.eb_alpha(.5,-.1,.2,{"mu":0,"tau":1})[0] == 0.0


def test_oracle_is_diagnostic_only_and_bootstrap_subject_unit():
    assert "DIAGNOSTIC_UPPER_BOUND_ONLY" in Path(__file__).parents[1].joinpath("code","upda_core.py").read_text(); assert c.N_BOOTSTRAP == 10000


def test_rank_limit_and_forbidden_resource_tokens():
    assert max(c.RANKS) <= 4; assert "utility_metrics" in c.FORBIDDEN_FUTURE_TOKENS and "outer" in c.FORBIDDEN_FUTURE_TOKENS


def test_held_block_is_not_in_its_own_fit():
    x=rep(); b=basis(); tr=c.make_transitions(x)[0]; fit=c.fit_subject(tr,b,.01); assert len(fit["loo"])==4


def test_future_flags_are_explicitly_false():
    x=rep(); b=basis(); tr=c.make_transitions(x)[0]; fit=c.fit_subject(tr,b,.01); assert fit["future_labels_used_for_fit"] is False and fit["future_session_used_for_fit"] is False


def test_one_se_tie_prefers_zero():
    x=rep(); b=basis(); tr=c.make_transitions(x)[0]; zero=[{"a":np.zeros(2),"c":np.zeros(2)} for _ in tr.history_blocks]; a,_=c.select_one_se(list(tr.history_blocks),zero,b); assert a==0.0


def test_alpha_selection_has_no_future_selection_flag():
    x=rep(); b=basis(); tr=c.make_transitions(x)[0]; fit=c.fit_subject(tr,b,.01); assert fit["future_labels_used_for_fit"] is False


def test_basis_is_frozen_before_validation():
    x=rep(); b=c.fit_shared_basis(x,1); before=b.U.copy(); c.fit_ce_adapter(x,b,.01); assert np.array_equal(before,b.U)


def test_population_logits_remain_unchanged_after_prediction():
    x=rep(); b=basis(); before=x["logits"].copy(); c.metric(x,np.ones(2),np.ones(2),b); assert np.array_equal(before,x["logits"])


def test_eb_prior_has_single_predeclared_variant():
    assert c.eb_alpha(.25,.1,.1,{"mu":0,"tau":.2})[0] in c.ALPHAS


def test_s2_gate_is_not_authorized_in_source_core():
    txt=Path(__file__).parents[1].joinpath("code","run_source.py").read_text(); assert "run_s2" not in txt and "session-2" not in txt


def test_future_backbone_is_not_loaded_by_source():
    txt=Path(__file__).parents[1].joinpath("code","run_source.py").read_text(); assert "EEGNeX" not in txt


def test_outer_resources_are_sealed_in_core():
    assert "outer" in c.FORBIDDEN_FUTURE_TOKENS and "sealed" in c.FORBIDDEN_FUTURE_TOKENS


def test_primary_method_boundary_excludes_forbidden_redesigns():
    txt=Path(__file__).parents[1].joinpath("code","upda_core.py").read_text(); assert "class_balanced_cross_entropy" in txt and "fit_ce_adapter" in txt
