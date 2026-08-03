import torch

from hsc_tta.models.token_heads import make_token_head


def test_token_heads_shapes_and_hidden():
    cases=(("old_mean_mlp",torch.randn(2,1,30,200),5),("temporal_attention_head",torch.randn(2,1,30,200),5),
           ("channel_temporal_head",torch.randn(2,64,4,200),4),("official_downstream_head",torch.randn(2,64,4,200),4))
    for name,x,k in cases:
        model=make_token_head(name,k); logits,hidden=model(x,return_hidden=True)
        assert logits.shape==(2,k); assert hidden.shape[0]==2


def test_source_model_payload_is_subject_isolated():
    payload={"fit_subjects":["a"],"val_subjects":["b"]}
    assert set(payload["fit_subjects"]).isdisjoint(payload["val_subjects"])
