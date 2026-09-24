"""Synthetic pre-heldout check against the canonical EEGNet forward."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

spec=importlib.util.spec_from_file_location("native_transfer_forward_check",Path(__file__).with_name("run.py"))
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.B.seed_all(17)
model=mod.B.MODELS.EEGNet(20,1000,2).to(mod.DEVICE).eval()
x=np.random.default_rng(17).normal(size=(5,20,1000)).astype(np.float32)
with torch.inference_mode():
    direct=model(torch.from_numpy(x).to(mod.DEVICE)).cpu().numpy()
rs,rd=2,3
def orthogonal(n,r):
    q,_=np.linalg.qr(np.random.default_rng(n).normal(size=(n,r)))
    return q.astype(np.float32)
basis={"qs":orthogonal(16*250,rs),"qd":orthogonal(16*31,rd),
       "rs":orthogonal(16*250,rs),"rd":orthogonal(16*31,rd),
       "ms":np.zeros(16*250,dtype=np.float32),"md":np.zeros(16*31,dtype=np.float32)}
feat=mod.features(model,x,basis,"PROTECTED_NATIVE_TRANSFER_GATE")
gate=mod.make_gate(basis,"PROTECTED_NATIVE_TRANSFER_GATE","OpenBMI_MI",0)
identity=mod.identity(model,gate,feat,basis,"PROTECTED_NATIVE_TRANSFER_GATE")
direct_error=float(np.max(np.abs(direct-feat["baseline"].cpu().numpy())))
assert direct_error<1e-6 and identity["status"]=="PASS"
with torch.no_grad():gate.w2.weight.fill_(0.1)
_,_,correction=mod.gate_logits(model,gate,feat,basis,"PROTECTED_NATIVE_TRANSFER_GATE")
delta=correction.detach().cpu().numpy()@basis["qd"].T
complement=delta-(delta@basis["qd"])@basis["qd"].T
complement_error=float(np.max(np.abs(complement)))
assert complement_error<1e-5
print(json.dumps({"status":"PASS","canonical_forward_max_abs_diff":direct_error,
                  "zero_init_logit_max_abs_diff":identity["logits_max_abs_diff"],
                  "successor_complement_leak_max_abs":complement_error}))
