"""One-pass population BN moments, with dropout disabled and bitwise state audit."""
import hashlib
import torch
from torch import nn

def tensor_bytes(t): return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
def fingerprint(items):
    h=hashlib.sha256()
    for name,t in sorted(items):
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(tensor_bytes(t))
    return h.hexdigest()

@torch.no_grad()
def calibrate(model,batches):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    before_param=fingerprint(model.named_parameters())
    layers={k:m for k,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)}
    assert layers and all(m.track_running_stats for m in layers.values())
    allowed={f'{k}.{v}' for k in layers for v in ['running_mean','running_var','num_batches_tracked']}
    non_bn_before=fingerprint((k,v) for k,v in model.named_buffers() if k not in allowed)
    moments={};hooks=[];settings={};calls={k:0 for k in layers};trials=0;nb=0
    def hook(name):
        def collect(module,args):
            assert not model.training and not torch.is_grad_enabled()
            x=args[0].detach().double();axes=tuple(i for i in range(x.ndim) if i!=1)
            n=x.numel()//x.shape[1];s=x.sum(axes);q=x.square().sum(axes)
            if name not in moments:moments[name]=[n,s,q]
            else:moments[name][0]+=n;moments[name][1].add_(s);moments[name][2].add_(q)
            calls[name]+=1
        return collect
    try:
        for k,m in layers.items():
            settings[k]=m.momentum;m.momentum=0.;m.train();hooks.append(m.register_forward_pre_hook(hook(k)))
        for x in batches:
            assert all(not m.training for m in model.modules() if not isinstance(m,nn.modules.batchnorm._BatchNorm))
            assert torch.isfinite(x).all();model(x);trials+=len(x);nb+=1
        assert nb>0 and all(v==nb for v in calls.values())
        stats=[]
        for k,m in layers.items():
            n,s,q=moments[k];mu=s/n;var=(q/n-mu.square()).clamp_min(0)
            assert torch.isfinite(mu).all() and torch.isfinite(var).all()
            m.running_mean.copy_(mu);m.running_var.copy_(var);m.num_batches_tracked.fill_(nb)
            old_mu=before[k+'.running_mean'].double();old_var=before[k+'.running_var'].double()
            new_mu=m.running_mean.cpu().double();new_var=m.running_var.cpu().double()
            dm=(new_mu-old_mu).norm().item();dv=(new_var-old_var).norm().item()
            stats.append({'layer':k,'calibration_count':n,'batches':nb,'old_mean_norm':old_mu.norm().item(),'new_mean_norm':new_mu.norm().item(),'mean_shift':dm,'old_variance_norm':old_var.norm().item(),'new_variance_norm':new_var.norm().item(),'variance_shift':dv,'relative_mean_shift':dm/max(old_mu.norm().item(),1e-12),'relative_variance_shift':dv/max(old_var.norm().item(),1e-12)})
    finally:
        for h in hooks:h.remove()
        for k,m in layers.items():m.momentum=settings[k]
        model.eval()
    after_param=fingerprint(model.named_parameters());after_non=fingerprint((k,v) for k,v in model.named_buffers() if k not in allowed)
    diffs=[(k,(v.detach().cpu()-before[k]).abs().max().item()) for k,v in model.named_parameters()]
    bad=[k for k,v in model.state_dict().items() if k not in allowed and tensor_bytes(v)!=tensor_bytes(before[k])]
    assert before_param==after_param and non_bn_before==after_non and not bad,'IMPLEMENTATION_INVALID'
    assert all(p.grad is None for p in model.parameters())
    audit={'status':'PASS','parameters_bitwise_identical':True,'max_abs_parameter_diff':max(v for k,v in diffs),'non_BN_buffers_identical':True,'parameter_hash_before':before_param,'parameter_hash_after':after_param,'non_BN_buffer_hash_before':non_bn_before,'non_BN_buffer_hash_after':after_non,'calibration_trials':trials,'calibration_batches':nb,'dropout_disabled':True}
    return stats,audit,{k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k in allowed}

def tests():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__();self.bn=nn.BatchNorm2d(2);self.drop=nn.Dropout(.8);self.scale=nn.Parameter(torch.ones(2));self.register_buffer('fixed',torch.tensor([7.]))
        def forward(self,x):return self.drop(self.bn(x))*self.scale[None,:,None,None]
    torch.manual_seed(0);m=Toy();x=torch.arange(40,dtype=torch.float32).reshape(5,2,2,2)/7
    expected_mu=x.double().mean((0,2,3));expected_var=x.double().var((0,2,3),unbiased=False)
    stats,audit,_=calibrate(m,[x[:3],x[3:]])
    torch.testing.assert_close(m.bn.running_mean,expected_mu.float(),rtol=0,atol=0)
    torch.testing.assert_close(m.bn.running_var,expected_var.float(),rtol=0,atol=0)
    assert stats[0]['calibration_count']==20 and audit['max_abs_parameter_diff']==0
    assert torch.equal(m(x),m(x))
    # Means differ across unequal-size batches: naive averaged batch variance is wrong.
    naive=(x[:3].double().var((0,2,3),unbiased=False)+x[3:].double().var((0,2,3),unbiased=False))/2
    assert not torch.allclose(naive,expected_var)
    return {'unequal_batches_global_moments':True,'all_nonchannel_dimensions_counted':True,'parameter_and_nonBN_buffer_bitwise_audit':True,'dropout_off_repeat_exact':True,'naive_batch_variance_rejected':True}
