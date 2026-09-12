"""Supplement the verified CompactLite forward; never reconstruct its backbone."""
import ast,inspect,textwrap
import torch
from torch import nn
import torch.nn.functional as F


class ReadoutMethods:
    def local_embed(self,m,shallow):
        h=torch.cat(shallow,1).squeeze(2)
        assert h.ndim==3 and h.shape[1]==self.shallow_channels
        # Stable second moments: no half-precision square/variance cancellation.
        with torch.autocast(device_type=h.device.type,enabled=False):
            mu=F.adaptive_avg_pool1d(h.float(),8)
            m2=F.adaptive_avg_pool1d(h.float().square(),8)
            lv=torch.log1p((m2-mu.square()).clamp_min(0))
            q=torch.cat([mu,lv],1).flatten(1)
        base=self.embedding[0](m)
        stats=self.local_U(self.local_V(q))
        if self.collect_diagnostics:
            with torch.no_grad():
                cut=self.shallow_channels*8
                cm=F.linear(F.linear(q[:,:cut],self.local_V.weight[:,:cut]),self.local_U.weight)
                cv=F.linear(F.linear(q[:,cut:],self.local_V.weight[:,cut:]),self.local_U.weight)
                bn=base.float().norm(dim=1);sn=stats.float().norm(dim=1)
                self.diagnostic_rows.append({'trials':len(q),'base_norm':bn.mean().item(),'stats_norm':sn.mean().item(),'stats_base_ratio':(sn/bn.clamp_min(1e-12)).mean().item(),'q_mean':q.mean().item(),'q_second_moment':q.square().mean().item(),'mu_contribution_norm':cm.float().norm(dim=1).mean().item(),'logvar_contribution_norm':cv.float().norm(dim=1).mean().item()})
        a=base+stats
        for layer in list(self.embedding.children())[1:]:a=layer(a)
        return a


def attach(base,subseed=741103):
    """Add independently seeded modules, preserving all RNG streams/shared tensors."""
    source=textwrap.dedent(inspect.getsource(type(base).forward));tree=ast.parse(source)
    captured=0;replaced=0
    class Edit(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal captured
            if ast.unparse(node.value)=='F.elu(sn(s(y)))':
                captured+=1
                return [node,ast.parse('shallow.append(y)').body[0]]
            return self.generic_visit(node)
        def visit_Call(self,node):
            nonlocal replaced
            if ast.unparse(node.func)=='self.embedding':
                replaced+=1
                return ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='local_embed',ctx=ast.Load()),args=[node.args[0],ast.Name(id='shallow',ctx=ast.Load())],keywords=[])
            return self.generic_visit(node)
    tree=Edit().visit(tree);tree.body[0].body.insert(0,ast.parse('shallow = []').body[0]);ast.fix_missing_locations(tree)
    assert captured==replaced==1,'Unsupported historical forward graph'
    ns={'torch':torch,'F':F};exec(compile(tree,'verified_CompactLite_plus_LocalStats','exec'),ns)
    parent=type(base);base.__class__=type('CompactLiteLocalStats',(ReadoutMethods,parent),{'forward':ns['forward']})
    base.shallow_channels=sum(layer.num_features for layer in base.spatial_norm)
    device=next(base.parameters()).device
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(subseed)
        base.local_V=nn.Linear(base.shallow_channels*8*2,8,bias=False).to(device)
        base.local_U=nn.Linear(8,64,bias=False).to(device)
        nn.init.zeros_(base.local_U.weight)
    base.collect_diagnostics=False;base.diagnostic_rows=[]
    base.transformed_forward=ast.unparse(tree)
    return base
