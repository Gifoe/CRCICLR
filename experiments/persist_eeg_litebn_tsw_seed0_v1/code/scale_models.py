"""Scale-only augmentations of the verified historical CompactLite forward."""
import ast,inspect,textwrap
import torch
from torch import nn
import torch.nn.functional as F

class ScaleMethods:
    def scale_branches(self,branches):
        if self.scale_kind=='StaticScale':
            alpha=F.softmax(self.beta,dim=0).expand(len(branches[0]),-1)
            scales=3*alpha
        else:
            descriptor=torch.cat([b.mean(dim=(2,3)) for b in branches],1)
            alpha=F.softmax(self.scale_mlp(descriptor),dim=1)
            scales=1+torch.tanh(self.lambda_scale)*(3*alpha-1)
        if self.collect_scale_diagnostics:
            self.scale_diagnostic_chunks.append((alpha.detach().float().cpu(),scales.detach().float().cpu()))
        return [b*scales[:,i,None,None,None] for i,b in enumerate(branches)]

def attach(base,kind,subseed=314159):
    assert kind in ('StaticScale','TSW')
    source=textwrap.dedent(inspect.getsource(type(base).forward));tree=ast.parse(source);changed=0
    class Edit(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal changed
            if ast.unparse(node.value)=='torch.cat(branches, 1)':
                changed+=1
                return [ast.parse('branches = self.scale_branches(branches)').body[0],node]
            return self.generic_visit(node)
    tree=Edit().visit(tree);ast.fix_missing_locations(tree);assert changed==1,'Historical concat insertion point not found'
    ns={'torch':torch,'F':F};exec(compile(tree,'exact_CompactLite_scale_only','exec'),ns)
    parent=type(base);base.__class__=type('CompactLite'+kind,(ScaleMethods,parent),{'forward':ns['forward']})
    base.scale_kind=kind;base.collect_scale_diagnostics=False;base.scale_diagnostic_chunks=[]
    device=next(base.parameters()).device
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(subseed)
        if kind=='StaticScale':base.beta=nn.Parameter(torch.zeros(3,device=device))
        else:
            base.scale_mlp=nn.Sequential(nn.Linear(48,24),nn.GELU(),nn.Linear(24,3)).to(device)
            base.lambda_scale=nn.Parameter(torch.zeros((),device=device))
    base.transformed_forward=ast.unparse(tree)
    return base
