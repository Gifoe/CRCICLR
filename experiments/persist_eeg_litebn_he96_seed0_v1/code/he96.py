"""Function-preserving 48->96->64 widening of an already exact CompactLite."""
import torch
from torch import nn
import torch.nn.functional as F

class HE96Forward:
 def _he96_point2(self,x):
  """Keep the historical 64-input Conv2d kernel path exact at initialization."""
  kw={'bias':None,'stride':self.point2.stride,'padding':self.point2.padding,'dilation':self.point2.dilation,'groups':self.point2.groups}
  main=F.conv2d(x[:,:64],self.point2.weight[:,:64],**kw)
  extra=F.conv2d(x[:,64:],self.point2.weight[:,64:],**kw)
  return main+extra
 def _he96_drop(self,x):
  if not self.training:return x
  # The channel slice of an NCHW 96-channel tensor is strided.  Materializing
  # it restores the historical contiguous 64-channel dropout kernel path.
  main=F.dropout(x[:,:64].contiguous(),.15,True)
  with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))): extra=F.dropout(x[:,64:],.15,True)
  return torch.cat((main,extra),1)
 def forward(self,x):
  x=x.unsqueeze(1);branches=[]
  for t,tn,s,sn in zip(self.temporal,self.temporal_norm,self.spatial,self.spatial_norm):
   y=F.elu(tn(t(x)));y=F.elu(sn(s(y)));y=F.avg_pool2d(y,(1,4));branches.append(F.dropout(y,.20,self.training))
  x=torch.cat(branches,1);x=self._he96_drop(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(x)))),(1,2)))
  x=F.dropout(F.avg_pool2d(F.elu(self.norm2(self._he96_point2(self.depth2(x)))),(1,2)),.15,self.training)
  z=self.drop(self.embedding(self.pool(x).flatten(1)))
  return self.head(z),z

def attach(base,kind='HE96',subseed=960096):
    assert kind=='HE96'; old1,oldn1,oldd2,oldp2=base.point1,base.norm1,base.depth2,base.point2; device=old1.weight.device
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        torch.manual_seed(subseed)
        p1=nn.Conv2d(48,96,1,bias=False).to(device);n1=nn.BatchNorm2d(96).to(device);d2=nn.Conv2d(96,96,(1,31),groups=96,padding='same',bias=False).to(device);p2=nn.Conv2d(96,64,1,bias=False).to(device)
    with torch.no_grad():
        p1.weight[:64].copy_(old1.weight);n1.weight[:64].copy_(oldn1.weight);n1.bias[:64].copy_(oldn1.bias);n1.running_mean[:64].copy_(oldn1.running_mean);n1.running_var[:64].copy_(oldn1.running_var);n1.num_batches_tracked.copy_(oldn1.num_batches_tracked)
        d2.weight[:64].copy_(oldd2.weight);p2.weight[:,:64].copy_(oldp2.weight);p2.weight[:,64:].zero_()
    base.point1,base.norm1,base.depth2,base.point2=p1,n1,d2,p2
    base.__class__=type('CompactLiteHE96',(HE96Forward,type(base)),{})
    return base
