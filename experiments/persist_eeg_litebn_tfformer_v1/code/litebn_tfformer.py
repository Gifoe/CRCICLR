"""Frozen LiteBN temporal encoder plus one fixed multi-resolution TFFormer head."""
from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F

def rope(x):
    d=x.shape[-1]; p=torch.arange(x.shape[-2],device=x.device,dtype=x.dtype)[:,None]; inv=10000**(-torch.arange(d//2,device=x.device,dtype=x.dtype)/(d//2)); a=p*inv
    c,s=a.cos()[None,None],a.sin()[None,None]; y=torch.empty_like(x); y[...,0::2]=x[...,0::2]*c-x[...,1::2]*s; y[...,1::2]=x[...,0::2]*s+x[...,1::2]*c; return y
class DropPath(nn.Module):
 def __init__(self,p=.05): super().__init__(); self.p=p
 def forward(self,x):
  if not self.training or not self.p:return x
  keep=1-self.p; return x/keep*(torch.rand((x.shape[0],)+(1,)*(x.ndim-1),device=x.device)<keep)
class GQA(nn.Module):
 def __init__(self):
  super().__init__(); self.n1=nn.LayerNorm(64); self.q=nn.Linear(64,64); self.k=nn.Linear(64,32); self.v=nn.Linear(64,32); self.o=nn.Linear(64,64); self.n2=nn.LayerNorm(64); self.ff=nn.Sequential(nn.Linear(64,192),nn.GELU(),nn.Dropout(.1),nn.Linear(192,64)); self.ga=nn.Parameter(torch.full((64,),.01)); self.gf=nn.Parameter(torch.full((64,),.01)); self.dp=DropPath(); self.ent=float('nan')
 def forward(self,x):
  b,l,_=x.shape; h=self.n1(x); q=rope(self.q(h).view(b,l,8,8).transpose(1,2)); k=rope(self.k(h).view(b,l,4,8).transpose(1,2)).repeat_interleave(2,1); v=self.v(h).view(b,l,4,8).transpose(1,2).repeat_interleave(2,1); a=(q@k.transpose(-1,-2)/math.sqrt(8)).softmax(-1); self.ent=float((-(a.clamp_min(1e-12)*a.clamp_min(1e-12).log()).sum(-1)).mean().detach()); z=x+self.ga*self.dp(self.o((F.dropout(a,.1,self.training)@v).transpose(1,2).reshape(b,l,64))); return z+self.gf*self.dp(self.ff(self.n2(z)))
class ConvMixer(nn.Module):
 def __init__(self): super().__init__(); self.n=nn.LayerNorm(64); self.fc=nn.Linear(64,128); self.dw=nn.Conv1d(64,64,15,padding=7,groups=64); self.pw=nn.Conv1d(64,64,1); self.g=nn.Parameter(torch.full((64,),.01)); self.dp=DropPath(); self.active=0.
 def forward(self,x):
  z=F.glu(self.fc(self.n(x)),-1).transpose(1,2); z=self.pw(F.gelu(self.dw(z))).transpose(1,2); self.active=float(z.detach().norm()); return x+self.g*self.dp(F.dropout(z,.1,self.training))
class Spectral(nn.Module):
 def __init__(self,c,fs):
  super().__init__(); self.fs=fs; self.proj=nn.ModuleList([nn.Linear(c,16,bias=False) for _ in range(3)]); self.cnn=nn.ModuleList([nn.Sequential(nn.Conv2d(16,32,(5,3),padding=(2,1)),nn.GroupNorm(8,32),nn.GELU(),nn.Conv2d(32,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.GroupNorm(8,64),nn.GELU()) for _ in range(3)]); self.scale=nn.Parameter(torch.empty(3,64)); self.freq=nn.Parameter(torch.empty(24,64)); self.time=nn.Parameter(torch.empty(4,64)); nn.init.trunc_normal_(self.scale,std=.02);nn.init.trunc_normal_(self.freq,std=.02);nn.init.trunc_normal_(self.time,std=.02)
 def forward(self,x):
  out=[]
  for i,sec in enumerate((.25,.5,1.0)):
   w=round(self.fs*sec); n=1<<(w-1).bit_length(); z=torch.stft(x.float().reshape(-1,x.shape[-1]),n_fft=n,hop_length=max(1,round(w/4)),win_length=w,window=torch.hann_window(w,device=x.device,dtype=torch.float32),center=True,return_complex=True).abs().log1p(); z=z.reshape(x.shape[0],x.shape[1],z.shape[-2],z.shape[-1])
   f=torch.fft.rfftfreq(n,1/self.fs).to(x.device); z=z[:,:, (f>=1)&(f<=45),:]; z=self.proj[i](z.permute(0,2,3,1)).permute(0,3,1,2); z=F.adaptive_avg_pool2d(self.cnn[i](z),(24,4)).permute(0,2,3,1); out.append((z+self.scale[i]+self.freq[:,None]+self.time[None,:]).reshape(x.shape[0],96,64))
  return torch.cat(out,1)
class Cross(nn.Module):
 def __init__(self): super().__init__(); self.nt=nn.LayerNorm(64);self.ns=nn.LayerNorm(64);self.q=nn.Linear(64,64);self.k=nn.Linear(64,64);self.v=nn.Linear(64,64);self.o=nn.Linear(64,64);self.g=nn.Parameter(torch.full((64,),.01));self.ent=float('nan');self.mass=None
 def forward(self,t,s):
  b,l,_=t.shape;q=self.q(self.nt(t)).view(b,l,4,16).transpose(1,2);k=self.k(self.ns(s)).view(b,288,4,16).transpose(1,2);v=self.v(self.ns(s)).view(b,288,4,16).transpose(1,2);a=(q@k.transpose(-1,-2)/4).softmax(-1);self.ent=float((-(a.clamp_min(1e-12)*a.clamp_min(1e-12).log()).sum(-1)).mean().detach());self.mass=a.detach().mean((0,1,2));return t+self.g*self.o((F.dropout(a,.1,self.training)@v).transpose(1,2).reshape(b,l,64))
class LiteBNTFFormer(nn.Module):
 def __init__(self,base,fs=250):
  super().__init__();self.base=base;self.spectral=Spectral(base.temporal[0].in_channels*0+base.spatial[0].in_channels*0+base.temporal[0].in_channels+61,fs) if False else Spectral(base.spatial[0].kernel_size[0],fs);self.cross=Cross();self.g1=GQA();self.conv=ConvMixer();self.g2=GQA()
  for m in self.base.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):
    for p in m.parameters(recurse=False):p.requires_grad_(False)
 def train(self,mode=True):
  super().train(mode)
  for m in self.base.modules():
   if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
  return self
 def forward(self,x):
  raw=x;z=x.unsqueeze(1);bs=[]
  for a,b,c,d in zip(self.base.temporal,self.base.temporal_norm,self.base.spatial,self.base.spatial_norm):
   q=F.avg_pool2d(F.elu(d(c(F.elu(b(a(z)))))),(1,4));bs.append(F.dropout(q,.2,self.base.training))
  z=torch.cat(bs,1);z=F.dropout(F.avg_pool2d(F.elu(self.base.norm1(self.base.point1(self.base.depth1(z)))),(1,2)),.15,self.base.training);z=F.dropout(F.avg_pool2d(F.elu(self.base.norm2(self.base.point2(self.base.depth2(z)))),(1,2)),.15,self.base.training);t=z.squeeze(2).transpose(1,2);t=self.g2(self.conv(self.g1(self.cross(t,self.spectral(raw)))));r=self.base.drop(self.base.embedding(self.base.pool(t.transpose(1,2).unsqueeze(2)).flatten(1)));return self.base.head(r),r
