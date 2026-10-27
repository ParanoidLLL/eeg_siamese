"""Record-conditioned group modulation of an existing log spectral residual.

One vector per record is shared across its windows. Hidden groups are not electrodes.
Missing spectral cells are excluded from each feature's context statistics.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SpectralModulation(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        if config.spectral_modulation=='global':
            self.logits=nn.Parameter(torch.zeros(config.spectral_modulation_groups))
        else:
            size=2*config.hidden_dim+2*config.channels*5+2
            self.scorer=nn.Sequential(nn.Linear(size,32),nn.GELU(),nn.Linear(32,config.spectral_modulation_groups))
            nn.init.zeros_(self.scorer[-1].weight);nn.init.zeros_(self.scorer[-1].bias)

    def context(self,h,residual,f,q,flat_valid,b,n):
        def padded(values):
            out=values.new_zeros(b*n,values.shape[-1]);out[flat_valid]=values
            return out.reshape(b,n,-1)
        def moments(values,weights):
            values=padded(values);weights=padded(weights)
            w=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
            mean=(values*w).sum(1)
            var=((values-mean[:,None]).square()*w).sum(1)
            return mean,(var+1e-8).sqrt()
        weights=q.sum((1,2))[:,None]
        feature_valid=(q>0).all(-1).repeat_interleave(5,-1)
        fm,fs=moments(f,weights*feature_valid)
        hm,_=moments(F.layer_norm(h,(h.shape[-1],)),weights)
        rm,_=moments(F.layer_norm(residual,(residual.shape[-1],)),weights*feature_valid.any(-1,keepdim=True))
        qm,qs=moments(q.mean((1,2))[:,None],weights)
        return torch.cat((hm,rm,fm,fs,qm,qs),-1)

    def forward(self,h,residual,f,q,flat_valid,b,n):
        if self.config.spectral_modulation=='global':
            logits=self.logits[None].expand(b,-1)
        else:
            with torch.no_grad():context=self.context(h.detach(),residual.detach(),f.detach(),q,flat_valid,b,n)
            logits=self.scorer(context)
        groups=1+self.config.spectral_modulation_bound*logits.tanh()
        expanded=groups.repeat_interleave(self.config.hidden_dim//self.config.spectral_modulation_groups,-1)
        owner=torch.arange(b,device=h.device).repeat_interleave(n)[flat_valid]
        return residual*expanded[owner],groups
