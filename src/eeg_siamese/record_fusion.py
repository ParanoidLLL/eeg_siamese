"""Protected waveform expert and independently supervised log-enhanced expert.

The gate observes one record only, never a gallery, identity, task, or repeat ID.
Both experts start in the reference Q coordinate system; Q remains frozen/eval.
"""
from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F


class RecordFusion(nn.Module):
    def __init__(self,config):
        super().__init__()
        from .model import EEGEncoder
        self.mode=config.record_fusion
        self.raw=EEGEncoder(replace(config,record_fusion='none',spectral_fusion='none',
            spectral_feature_mean=(),spectral_feature_scale=()))
        self.log=EEGEncoder(replace(config,record_fusion='none'))
        self.raw.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            self.gate=nn.Sequential(nn.Linear(3*config.embedding_dim+2*config.channels*5+2,32),
                nn.GELU(),nn.Linear(32,1)) if self.mode=='adaptive' else None
        if self.gate is not None:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        self.raw.eval()

    def train(self,mode=True):
        super().train(mode)
        self.raw.eval()
        return self

    def context(self,x,valid,quality):
        b,n,c,t=x.shape
        available=valid & (quality.sum((2,3))>0)
        selected=x[available]
        q=quality[available]
        # Same missing-cell isolation as the encoder, including NaN padding.
        selected=torch.where((q>0).repeat_interleave(self.log.config.patch_samples,-1),selected,0)
        f,active=self.log.spectral_branch.standardized_features(selected,q)
        features=x.new_zeros(b,n,c*5)
        features[available]=f
        weights=quality.sum((2,3))*available
        # Completely unavailable spectral windows cannot contribute a biased zero.
        spectral_active=torch.zeros_like(available)
        spectral_active[available]=active
        sw=weights*spectral_active
        w=sw/sw.sum(1,keepdim=True).clamp_min(1e-12)
        mean=(features*w[...,None]).sum(1)
        var=((features-mean[:,None]).square()*w[...,None]).sum(1)
        qw=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
        qm=quality.mean((2,3))
        qmean=(qm*qw).sum(1,keepdim=True)
        qvar=((qm-qmean).square()*qw).sum(1,keepdim=True)
        return torch.cat((mean,(var+1e-8).sqrt(),qmean,(qvar+1e-8).sqrt()),-1)

    def forward(self,x,valid,quality):
        with torch.no_grad():
            raw=self.raw(x,valid,quality)
        log=self.log(x,valid,quality)
        if self.mode=='single':
            # Keep construction/RNG/reference audits identical to fixed fusion,
            # but the frozen expert has no effect on predictions or gradients.
            return log,dict(raw=raw,log=log,alpha=log.new_ones((len(log),1)))
        alpha=log.new_full((len(log),1),.5)
        if self.gate is not None:
            # Gate gradients do not reshape experts just to make routing easier.
            with torch.no_grad():context=self.context(x,valid,quality)
            inputs=torch.cat((raw,log.detach(),(raw-log.detach()).abs(),context),-1)
            alpha=self.gate(inputs).sigmoid()
        result=F.normalize((1-alpha)*raw+alpha*log,dim=-1)
        return result,dict(raw=raw,log=log,alpha=alpha)
