"""Electrode/time quality proxies and identity-independent training corruption.

Weights are documented physical-amplitude proxies, not fitted identity features
or neural SNR estimates. Known hard-invalid cells always have zero weight.
"""
import math
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class QualityConfig:
    patch_samples: int = 125
    rms_uv: float = 100.0
    derivative_uv: float = 20.0

    def __post_init__(self):
        if not isinstance(self.patch_samples,int) or isinstance(self.patch_samples,bool) or self.patch_samples<2:
            raise ValueError('Quality patch_samples must be an integer >=2')
        if not all(math.isfinite(v) and v>0 for v in (self.rms_uv,self.derivative_uv)):
            raise ValueError('Quality proxy thresholds must be positive physical uV values')


def quality_weights(x_uv, sample_valid, config):
    """[...,C,T] physical uV -> [...,C,P] in [0,1], hard-invalid exactly zero."""
    if x_uv.shape != sample_valid.shape or sample_valid.dtype != torch.bool or x_uv.shape[-1]%config.patch_samples:
        raise ValueError('Quality needs matching boolean sample masks and complete patches')
    samples=config.patch_samples
    safe=torch.where(sample_valid, x_uv, 0)
    if not torch.isfinite(safe).all():
        raise ValueError('Non-finite value in a trusted sample')
    patches=safe.unfold(-1,samples,samples)
    hard=sample_valid.unfold(-1,samples,samples).all(-1)
    centered=patches-patches.mean(-1,keepdim=True)
    rms=centered.square().mean(-1).sqrt()
    slope=patches.diff(dim=-1).square().mean(-1).sqrt()
    score=1/(1+(rms/config.rms_uv).square()+(slope/config.derivative_uv).square())
    return torch.where(hard,score,0).detach()


def corrupt_view(x, quality, policy, scale_uv, probability=.8, max_channel_fraction=.2, noise_uv=30.):
    """Random policies take no identity/session labels; return new signal and weights.

    At most C-1 channels are dropped. Short bursts affect one random channel;
    severities vary independently for each batch item. Known masks never recover.
    The input is already fixed-scaled; augmentation changes only the training view.
    """
    if x.ndim!=4 or quality.shape!=(*x.shape[:3],x.shape[-1]//policy.patch_samples):
        raise ValueError('Corruption needs x [B,N,C,T] and electrode-patch quality')
    y=x.clone();q=quality.clone();patch=policy.patch_samples
    for b in range(len(x)):
        if float(torch.rand((),device=x.device))>=probability:continue
        available=torch.nonzero(q[b].sum(dim=(0,2))>0).flatten()
        if not len(available):raise ValueError('No usable channel for corruption')
        max_drop=min(len(available)-1,math.ceil(x.shape[2]*max_channel_fraction))
        drop=int(torch.randint(max_drop+1,(),device=x.device)) if max_drop else 0
        if drop:
            channels=available[torch.randperm(len(available),device=x.device)[:drop]]
            q[b,:,channels,:]=0;y[b,:,channels,:]=0
        # A noisy electrode over the whole sampled block, distinct from dropout.
        remaining_channels=torch.nonzero(q[b].sum(dim=(0,2))>0).flatten()
        if len(remaining_channels) and float(torch.rand((),device=x.device))<.5:
            ch=int(remaining_channels[torch.randint(len(remaining_channels),(),device=x.device)])
            amplitude=noise_uv*float(.5+1.5*torch.rand((),device=x.device))/scale_uv
            electrode_noise=torch.randn(x.shape[1],1,x.shape[-1],device=x.device,dtype=x.dtype)*amplitude
            electrode_noise=torch.nn.functional.avg_pool1d(electrode_noise,5,stride=1,padding=2)[:,0]
            y[b,:,ch]+=electrode_noise
        # One short, band-limited-looking transient (smooth random noise), not a subject-specific pattern.
        n=int(torch.randint(x.shape[1],(),device=x.device))
        remaining=torch.nonzero(q[b,n].sum(-1)>0).flatten()
        if not len(remaining):continue
        ch=int(remaining[torch.randint(len(remaining),(),device=x.device)])
        patches=torch.nonzero(q[b,n,ch]>0).flatten()
        k=int(patches[torch.randint(len(patches),(),device=x.device)])
        start=k*patch;amplitude=noise_uv*float(.25+.75*torch.rand((),device=x.device))/scale_uv
        noise=torch.randn(patch,device=x.device,dtype=x.dtype)*amplitude
        noise=torch.nn.functional.avg_pool1d(noise[None,None],5,stride=1,padding=2)[0,0]
        y[b,n,ch,start:start+patch]+=noise
        # Sometimes the brief interruption is known loss of contact rather than a soft burst.
        if float(torch.rand((),device=x.device))<.25 and int((q[b]>0).sum())>1:
            q[b,n,ch,k]=0;y[b,n,ch,start:start+patch]=0
    sample_mask=(q>0).repeat_interleave(patch,dim=-1)
    measured=quality_weights(y*scale_uv,sample_mask,policy)
    q=torch.minimum(q,measured)
    y=torch.where((q>0).repeat_interleave(patch,dim=-1),y,0)
    if not (q.sum(dim=(1,2,3))>0).all():raise ValueError('Corruption removed all usable cells')
    return y,q
