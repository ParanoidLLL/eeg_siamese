"""Per-window calibrated spectral residual; waveform identity path is preserved."""
import torch
from torch import nn


def window_band_power(x, sample_rate, window_seconds=4.):
    """Differentiable single-segment Welch, matching scipy's periodic Hann density."""
    if x.ndim != 3 or not torch.isfinite(x).all():
        raise ValueError('Expected finite [windows, channels, samples]')
    t=x.shape[-1]
    if window_seconds not in (4.,5.) or t!=round(window_seconds*sample_rate):
        raise ValueError('Spectral study requires complete protocol-matched windows')
    y=x-x.mean(-1,keepdim=True)
    hann=torch.hann_window(t,periodic=True,device=x.device,dtype=x.dtype)
    spectrum=torch.fft.rfft(y*hann,dim=-1).abs().square()/(sample_rate*hann.square().sum())
    factors=torch.full((spectrum.shape[-1],),2.,device=x.device,dtype=x.dtype)
    factors[0]=1
    if t%2==0:factors[-1]=1
    spectrum=spectrum*factors
    f=torch.fft.rfftfreq(t,1/sample_rate,device=x.device)
    return torch.stack([spectrum[...,(f>=a)&(f<b)].sum(-1)*(sample_rate/t)
                        for a,b in ((1,4),(4,8),(8,13),(13,30),(30,40.001))],-1)


class SpectralResidual(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.register_buffer('feature_mean',torch.tensor(config.spectral_feature_mean,dtype=torch.float32))
        self.register_buffer('feature_scale',torch.tensor(config.spectral_feature_scale,dtype=torch.float32))
        self.mlp=nn.Sequential(nn.Linear(config.channels*5,config.hidden_dim),nn.GELU(),
                               nn.Linear(config.hidden_dim,config.hidden_dim))
        nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)

    def standardized_features(self,x,q):
        active=(q>0).all(-1)
        # Partial missingness cannot create artificial spectral features.
        x=torch.where(active[...,None],x,0)
        p=window_band_power(x*self.config.spectral_input_scale_uv,self.config.sample_rate,self.config.spectral_window_seconds)
        f=p.clamp_min(1e-12).log() if self.config.spectral_fusion=='log' else p
        f=(f.flatten(1)-self.feature_mean)/self.feature_scale
        f=torch.where(active.repeat_interleave(5,-1),f,0)
        return f,active.any(-1)

    def forward(self,x,q,return_features=False):
        f,active=self.standardized_features(x,q)
        residual=self.mlp(f).masked_fill(~active[:,None],0)
        return (residual,f,active) if return_features else residual
