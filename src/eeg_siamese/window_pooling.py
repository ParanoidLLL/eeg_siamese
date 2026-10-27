"""Bounded residual window attention and detached perturbation-stability targets.

Stability is a training proxy, not a calibrated probability of identity correctness.
The identity loss remains necessary: stable nuisance features can also agree.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def residual_weights(base, multiplier, mix):
    adaptive=base*multiplier
    adaptive=adaptive/adaptive.sum(-1,keepdim=True).clamp_min(1e-12)
    return (1-mix)*base+mix*adaptive


class ResidualWindowPooling(nn.Module):
    def __init__(self,hidden,scorer_hidden=32,mix=.5,bound=.5):
        super().__init__()
        if not isinstance(scorer_hidden,int) or isinstance(scorer_hidden,bool) or scorer_hidden<1:
            raise ValueError('Window scorer size must be positive')
        if not math.isfinite(mix) or not 0<mix<=.5 or not math.isfinite(bound) or not 0<bound<1:
            raise ValueError('Window attention needs a >=half baseline path and bounded positive multipliers')
        self.mix,self.bound=mix,bound
        self.scorer=nn.Sequential(nn.Linear(2*hidden+2,scorer_hidden),nn.GELU(),nn.Linear(scorer_hidden,1))
        nn.init.zeros_(self.scorer[-1].weight);nn.init.zeros_(self.scorer[-1].bias)

    def forward(self,features,raw_weights):
        if features.ndim!=3 or raw_weights.shape!=features.shape[:2]:raise ValueError('Expected windows[B,N,H] and weights[B,N]')
        if not torch.isfinite(raw_weights).all() or (raw_weights<0).any() or (raw_weights.sum(-1)<=0).any():raise ValueError('Invalid window support')
        features=torch.where((raw_weights>0).unsqueeze(-1),features,0)
        if not torch.isfinite(features).all():raise ValueError('Nonfinite usable window features')
        base=raw_weights/raw_weights.sum(-1,keepdim=True)
        # Descriptors do not create an auxiliary route for changing the backbone.
        # Backbone still learns directly through the weighted identity representation.
        detached=F.layer_norm(features.detach(),(features.shape[-1],))
        detached=torch.where((base>0).unsqueeze(-1),detached,0)
        summary=(detached*base.unsqueeze(-1)).sum(1,keepdim=True)
        other=(summary-base.unsqueeze(-1)*detached)/(1-base).clamp_min(1e-8).unsqueeze(-1)
        other=torch.where(((base>0)&(base<1-1e-7)).unsqueeze(-1),other,0)
        agreement=F.cosine_similarity(detached,other,dim=-1).unsqueeze(-1)
        descriptor=torch.cat([detached,other,agreement,base.unsqueeze(-1)],-1)
        multiplier=1+self.bound*self.scorer(descriptor).squeeze(-1).tanh()
        weights=residual_weights(base,multiplier,self.mix)
        pooled=(features*weights.unsqueeze(-1)).sum(1)
        return pooled,dict(base_weights=base,weights=weights,multipliers=multiplier,
            window_features=features,valid=base>0)


def reliability_constraint(clean,noisy,mix=.5,bound=.5,temperature=.1):
    """KL target on noisy weights; target uses detached clean/noisy feature drift.

    Clean/noisy training dropout also contributes to drift. This is not an EMA
    teacher and not a ground-truth acquisition-quality or identity-utility label.
    """
    if not math.isfinite(temperature) or temperature<=0:raise ValueError('Reliability temperature must be positive')
    with torch.no_grad():
        cosine=F.cosine_similarity(clean['window_features'].detach(),noisy['window_features'].detach(),dim=-1)
        support=clean['valid']&noisy['valid']
        stability=torch.exp(-(1-cosine.clamp(-1,1))/temperature)*support
        multiplier=1-bound+2*bound*stability
        target=residual_weights(noisy['base_weights'].detach(),multiplier,mix)
    actual=noisy['weights']
    divergence=(target*(target.clamp_min(1e-12).log()-actual.clamp_min(1e-12).log())).sum(-1).mean().clamp_min(0)
    return divergence,dict(target=target,stability=stability,support=support)
