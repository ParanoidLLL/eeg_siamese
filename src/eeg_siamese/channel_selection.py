"""Budgeted original-electrode selection with a hard forward path.

Gumbel perturbation chooses a subset without replacement. Sequential masked
softmax supplies a biased straight-through surrogate; it is not an exact
reimplementation of the EEG Concrete-selector paper. Selectors are global or
instruction-task conditioned, never waveform/identity conditioned.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ElectrodeSubsetSelector(nn.Module):
    def __init__(self, channels, budget, mode, tasks=0, random_seed=12345,
                 temperature_start=2., temperature_end=.25, anneal_epochs=30,
                 noise_start=1., noise_end=1., noise_anneal_epochs=30, fixed_scores=()):
        super().__init__()
        if mode not in ('learned_global','learned_task','random_fixed','fixed_scores'):
            raise ValueError('Unknown electrode selection mode')
        if not isinstance(budget,int) or isinstance(budget,bool) or not 1 <= budget <= channels:
            raise ValueError('Selection budget must be an integer in [1,channels]')
        if mode=='learned_task' and tasks<1:
            raise ValueError('Task selector requires a frozen task vocabulary')
        if not all(math.isfinite(v) and v>0 for v in (temperature_start,temperature_end)):
            raise ValueError('Selection temperatures must be finite and positive')
        if not isinstance(anneal_epochs,int) or isinstance(anneal_epochs,bool) or anneal_epochs<1:
            raise ValueError('anneal_epochs must be positive')
        self.channels,self.budget,self.mode=channels,budget,mode
        self.temperature_start,self.temperature_end=temperature_start,temperature_end
        self.anneal_epochs=anneal_epochs
        if not all(math.isfinite(v) and v>=0 for v in (noise_start,noise_end)):
            raise ValueError('Selection noise must be finite and nonnegative')
        if not isinstance(noise_anneal_epochs,int) or isinstance(noise_anneal_epochs,bool) or noise_anneal_epochs<1:
            raise ValueError('noise_anneal_epochs must be positive')
        self.noise_start,self.noise_end,self.noise_anneal_epochs=noise_start,noise_end,noise_anneal_epochs
        if mode=='fixed_scores':
            if len(fixed_scores)!=channels or not all(math.isfinite(v) for v in fixed_scores):
                raise ValueError('Fixed scores require one finite score per original electrode')
            self.register_buffer('scores',torch.tensor([fixed_scores],dtype=torch.float32))
        elif mode=='random_fixed':
            generator=torch.Generator().manual_seed(random_seed)
            self.register_buffer('scores',torch.rand(1,channels,generator=generator))
        else:
            self.scores=nn.Parameter(torch.zeros(tasks if mode=='learned_task' else 1,channels))
        self.register_buffer('temperature',torch.tensor(float(temperature_start)))
        self.register_buffer('noise_scale',torch.tensor(float(noise_start)),persistent=False)

    def set_epoch(self,epoch):
        fraction=min(max(epoch,0)/max(self.anneal_epochs-1,1),1.)
        self.temperature.fill_(self.temperature_start*(self.temperature_end/self.temperature_start)**fraction)
        noise_fraction=min(max(epoch,0)/max(self.noise_anneal_epochs-1,1),1.)
        self.noise_scale.fill_(self.noise_start+(self.noise_end-self.noise_start)*noise_fraction)

    def forward(self,x,quality,tasks=None):
        # x[M,C,T], q[M,C,P]. Quality defines availability, not relevance.
        if x.ndim!=3 or x.shape[1]!=self.channels or quality.ndim!=3 or quality.shape[:2]!=x.shape[:2]:
            raise ValueError('Selector requires x[M,C,T], quality[M,C,P]')
        eligible=quality.sum(-1)>0
        if (eligible.sum(-1)<self.budget).any():
            raise ValueError('Fewer usable electrodes than the fixed selection budget')
        if self.mode=='learned_task':
            if tasks is None or tasks.shape!=(len(x),) or tasks.dtype!=torch.long or tasks.device!=x.device:
                raise ValueError('Task selector requires task indices[M] on input device')
            if ((tasks<0)|(tasks>=len(self.scores))).any():
                raise ValueError('Unknown task selector index')
            logits=self.scores[tasks]
        else:
            logits=self.scores.expand(len(x),-1)
        perturbed=logits
        learning=self.mode in ('learned_global','learned_task')
        if self.training and learning and float(self.noise_scale)>0:
            u=torch.rand_like(logits).clamp_(1e-6,1-1e-6)
            noise=-(-u.log()).log()
            # Avoid a redundant multiplication for exact legacy behavior at scale1.
            perturbed=logits+noise if float(self.noise_scale)==1. else logits+self.noise_scale*noise
        ranked=perturbed.masked_fill(~eligible,-torch.inf)
        indices=torch.argsort(ranked,dim=-1,descending=True,stable=True)[:,:self.budget]
        remaining=eligible.clone()
        rows=[];entropies=[]
        for slot in range(self.budget):
            index=indices[:,slot]
            hard=F.one_hot(index,self.channels).to(x.dtype)
            if self.training and learning:
                soft=(perturbed/self.temperature).masked_fill(~remaining,-torch.inf).softmax(-1)
                # Forward values are exactly hard; only backward uses the surrogate.
                row=hard+(soft-soft.detach()).to(x.dtype)
                entropies.append(-(soft*soft.clamp_min(1e-12).log()).sum(-1))
            else:
                row=hard
            rows.append(row)
            remaining=remaining.scatter(1,index[:,None],False)
        selection=torch.stack(rows,1)
        # Canonical electrode order, so selector ranks are not another code path.
        order=indices.argsort(-1)
        indices=indices.gather(1,order)
        selection=selection.gather(1,order[:,:,None].expand(-1,-1,self.channels))
        selected=torch.bmm(selection,x)
        selected_quality=quality.gather(1,indices[:,:,None].expand(-1,-1,quality.shape[-1]))
        entropy=torch.stack(entropies,1).mean() if entropies else x.new_zeros(())
        return selected,selected_quality,dict(indices=indices,entropy=entropy,
            temperature=self.temperature.detach().clone(),noise_scale=self.noise_scale.detach().clone(),budget=self.budget)
