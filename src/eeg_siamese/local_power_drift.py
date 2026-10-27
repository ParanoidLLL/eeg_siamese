"""Controlled local spectral-amplitude intervention, not a physiological cause label."""
import math
import torch

REGIONS={
    'right_temporal':('FT8','T8','T10','TP8'),
    'left_temporal':('FT7','T7','T9','TP7'),
    'central':('FCz','Cz','CPz','Pz'),
    'frontal':('Fp1','Fp2','F3','F4'),
}
BANDS={'alpha':(8.,13.),'beta':(13.,30.),'high':(30.,40.)}
POWER_FACTORS=(.25,.5,2.,4.)


def spectral_gain(x,sample_rate,indices,band,power_factor,quality=None,patch_samples=None,taper_hz=1.):
    """Apply constant per-record gain to a band in selected original electrodes.

    Nominal power factor holds in the flat part of the raised-cosine band mask;
    actual integrated band power is measured separately. No DC normalization,
    phase randomization, added EEG, label input, or resurrection of lost cells.
    """
    if x.ndim not in (3,4) or not x.is_floating_point():raise ValueError('Expected [N,C,T] or [B,N,C,T]')
    low,high=band
    if not all(math.isfinite(v) for v in (sample_rate,low,high,power_factor,taper_hz)) or sample_rate<=0 or not 0<low<high<=sample_rate/2 or power_factor<=0 or not 0<taper_hz<=(high-low)/2:
        raise ValueError('Invalid spectral gain policy')
    indices=tuple(indices)
    if not indices or len(set(indices))!=len(indices) or min(indices)<0 or max(indices)>=x.shape[-2]:raise ValueError('Invalid electrode indices')
    safe=x
    hard=None
    if quality is not None:
        if patch_samples is None or quality.shape!=(*x.shape[:-1],x.shape[-1]//patch_samples) or x.shape[-1]%patch_samples:raise ValueError('Quality/patch mismatch')
        hard=(quality>0).repeat_interleave(patch_samples,-1);safe=torch.where(hard,x,0)
    if not torch.isfinite(safe).all():raise ValueError('Nonfinite usable EEG')
    if power_factor==1.:return safe.clone()
    freq=torch.fft.rfftfreq(x.shape[-1],1/sample_rate,device=x.device)
    distance=torch.minimum(freq-low,high-freq)
    ramp=(distance/taper_hz).clamp(0,1)
    mask=.5-.5*torch.cos(math.pi*ramp)
    gain=1+(math.sqrt(power_factor)-1)*mask
    part=torch.fft.rfft(safe[...,list(indices),:],dim=-1)
    changed=torch.fft.irfft(part*gain,n=x.shape[-1],dim=-1)
    result=safe.clone();result[...,list(indices),:]=changed
    return torch.where(hard,result,0) if hard is not None else result


def band_power(x,sample_rate,indices,band):
    """Per-record sum of squared FFT coefficients, same scale before/after."""
    freq=torch.fft.rfftfreq(x.shape[-1],1/sample_rate,device=x.device)
    selected=(freq>=band[0])&(freq<=band[1])
    spectrum=torch.fft.rfft(x[...,list(indices),:],dim=-1)
    values=spectrum[...,selected].abs().square()
    return values.flatten(-3).sum(-1)


def random_local_drift(x,quality,channel_names,sample_rate,patch_samples):
    """Train-only random region/band/gain, constant across all windows per record."""
    result=x.clone();policies=[]
    names=list(REGIONS);bands=list(BANDS)
    for item in range(len(x)):
        region=names[int(torch.randint(len(names),(),device=x.device))]
        band=bands[int(torch.randint(len(bands),(),device=x.device))]
        factor=POWER_FACTORS[int(torch.randint(len(POWER_FACTORS),(),device=x.device))]
        indices=[channel_names.index(name) for name in REGIONS[region]]
        result[item]=spectral_gain(x[item],sample_rate,indices,BANDS[band],factor,quality[item],patch_samples)
        policies.append(dict(region=region,band=band,power_factor=factor))
    return result,policies


def preserve_identity_margin(clean,changed,identities,tasks):
    """Changed queries against detached clean gallery; protect cross-task margins.

    Clean target cannot be weakened to avoid the loss. Contrastive identity losses
    on both views remain necessary to learn discrimination rather than collapse.
    """
    from torch.nn import functional as F
    clean=F.normalize(clean,dim=-1);changed=F.normalize(changed,dim=-1)
    same=identities[:,None]==identities[None,:]
    positive=same&(tasks[:,None]!=tasks[None,:]);negative=~same
    if not positive.any(1).all() or not negative.any(1).all():raise ValueError('Need cross-task positives and other identities')
    gallery=clean.detach();base=clean.detach()@gallery.T;shifted=changed@gallery.T
    def margin(scores):
        positive_mean=(scores*positive).sum(1)/positive.sum(1)
        hard_negative=scores.masked_fill(~negative,-torch.inf).max(1).values
        return positive_mean-hard_negative
    target=margin(base).clamp_min(0)
    return (target-margin(shifted)).clamp_min(0).mean()
