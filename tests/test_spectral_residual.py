from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from scipy.signal import welch
from eeg_siamese.model import EEGEncoder,EncoderConfig
from eeg_siamese.spectral_residual import window_band_power


def cfg(mode='none'):
    return EncoderConfig(channels=4,hidden_dim=16,embedding_dim=8,attention_heads=2,
        patch_samples=64,dropout=0,sample_rate=128,spectral_fusion=mode,
        spectral_feature_mean=[0.]*20 if mode!='none' else (),
        spectral_feature_scale=[1.]*20 if mode!='none' else ())


def test_welch_parity_physical_gain_and_finite_gradients():
    x=torch.randn(2,4,512,dtype=torch.float64,requires_grad=True)
    f,p=welch(x.detach().numpy(),fs=128,nperseg=512,axis=-1)
    expected=np.stack([p[...,(f>=a)&(f<b)].sum(-1)*.25 for a,b in ((1,4),(4,8),(8,13),(13,30),(30,40.001))],-1)
    actual=window_band_power(x,128)
    np.testing.assert_allclose(actual.detach(),expected,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(window_band_power(2*x,128),4*actual)
    actual.sum().backward();assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('mode',['absolute','log'])
def test_baseline_parity_rng_invalid_signal_isolation_and_branch_gradient(mode):
    torch.manual_seed(43);base=EEGEncoder(cfg()).eval();state=torch.get_rng_state()
    torch.manual_seed(43);net=EEGEncoder(cfg(mode)).eval();assert torch.equal(state,torch.get_rng_state())
    x=torch.randn(2,3,4,512);v=torch.tensor([[1,0,1],[1,1,1]],dtype=torch.bool);q=torch.ones(2,3,4,8);q[:,:,1,2]=0
    torch.testing.assert_close(base(x,v,q),net(x,v,q),rtol=0,atol=0)
    net(x,v,q)[:,0].sum().backward();assert net.spectral_branch.mlp[-1].weight.grad.abs().sum()>0
    with torch.no_grad():net.spectral_branch.mlp[-1].weight.normal_(0,.1)
    expected=net(x,v,q);x[~v]=float('nan');x[:,:,1,128:192]=float('nan')
    torch.testing.assert_close(net(x,v,q),expected,rtol=0,atol=0)
    xx=torch.randn(2,4,512);qq=torch.ones(2,4,8);qq[:,:,0]=0
    assert (net.spectral_branch(xx,qq)==0).all()


def test_absolute_log_parameterization_and_invalid_constants():
    torch.manual_seed(1);a=EEGEncoder(cfg('absolute'));torch.manual_seed(1);b=EEGEncoder(cfg('log'))
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    for (k,p),(kk,q) in zip(a.named_parameters(),b.named_parameters()):
        assert k==kk;torch.testing.assert_close(p,q,rtol=0,atol=0)
    with pytest.raises(ValueError):replace(cfg('log'),spectral_feature_scale=[0.]*20)


def test_openbmi_five_second_welch_and_initial_waveform_parity():
    x=torch.randn(2,4,1250,dtype=torch.float64,requires_grad=True)
    freq,power=welch(x.detach().numpy(),fs=250,nperseg=1250,axis=-1)
    expected=np.stack([power[...,(freq>=a)&(freq<b)].sum(-1)*.2
        for a,b in ((1,4),(4,8),(8,13),(13,30),(30,40.001))],-1)
    actual=window_band_power(x,250,5.)
    np.testing.assert_allclose(actual.detach(),expected,rtol=1e-10,atol=1e-10)
    actual.sum().backward();assert torch.isfinite(x.grad).all()
    with pytest.raises(ValueError):window_band_power(x,250)  # Historical4s remains strict.
    c=replace(cfg(),sample_rate=250,patch_samples=125,spectral_window_seconds=5.)
    torch.manual_seed(42);base=EEGEncoder(c).eval();rng=torch.get_rng_state()
    torch.manual_seed(42);net=EEGEncoder(replace(c,spectral_fusion='log',
        spectral_feature_mean=[0.]*20,spectral_feature_scale=[1.]*20)).eval()
    assert torch.equal(rng,torch.get_rng_state())
    xx=x.detach().float().unsqueeze(0);v=torch.ones(1,2,dtype=torch.bool);q=torch.ones(1,2,4,10)
    torch.testing.assert_close(base(xx,v,q),net(xx,v,q),rtol=0,atol=0)
    with pytest.raises(ValueError):replace(net.config,spectral_window_seconds=3.)


def test_log_branch_checkpoint_inference_and_exact_resume(tmp_path):
    from eeg_siamese.data import Record
    from eeg_siamese.training import train
    from eeg_siamese.inference import load_encoder,encode_record
    records=[]
    for p in range(2):
        for task in range(2):
            path=tmp_path/f'p{p}t{task}.npz'
            np.savez(path,eeg=np.random.default_rng(p*2+task).normal(0,10,(4,1024)).astype('float32'),
                sample_rate=128.,channel_names=np.array(['a','b','c','d']),unit='uV',valid_channel_samples=np.ones((4,1024),bool))
            records.append(Record(f'p{p}',f't{task}',path.stem,path))
    config=dict(data=dict(sample_rate=128,channel_names=['a','b','c','d'],window_seconds=4,block_windows=2,quality=dict(patch_samples=64)),
        encoder=cfg('log').__dict__,training=dict(epochs=1,steps_per_epoch=2,identities_per_batch=2,samples_per_identity=2,ce_weight=0,consistency_weight=.5))
    path=tmp_path/'cfg.json';path.write_text(json.dumps(config));train(records,path,tmp_path/'model.pt','cpu')
    config['training']['epochs']=2;path.write_text(json.dumps(config))
    resumed=train(records,path,tmp_path/'model.pt','cpu',resume=tmp_path/'model.last.pt');full=train(records,path,tmp_path/'full.pt','cpu')
    assert resumed['history']==full['history']
    for k,v in full['encoder_state'].items():torch.testing.assert_close(v,resumed['encoder_state'][k],rtol=0,atol=0)
    net,dc,_,_=load_encoder(tmp_path/'model.pt','cpu');assert np.isfinite(encode_record(net,dc,records[0],'cpu')).all()
