from dataclasses import replace
import hashlib,json
import numpy as np
import pytest
import torch
from eeg_siamese.model import EEGEncoder
from test_spectral_residual import cfg


def prepared(mode):
    torch.manual_seed(42);base=EEGEncoder(cfg('log'))
    with torch.no_grad():base.spectral_branch.mlp[-1].weight.normal_(0,.05)
    torch.manual_seed(43);net=EEGEncoder(replace(cfg('log'),spectral_modulation=mode))
    net.load_state_dict(base.state_dict(),strict=False)
    return base.eval(),net.eval()


@pytest.mark.parametrize('mode',['global','conditional'])
def test_initial_function_rng_masks_and_gradients(mode):
    torch.manual_seed(43);original=EEGEncoder(cfg('log'));state=torch.get_rng_state()
    torch.manual_seed(43);EEGEncoder(replace(cfg('log'),spectral_modulation=mode));assert torch.equal(state,torch.get_rng_state())
    base,net=prepared(mode)
    x=torch.randn(2,3,4,512);v=torch.tensor([[1,0,1],[1,1,1]],dtype=torch.bool);q=torch.ones(2,3,4,8);q[:,:,1,2]=0
    expected=base(x,v,q);z,d=net(x,v,q,return_modulation=True)
    torch.testing.assert_close(z,expected,rtol=0,atol=0)
    assert (d['groups']==1).all()
    z[:,0].sum().backward()
    parameter=net.spectral_modulator.logits if mode=='global' else net.spectral_modulator.scorer[-1].weight
    assert parameter.grad.abs().sum()>0
    with torch.no_grad():
        if mode=='global':parameter.fill_(.25)
        else:parameter.normal_(0,.05)
    expected,diag=net(x,v,q,return_modulation=True)
    assert ((diag['groups']>.5)&(diag['groups']<1.5)).all()
    if mode=='global':torch.testing.assert_close(diag['groups'][0],diag['groups'][1],rtol=0,atol=0)
    else:assert not torch.equal(diag['groups'][0],diag['groups'][1])
    x[~v]=float('nan');x[:,:,1,128:192]=float('nan')
    torch.testing.assert_close(net(x,v,q),expected,rtol=0,atol=0)
    torch.testing.assert_close(net(x[:,[2,0,1]],v[:,[2,0,1]],q[:,[2,0,1]]),expected,rtol=1e-5,atol=1e-6)
    x2=x.clone();x2[1]*=5
    _,d2=net(x2,v,q,return_modulation=True)
    torch.testing.assert_close(d2['groups'][0],diag['groups'][0],rtol=0,atol=0)


def test_context_is_detached_and_empty_spectral_support_stays_zero():
    _,net=prepared('conditional');mod=net.spectral_modulator
    with torch.no_grad():mod.scorer[-1].weight.normal_(0,.05)
    h=torch.randn(4,16,requires_grad=True);r=torch.randn(4,16,requires_grad=True);f=torch.randn(4,20,requires_grad=True)
    q=torch.ones(4,4,8);valid=torch.ones(4,dtype=torch.bool)
    out,g=mod(h,r,f,q,valid,2,2);out.sum().backward()
    assert h.grad is None and f.grad is None and r.grad is not None
    x=torch.randn(2,2,4,512);v=torch.ones(2,2,dtype=torch.bool);quality=torch.ones(2,2,4,8);quality[:,:,:,0]=0
    zero=[];hook=mod.register_forward_hook(lambda m,i,o:zero.append(o[0].detach()))
    net(x,v,quality);hook.remove();assert (zero[0]==0).all()
    with pytest.raises(ValueError):replace(cfg('log'),spectral_modulation='global',spectral_modulation_groups=3)


@pytest.mark.parametrize('mode',['none','global','conditional'])
def test_w_initialization_exact_resume_and_self_contained_checkpoint(tmp_path,mode):
    from eeg_siamese.data import Record
    from eeg_siamese.training import train
    from eeg_siamese.inference import load_encoder,encode_record
    records=[]
    for p in range(2):
        for task in range(2):
            path=tmp_path/f'p{p}t{task}.npz'
            np.savez(path,eeg=np.random.default_rng(p*2+task).normal(0,10,(4,1024)).astype('float32'),sample_rate=128.,channel_names=np.array(['a','b','c','d']),unit='uV',valid_channel_samples=np.ones((4,1024),bool))
            records.append(Record(f'p{p}',f't{task}',path.stem,path))
    config=dict(data=dict(sample_rate=128,channel_names=['a','b','c','d'],window_seconds=4,block_windows=2,quality=dict(patch_samples=64)),encoder=cfg('log').__dict__,training=dict(epochs=1,steps_per_epoch=2,identities_per_batch=2,samples_per_identity=2,ce_weight=0,consistency_weight=.5))
    path=tmp_path/'cfg.json';path.write_text(json.dumps(config));train(records,path,tmp_path/'w.pt','cpu')
    config['encoder']=replace(cfg('log'),spectral_modulation=mode).__dict__
    config['training'].update(modulation_reference=str(tmp_path/'w.pt'),modulation_reference_sha256=hashlib.sha256((tmp_path/'w.pt').read_bytes()).hexdigest())
    path.write_text(json.dumps(config));train(records,path,tmp_path/'model.pt','cpu')
    config['training']['epochs']=2;path.write_text(json.dumps(config))
    resumed=train(records,path,tmp_path/'model.pt','cpu',resume=tmp_path/'model.last.pt');full=train(records,path,tmp_path/'full.pt','cpu')
    assert resumed['history']==full['history']
    for k,v in full['encoder_state'].items():torch.testing.assert_close(v,resumed['encoder_state'][k],rtol=0,atol=0)
    if mode!='conditional':assert max(full['history'][-1]['modulation_within_batch_std'])<1e-7
    (tmp_path/'w.pt').unlink();net,dc,_,_=load_encoder(tmp_path/'model.pt','cpu');assert np.isfinite(encode_record(net,dc,records[0],'cpu')).all()
