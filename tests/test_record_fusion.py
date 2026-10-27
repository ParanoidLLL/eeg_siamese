from dataclasses import replace
import hashlib,json
import numpy as np
import torch
from eeg_siamese.model import EEGEncoder
from test_spectral_residual import cfg


def model(mode):
    net=EEGEncoder(replace(cfg('log'),record_fusion=mode))
    raw=net.record_fuser.raw
    net.record_fuser.log.load_state_dict(raw.state_dict(),strict=False)
    return net


def test_protected_reference_mask_permutation_and_adaptive_gradients():
    net=model('adaptive').train()
    assert not net.record_fuser.raw.training
    assert all(not p.requires_grad for p in net.record_fuser.raw.parameters())
    x=torch.randn(2,3,4,512);v=torch.tensor([[1,0,1],[1,1,1]],dtype=torch.bool)
    q=torch.ones(2,3,4,8);q[:,:,1,2]=0
    net.eval();z,d=net(x,v,q,return_branches=True)
    torch.testing.assert_close(d['alpha'],torch.full((2,1),.5))
    torch.testing.assert_close(z,d['raw'],atol=1e-6,rtol=1e-6)
    x[~v]=float('nan');x[:,:,1,128:192]=float('nan')
    torch.testing.assert_close(net(x,v,q),z,rtol=0,atol=0)
    torch.testing.assert_close(net(x[:,[2,0,1]],v[:,[2,0,1]],q[:,[2,0,1]]),z)
    with torch.no_grad():net.record_fuser.log.spectral_branch.mlp[-1].weight.normal_(0,.1)
    z,d=net(x,v,q,return_branches=True)
    (z[:,0].sum()+d['log'][:,1].sum()).backward()
    assert net.record_fuser.gate[-1].weight.grad.abs().sum()>0
    assert net.record_fuser.log.spectral_branch.mlp[-1].weight.grad.abs().sum()>0
    assert all(p.grad is None for p in net.record_fuser.raw.parameters())


def test_fusion_checkpoint_resume_preserves_q_and_is_self_contained(tmp_path):
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
        encoder=cfg().__dict__,training=dict(epochs=1,steps_per_epoch=2,identities_per_batch=2,samples_per_identity=2,ce_weight=0,consistency_weight=.5))
    path=tmp_path/'cfg.json';path.write_text(json.dumps(config));reference=train(records,path,tmp_path/'q.pt','cpu')
    config['encoder']=replace(cfg('log'),record_fusion='adaptive').__dict__
    config['training'].update(fusion_reference=str(tmp_path/'q.pt'),fusion_reference_sha256=hashlib.sha256((tmp_path/'q.pt').read_bytes()).hexdigest())
    path.write_text(json.dumps(config));train(records,path,tmp_path/'model.pt','cpu')
    config['training']['epochs']=2;path.write_text(json.dumps(config))
    resumed=train(records,path,tmp_path/'model.pt','cpu',resume=tmp_path/'model.last.pt')
    full=train(records,path,tmp_path/'full.pt','cpu')
    assert resumed['history']==full['history']
    for k,v in full['encoder_state'].items():torch.testing.assert_close(v,resumed['encoder_state'][k],rtol=0,atol=0)
    for k,v in reference['encoder_state'].items():torch.testing.assert_close(v,full['encoder_state']['record_fuser.raw.'+k],rtol=0,atol=0)
    (tmp_path/'q.pt').unlink()
    net,dc,_,_=load_encoder(tmp_path/'model.pt','cpu')
    assert np.isfinite(encode_record(net,dc,records[0],'cpu')).all()
