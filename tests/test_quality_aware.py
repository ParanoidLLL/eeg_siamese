"""Quality masks, perturbation exclusion and full training/resume integration."""
import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from eeg_siamese.data import DataConfig,Record,load_windows
from eeg_siamese.model import EEGEncoder,EncoderConfig,quality_attention,make_transformer
from eeg_siamese.preprocessing import PreprocessingConfig,preprocess,save_processed
from eeg_siamese.quality import QualityConfig,quality_weights,corrupt_view
from eeg_siamese.inference import load_encoder,encode_record
from eeg_siamese.training import train


def test_channel_failure_does_not_discard_other_channels_and_guards_propagate(tmp_path):
    fs=100;t=np.arange(4000)/fs
    x=np.array([20*np.sin(2*np.pi*9*t),15*np.sin(2*np.pi*12*t),np.zeros(len(t))])
    x[0,1800:1805]=np.nan
    cfg=PreprocessingConfig(target_sample_rate=100,high_cut_hz=35,mask_scope='channel_sample')
    out=preprocess(x,fs,['A','B','DEAD'],cfg,invalid_intervals=[(2500,2510,'gap')])
    hard=out['valid_channel_samples']
    assert hard.dtype==np.bool_ and hard.shape==x.shape
    assert not hard[2].any()
    assert not hard[0,1800:1805].any() and hard[1,1800:1805].all()
    guard=out['qc']['guard_input_samples']
    assert not hard[:,:guard].any() and not hard[:,-guard:].any()
    assert not hard[:,2500:2510].any()
    assert np.isfinite(out['eeg']).all() and (out['eeg'][~hard]==0).all()
    path=tmp_path/'x.npz';save_processed(out,path)
    c=DataConfig(100,('A','B','DEAD'),preprocessing_id=cfg.preprocessing_id,quality=dict(patch_samples=50))
    windows,valid,q=load_windows(Record('id','s1','x',path),c)
    assert valid.any() and q[:,2].count_nonzero()==0
    assert q.shape==(8,3,10)
    with pytest.raises(ValueError,match='quality mode requires'):
        legacy=tmp_path/'legacy.npz'
        np.savez(legacy,eeg=out['eeg'],sample_rate=100,channel_names=['A','B','DEAD'],unit='uV',preprocessing_id=cfg.preprocessing_id)
        load_windows(Record('id','s1','legacy',legacy),c)


def test_saturation_is_hard_masked_only_with_documented_limits():
    fs=100;t=np.arange(4000)/fs
    x=np.array([10*np.sin(2*np.pi*9*t),15*np.sin(2*np.pi*12*t)])
    x[0,1800:1820]=200
    c=PreprocessingConfig(target_sample_rate=100,high_cut_hz=35,mask_scope='channel_sample',adc_min_uv=-200,adc_max_uv=200)
    r=preprocess(x,fs,['A','B'],c)
    assert not r['valid_channel_samples'][0,1800:1820].any()
    assert r['valid_channel_samples'][1,1800:1820].all()
    assert any(v['reason']=='documented_adc_limit' for v in r['qc']['raw_invalid_intervals'])


def small_model():
    return EEGEncoder(EncoderConfig(3,hidden_dim=16,embedding_dim=8,transformer_layers=1,
        attention_heads=2,dropout=0,attention_scope='axial_patches',patch_samples=8,quality_aware=True, token_stride=2))


def test_hard_invalid_cells_cannot_change_output_or_receive_gradients():
    torch.manual_seed(42);m=small_model().eval()
    x=torch.randn(2,2,3,32,requires_grad=True);mask=torch.ones(2,2,dtype=torch.bool)
    q=torch.ones(2,2,3,4);q[:,:,1]=0;q[:,0,:,2]=0;q[0,1]=0
    a=m(x,mask,q)
    corrupt=x.detach().clone();corrupt[(q==0).repeat_interleave(8,-1)]=float('nan')
    torch.testing.assert_close(a,m(corrupt,mask,q),rtol=0,atol=0)
    a[:,0].sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[(q==0).repeat_interleave(8,-1)])==0
    with pytest.raises(ValueError,match='at least one valid'):
        m(x,mask,torch.zeros_like(q))
    with pytest.raises(ValueError,match='requires electrode-time'):
        m(x,mask)


def test_soft_quality_changes_attention_and_uniform_quality_matches_unweighted():
    torch.manual_seed(5)
    attention=make_transformer(16,2,1,0).eval()
    x=torch.randn(2,4,16)
    torch.testing.assert_close(quality_attention(attention,x,torch.ones(2,4),2),attention(x),rtol=1e-5,atol=1e-6)
    q=torch.ones(2,4);q[:,0]=.05
    assert not torch.allclose(quality_attention(attention,x,q,2),attention(x))
    q[0]=0
    output=quality_attention(attention,x,q,2)
    assert torch.isfinite(output).all() and output[0].count_nonzero()==0
    output[1].sum().backward()
    assert attention.layers[0].self_attn.in_proj_weight.grad is not None


def test_quality_proxy_is_monotonic_and_corruption_preserves_masks_rng_and_input():
    policy=QualityConfig(patch_samples=8)
    x=torch.randn(3,2,3,32)*.1;q=torch.ones(3,2,3,4);q[:,:,1,0]=0
    hard=(q>0).repeat_interleave(8,-1)
    low=quality_weights(x*100,hard,policy)
    high=quality_weights(x*1000,hard,policy)
    assert torch.all(low>=high) and low[q==0].count_nonzero()==0
    original=x.clone();torch.manual_seed(7)
    y,weights=corrupt_view(x,low,policy,100,1,.5,80)
    torch.manual_seed(7);y2,weights2=corrupt_view(x,low,policy,100,1,.5,80)
    torch.testing.assert_close(y,y2,rtol=0,atol=0);torch.testing.assert_close(weights,weights2,rtol=0,atol=0)
    torch.testing.assert_close(x,original,rtol=0,atol=0)
    assert weights[low==0].count_nonzero()==0 and (weights.sum((1,2,3))>0).all()
    assert not torch.equal(y,x)


def test_quality_consistency_training_resume_and_record_inference(tmp_path):
    records=[]
    for ident in range(2):
        for session in range(2):
            path=tmp_path/f'id{ident}_s{session}.npz'
            rng=np.random.default_rng(ident*2+session)
            eeg=rng.normal(0,10,(3,192)).astype(np.float32)
            hard=np.ones_like(eeg,dtype=bool);hard[1,64:80]=False;eeg[~hard]=0
            np.savez(path,eeg=eeg,valid_channel_samples=hard,sample_rate=32,channel_names=['A','B','C'],unit='uV')
            records.append(Record(str(ident),str(session),path.stem,path))
    config=dict(data=dict(sample_rate=32,channel_names=['A','B','C'],window_seconds=1,block_windows=2,quality=dict(patch_samples=8)),
        encoder=dict(hidden_dim=16,embedding_dim=8,attention_scope='axial_patches',transformer_layers=1,attention_heads=2,patch_samples=8,quality_aware=True,dropout=.1),
        training=dict(epochs=2,steps_per_epoch=2,identities_per_batch=2,samples_per_identity=2,ce_weight=0,consistency_weight=.5,corruption_probability=1.,seed=7))
    def cfg(epochs,name):
        c=json.loads(json.dumps(config));c['training']['epochs']=epochs
        path=tmp_path/(name+'.json');path.write_text(json.dumps(c));return path
    full=train(records,cfg(2,'full'),tmp_path/'full.pt','cpu')
    train(records,cfg(1,'one'),tmp_path/'one.pt','cpu')
    resumed=train(records,cfg(2,'resume'),tmp_path/'resumed.pt','cpu',resume=tmp_path/'one.last.pt')
    assert full['history']==resumed['history']
    for name,weight in full['encoder_state'].items():
        torch.testing.assert_close(weight,resumed['encoder_state'][name],rtol=0,atol=0)
    assert all(np.isfinite(h['consistency_loss']) for h in full['history'])
    assert full['history'][0]['consistency_loss']>0
    encoder,data,state,mid=load_encoder(tmp_path/'full.pt','cpu')
    z=encode_record(encoder,data,records[0],'cpu')
    assert np.isfinite(z).all() and np.isclose(np.linalg.norm(z),1)


def test_soft_attention_probabilities_follow_quality_prior():
    attention=make_transformer(16,2,1,0).eval()
    layer=attention.layers[0]
    with torch.no_grad():
        # Equal QK logits isolate the quality prior; V keeps the actual features.
        layer.self_attn.in_proj_weight[:32].zero_();layer.self_attn.in_proj_bias.zero_()
        layer.self_attn.in_proj_weight[32:].copy_(torch.eye(16))
        layer.self_attn.out_proj.weight.copy_(torch.eye(16));layer.self_attn.out_proj.bias.zero_()
    tokens=torch.randn(1,3,16);q=torch.tensor([[1.,.2,0.]])
    observed=[]
    hook=layer.self_attn.register_forward_hook(lambda module,args,out:observed.append(out[0]))
    quality_attention(attention,tokens,q,2);hook.remove()
    values=layer.norm1(tokens)
    expected=(values*q.unsqueeze(-1)).sum(1)/q.sum(1,keepdim=True)
    torch.testing.assert_close(observed[0],expected[:,None].expand(-1,3,-1),rtol=1e-5,atol=1e-6)


def test_cli_conversion_exports_masks_gap_and_ready_quality_config(tmp_path):
    from eeg_siamese.cli import main
    fs=100;t=np.arange(4000)/fs
    raw=tmp_path/'raw.npz'
    np.savez(raw,eeg=np.array([20*np.sin(2*np.pi*9*t),15*np.sin(2*np.pi*12*t)]),
             sample_rate=fs,channel_names=['A','B'],unit='uV',invalid_intervals=np.array([[2500,2510]]))
    manifest=tmp_path/'raw.csv'
    manifest.write_text('subject_id,session_id,record_id,path\nid,s1,raw,raw.npz\n')
    cfg=tmp_path/'pre.json'
    cfg.write_text(json.dumps(dict(target_sample_rate=100,high_cut_hz=35,mask_scope='channel_sample')))
    out=tmp_path/'converted'
    main(['preprocess','--manifest',str(manifest),'--config',str(cfg),'--output-dir',str(out)])
    config=json.loads((out/'general.json').read_text())
    data=DataConfig(**config['data']);encoder=EncoderConfig(channels=2,**config['encoder'])
    assert data.quality['patch_samples']==encoder.patch_samples==50
    assert encoder.quality_aware and config['training']['consistency_weight']>.0
    with np.load(out/'records/raw.npz') as archive:
        assert not archive['valid_channel_samples'][:,2500:2510].any()


