import json
from pathlib import Path
from dataclasses import replace

import numpy as np
import pytest
import torch

from eeg_siamese.data import Record, DataConfig
from eeg_siamese.inference import load_encoder, encode_record
from eeg_siamese.w_model import fit_training_statistics, train_w, export_w_checkpoint
from test_spectral_residual import cfg


def test_portable_w_recipe_export_and_training_only_statistics(tmp_path):
    import csv
    records=[]
    for person in range(2):
        for day in range(2):
            path=tmp_path/f'p{person}d{day}.npz'
            rng=np.random.default_rng(person*2+day)
            np.savez(path,eeg=rng.normal(0,10,(4,1024)).astype('float32'),sample_rate=128.,
                channel_names=np.array(['a','b','c','d']),unit='uV',valid_channel_samples=np.ones((4,1024),bool))
            records.append(Record(f'p{person}',f'd{day}',path.stem,path))
    manifest=tmp_path/'train.csv'
    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['subject_id','session_id','record_id','path'])
        writer.writeheader();writer.writerows(r.metadata() for r in records)
    config=dict(data=dict(sample_rate=128,channel_names=['a','b','c','d'],window_seconds=4,block_windows=2,quality=dict(patch_samples=64)),
        encoder=cfg().__dict__,training=dict(epochs=1,steps_per_epoch=2,identities_per_batch=2,samples_per_identity=2,ce_weight=0,consistency_weight=.5))
    path=tmp_path/'config.json';path.write_text(json.dumps(config));out=tmp_path/'w'
    result=train_w(manifest,path,out,'cpu',control=True)
    assert not result['evaluated'] and result['control_trained']
    stats=json.loads((out/'spectral_statistics.json').read_text())
    assert {r['record_id'] for r in stats['records']}=={r.record_id for r in records}
    assert len(stats['mean'])==20 and not stats['target_data_used']
    wrapped,dc,_,_=load_encoder(out/'W/model.pt','cpu')
    standalone,_,checkpoint,_=load_encoder(out/'model.pt','cpu')
    assert standalone.config.record_fusion=='none'
    assert not any(k.startswith('record_fuser.') for k in checkpoint['encoder_state'])
    assert 'optimizer_state' not in checkpoint
    np.testing.assert_allclose(encode_record(wrapped,dc,records[0],'cpu'),encode_record(standalone,dc,records[0],'cpu'),rtol=0,atol=1e-7)
    assert (out/'U/model.pt').is_file()
    with pytest.raises(ValueError):train_w(manifest,path,out,'cpu')
    with pytest.raises(ValueError):export_w_checkpoint(out/'W/model.pt',out/'model.pt')
    # A held-out-like file exists alongside training files but cannot affect fit.
    np.savez(tmp_path/'unlisted_query.npz',eeg=np.full((4,1024),1e6,dtype='float32'))
    again=fit_training_statistics(records,DataConfig(**config['data']))
    np.testing.assert_array_equal(again['mean'],stats['mean'])


def test_w_template_is_packaged_and_has_no_fitted_or_machine_specific_paths():
    from importlib.resources import files
    root=Path(__file__).resolve().parents[1]
    text=files('eeg_siamese').joinpath('configs/w_openbmi.json').read_text()
    cfg=json.loads(text)
    assert cfg==json.loads((root/'configs/w_openbmi.json').read_text())
    assert cfg['encoder']['spectral_fusion']=='log'
    assert 'spectral_feature_mean' not in cfg['encoder']
    assert cfg['training']['cross_session'] and cfg['training']['epochs']==35
