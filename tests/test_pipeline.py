"""Exercise the retained three-model wrapper without real EEG or target tuning."""
import csv,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pytest


@pytest.mark.parametrize('model',['axial_transformer'])
def test_train_register_query_wrapper(tmp_path,model):
    root=Path(__file__).resolve().parents[1]
    fields=['subject_id','session_id','record_id','path'];parts={'train':[],'enroll':[],'query':[]}
    rng=np.random.default_rng(9)
    for role,ids,sessions in [('train',['source_A','source_B'],['s1','s2']),('enroll',['new_A','new_B'],['s1']),('query',['new_A','new_B'],['s2'])]:
        for identity in ids:
            for session in sessions:
                rid=f'{identity}_{session}';path=tmp_path/(rid+'.npz')
                np.savez(path,eeg=rng.normal(0,10,(3,96)).astype(np.float32),sample_rate=32,channel_names=['A','B','C'],unit='uV',valid_channel_samples=np.ones((3,96),dtype=bool))
                parts[role].append(dict(subject_id=identity,session_id=session,record_id=rid,path=str(path)))
    for role,rows in parts.items():
        with (tmp_path/(role+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    config=dict(data=dict(sample_rate=32,channel_names=['A','B','C'],window_seconds=1,block_windows=2),encoder=dict(hidden_dim=16,embedding_dim=8,attention_heads=2,dropout=0,token_stride=2,transformer_layers=0 if model=='cnn' else 1),training=dict(epochs=1,steps_per_epoch=1,identities_per_batch=2,samples_per_identity=2,seed=9,ce_weight=0,metric_loss='pair' if model=='cnn' else 'supcon',cross_session=True))
    if model=='axial_transformer':
        config['data']['quality']=dict(patch_samples=8)
        config['encoder'].update(attention_scope='axial_patches',patch_samples=8,quality_aware=True)
        config['training']['consistency_weight']=.5
    cp=tmp_path/'config.json';cp.write_text(json.dumps(config));out=tmp_path/'output'
    command=[sys.executable,str(root/'scripts/run_pipeline.py'),'--model',model,'--config',str(cp),'--train-manifest',str(tmp_path/'train.csv'),'--enroll-manifest',str(tmp_path/'enroll.csv'),'--query-manifest',str(tmp_path/'query.csv'),'--output-dir',str(out),'--device','cpu','--threads','1']
    env=dict(os.environ);env['PYTHONPATH']=str(root/'src')
    with (tmp_path/'target.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(parts['enroll']+parts['query'])
    prepare=[sys.executable,str(root/'scripts/prepare_manifests.py'),'--source',str(tmp_path/'train.csv'),'--target',str(tmp_path/'target.csv'),'--output-dir',str(tmp_path/'split'),'--enroll-session','s1','--query-session','s2']
    prepared=subprocess.run(prepare,env=env,capture_output=True,text=True,timeout=30)
    assert prepared.returncode==0,prepared.stderr
    for role in ('train','enroll','query'):
        command[command.index('--'+('train-manifest' if role=='train' else role+'-manifest'))+1]=str(tmp_path/'split'/f'{role}.csv')
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr+result.stdout
    report=json.loads((out/'final_test.json').read_text());assert report['metrics']['records']==2
    assert report['metrics']['failed_records']==0
    gallery=json.loads((out/'gallery.json').read_text());assert gallery['identities']==['new_A','new_B']
    plotted=subprocess.run([sys.executable,str(root/'scripts/plot_results.py'),'--run-dir',str(out),'--tensorboard-dir',str(tmp_path/'tensorboard')],env=env,capture_output=True,text=True,timeout=30)
    assert plotted.returncode==0,plotted.stderr
    assert (out/'summary.png').is_file()


def test_packaged_default_templates_are_available_and_current():
    from importlib.resources import files
    root=Path(__file__).resolve().parents[1]
    for name in ('axial_transformer.json','preprocessing.quality.json'):
        packaged=json.loads(files('eeg_siamese').joinpath('configs/'+name).read_text(encoding='utf-8'))
        assert packaged==json.loads((root/'configs'/name).read_text())
