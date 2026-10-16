"""Train one retained model, freeze it, enroll S1 and evaluate S2.

Use processed manifests. Conversion and advanced development options are exposed
through eeg-siamese. This script never uses query scores for model selection.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS=('axial_transformer',)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,default='axial_transformer')
    for name in ('train-manifest','enroll-manifest','query-manifest','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,help='Defaults to configs/<model>.json; adapt channel names for new datasets')
    p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume',action='store_true',help='Resume from output-dir/model.last.pt')
    a=p.parse_args()
    if a.threads<1:p.error('threads must be positive')
    from eeg_siamese.data import read_manifest,audit_episode
    train=read_manifest(a.train_manifest);enroll=read_manifest(a.enroll_manifest);query=read_manifest(a.query_manifest,allow_missing=True)
    audit_episode(train,enroll,query)
    if {r.subject_id for r in enroll}!={r.subject_id for r in query}:p.error('Closed-set query identities must equal the registered identities')
    config=(a.config or ROOT/f'configs/{a.model}.json').resolve()
    from eeg_siamese.model import EncoderConfig
    values=json.loads(config.read_text());ec=EncoderConfig(channels=len(values['data']['channel_names']),**values['encoder'])
    if not ec.quality_aware:p.error('The axial pipeline requires quality-aware inputs')
    if a.model=='axial_transformer' and values['training'].get('consistency_weight',0)<=0:p.error('Retained axial training includes pollution consistency')
    if values['training'].get('early_stopping_evaluations',0):p.error('This final-training wrapper has no development set; use CLI train with --dev-episodes for early stopping')
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    checkpoint=out/'model.pt';gallery=out/'gallery.json';last=out/'model.last.pt'
    if not a.resume and (checkpoint.exists() or last.exists()):p.error('Output contains weights; choose a new directory or --resume')
    if a.resume and not last.exists():p.error('--resume requires model.last.pt')
    env=dict(os.environ);env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
    bootstrap=f'import torch;torch.set_num_threads({a.threads});from eeg_siamese.cli import main;main()'
    def run(args):subprocess.run([sys.executable,'-u','-c',bootstrap,*map(str,args)],env=env,check=True)
    args=['train','--manifest',a.train_manifest.resolve(),'--config',config,'--output',checkpoint,'--device',a.device,'--tensorboard-dir',out/'tensorboard']
    if a.resume:args+=['--resume',last]
    run(args)
    run(['enroll','--checkpoint',checkpoint,'--manifest',a.enroll_manifest.resolve(),'--output',gallery,'--device',a.device])
    run(['evaluate','--checkpoint',checkpoint,'--gallery',gallery,'--manifest',a.query_manifest.resolve(),'--closed-set','--output',out/'final_test.json','--device',a.device])


if __name__=='__main__':main()
