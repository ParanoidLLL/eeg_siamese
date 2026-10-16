"""Training lifecycle tests; synthetic signals validate mechanics, not accuracy."""
import json
import sys
import types

import numpy as np
import pytest
import torch

from eeg_siamese.data import DataConfig, Record
from eeg_siamese.evaluation import Episode
from eeg_siamese.model import EEGEncoder, EncoderConfig
from eeg_siamese.training import _load_initialization, train


torch.set_num_threads(1)


def records_at(folder, prefix='train', channels=2):
    records = []
    for identity in range(2):
        for session in range(2):
            path = folder / f'{prefix}_{identity}_{session}.npz'
            rng = np.random.default_rng(identity * 10 + session)
            np.savez(path, eeg=rng.normal(0, 10, (channels, 192)).astype(np.float32),
                     sample_rate=np.array(32), channel_names=np.array([f'C{i}' for i in range(channels)]),
                     unit=np.array('uV'), valid_channel_samples=np.ones((channels,192),dtype=bool))
            records.append(Record(f'{prefix}_{identity}', f's{session}', path.stem, path.resolve()))
    return records


def config_at(folder, epochs, name='config', **overrides):
    config = dict(data=dict(sample_rate=32, channel_names=['C0', 'C1'], window_seconds=.5,
                           block_windows=3, quality=dict(patch_samples=4)),
                  encoder=dict(hidden_dim=16, embedding_dim=8, transformer_layers=1,
                               attention_heads=2, dropout=.2, token_stride=2, patch_samples=4),
                  training=dict(epochs=epochs, steps_per_epoch=2, identities_per_batch=2,
                                samples_per_identity=2, seed=15))
    config['training'].update(overrides)
    path = folder / f'{name}.json'
    path.write_text(json.dumps(config))
    return path


def development_at(folder, training=None):
    records = records_at(folder, 'dev')
    # Both registered and unregistered identities occur in the query session.
    return [Episode(records[:1], [records[1], records[3]], 'development')]


def test_resume_restores_encoder_classifier_optimizer_and_rng(tmp_path):
    records = records_at(tmp_path)
    full = train(records, config_at(tmp_path, 3, 'full'), tmp_path / 'full.pt', torch.device('cpu'))
    train(records, config_at(tmp_path, 1, 'initial'), tmp_path / 'initial.pt', torch.device('cpu'))
    resumed = train(records, config_at(tmp_path, 3, 'resumed'), tmp_path / 'resumed.pt',
                    torch.device('cpu'), resume=tmp_path / 'initial.last.pt')
    assert full['history'] == resumed['history']
    assert resumed['completed_epochs'] == 3
    assert full['identity_labels'] == resumed['identity_labels']
    for state_key in ('encoder_state', 'classifier_state'):
        for key, value in full[state_key].items():
            torch.testing.assert_close(value, resumed[state_key][key], rtol=0, atol=0)
    for key, values in full['optimizer_state']['state'].items():
        for name, value in values.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, resumed['optimizer_state']['state'][key][name], rtol=0, atol=0)
    assert resumed['selection'] == 'final_epoch_no_development'






def test_resume_rejects_changed_signal_and_optimization_config(tmp_path):
    records = records_at(tmp_path)
    train(records, config_at(tmp_path, 1), tmp_path / 'first.pt', torch.device('cpu'))
    with pytest.raises(ValueError, match='config changed'):
        train(records, config_at(tmp_path, 2, 'changed', learning_rate=.01), tmp_path / 'changed.pt',
              torch.device('cpu'), resume=tmp_path / 'first.last.pt')
    with np.load(records[0].path) as source:
        values = {key: source[key].copy() for key in source.files}
    values['eeg'][0, 0] += 1
    np.savez(records[0].path, **values)
    with pytest.raises(ValueError, match='file contents changed'):
        train(records, config_at(tmp_path, 2, 'contents'), tmp_path / 'changed.pt',
              torch.device('cpu'), resume=tmp_path / 'first.last.pt')


def test_best_development_checkpoint_is_distinct_from_last_and_survives_resume(tmp_path, monkeypatch):
    records = records_at(tmp_path)
    episodes = development_at(tmp_path)
    accuracies = iter([.9, .5, .6, .4])

    def fake_evaluate(*args):
        value = next(accuracies)
        return dict(threshold=.75, metrics=dict(accuracy=value, known_identification_rate=value), rows=[])

    monkeypatch.setattr('eeg_siamese.evaluation.evaluate_development', fake_evaluate)
    best = train(records, config_at(tmp_path, 3), tmp_path / 'best.pt', torch.device('cpu'),
                 dev_episodes=episodes)
    last = torch.load(tmp_path / 'best.last.pt', weights_only=True)
    assert best['completed_epochs'] == 1
    assert last['completed_epochs'] == 3
    assert best['selection'] == 'best_development_accuracy'
    assert best['development_subject_ids'] == ['dev_0', 'dev_1']
    continued = train(records, config_at(tmp_path, 4, 'continued'), tmp_path / 'continued.pt',
                      torch.device('cpu'), resume=tmp_path / 'best.last.pt', dev_episodes=episodes)
    assert continued['completed_epochs'] == 1
    final_last = torch.load(tmp_path / 'continued.last.pt', weights_only=True)
    assert final_last['completed_epochs'] == 4
    assert len(json.loads((tmp_path / 'continued.history.json').read_text())) == 4
    assert final_last['best_snapshot']['completed_epochs'] == 1


def test_development_cannot_include_any_pretraining_identity(tmp_path):
    prior_records = records_at(tmp_path, 'prior')
    initial = tmp_path / 'prior.pt'
    train(prior_records, config_at(tmp_path, 1), initial, torch.device('cpu'))
    current = records_at(tmp_path, 'new')
    other = records_at(tmp_path, 'overlap')
    # Different files and IDs, same identity as prior training: identity audit is essential.
    enroll = Record(prior_records[0].subject_id, 's2', other[0].record_id, other[0].path)
    query = Record(prior_records[0].subject_id, 's3', other[1].record_id, other[1].path)
    episode = Episode([enroll], [query, other[3]])
    with pytest.raises(ValueError, match='training identities'):
        train(current, config_at(tmp_path, 1), tmp_path / 'invalid.pt', torch.device('cpu'),
              initialization=initial, dev_episodes=[episode])




def test_selected_development_subjects_cannot_later_be_training_data(tmp_path, monkeypatch):
    monkeypatch.setattr('eeg_siamese.evaluation.evaluate_development', lambda *args: dict(
        threshold=.7, metrics=dict(accuracy=.5, known_identification_rate=.5), rows=[]))
    records = records_at(tmp_path)
    episodes = development_at(tmp_path)
    source = tmp_path / 'source.pt'
    train(records, config_at(tmp_path, 1), source, torch.device('cpu'), dev_episodes=episodes)
    with pytest.raises(ValueError, match='previously used for development'):
        train(records_at(tmp_path, 'dev'), config_at(tmp_path, 1), tmp_path / 'invalid.pt',
              torch.device('cpu'), initialization=source)


@pytest.fixture
def tensorboard_writers(monkeypatch):
    writers = []

    class FakeWriter:
        def __init__(self, **kwargs):
            self.options, self.scalars = kwargs, []
            self.flushed, self.closed = False, False
            writers.append(self)

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, value, step))

        def flush(self):
            self.flushed = True

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', types.SimpleNamespace(SummaryWriter=FakeWriter))
    return writers


def test_tensorboard_resume_preserves_completed_steps_and_training_results(tmp_path, tensorboard_writers):
    records = records_at(tmp_path)
    reference = train(records, config_at(tmp_path, 2), tmp_path / 'reference.pt', torch.device('cpu'))
    assert not tensorboard_writers  # Optional dependency is unused unless requested.
    log_dir = tmp_path / 'logs'
    train(records, config_at(tmp_path, 1), tmp_path / 'logged.pt', torch.device('cpu'), tensorboard_dir=log_dir)
    resumed = train(records, config_at(tmp_path, 2), tmp_path / 'logged.pt', torch.device('cpu'),
                    resume=tmp_path / 'logged.last.pt', tensorboard_dir=log_dir)
    assert reference['history'] == resumed['history']
    first, second = tensorboard_writers
    assert first.options == dict(log_dir=str(log_dir), purge_step=None)
    assert second.options == dict(log_dir=str(log_dir), purge_step=2)
    for writer, epoch in ((first, 1), (second, 2)):
        assert writer.closed and writer.flushed
        assert {step for _, _, step in writer.scalars} == {epoch}
        values = {tag: value for tag, value, _ in writer.scalars}
        assert values['train/training_loss'] == resumed['history'][epoch - 1]['training_loss']
        assert values['train/epoch_seconds'] > 0
    for key, value in reference['encoder_state'].items():
        torch.testing.assert_close(value, resumed['encoder_state'][key], rtol=0, atol=0)


def test_tensorboard_closes_writer_when_training_fails(tmp_path, tensorboard_writers, monkeypatch):
    records = records_at(tmp_path)
    monkeypatch.setattr('eeg_siamese.training.supervised_contrastive_loss',
                        lambda z, *args: z.new_tensor(float('nan')))
    with pytest.raises(ValueError, match='Non-finite training loss'):
        train(records, config_at(tmp_path, 1), tmp_path / 'failed.pt', torch.device('cpu'),
              tensorboard_dir=tmp_path / 'logs')
    assert tensorboard_writers[0].closed and tensorboard_writers[0].flushed


def test_tensorboard_dependency_is_optional_and_cli_forwards_log_directory(tmp_path, monkeypatch):
    from eeg_siamese.cli import main
    from eeg_siamese.training import _tensorboard_writer
    monkeypatch.setitem(sys.modules, 'torch.utils.tensorboard', None)
    with _tensorboard_writer(None, 0) as writer:
        assert writer is None
    with pytest.raises(ValueError, match='TensorBoard logging requires'):
        with _tensorboard_writer(tmp_path / 'logs', 0):
            pass
    calls = []
    monkeypatch.setattr('eeg_siamese.cli.read_manifest', lambda path: [])
    monkeypatch.setattr('eeg_siamese.cli.train', lambda *args, **kwargs: calls.append(kwargs))
    main(['train', '--manifest', 'train.csv', '--config', 'config.json', '--output', 'model.pt',
          '--device', 'cpu', '--tensorboard-dir', str(tmp_path / 'logs')])
    assert calls[0]['tensorboard_dir'] == str(tmp_path / 'logs')


def test_real_development_episode_is_logged_and_can_be_reused_for_finetuning(tmp_path):
    records = records_at(tmp_path)
    episodes = development_at(tmp_path)
    source = tmp_path / 'source.pt'
    result = train(records, config_at(tmp_path, 1), source, torch.device('cpu'), dev_episodes=episodes)
    development = result['history'][0]['development']
    assert development['metrics']['records'] == 2
    assert development['metrics']['failed_records'] == 0
    assert np.isfinite(development['threshold'])
    assert {r['target_id'] for r in development['rows']} == {'dev_0', 'Unknown'}
    adapted = train(records, config_at(tmp_path, 1), tmp_path / 'adapted.pt', torch.device('cpu'),
                    initialization=source, dev_episodes=episodes)
    assert adapted['development_subject_ids'] == ['dev_0', 'dev_1']


def test_closed_set_development_selects_epochs_without_threshold_and_guards_identity_leakage(tmp_path):
    records=records_at(tmp_path)
    dev=records_at(tmp_path,'dev_closed')
    episode=Episode([dev[0],dev[2]],[dev[1],dev[3]],'closed')
    result=train(records,config_at(tmp_path,2),tmp_path/'closed.pt',torch.device('cpu'),
                 dev_episodes=[episode],dev_closed_set=True)
    assert result['dev_closed_set'] is True
    report=result['history'][-1]['development']
    assert report['threshold'] is None and report['metrics']['unknown_records']==0
    assert report['metrics']['records']==2
    assert set(result['development_subject_ids'])=={'dev_closed_0','dev_closed_1'}
    with pytest.raises(ValueError,match='mode changed'):
        train(records,config_at(tmp_path,3),tmp_path/'changed.pt',torch.device('cpu'),
              resume=tmp_path/'closed.last.pt',dev_episodes=[episode])
    with pytest.raises(ValueError,match='overlap'):
        train(records,config_at(tmp_path,1),tmp_path/'leak.pt',torch.device('cpu'),
              dev_episodes=[Episode(records[:1],records[1:2])],dev_closed_set=True)
    from eeg_siamese.evaluation import evaluate_development
    with pytest.raises(ValueError,match='unregistered'):
        evaluate_development(EEGEncoder(EncoderConfig(2,hidden_dim=16,embedding_dim=8,transformer_layers=1, patch_samples=4, token_stride=2)),
                             DataConfig(32,('C0','C1'),window_seconds=.5,block_windows=3,quality=dict(patch_samples=4)),
                             [Episode(dev[:1],[dev[1],dev[3]])],'cpu',closed_set=True)


def test_closed_set_episode_loader_requires_registered_queries(tmp_path):
    import csv
    from eeg_siamese.evaluation import load_episodes
    dev=records_at(tmp_path,'loader')
    for name,group in [('enroll',[dev[0],dev[2]]),('query',[dev[1],dev[3]])]:
        with (tmp_path/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['subject_id','session_id','record_id','path']);w.writeheader()
            w.writerows(r.metadata() for r in group)
    path=tmp_path/'episodes.json';path.write_text(json.dumps([dict(enrollment='enroll.csv',query='query.csv')]))
    assert len(load_episodes(path,closed_set=True))==1
    with pytest.raises(ValueError,match='unregistered'):
        load_episodes(path)


def test_early_stopping_counts_only_new_evaluations_after_resume_and_keeps_best(tmp_path, monkeypatch):
    import eeg_siamese.evaluation as evaluation
    def unchanged(*args, **kwargs):
        return dict(metrics=dict(accuracy=1.,known_identification_rate=1.),rows=[])
    monkeypatch.setattr(evaluation,'evaluate_development',unchanged)
    records=records_at(tmp_path);episodes=development_at(tmp_path)
    train(records,config_at(tmp_path,2,'before',eval_every=1),tmp_path/'before.pt','cpu',dev_episodes=episodes)
    selected=train(records,config_at(tmp_path,10,'continue',eval_every=1,early_stopping_evaluations=3),
                   tmp_path/'continued.pt','cpu',resume=tmp_path/'before.last.pt',dev_episodes=episodes)
    last=torch.load(tmp_path/'continued.last.pt',map_location='cpu',weights_only=True)
    assert last['completed_epochs']==5
    assert selected['completed_epochs']==1
    assert last['training_stop_reason']=='development_plateau'
    assert last['stopping_state']==dict(stale_evaluations=3,monitor_start_epoch=2)
    # A partial continuation preserves patience state across process restarts.
    train(records,config_at(tmp_path,3,'partial',eval_every=1,early_stopping_evaluations=3),
          tmp_path/'partial.pt','cpu',resume=tmp_path/'before.last.pt',dev_episodes=episodes)
    train(records,config_at(tmp_path,10,'restart',eval_every=1,early_stopping_evaluations=3),
          tmp_path/'restarted.pt','cpu',resume=tmp_path/'partial.last.pt',dev_episodes=episodes)
    restarted=torch.load(tmp_path/'restarted.last.pt',map_location='cpu',weights_only=True)
    assert last['history']==restarted['history']
    for key,weight in last['encoder_state'].items():
        torch.testing.assert_close(weight,restarted['encoder_state'][key],rtol=0,atol=0)
