import math
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import DataConfig, Record, RecordingDataset, SessionBatchSampler, audit_episode
from .inference import checkpoint_id, read_json, write_json
from .losses import pairwise_contrastive_loss, supervised_contrastive_loss
from .model import EEGEncoder, EncoderConfig


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    steps_per_epoch: int = 100
    identities_per_batch: int = 4
    samples_per_identity: int = 2
    learning_rate: float = 0.0003
    weight_decay: float = 0.001
    ce_weight: float = 0.1
    temperature: float = 0.1
    metric_loss: str = "supcon"
    pair_margin: float = 1.0
    cross_session: bool = True
    seed: int = 42
    eval_every: int = 1
    cache_records: bool = True
    cache_max_bytes: int = 512 * 1024 * 1024
    consistency_weight: float = 0.0
    corruption_probability: float = 0.8
    corruption_channel_fraction: float = 0.2
    corruption_noise_uv: float = 30.0
    early_stopping_evaluations: int = 0

    def __post_init__(self):
        counts = (self.epochs, self.steps_per_epoch, self.identities_per_batch, self.samples_per_identity,
                  self.eval_every)
        if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in counts):
            raise ValueError("Training counts must be positive integers")
        numeric = (self.learning_rate, self.weight_decay, self.ce_weight, self.temperature, self.pair_margin)
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError("Optimization settings must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.ce_weight < 0:
            raise ValueError("Invalid optimization settings")
        if self.temperature <= 0 or not 0 < self.pair_margin <= 2:
            raise ValueError("Invalid temperature or pair margin")
        if self.metric_loss not in ("supcon", "pair"):
            raise ValueError("metric_loss must be supcon or pair")
        if not isinstance(self.cache_records, bool):
            raise ValueError("cache_records must be boolean")
        if not isinstance(self.early_stopping_evaluations, int) or isinstance(self.early_stopping_evaluations, bool) or self.early_stopping_evaluations < 0:
            raise ValueError('early_stopping_evaluations must be a nonnegative integer')
        if not all(math.isfinite(v) for v in (self.consistency_weight, self.corruption_probability,
                                            self.corruption_channel_fraction, self.corruption_noise_uv)):
            raise ValueError("Consistency/corruption settings must be finite")
        if self.consistency_weight < 0 or not 0 <= self.corruption_probability <= 1 or not 0 <= self.corruption_channel_fraction < 1 or self.corruption_noise_uv < 0:
            raise ValueError("Invalid consistency/corruption settings")
        if not isinstance(self.cache_max_bytes, int) or isinstance(self.cache_max_bytes, bool) or self.cache_max_bytes < 1:
            raise ValueError("cache_max_bytes must be a positive integer")


def select_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this computer")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable on this computer")
    return device



def _rng_state():
    numpy_state = np.random.get_state()
    state = dict(python=random.getstate(), torch=torch.get_rng_state(),
                 numpy=[numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]])
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, 'get_rng_state'):
        state['mps'] = torch.mps.get_rng_state()
    return state


def _restore_rng(state):
    random.setstate(state['python'])
    values = state['numpy']
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), *values[2:]))
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if torch.backends.mps.is_available() and 'mps' in state:
        torch.mps.set_rng_state(state['mps'])


def _file_provenance(records, allow_missing=False):
    return [dict(r.metadata(), sha256=(checkpoint_id(r.path) if r.path.is_file() else None))
            if allow_missing else dict(r.metadata(), sha256=checkpoint_id(r.path)) for r in records]


def _development_provenance(episodes):
    return [dict(enrollment=_file_provenance(episode.enrollment),
                 queries=_file_provenance(episode.queries, allow_missing=True)) for episode in episodes]


def _record(metadata):
    return Record(metadata['subject_id'], metadata['session_id'], metadata['record_id'],
                  Path(metadata['path']))


def _merge_records(*groups):
    merged = {}
    physical_files = {}
    for group in groups:
        for row in group:
            previous = merged.get(row['record_id'])
            if previous is not None and (previous['subject_id'], previous['session_id'], previous['path']) != (
                    row['subject_id'], row['session_id'], row['path']):
                raise ValueError('record_id collides with prior data; use globally unique record identifiers')
            physical = physical_files.get(row['path'])
            if physical is not None and physical != (row['record_id'], row['subject_id'], row['session_id']):
                raise ValueError('A prior EEG file was relabeled; preserve its identity and record_id')
            physical_files[row['path']] = (row['record_id'], row['subject_id'], row['session_id'])
            merged[row['record_id']] = row
    return list(merged.values())


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _atomic_save(checkpoint, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _load_initialization(encoder, path, data_config, init_mode):
    initial = torch.load(path, map_location='cpu', weights_only=True)
    if initial.get('schema_version') != 1:
        raise ValueError('Initialization needs a compatible project checkpoint')
    if init_mode == 'strict':
        if EncoderConfig(**initial['encoder_config']) != encoder.config:
            raise ValueError('Pretrained encoder architecture is incompatible')
        if DataConfig(**initial['data_config']) != data_config:
            raise ValueError('Pretrained preprocessing is incompatible; adapt it explicitly')
        encoder.load_state_dict(initial['encoder_state'], strict=True)
        loaded = list(initial['encoder_state'])
    else:
        raise ValueError('Only strict axial initialization is supported')
    return initial, loaded


@contextmanager
def _tensorboard_writer(log_dir, start_epoch):
    if log_dir is None:
        yield None
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:
        raise ValueError('TensorBoard logging requires pip install -e ".[monitor]"') from error
    # Epochs are logged with one-based steps. Keep completed epochs on resume and
    # hide any later events left behind by a run interrupted before checkpointing.
    writer = SummaryWriter(log_dir=str(log_dir), purge_step=start_epoch + 1 if start_epoch else None)
    try:
        yield writer
    finally:
        try:
            writer.flush()
        finally:
            writer.close()


def train(records, config_path, output, device, initialization=None, resume=None,
          dev_episodes=None, init_mode='strict', tensorboard_dir=None, dev_closed_set=False):
    """Train an encoder; best model is selected on unseen development episodes.

    ``output`` contains the best evaluated epoch (the final epoch without dev).
    ``output.with_suffix('.last.pt')`` contains the latest optimizer/classifier/RNG
    state, suitable for reproducible epoch-boundary resume on the same runtime.
    Development metrics are model-selection estimates, not held-out results.
    ``tensorboard_dir`` enables optional epoch-level scalars without changing
    checkpoint history or requiring TensorBoard for ordinary training.
    """
    if initialization is not None and resume is not None:
        raise ValueError('Choose initialization or resume, not both')
    if init_mode != 'strict':
        raise ValueError('init_mode must be strict for the local axial-only project')
    if initialization is None and init_mode != 'strict':
        raise ValueError('Temporal initialization requires an initialization checkpoint')
    records = list(records)
    episodes = list(dev_episodes or [])
    if dev_closed_set and not episodes:
        raise ValueError("Closed-set development requires development episodes")
    config = read_json(config_path)
    data_config = DataConfig(**config['data'])
    from .window_qc import assert_reference_training_records
    assert_reference_training_records(data_config.window_qc_reference, records)
    options = dict(config.get('encoder', {}))
    options.setdefault('channels', len(data_config.channel_names))
    encoder_config = EncoderConfig(**options)
    train_config = TrainConfig(**config.get('training', {}))
    if train_config.early_stopping_evaluations and not episodes:
        raise ValueError('Early stopping requires development episodes')
    if encoder_config.quality_aware != (data_config.quality is not None):
        raise ValueError('Encoder and data quality mode must agree')
    if encoder_config.quality_aware and encoder_config.patch_samples != data_config.quality['patch_samples']:
        raise ValueError('Encoder and data quality patch sizes must agree')
    if train_config.consistency_weight and not encoder_config.quality_aware:
        raise ValueError('Pollution consistency requires quality-aware data/encoder')
    if encoder_config.channels != len(data_config.channel_names):
        raise ValueError('Encoder and data channel counts disagree')
    output = Path(output)
    last_path = output.with_suffix('.last.pt')
    if output.resolve() == last_path.resolve():
        raise ValueError('Output must not end in .last.pt; that name is reserved for resume state')
    output.parent.mkdir(parents=True, exist_ok=True)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    dataset = RecordingDataset(records, data_config, cache=train_config.cache_records,
                               cache_max_bytes=train_config.cache_max_bytes)
    sampler = SessionBatchSampler(
        records, train_config.identities_per_batch, train_config.samples_per_identity,
        train_config.steps_per_epoch, train_config.seed, train_config.cross_session,
    )
    # Validate all records and eligible blocks before optimization. Cache controls IO.
    for index in range(len(dataset)):
        dataset[index]
    encoder = EEGEncoder(encoder_config)
    prior_records, prior_development = [], []
    initialization_id, loaded_keys = None, []
    initial = None
    if initialization is not None:
        initial, loaded_keys = _load_initialization(encoder, initialization, data_config, init_mode)
        prior_records = initial['training_records']
        prior_development = initial.get('development_records', [])
        initialization_id = checkpoint_id(initialization)
    resumed = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if resumed is not None:
        if resumed.get('schema_version') != 1 or 'optimizer_state' not in resumed:
            raise ValueError('Resume requires a complete resumable checkpoint')
        if EncoderConfig(**resumed['encoder_config']) != encoder_config:
            raise ValueError('Resume encoder config changed')
        if DataConfig(**resumed['data_config']) != data_config:
            raise ValueError('Resume preprocessing config changed')
        old_training = asdict(TrainConfig(**resumed['training_config']))
        old_training.setdefault('cache_max_bytes', 512 * 1024 * 1024)
        new_training = asdict(train_config)
        old_training.pop('epochs')
        new_training.pop('epochs')
        old_training.pop('early_stopping_evaluations')
        new_training.pop('early_stopping_evaluations')
        if old_training != new_training:
            raise ValueError('Resume training config changed; only epochs and stopping patience may change')
        if resumed.get('device_type') != torch.device(device).type:
            raise ValueError('Resume requires the same device type for RNG restoration')
        if resumed.get('versions') != dict(torch=str(torch.__version__), numpy=str(np.__version__)):
            raise ValueError('Resume requires the same torch and numpy versions')
        prior_records = resumed['training_records']
        prior_development = resumed.get('development_records', [])
        initialization_id = resumed.get('initialization_id')
        loaded_keys = resumed.get('initialization_loaded_keys', [])
        encoder.load_state_dict(resumed['encoder_state'], strict=True)
    all_records = _merge_records(prior_records, [r.metadata() for r in records])
    if {r.subject_id for r in records} & {r['subject_id'] for r in prior_development}:
        raise ValueError('Training identities were previously used for development model selection')
    # Check paths/IDs as well, so relabeling a development signal cannot bypass this guard.
    from .data import assert_records_disjoint
    assert_records_disjoint(records, [_record(row) for row in prior_development])
    # Audits include every identity seen during pretraining, not just this run.
    for episode in episodes:
        audit_episode([_record(row) for row in all_records], episode.enrollment,
                      episode.queries, unseen=True)
    development_records = _merge_records(prior_development,
        [r.metadata() for episode in episodes for r in episode.enrollment + episode.queries])
    training_provenance = _file_provenance(records)
    development_provenance = _development_provenance(episodes)
    if resumed is not None:
        if resumed.get('run_provenance') != training_provenance:
            raise ValueError('Resume training manifest or EEG file contents changed')
        if resumed.get('dev_closed_set', False) != dev_closed_set:
            raise ValueError('Resume development mode changed')
        if resumed.get('development_protocol') != development_provenance:
            raise ValueError('Resume development protocol or EEG file contents changed')
        if resumed.get('identity_labels') != dataset.identities:
            raise ValueError('Resume identity label order changed')
    encoder.to(device).train()
    classifier = nn.Linear(encoder_config.embedding_dim, len(dataset.identities)).to(device)
    parameters = list(encoder.parameters()) + (list(classifier.parameters()) if train_config.ce_weight else [])
    optimizer = torch.optim.AdamW(parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    history, start_epoch, best_score, best_snapshot = [], 0, None, None
    if resumed is not None:
        if resumed['classifier_state'] is not None:
            classifier.load_state_dict(resumed['classifier_state'], strict=True)
        optimizer.load_state_dict(resumed['optimizer_state'])
        history = list(resumed['history'])
        start_epoch = resumed['completed_epochs']
        if train_config.epochs <= start_epoch:
            raise ValueError('Resume target epochs must exceed completed epochs')
        best_score = resumed.get('best_score')
        best_snapshot = resumed.get('best_snapshot')
        if best_snapshot is None and resumed.get('selection') == 'best_development_accuracy':
            best_snapshot = {k: v for k, v in resumed.items() if k != 'best_snapshot'}
        # Restore after constructing/loading modules and validating data.
        _restore_rng(resumed['rng_state'])
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    stopping = dict(stale_evaluations=0, monitor_start_epoch=start_epoch)
    if resumed and resumed.get('training_config', {}).get('early_stopping_evaluations', 0) == train_config.early_stopping_evaluations:
        stopping.update(resumed.get('stopping_state', {}))
    stop_reason = None

    def checkpoint(epoch):
        return dict(
            schema_version=1, encoder_config=asdict(encoder_config), data_config=asdict(data_config),
            training_config=asdict(train_config), dev_closed_set=dev_closed_set, encoder_state=_cpu_state(encoder),
            classifier_state=_cpu_state(classifier) if train_config.ce_weight else None,
            identity_labels=dataset.identities, training_records=all_records,
            run_provenance=training_provenance, development_protocol=development_provenance,
            development_records=development_records,
            development_subject_ids=sorted({r['subject_id'] for r in development_records}),
            initialization_id=initialization_id,
            initialization_mode=resumed.get('initialization_mode', 'strict') if resumed else init_mode,
            initialization_loaded_keys=loaded_keys, history=list(history), completed_epochs=epoch,
            optimizer_state=optimizer.state_dict(), rng_state=_rng_state(), best_score=best_score,
            stopping_state=dict(stopping), training_stop_reason=stop_reason,
            selection='best_development_accuracy' if episodes else 'final_epoch_no_development',
            versions=dict(torch=str(torch.__version__), numpy=str(np.__version__)),
            device_type=torch.device(device).type,
        )

    with _tensorboard_writer(tensorboard_dir, start_epoch) as writer:
        for epoch in range(start_epoch, train_config.epochs):
            epoch_started = perf_counter()
            sampler.epoch = epoch
            encoder.train()
            total = metric_total = ce_total = consistency_total = 0.0
            for batch in loader:
                if encoder_config.quality_aware:
                    x, mask, quality, identities, sessions = batch
                    quality = quality.to(device)
                else:
                    x, mask, identities, sessions = batch
                    quality = None
                x, mask = x.to(device), mask.to(device)
                identities, sessions = identities.to(device), sessions.to(device)
                optimizer.zero_grad(set_to_none=True)
                z = encoder(x, mask, quality) if quality is not None else encoder(x, mask)
                if train_config.metric_loss == 'supcon':
                    metric = supervised_contrastive_loss(
                        z, identities, sessions, train_config.temperature, train_config.cross_session,
                    )
                else:
                    metric = pairwise_contrastive_loss(
                        z, identities, sessions, train_config.pair_margin, train_config.cross_session,
                    )
                ce = F.cross_entropy(16 * classifier(z), identities) if train_config.ce_weight else z.new_zeros(())
                loss = metric + train_config.ce_weight * ce
                consistency = z.new_zeros(())
                if train_config.consistency_weight:
                    from .quality import QualityConfig, corrupt_view
                    noisy, noisy_quality = corrupt_view(x, quality, QualityConfig(**data_config.quality),
                        data_config.signal_scale_uv, train_config.corruption_probability,
                        train_config.corruption_channel_fraction, train_config.corruption_noise_uv)
                    noisy_z = encoder(noisy, mask, noisy_quality)
                    consistency = (1 - (z * noisy_z).sum(-1)).clamp_min(0).mean()
                    loss = loss + train_config.consistency_weight * consistency
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite training loss')
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                total += float(loss.detach())
                metric_total += float(metric.detach())
                ce_total += float(ce.detach())
                consistency_total += float(consistency.detach())
            entry = dict(epoch=epoch + 1, training_loss=total / len(loader),
                         metric_loss=metric_total / len(loader), ce_loss=ce_total / len(loader))
            if train_config.consistency_weight:
                entry['consistency_loss'] = consistency_total / len(loader)
            improved = not episodes
            if episodes and ((epoch + 1) % train_config.eval_every == 0 or epoch + 1 == train_config.epochs):
                from .evaluation import evaluate_development
                encoder.eval()
                development = (evaluate_development(encoder, data_config, episodes, device, closed_set=True)
                               if dev_closed_set else evaluate_development(encoder, data_config, episodes, device))
                entry['development'] = development
                metrics = development['metrics']
                score = [float(metrics['accuracy']), float(metrics.get('known_identification_rate') or 0)]
                improved = best_score is None or tuple(score) > tuple(best_score)
                if improved:
                    best_score = score
                if train_config.early_stopping_evaluations:
                    stopping['stale_evaluations'] = 0 if improved else stopping['stale_evaluations'] + 1
                    entry['stale_development_evaluations'] = stopping['stale_evaluations']
                    if stopping['stale_evaluations'] >= train_config.early_stopping_evaluations:
                        stop_reason = 'development_plateau'
            history.append(entry)
            print(f"epoch={epoch + 1} training_loss={entry['training_loss']:.6f}" +
                  (f" development_accuracy={entry['development']['metrics']['accuracy']:.6f}" if 'development' in entry else ''),
                  flush=True)
            current = checkpoint(epoch + 1)
            # Snapshot optimizer tensors now; otherwise later steps mutate an earlier best.
            if improved:
                _atomic_save(current, output)
                best_snapshot = torch.load(output, map_location='cpu', weights_only=True) if episodes else None
            if episodes and best_snapshot is not None and not output.exists():
                _atomic_save(best_snapshot, output)
            current['best_snapshot'] = best_snapshot
            _atomic_save(current, last_path)
            write_json(output.with_suffix('.history.json'), history)
            if writer is not None:
                for name in ('training_loss', 'metric_loss', 'ce_loss'):
                    writer.add_scalar(f'train/{name}', entry[name], epoch + 1)
                if 'consistency_loss' in entry:
                    writer.add_scalar('train/consistency_loss', entry['consistency_loss'], epoch + 1)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
                writer.add_scalar('train/epoch_seconds', perf_counter() - epoch_started, epoch + 1)
                if 'development' in entry:
                    for name, value in entry['development']['metrics'].items():
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            writer.add_scalar(f'development/{name}', value, epoch + 1)
                writer.flush()
            if stop_reason:
                print(f"early_stop epoch={epoch+1} reason={stop_reason} stale_evaluations={stopping['stale_evaluations']}", flush=True)
                break
    if not output.exists():
        raise ValueError('No evaluated checkpoint was selected')
    # A resumed run into a new output path must retain the historical best.
    if episodes and best_snapshot is not None:
        _atomic_save(best_snapshot, output)
    return torch.load(output, map_location='cpu', weights_only=True)
