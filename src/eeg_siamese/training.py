import math
import json
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
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
    same_session_batch: bool = False
    seed: int = 42
    eval_every: int = 1
    cache_records: bool = True
    cache_max_bytes: int = 512 * 1024 * 1024
    consistency_weight: float = 0.0
    corruption_probability: float = 0.8
    corruption_channel_fraction: float = 0.2
    corruption_noise_uv: float = 30.0
    early_stopping_evaluations: int = 0
    gate_regularization_weight: float = 0.0
    selector_lr_multiplier: float = 1.0
    window_reliability_weight: float = 0.0
    window_reliability_temperature: float = .1
    local_power_mode: str = 'none'
    local_power_margin_weight: float = .5
    fusion_reference: str = ''
    fusion_reference_sha256: str = ''
    fusion_branch_weight: float = 1.
    modulation_reference: str = ''
    modulation_reference_sha256: str = ''

    def __post_init__(self):
        if not math.isfinite(self.fusion_branch_weight) or self.fusion_branch_weight<0:
            raise ValueError('Fusion branch supervision must be nonnegative')
        if not isinstance(self.same_session_batch, bool) or (self.same_session_batch and self.cross_session):
            raise ValueError('Same-session batches require cross_session=False')
        if self.local_power_mode not in ('none','augment','margin'):
            raise ValueError('Unknown local_power_mode')
        if not math.isfinite(self.local_power_margin_weight) or self.local_power_margin_weight<=0:
            raise ValueError('Local power margin weight must be positive')
        if not math.isfinite(self.window_reliability_weight) or self.window_reliability_weight<0:
            raise ValueError('Window reliability weight must be nonnegative')
        if not math.isfinite(self.window_reliability_temperature) or self.window_reliability_temperature<=0:
            raise ValueError('Window reliability temperature must be positive')
        if not math.isfinite(self.selector_lr_multiplier) or self.selector_lr_multiplier<=0:
            raise ValueError('selector_lr_multiplier must be finite and positive')
        if not math.isfinite(self.gate_regularization_weight) or self.gate_regularization_weight < 0:
            raise ValueError('Invalid gate regularization weight')
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
                  Path(metadata['path']), metadata.get('task_id'), metadata.get('run_id'))


def _merge_records(*groups):
    merged = {}
    physical_files = {}
    for group in groups:
        for row in group:
            previous = merged.get(row['record_id'])
            if previous is not None and (previous['subject_id'], previous['session_id'], previous['path']) != (
                    row['subject_id'], row['session_id'], row['path']):
                raise ValueError('record_id collides with prior data; use globally unique record identifiers')
            if previous is not None and any(previous.get(k) != row.get(k) for k in ('task_id', 'run_id')):
                raise ValueError('Prior task_id/run_id metadata changed')
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


def _load_modulation_reference(encoder,path,data_config):
    initial=torch.load(path,map_location='cpu',weights_only=True)
    if initial.get('schema_version')!=1 or EncoderConfig(**initial['encoder_config'])!=replace(encoder.config,spectral_modulation='none'):
        raise ValueError('Modulation reference must be identical W without modulation')
    if DataConfig(**initial['data_config'])!=data_config:
        raise ValueError('Modulation reference preprocessing changed')
    expected=set(encoder.state_dict())-set(initial['encoder_state'])
    if any(not k.startswith('spectral_modulator.') for k in expected):
        raise ValueError('Only modulation parameters may be newly initialized')
    mismatch=encoder.load_state_dict(initial['encoder_state'],strict=False)
    if mismatch.unexpected_keys or set(mismatch.missing_keys)!=expected:
        raise ValueError('Unexpected modulation reference keys')
    return initial,list(initial['encoder_state'])


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
          dev_episodes=None, init_mode='strict', tensorboard_dir=None, dev_closed_set=False,
          dev_evaluator=None):
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
    if encoder_config.spectral_fusion!='none':
        if not math.isclose(encoder_config.sample_rate,data_config.sample_rate) or not math.isclose(encoder_config.spectral_input_scale_uv,data_config.signal_scale_uv) or not math.isclose(data_config.window_seconds,encoder_config.spectral_window_seconds):
            raise ValueError('Spectral residual sample rate, physical scale and window must match data')
    train_config = TrainConfig(**config.get('training', {}))
    if encoder_config.spectral_modulation!='none' and not train_config.modulation_reference:
        raise ValueError('Spectral modulation requires an audited W reference')
    if train_config.modulation_reference and (encoder_config.record_fusion!='none' or encoder_config.spectral_fusion!='log' or initialization is not None):
        raise ValueError('W modulation initialization requires standalone log model')
    if bool(train_config.fusion_reference)!=(encoder_config.record_fusion!='none'):
        raise ValueError('Protected record fusion requires its reference checkpoint')
    if encoder_config.record_fusion!='none' and (train_config.metric_loss!='supcon' or train_config.ce_weight or train_config.local_power_mode!='none'):
        raise ValueError('Record fusion uses SupCon without classifier or local power auxiliary')
    if encoder_config.record_fusion=='single' and train_config.fusion_branch_weight!=0:
        raise ValueError('Single expert control must not double its identity loss')
    if train_config.local_power_mode!='none':
        if encoder_config.window_pooling!='quality' or encoder_config.channel_gate!='none' or encoder_config.selection_mode!='none' or train_config.metric_loss!='supcon' or not train_config.cross_session:
            raise ValueError('Local power study requires full ungated original pooling and cross-task SupCon')
        from .local_power_drift import REGIONS
        if any(name not in data_config.channel_names for group in REGIONS.values() for name in group):
            raise ValueError('Local power policy requires explicit region electrode mapping')
    if bool(train_config.window_reliability_weight)!=(encoder_config.window_pooling=='reliable'):
        raise ValueError('Reliability pooling requires its auxiliary loss; controls must disable it')
    if train_config.window_reliability_weight and not train_config.consistency_weight:
        raise ValueError('Window reliability study reuses the existing paired corruption view')
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
                               cache_max_bytes=train_config.cache_max_bytes,
                               task_names=encoder_config.task_names if encoder_config.requires_task else ())
    sampler = SessionBatchSampler(
        records, train_config.identities_per_batch, train_config.samples_per_identity,
        train_config.steps_per_epoch, train_config.seed, train_config.cross_session,
        train_config.same_session_batch,
    )
    # Validate all records and eligible blocks before optimization. Cache controls IO.
    for index in range(len(dataset)):
        dataset[index]
    encoder = EEGEncoder(encoder_config)
    prior_records, prior_development = [], []
    initialization_id, loaded_keys = None, []
    initial = None
    if train_config.modulation_reference:
        if checkpoint_id(train_config.modulation_reference)!=train_config.modulation_reference_sha256:
            raise ValueError('W reference checksum changed')
        initial,loaded_keys=_load_modulation_reference(encoder,train_config.modulation_reference,data_config)
        if {r['record_id'] for r in initial['training_records']}!={r.record_id for r in records}:
            raise ValueError('W reference training records differ')
        prior_records=initial['training_records'];prior_development=initial.get('development_records',[])
        initialization_id=train_config.modulation_reference_sha256
    if encoder_config.record_fusion!='none':
        if initialization is not None:
            raise ValueError('Fusion reference and initialization cannot be combined')
        if checkpoint_id(train_config.fusion_reference)!=train_config.fusion_reference_sha256:
            raise ValueError('Fusion reference checksum changed')
        initial,raw_keys=_load_initialization(encoder.record_fuser.raw,
            train_config.fusion_reference,data_config,'strict')
        mismatch=encoder.record_fuser.log.load_state_dict(initial['encoder_state'],strict=False)
        if mismatch.unexpected_keys or any(not k.startswith('spectral_branch.') for k in mismatch.missing_keys):
            raise ValueError('Log expert initialization must differ only by spectral branch')
        # Every reference training record must belong to this audited training set.
        if {r['record_id'] for r in initial['training_records']}!={r.record_id for r in records}:
            raise ValueError('Fusion reference training records differ')
        prior_records=initial['training_records']
        prior_development=initial.get('development_records',[])
        initialization_id=train_config.fusion_reference_sha256
        loaded_keys=['record_fuser.raw.'+k for k in raw_keys]+['record_fuser.log.'+k for k in raw_keys]
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
    optimization_parameters = parameters
    if train_config.selector_lr_multiplier != 1:
        if encoder_config.selection_mode not in ('learned_global','learned_task'):
            raise ValueError('Separate selector learning rate requires a learned selector')
        scores = encoder.selector.scores
        optimization_parameters = [dict(params=[p for p in parameters if p is not scores]),
            dict(params=[scores],lr=train_config.learning_rate*train_config.selector_lr_multiplier,weight_decay=0.)]
    optimizer = torch.optim.AdamW(optimization_parameters, lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
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
            if encoder_config.selection_mode != 'none':
                encoder.selector.set_epoch(epoch)
            total = metric_total = ce_total = consistency_total = 0.0
            gate_total = gate_min = gate_max = gate_saturation = 0.0
            selection_entropy = 0.0
            reliability_total=window_entropy=window_max_weight=0.0
            clean_metric_total=drift_metric_total=drift_margin_total=0.0
            fusion_branch_total=fusion_alpha_total=fusion_alpha_square_total=0.0
            modulation_means=[];modulation_stds=[]
            for batch in loader:
                tasks = None
                if encoder_config.requires_task:
                    *batch, tasks = batch
                    tasks = tasks.to(device)
                if encoder_config.quality_aware:
                    x, mask, quality, identities, sessions = batch
                    quality = quality.to(device)
                else:
                    x, mask, identities, sessions = batch
                    quality = None
                x, mask = x.to(device), mask.to(device)
                identities, sessions = identities.to(device), sessions.to(device)
                optimizer.zero_grad(set_to_none=True)
                pooling=None
                if train_config.modulation_reference:
                    z,modulation=encoder(x,mask,quality,return_modulation=True)
                    groups=modulation['groups'].detach()
                    modulation_means.append(groups.mean(0).cpu())
                    modulation_stds.append(groups.std(0,unbiased=False).cpu())
                elif encoder_config.record_fusion!='none':
                    z,branches=encoder(x,mask,quality,return_branches=True)
                elif encoder_config.window_pooling!='quality':
                    z,pooling=encoder(x,mask,quality,tasks=tasks,return_pooling=True)
                    window_entropy+=float(-(pooling['weights'].detach()*pooling['weights'].detach().clamp_min(1e-12).log()).sum(-1).mean())
                    window_max_weight+=float(pooling['weights'].detach().max(-1).values.mean())
                elif encoder_config.selection_mode != 'none':
                    z, selection = encoder(x, mask, quality, tasks=tasks, return_selection=True)
                    selection_entropy += float(selection['entropy'].detach())
                elif encoder_config.channel_gate != 'none':
                    z, gate = encoder(x, mask, quality, tasks=tasks, return_gate=True)
                else:
                    z = encoder(x, mask, quality) if quality is not None else encoder(x, mask)
                if train_config.metric_loss == 'supcon':
                    metric = supervised_contrastive_loss(
                        z, identities, sessions, train_config.temperature, train_config.cross_session,
                    )
                else:
                    metric = pairwise_contrastive_loss(
                        z, identities, sessions, train_config.pair_margin, train_config.cross_session,
                    )
                drift_margin=z.new_zeros(())
                if train_config.local_power_mode!='none':
                    from .local_power_drift import random_local_drift,preserve_identity_margin
                    from .quality import QualityConfig,quality_weights
                    shifted,_=random_local_drift(x,quality,data_config.channel_names,data_config.sample_rate,encoder_config.patch_samples)
                    hard=(quality>0).repeat_interleave(encoder_config.patch_samples,-1)
                    shifted_quality=quality_weights(shifted*data_config.signal_scale_uv,hard,QualityConfig(**data_config.quality))
                    shifted_z=encoder(shifted,mask,shifted_quality)
                    shifted_metric=supervised_contrastive_loss(shifted_z,identities,sessions,train_config.temperature,True)
                    clean_metric_total+=float(metric.detach());drift_metric_total+=float(shifted_metric.detach())
                    metric=.5*(metric+shifted_metric)
                    if train_config.local_power_mode=='margin':
                        drift_margin=preserve_identity_margin(z,shifted_z,identities,sessions)
                        drift_margin_total+=float(drift_margin.detach())
                ce = F.cross_entropy(16 * classifier(z), identities) if train_config.ce_weight else z.new_zeros(())
                loss = metric + train_config.ce_weight * ce+train_config.local_power_margin_weight*drift_margin
                if encoder_config.record_fusion!='none':
                    branch_loss=supervised_contrastive_loss(branches['log'],identities,sessions,
                        train_config.temperature,train_config.cross_session)
                    loss=loss+train_config.fusion_branch_weight*branch_loss
                    fusion_branch_total+=float(branch_loss.detach())
                    fusion_alpha_total+=float(branches['alpha'].detach().mean())
                    fusion_alpha_square_total+=float(branches['alpha'].detach().square().mean())
                if encoder_config.channel_gate != 'none':
                    loss = loss + train_config.gate_regularization_weight * gate['regularizer']
                    weights = gate['weights'].detach()[gate['active']]
                    gate_total += float(gate['regularizer'].detach())
                    gate_min += float(weights.min())
                    gate_max += float(weights.max())
                    gate_saturation += float(((weights < .51) | (weights > 1.49)).float().mean())
                consistency = z.new_zeros(())
                if train_config.consistency_weight:
                    from .quality import QualityConfig, corrupt_view
                    noisy, noisy_quality = corrupt_view(x, quality, QualityConfig(**data_config.quality),
                        data_config.signal_scale_uv, train_config.corruption_probability,
                        train_config.corruption_channel_fraction, train_config.corruption_noise_uv)
                    if train_config.window_reliability_weight:
                        noisy_z,noisy_pooling=encoder(noisy,mask,noisy_quality,tasks=tasks,return_pooling=True)
                        from .window_pooling import reliability_constraint
                        reliability,_=reliability_constraint(pooling,noisy_pooling,
                            encoder_config.window_residual_mix,encoder_config.window_multiplier_bound,
                            train_config.window_reliability_temperature)
                        loss=loss+train_config.window_reliability_weight*reliability
                        reliability_total+=float(reliability.detach())
                    else:
                        noisy_z = encoder(noisy, mask, noisy_quality, tasks=tasks)
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
            if train_config.modulation_reference:
                entry.update(modulation_group_mean=torch.stack(modulation_means).mean(0).tolist(),
                    modulation_within_batch_std=torch.stack(modulation_stds).mean(0).tolist())
            if encoder_config.record_fusion!='none':
                entry.update(fusion_branch_loss=fusion_branch_total/len(loader),
                    fusion_alpha_mean=fusion_alpha_total/len(loader),
                    fusion_alpha_std=math.sqrt(max(0.,fusion_alpha_square_total/len(loader)-(fusion_alpha_total/len(loader))**2)))
            if encoder_config.selection_mode != 'none':
                entry.update(selection_entropy=selection_entropy / len(loader),
                             selection_temperature=float(encoder.selector.temperature),
                             selection_noise_scale=(float(encoder.selector.noise_scale) if encoder_config.selection_mode in ('learned_global','learned_task') else 0.),
                             selected_channels=encoder_config.selected_channels)
            if train_config.consistency_weight:
                entry['consistency_loss'] = consistency_total / len(loader)
            if encoder_config.window_pooling!='quality':
                entry.update(window_entropy=window_entropy/len(loader),window_max_weight=window_max_weight/len(loader))
            if train_config.window_reliability_weight:
                entry['window_reliability_loss']=reliability_total/len(loader)
            if train_config.local_power_mode!='none':
                entry.update(clean_metric_loss=clean_metric_total/len(loader),local_power_metric_loss=drift_metric_total/len(loader))
            if train_config.local_power_mode=='margin':entry['local_power_margin_loss']=drift_margin_total/len(loader)
            if encoder_config.channel_gate != 'none':
                entry.update(gate_regularizer=gate_total / len(loader),
                             gate_mean_batch_min=gate_min / len(loader),
                             gate_mean_batch_max=gate_max / len(loader),
                             gate_saturation_fraction=gate_saturation / len(loader))
            improved = not episodes
            if episodes and ((epoch + 1) % train_config.eval_every == 0 or epoch + 1 == train_config.epochs):
                from .evaluation import evaluate_development
                encoder.eval()
                evaluator = dev_evaluator or evaluate_development
                development = (evaluator(encoder, data_config, episodes, device, closed_set=True)
                               if dev_closed_set else evaluator(encoder, data_config, episodes, device))
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
            with output.with_suffix('.timing.jsonl').open('a',encoding='utf-8') as timing:
                timing.write(json.dumps(dict(epoch=epoch+1,seconds=perf_counter()-epoch_started))+'\n')
            if stop_reason:
                print(f"early_stop epoch={epoch+1} reason={stop_reason} stale_evaluations={stopping['stale_evaluations']}", flush=True)
                break
    if not output.exists():
        raise ValueError('No evaluated checkpoint was selected')
    # A resumed run into a new output path must retain the historical best.
    if episodes and best_snapshot is not None:
        _atomic_save(best_snapshot, output)
    return torch.load(output, map_location='cpu', weights_only=True)
