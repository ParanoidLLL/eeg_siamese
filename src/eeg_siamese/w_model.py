"""Portable W recipe and prediction-equivalent single-encoder export."""
from dataclasses import asdict, replace
from pathlib import Path
import copy
import math

import numpy as np
import torch

from .data import DataConfig, read_manifest, load_windows
from .inference import checkpoint_id, load_encoder, read_json, write_json
from .model import EEGEncoder, EncoderConfig
from .spectral_residual import window_band_power
from .training import train, select_device


@torch.no_grad()
def fit_training_statistics(records, data_config):
    """Match W's record-balanced estimator; never fit on enrollment/query data."""
    if data_config.window_seconds not in (4., 5.):
        raise ValueError('W requires an explicit four/five-second window protocol')
    if data_config.quality is None:
        raise ValueError('W requires the original quality-aware waveform path')
    means, provenance = [], []
    for record in records:
        x, valid, quality = load_windows(record, data_config)
        good = (quality > 0).all(-1) & valid[:, None]
        count = good.sum(0)
        if not (count > 0).all():
            raise ValueError(f'{record.record_id}: each channel needs a complete valid spectral window')
        total = torch.zeros(len(data_config.channel_names), 5, dtype=torch.float64)
        for start in range(0, len(x), 32):
            power = window_band_power(x[start:start+32].double() * data_config.signal_scale_uv,
                                      data_config.sample_rate, data_config.window_seconds)
            total += (power.clamp_min(1e-12).log() * good[start:start+32, :, None]).sum(0)
        means.append((total / count[:, None]).flatten().numpy())
        provenance.append(dict(record.metadata(), sha256=checkpoint_id(record.path)))
    if not means:
        raise ValueError('No training records')
    matrix = np.stack(means)
    mean, scale = matrix.mean(0), matrix.std(0)
    constant = scale < 1e-8
    scale[constant] = 1.
    return dict(mean=mean.tolist(), scale=scale.tolist(), records=provenance,
                constant_features=int(constant.sum()), fit='training-record mean log-power, then mean/std across records',
                bands_hz=[[1,4],[4,8],[8,13],[13,30],[30,40]], target_data_used=False)


def export_w_checkpoint(source, output):
    """Remove W's unused frozen Q and optimizer; retain origin/provenance."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Export output exists; use a new path')
    if source.resolve() == output.resolve():
        raise ValueError('Export cannot replace its source')
    original = torch.load(source, map_location='cpu', weights_only=True)
    config = EncoderConfig(**original['encoder_config'])
    if config.spectral_fusion != 'log' or config.spectral_modulation != 'none':
        raise ValueError('Expected the unmodulated W log-residual model')
    if config.record_fusion == 'single':
        config = replace(config, record_fusion='none')
        state = {k.removeprefix('record_fuser.log.'):v for k,v in original['encoder_state'].items()
                 if k.startswith('record_fuser.log.')}
    elif config.record_fusion == 'none':
        state = original['encoder_state']
    else:
        raise ValueError('A fused F/D model cannot be exported as W')
    # Validate every key/shape before writing a self-contained artifact.
    with torch.random.fork_rng(devices=[]):
        probe = EEGEncoder(config)
    probe.load_state_dict(state, strict=True)
    retained = ('schema_version','data_config','training_config','training_records',
                'development_records','development_subject_ids','identity_labels',
                'run_provenance','development_protocol','completed_epochs')
    result = {key:original[key] for key in retained if key in original}
    result.update(encoder_config=asdict(config), encoder_state=state,
                  derived_from=dict(source=str(source.resolve()), source_sha256=checkpoint_id(source),
                                    operation='Prediction-equivalent active W encoder; optimizer and unused Q omitted'))
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    # Check inference metadata as well as state shape compatibility.
    load_encoder(output, 'cpu')
    return dict(output=str(output), model_sha256=checkpoint_id(output), source_sha256=checkpoint_id(source))


def train_w(manifest, config_path, output_dir, device='auto', warmup_epochs=None, control=False):
    """Q warmup -> matched W continuation; optional equal-budget U control.

    Only training signals are read. No enrollment, query, threshold fitting, or
    held-out selection happens in this command. Outputs use fixed endpoints.
    """
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory; existing runs are preserved')
    config = read_json(config_path)
    dc = DataConfig(**config['data'])
    records = read_manifest(manifest)
    epochs = config.get('training', {}).get('epochs', 35)
    warmup_epochs = epochs if warmup_epochs is None else warmup_epochs
    if not isinstance(warmup_epochs, int) or isinstance(warmup_epochs, bool) or warmup_epochs < 1:
        raise ValueError('warmup_epochs must be a positive integer')
    options = dict(config['encoder'])
    options.update(channels=len(dc.channel_names), spectral_fusion='none',
                   spectral_feature_mean=[], spectral_feature_scale=[],
                   sample_rate=dc.sample_rate, spectral_input_scale_uv=dc.signal_scale_uv,
                   spectral_window_seconds=dc.window_seconds, record_fusion='none')
    baseline = EncoderConfig(**options)
    if baseline.channel_gate != 'none' or baseline.selection_mode != 'none' or baseline.window_pooling != 'quality' or baseline.spectral_modulation != 'none':
        raise ValueError('W recipe requires the original full-channel quality pooling model')
    policy = config.get('training', {})
    if policy.get('early_stopping_evaluations', 0) or policy.get('ce_weight', 0) or policy.get('metric_loss', 'supcon') != 'supcon':
        raise ValueError('W recipe uses fixed endpoints and SupCon without classifier loss')
    if not math.isclose(policy.get('consistency_weight', .5), .5):
        raise ValueError('W recipe preserves consistency_weight=0.5')
    if policy.get('fusion_reference') or policy.get('modulation_reference'):
        raise ValueError('The W recipe creates its own matched Q reference')
    if policy.get('local_power_mode', 'none') != 'none' or policy.get('window_reliability_weight', 0):
        raise ValueError('The W recipe does not add experimental auxiliary objectives')
    stats = fit_training_statistics(records, dc)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'spectral_statistics.json', stats)
    qconfig = copy.deepcopy(config)
    options.pop('channels')
    qconfig['encoder'] = options
    qconfig.setdefault('training', {}).update(epochs=warmup_epochs, early_stopping_evaluations=0,
        fusion_branch_weight=0., ce_weight=0., metric_loss='supcon', consistency_weight=.5)
    write_json(output/'Q/config.json', qconfig)
    device = select_device(device)
    train(records, output/'Q/config.json', output/'Q/model.pt', device)
    qhash = checkpoint_id(output/'Q/model.pt')
    for arm in (['U', 'W'] if control else ['W']):
        cfg = copy.deepcopy(qconfig)
        cfg['encoder'].update(record_fusion='single', spectral_fusion='log' if arm=='W' else 'none')
        if arm=='W':
            cfg['encoder'].update(spectral_feature_mean=stats['mean'], spectral_feature_scale=stats['scale'])
        cfg['training'].update(epochs=epochs, fusion_reference=str(output/'Q/model.pt'), fusion_reference_sha256=qhash)
        write_json(output/f'{arm}/config.json', cfg)
        train(records, output/f'{arm}/config.json', output/f'{arm}/model.pt', device)
    exported = export_w_checkpoint(output/'W/model.pt', output/'model.pt')
    write_json(output/'COMPLETE.json', dict(model='W', fixed_endpoint=True, training_records=len(records),
               warmup_epochs=warmup_epochs, continuation_epochs=epochs, control_trained=control,
               evaluated=False, config_sha256=checkpoint_id(config_path),
               manifest_sha256=checkpoint_id(manifest), **exported))
    return read_json(output/'COMPLETE.json')
