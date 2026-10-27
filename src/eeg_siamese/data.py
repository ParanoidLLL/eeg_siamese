import csv
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class DataConfig:
    sample_rate: float
    channel_names: tuple[str, ...]
    window_seconds: float = 5.0
    block_windows: int = 6
    signal_scale_uv: float = 100.0
    demean: bool = True
    preprocessing_id: str | None = None
    min_valid_windows: int = 1
    min_valid_fraction: float = 0.0
    exclude_suspect_windows: bool = False
    window_qc_reference: dict | None = None
    quality: dict | None = None

    def __post_init__(self):
        if self.quality is not None:
            from .quality import QualityConfig
            policy = QualityConfig(**self.quality)
            if self.window_samples % policy.patch_samples:
                raise ValueError('Quality patches must divide window_samples')
            if self.exclude_suspect_windows:
                raise ValueError('Quality mode retains mild artifacts; do not exclude suspect windows')
            object.__setattr__(self, 'quality', asdict(policy))
        object.__setattr__(self, "channel_names", tuple(self.channel_names))
        if not self.channel_names or len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Provide unique channel names in the expected order")
        if not all(isinstance(name, str) and name for name in self.channel_names):
            raise ValueError("Channel names must be nonempty strings")
        values = (self.sample_rate, self.window_seconds, self.signal_scale_uv)
        if not all(math.isfinite(v) and v > 0 for v in values) or self.block_windows < 1:
            raise ValueError("Sampling, window, block and scale values must be positive")
        exact = self.sample_rate * self.window_seconds
        if not math.isclose(exact, round(exact), abs_tol=1e-6) or round(exact) < 4:
            raise ValueError("Window duration must produce an integer number of >=4 samples")
        if not isinstance(self.min_valid_windows, int) or not 1 <= self.min_valid_windows <= self.block_windows:
            raise ValueError("min_valid_windows must be within block length")
        if not math.isfinite(self.min_valid_fraction) or not 0 <= self.min_valid_fraction <= 1:
            raise ValueError("min_valid_fraction must be in [0,1]")
        if self.preprocessing_id is not None and (not isinstance(self.preprocessing_id, str) or not self.preprocessing_id):
            raise ValueError("preprocessing_id must be a nonempty string")
        if not isinstance(self.exclude_suspect_windows, bool):
            raise ValueError("exclude_suspect_windows must be boolean")
        if self.exclude_suspect_windows and self.window_qc_reference is None:
            raise ValueError("Exclusion requires a training-fitted window QC reference")
        if self.window_qc_reference is not None:
            from .window_qc import validate_reference
            validate_reference(self.window_qc_reference, self.channel_names, self.sample_rate,
                               self.window_seconds, self.preprocessing_id, training_only=True)

    @property
    def window_samples(self):
        return round(self.sample_rate * self.window_seconds)


@dataclass(frozen=True)
class Record:
    subject_id: str
    session_id: str
    record_id: str
    path: Path
    task_id: str | None = None
    run_id: str | None = None

    def metadata(self):
        result = dict(subject_id=self.subject_id, session_id=self.session_id,
                      record_id=self.record_id, path=str(self.path))
        if self.task_id is not None:
            result['task_id'] = self.task_id
        if self.run_id is not None:
            result['run_id'] = self.run_id
        return result


def read_manifest(path, allow_missing=False):
    path = Path(path).resolve()
    records = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"subject_id", "session_id", "record_id", "path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires {sorted(required)}")
        for line, row in enumerate(reader, 2):
            if any(not (row.get(key) or "").strip() for key in required):
                raise ValueError(f"Empty metadata on manifest line {line}")
            values = {key: row[key].strip() for key in required}
            source = Path(values["path"])
            source = (path.parent / source).resolve() if not source.is_absolute() else source.resolve()
            if not allow_missing and not source.is_file():
                raise ValueError(f"Missing EEG file: {source}")
            if values["subject_id"] == "Unknown":
                raise ValueError("Use actual identity IDs; Unknown is reserved for predictions")
            records.append(Record(values["subject_id"], values["session_id"], values["record_id"], source,
                                  (row.get('task_id') or '').strip() or None,
                                  (row.get('run_id') or '').strip() or None))
    if not records:
        raise ValueError("Manifest is empty")
    if len({r.record_id for r in records}) != len(records):
        raise ValueError("record_id must be unique within a manifest")
    if len({r.path for r in records}) != len(records):
        raise ValueError("Each physical EEG file can occur only once in a manifest")
    return records


def load_windows(record, config):
    """Strict adapter boundary; filtering/resampling belongs to source adapters."""
    with np.load(record.path, allow_pickle=False) as archive:
        if "requires_event_windowing" in archive and bool(archive["requires_event_windowing"].item()):
            raise ValueError(f"{record.record_id}: continuous event-preserving data requires event-aware windowing first")
        required = {"eeg", "sample_rate", "channel_names", "unit"}
        if not required.issubset(archive.files):
            raise ValueError(f"{record.path}: missing NPZ metadata {required - set(archive.files)}")
        if not math.isclose(float(archive["sample_rate"].item()), config.sample_rate):
            raise ValueError(f"{record.record_id}: sampling rate does not match config")
        if tuple(archive["channel_names"].tolist()) != config.channel_names:
            raise ValueError(f"{record.record_id}: channel names/order do not match config")
        if str(archive["unit"].item()) != "uV":
            raise ValueError(f"{record.record_id}: convert amplitude units to uV first")
        actual_id = str(archive["preprocessing_id"].item()) if "preprocessing_id" in archive else None
        if actual_id != config.preprocessing_id:
            raise ValueError(f"{record.record_id}: preprocessing_id does not match config")
        source_eeg = archive["eeg"]
        if source_eeg.dtype.kind not in "fiu":
            raise ValueError(f"{record.record_id}: EEG must be a real numeric array")
        eeg = np.asarray(source_eeg, dtype=np.float32)
        if eeg.ndim != 2 or eeg.shape[0] != len(config.channel_names):
            raise ValueError(f"{record.record_id}: expected EEG [channels,samples]")
        count = eeg.shape[1] // config.window_samples
        if 'trial_samples' in archive:
            trial_samples = int(archive['trial_samples'].item())
            if config.window_samples != trial_samples or eeg.shape[1] % trial_samples:
                raise ValueError(f'{record.record_id}: windows must preserve stored trial boundaries')
        if count == 0:
            raise ValueError(f"{record.record_id}: shorter than one complete window")
        windows = eeg[:, :count * config.window_samples].reshape(
            len(config.channel_names), count, config.window_samples,
        ).transpose(1, 0, 2).copy()
        sample_mask = None
        if config.quality is not None:
            if 'valid_channel_samples' not in archive:
                raise ValueError(f'{record.record_id}: quality mode requires fresh channel_sample preprocessing masks')
            supplied_samples = archive['valid_channel_samples']
            if supplied_samples.dtype != np.bool_ or supplied_samples.shape != eeg.shape:
                raise ValueError('Invalid per-channel sample mask shape/type')
            sample_mask = supplied_samples[:, :count * config.window_samples].reshape(
                len(config.channel_names), count, config.window_samples).transpose(1, 0, 2).copy()
            if not np.isfinite(windows[sample_mask]).all():
                raise ValueError('Non-finite EEG in a trusted channel sample')
            windows[~sample_mask] = 0
            valid = sample_mask.any(axis=(1, 2))
        else:
            valid = np.isfinite(windows).all(axis=(1, 2))
            if 'valid_channel_samples' in archive:
                conservative = archive['valid_channel_samples']
                if conservative.dtype != np.bool_ or conservative.shape != eeg.shape:
                    raise ValueError('Invalid per-channel sample mask shape/type')
                # Legacy encoders cannot interpret local missing cells. Never
                # present their zero placeholders as a fully trusted window.
                valid &= conservative[:, :count * config.window_samples].reshape(
                    len(config.channel_names), count, config.window_samples).all(axis=(0,2))
        if "valid_windows" in archive:
            supplied = archive["valid_windows"]
            if supplied.dtype != np.bool_ or supplied.shape != (count,):
                raise ValueError(f"{record.record_id}: valid_windows must be boolean [{count}]")
            valid &= supplied
        if config.exclude_suspect_windows:
            from .window_qc import measure_windows, score_windows
            metrics = measure_windows(eeg, valid, config.window_samples)
            _, suspect = score_windows(metrics, config.window_qc_reference)
            valid &= ~suspect
    if not valid.any():
        raise ValueError(f"{record.record_id}: no valid windows")
    windows[~valid] = 0
    if sample_mask is not None:
        sample_mask[~valid] = False
        if config.demean:
            counts = sample_mask.sum(-1, keepdims=True)
            means = np.divide(windows.sum(-1, keepdims=True), counts,
                              out=np.zeros_like(windows[..., :1]), where=counts > 0)
            windows = np.where(sample_mask, windows - means, 0)
        from .quality import QualityConfig, quality_weights
        quality = quality_weights(torch.from_numpy(windows), torch.from_numpy(sample_mask), QualityConfig(**config.quality))
        valid &= quality.sum(dim=(1,2)).numpy()>0
        quality[~torch.from_numpy(valid)] = 0
        windows[~valid] = 0
    elif config.demean:
        windows -= windows.mean(axis=-1, keepdims=True)
    windows /= config.signal_scale_uv
    if not np.isfinite(windows).all():
        raise ValueError(f"{record.record_id}: non-finite preprocessed values")
    if not valid.any():
        raise ValueError(f'{record.record_id}: no usable electrode-time patches')
    tensors = (torch.from_numpy(windows), torch.from_numpy(valid))
    return (*tensors, quality) if sample_mask is not None else tensors


def block_is_usable(valid, config):
    """Apply a shared duration rule to real windows, before padding.

    A short final block needs all available windows if shorter than the absolute
    minimum; the fraction denominator never includes fictitious padded windows.
    """
    count = len(valid)
    return count > 0 and int(valid.sum()) >= min(config.min_valid_windows, count) and float(valid.float().mean()) >= config.min_valid_fraction


def padded_block(windows, valid, start, block_windows):
    x = windows.new_zeros((block_windows, *windows.shape[1:]))
    mask = torch.zeros(block_windows, dtype=torch.bool)
    length = min(block_windows, len(windows) - start)
    x[:length] = windows[start:start + length]
    mask[:length] = valid[start:start + length]
    return x, mask


def pad_quality(quality, start, block_windows):
    result = quality.new_zeros((block_windows, *quality.shape[1:]))
    length = min(block_windows, len(quality) - start)
    result[:length] = quality[start:start+length]
    return result


class RecordingDataset(Dataset):
    def __init__(self, records, config, cache=False, cache_max_bytes=512 * 1024 * 1024, task_names=()):
        self.records = records
        self.config = config
        self.identities = {s: i for i, s in enumerate(sorted({r.subject_id for r in records}))}
        self.sessions = {s: i for i, s in enumerate(sorted({r.session_id for r in records}))}
        self.task_names = tuple(task_names)
        if self.task_names and any(r.task_id not in self.task_names for r in records):
            raise ValueError('Every training record requires a known explicit task_id')
        self.cache = cache
        self.cache_max_bytes = cache_max_bytes
        self._cache = OrderedDict()
        self._cache_bytes = 0

    def load(self, index):
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        loaded = load_windows(self.records[index], self.config)
        if self.cache:
            size = sum(t.numel() * t.element_size() for t in loaded)
            while self._cache and self._cache_bytes + size > self.cache_max_bytes:
                _, removed = self._cache.popitem(last=False)
                self._cache_bytes -= sum(t.numel() * t.element_size() for t in removed)
            if size <= self.cache_max_bytes:
                self._cache[index] = loaded
                self._cache_bytes += size
        return loaded

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        loaded = self.load(index)
        windows, valid = loaded[:2]
        k = self.config.block_windows
        starts = torch.arange(max(1, len(windows) - k + 1))
        prefix = torch.cat([torch.zeros(1, dtype=torch.long), valid.long().cumsum(0)])
        ends = (starts + k).clamp_max(len(windows))
        valid_counts = prefix[ends] - prefix[starts]
        required = torch.minimum(torch.full_like(ends, self.config.min_valid_windows), ends - starts)
        eligible = starts[(valid_counts >= required) & (valid_counts / (ends - starts) >= self.config.min_valid_fraction)]
        if not len(eligible):
            raise ValueError(f"{record.record_id}: no usable block under duration rules")
        start = int(eligible[torch.randint(len(eligible), ())])
        x, mask = padded_block(windows, valid, start, k)
        labels = (self.identities[record.subject_id], self.sessions[record.session_id])
        if self.task_names:
            labels += (self.task_names.index(record.task_id),)
        if len(loaded) == 3:
            q = pad_quality(loaded[2], start, k)
            return x, mask, q, *labels
        return x, mask, *labels


class SessionBatchSampler(Sampler):
    def __init__(self, records, identities_per_batch, samples_per_identity,
                 steps, seed=42, cross_session=True, same_session_batch=False):
        self.groups = {}
        for index, record in enumerate(records):
            self.groups.setdefault(record.subject_id, {}).setdefault(record.session_id, []).append(index)
        self.p = identities_per_batch
        self.k = samples_per_identity
        self.steps = steps
        self.seed = seed
        self.epoch = 0
        self.same_session_batch = same_session_batch
        if same_session_batch:
            if cross_session:
                raise ValueError('Same-session batches cannot require cross-session positives')
            common = set.intersection(*(set(g) for g in self.groups.values()))
            self.common_sessions = sorted(s for s in common if all(len(g[s]) >= self.k for g in self.groups.values()))
            if not self.common_sessions:
                raise ValueError('Need a shared task with enough distinct records per identity')
        if not 2 <= self.p <= len(self.groups) or self.k < 2 or steps < 1:
            raise ValueError("Batch requires >=2 identities, >=2 samples each and positive steps")
        if cross_session and any(len(sessions) < 2 for sessions in self.groups.values()):
            raise ValueError("Cross-session training requires >=2 sessions for every training identity")

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        identities = list(self.groups)
        for _ in range(self.steps):
            batch = []
            if self.same_session_batch:
                session = str(rng.choice(self.common_sessions))
                for identity in rng.choice(identities, self.p, replace=False):
                    batch.extend(int(i) for i in rng.choice(self.groups[identity][session], self.k, replace=False))
                yield batch
                continue
            for identity in rng.choice(identities, self.p, replace=False):
                sessions = list(self.groups[identity])
                rng.shuffle(sessions)
                for sample in range(self.k):
                    choices = self.groups[identity][sessions[sample % len(sessions)]]
                    batch.append(int(rng.choice(choices)))
            yield batch


def assert_records_disjoint(*groups):
    for i, left in enumerate(groups):
        for right in groups[i + 1:]:
            if {r.record_id for r in left} & {r.record_id for r in right}:
                raise ValueError("Record IDs overlap across data roles; split before windowing")
            if {r.path for r in left} & {r.path for r in right}:
                raise ValueError("EEG file paths overlap across data roles")


def audit_episode(training, enrollment, queries, unseen=True):
    assert_records_disjoint(training, enrollment, queries)
    seen = {r.subject_id for r in training}
    episode = {r.subject_id for r in enrollment + queries}
    if unseen and seen & episode:
        raise ValueError("Unseen-identity episode contains encoder-training identities")
    registered = {r.subject_id for r in enrollment}
    for identity in registered:
        earlier = {r.session_id for r in enrollment if r.subject_id == identity}
        later = {r.session_id for r in queries if r.subject_id == identity}
        if earlier & later:
            raise ValueError(f"{identity}: enrollment/query sessions overlap")
