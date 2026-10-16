"""Local OpenBMI adapters and identity-safe experimental protocols.

The MAT layout follows the official MOABB Lee2019 reader:
https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/Lee2019.py
No EEG downloads, pickle loading, task-label-to-identity conversion, or window
splitting are performed here. OpenBMI's train/test phases share one session.
"""

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt

from .data import read_manifest


OPENBMI_READER_VERSION = "openbmi-mi-v1"
OPENBMI_FILE = re.compile(r"sess(\d{2})_subj(\d{2})_EEG_MI\.mat$")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_openbmi(directory):
    """Find standard MI files; return (path, subject, independent session)."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"OpenBMI directory does not exist: {directory}")
    discovered = []
    identities = set()
    for path in sorted(directory.rglob("*_EEG_MI.mat")):
        match = OPENBMI_FILE.fullmatch(path.name)
        if not match:
            raise ValueError(f"Nonstandard OpenBMI filename: {path.name}")
        session, subject = map(int, match.groups())
        if (subject, session) in identities:
            raise ValueError(f"Duplicate OpenBMI file for subject {subject}, session {session}")
        identities.add((subject, session))
        discovered.append((path.resolve(), subject, session))
    if not discovered:
        raise ValueError("No sessXX_subjXX_EEG_MI.mat files found")
    return discovered


def _numeric(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric data")
    return array


def _channels(value):
    names = []
    for entry in np.asarray(value).ravel():
        while isinstance(entry, np.ndarray) and entry.size == 1:
            entry = entry.item()
        if isinstance(entry, np.ndarray) and entry.dtype.kind in "US":
            entry = "".join(entry.ravel().tolist())
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError("chan must contain nonempty channel names; no fallback montage is guessed")
        names.append(entry.strip())
    if not names or len(set(names)) != len(names):
        raise ValueError("chan requires unique EEG channel names")
    if any(name.upper().startswith("EMG") for name in names):
        raise ValueError("x/chan must contain EEG only; EMG is not an EEG or EOG channel")
    return tuple(names)


def _events(fields, samples, event_index_base):
    if "t" not in fields or np.asarray(fields["t"]).size == 0:
        if "y_dec" in fields and np.asarray(fields["y_dec"]).size:
            raise ValueError("Event labels supplied without event sample times")
        return np.empty((0, 2), dtype=np.int64), [], False
    times = _numeric(fields["t"], "t").ravel()
    if not np.isfinite(times).all() or not np.equal(times, np.floor(times)).all():
        raise ValueError("t must contain integer sample indices")
    times = times.astype(np.int64)
    positions = times - event_index_base
    if np.any(positions < 0) or np.any(positions >= samples) or np.any(np.diff(positions) < 0):
        raise ValueError("Event sample positions are outside x or not sorted; check event_index_base")
    available = "y_dec" in fields and np.asarray(fields["y_dec"]).size > 0
    labels = _numeric(fields["y_dec"], "y_dec").ravel() if available else np.zeros(len(times))
    if (len(labels) != len(times) or not np.isfinite(labels).all()
            or not np.equal(labels, np.floor(labels)).all() or np.any(labels < 0)):
        raise ValueError("y_dec must contain one nonnegative integer per event (or be absent)")
    return np.column_stack([positions, labels.astype(np.int64)]), times.tolist(), available


def read_openbmi(path, subject_id, session_id, input_unit, *, event_index_base=0):
    """Read classic MAT continuous MI phases into separate record dictionaries.

    x is strictly [samples, EEG channels] with its order supplied by chan.
    Input units are explicit (uV or V). Source t uses the MOABB convention of
    direct 0-based sample positions by default; use event_index_base=1 only
    for an explicitly documented MATLAB one-based conversion. Original t
    values and the chosen convention are retained for inspection. Unlabelled
    online events receive code 0; identity always comes from subject_id.
    """
    path = Path(path).resolve()
    if (not isinstance(subject_id, (int, np.integer)) or not 1 <= subject_id <= 54
            or not isinstance(session_id, (int, np.integer)) or session_id not in (1, 2)):
        raise ValueError("OpenBMI requires integer subject_id 1..54 and session_id 1 or 2")
    if input_unit not in ("uV", "V"):
        raise ValueError("Specify input_unit as uV or V; amplitude units are never guessed")
    if event_index_base not in (0, 1):
        raise ValueError("event_index_base must be 0 or 1")
    filename = OPENBMI_FILE.fullmatch(path.name)
    if filename and tuple(map(int, filename.groups())) != (session_id, subject_id):
        raise ValueError("Supplied subject/session do not match the official MAT filename")
    with path.open("rb") as stream:
        header = stream.read(1024)
    if header.startswith(b"\x89HDF\r\n\x1a\n") or b"MATLAB 7.3 MAT-file" in header:
        raise ValueError("MAT v7.3/HDF5 is unsupported by this reader. Convert to classic MAT "
                         "with MATLAB or provide a reviewed h5py adapter; do not rename the file.")
    try:
        mat = loadmat(path, squeeze_me=False, struct_as_record=True,
                      variable_names=["EEG_MI_train", "EEG_MI_test"])
    except NotImplementedError as error:
        raise ValueError("MAT v7.3 requires a separate reviewed HDF5 adapter or classic MAT conversion") from error
    checksum = file_sha256(path)
    records = []
    for phase in ("train", "test"):
        name = f"EEG_MI_{phase}"
        if name not in mat:
            raise ValueError(f"Missing official OpenBMI phase {name}")
        structure = mat[name]
        if not isinstance(structure, np.ndarray) or structure.size != 1 or not structure.dtype.names:
            raise ValueError(f"{name} must be a scalar MATLAB struct")
        fields = {field: structure.reshape(-1)[0][field] for field in structure.dtype.names}
        required = {"x", "fs", "chan"}
        if not required.issubset(fields):
            raise ValueError(f"{name} missing fields {sorted(required - fields.keys())}")
        fs_array = _numeric(fields["fs"], "fs")
        if fs_array.size != 1:
            raise ValueError("fs must be a scalar sampling rate")
        fs = float(fs_array.item())
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError("fs must be positive and finite")
        channels = _channels(fields["chan"])
        signal = _numeric(fields["x"], "x")
        if signal.ndim != 2 or signal.shape[1] != len(channels) or signal.shape[0] < 1:
            raise ValueError(f"{name}: x must be continuous [samples,{len(channels)}]; "
                             "epochs and transposed data are not silently repaired")
        events, original_times, labels_available = _events(fields, len(signal), event_index_base)
        subject = f"openbmi_S{int(subject_id):02d}"
        session = f"session_{int(session_id)}"
        provenance = {
            "dataset": "OpenBMI_MI", "dataset_doi": "10.5524/100542",
            "source_path": str(path), "source_sha256": checksum,
            "reader_version": OPENBMI_READER_VERSION, "input_unit": input_unit,
            "output_unit": "uV", "phase": phase,
            "phase_condition": "offline" if phase == "train" else "online_feedback",
            "event_index_base": event_index_base, "original_event_samples": original_times,
            "event_labels_available": labels_available, "unknown_event_code": 0,
            "session_is_collection_day": True,
        }
        records.append({
            "subject_id": subject, "session_id": session,
            "record_id": f"{subject}_{session}_MI_{phase}",
            "phase": phase, "eeg_uv": signal.T.astype(np.float64) * (1e6 if input_unit == "V" else 1),
            "sample_rate": fs, "channel_names": channels, "events": events,
            "provenance": provenance,
        })
    if (records[0]["channel_names"] != records[1]["channel_names"]
            or records[0]["sample_rate"] != records[1]["sample_rate"]):
        raise ValueError("OpenBMI phases disagree on sampling rate or channel order")
    return records


def _write_manifest(path, records):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "session_id", "record_id", "path"])
        writer.writeheader()
        writer.writerows(record.metadata() for record in records)


def split_manifest(manifest_path, output_dir, train_count, dev_count, eval_count,
                   seed=42, registered_count=3, *, episodes=5,
                   enrollment_session="session_1", query_session="session_2"):
    """Freeze subject-disjoint roles BEFORE any pretraining/window extraction.

    Each held-out episode chooses registered_count enrolled and two unknown
    people, with S1 enrollment and S2 queries. Repeated episodes are protocol
    repetitions (not independent populations); their overlap is recorded.
    """
    records = read_manifest(manifest_path)
    counts = (train_count, dev_count, eval_count, registered_count, episodes)
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in counts):
        raise ValueError("Split counts and episodes must be positive integers")
    if train_count < 2 or min(dev_count, eval_count) < registered_count + 2:
        raise ValueError("Need >=2 training identities and >=registered_count+2 held-out identities per role")
    if enrollment_session == query_session:
        raise ValueError("Enrollment and query must be independent sessions")
    subjects = sorted({record.subject_id for record in records})
    if sum((train_count, dev_count, eval_count)) > len(subjects):
        raise ValueError("Requested more identities than available in manifest")
    rng = np.random.default_rng(seed)
    ordered = [str(subject) for subject in rng.permutation(subjects)]
    roles = {"train": ordered[:train_count],
             "dev": ordered[train_count:train_count + dev_count],
             "eval": ordered[train_count + dev_count:train_count + dev_count + eval_count]}
    for subject in sum(roles.values(), []):
        sessions = {r.session_id for r in records if r.subject_id == subject}
        if not {enrollment_session, query_session}.issubset(sessions):
            raise ValueError(f"{subject} lacks {enrollment_session}/{query_session}; phases are not sessions")
    output_dir = Path(output_dir).resolve()
    # Refuse overwriting a previously frozen protocol without an explicit new path.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Split output directory must be new or empty; preserve previously frozen protocols")
    output_dir.mkdir(parents=True, exist_ok=True)
    for role, role_subjects in roles.items():
        _write_manifest(output_dir / f"{role}.csv", [r for r in records if r.subject_id in role_subjects])
    episode_metadata = {}
    for role in ("dev", "eval"):
        role_dir = output_dir / role
        role_dir.mkdir()
        role_episodes = []
        combinations = set()
        # Finite attempts bound repeated-combination handling for a five-person role.
        for number in range(episodes):
            for _ in range(100):
                choice = [str(subject) for subject in rng.choice(roles[role], registered_count + 2, replace=False)]
                registered = sorted(choice[:registered_count])
                unknown = sorted(choice[registered_count:])
                key = (tuple(registered), tuple(unknown))
                if key not in combinations:
                    break
            combinations.add(key)
            enrollment = [r for r in records if r.subject_id in registered and r.session_id == enrollment_session]
            queries = [r for r in records if r.subject_id in registered + unknown and r.session_id == query_session]
            enroll_path, query_path = (role_dir / f"episode_{number:03d}_{part}.csv" for part in ("enrollment", "query"))
            _write_manifest(enroll_path, enrollment)
            _write_manifest(query_path, queries)
            role_episodes.append({"enrollment": str(enroll_path), "query": str(query_path),
                                  "registered_subject_ids": registered, "unknown_subject_ids": unknown})
        (role_dir / "episodes.json").write_text(json.dumps(role_episodes, indent=2) + "\n", encoding="utf-8")
        episode_metadata[role] = str(role_dir / "episodes.json")
    metadata = {"version": "identity-split-v1", "seed": seed,
                "source_manifest": str(Path(manifest_path).resolve()),
                "source_manifest_sha256": file_sha256(manifest_path), "subjects": roles,
                "unused_subject_ids": ordered[sum((train_count, dev_count, eval_count)):],
                "enrollment_session": enrollment_session, "query_session": query_session,
                "registered_count": registered_count, "unknown_count": 2,
                "episodes": episode_metadata,
                "pretraining_policy": "Encoder pretraining may use train identities only. "
                "Dev/eval identities stay unseen across paradigms. Episodes can share identities."}
    (output_dir / "split.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def inject_noise(eeg_uv, sample_rate, *, kind="broadband", snr_db=10.0, seed=42,
                 line_hz=60.0, dropout_fraction=0.05):
    """Generate deterministic stress conditions, not measured neural SNR.

    Additive noise is scaled to the power of finite observed EEG (per channel,
    after removing its mean). Dropout has a duration fraction, not a dB level.
    Caller must use independent seeds for records/data roles to avoid coupling
    noise templates to identity. Existing invalid samples remain invalid.
    """
    eeg = np.asarray(eeg_uv)
    if eeg.dtype.kind not in "fiu" or eeg.ndim != 2 or min(eeg.shape) < 1:
        raise ValueError("eeg_uv must be a real numeric [channels,samples] array")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if kind not in {"broadband", "line", "burst", "drift", "dropout", "jump"}:
        raise ValueError("Unknown noise kind")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    signal = eeg.astype(np.float64, copy=True)
    rng = np.random.default_rng(seed)
    channels, samples = signal.shape
    valid = np.isfinite(signal)
    time = np.arange(samples) / sample_rate
    metadata = {"version": "observed-eeg-noise-v1", "kind": kind, "seed": int(seed),
                "snr_definition": "Nominal added-noise / observed mean-centred EEG power; "
                "not true neural SNR", "sample_rate": sample_rate}
    if kind == "dropout":
        if not 0 < dropout_fraction <= 1:
            raise ValueError("dropout_fraction must be in (0,1]")
        length = min(samples, max(1, round(samples * dropout_fraction)))
        channel = int(rng.integers(channels))
        start = int(rng.integers(samples - length + 1))
        signal[channel, start:start + length] = np.nan
        metadata.update(channel=channel, interval_samples=[start, start + length],
                        dropout_fraction=length / samples, nominal_snr_db=None)
        return signal, metadata
    if kind == "line":
        if not math.isfinite(line_hz) or not 0 < line_hz < sample_rate / 2:
            raise ValueError("line_hz must be below Nyquist and above zero")
        noise = np.sin(2 * np.pi * line_hz * time[None, :] + rng.uniform(0, 2 * np.pi, (channels, 1)))
        metadata["line_hz"] = line_hz
    elif kind == "drift":
        frequencies = rng.uniform(0.05, min(0.4, sample_rate / 4), (channels, 1))
        noise = np.sin(2 * np.pi * frequencies * time[None, :] + rng.uniform(0, 2 * np.pi, (channels, 1)))
    elif kind == "burst":
        if sample_rate <= 80:
            raise ValueError("High-frequency burst requires sample_rate >80 Hz")
        noise = rng.standard_normal(signal.shape)
        sos = butter(4, [30, min(100, sample_rate * 0.45)], btype="bandpass", fs=sample_rate, output="sos")
        noise = sosfilt(sos, noise, axis=-1)
        length = min(samples, max(1, round(samples * 0.1)))
        start = int(rng.integers(samples - length + 1))
        noise[:, :start] = 0
        noise[:, start + length:] = 0
        metadata["interval_samples"] = [start, start + length]
    elif kind == "jump":
        start = int(rng.integers(1, samples)) if samples > 1 else 0
        noise = np.zeros_like(signal)
        noise[:, start:] = rng.choice([-1.0, 1.0], (channels, 1))
        metadata["jump_sample"] = start
    else:
        noise = rng.standard_normal(signal.shape)
    powers, added_powers = [], []
    for channel in range(channels):
        usable = valid[channel]
        if not usable.any():
            raise ValueError("Cannot define added-noise strength for a channel with no finite signal")
        observed = signal[channel, usable]
        power = float(np.mean((observed - observed.mean()) ** 2))
        if power == 0 or not math.isfinite(power):
            raise ValueError("Cannot define SNR relative to a constant channel or non-finite power")
        noise_power = float(np.mean(noise[channel, usable] ** 2))
        if noise_power == 0:
            raise ValueError("Generated noise has no power on valid samples")
        try:
            factor = math.sqrt(power / noise_power) * 10 ** (-snr_db / 20)
        except OverflowError as error:
            raise ValueError("snr_db produces non-finite injected noise") from error
        if not math.isfinite(factor):
            raise ValueError("snr_db produces non-finite injected noise")
        delta = noise[channel] * factor
        signal[channel, usable] += delta[usable]
        powers.append(power)
        added_powers.append(float(np.mean(delta[usable] ** 2)))
    metadata.update(nominal_snr_db=float(snr_db), observed_power_uv2=powers,
                    added_noise_power_uv2=added_powers)
    return signal, metadata
