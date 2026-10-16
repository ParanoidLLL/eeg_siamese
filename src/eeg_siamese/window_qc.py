"""Amplitude diagnostics, separate from physical filtering and hard validity.

Scores use log peak-to-peak and log RMS, per channel. A record-local reference
is descriptive only. Optional exclusion requires a frozen training reference.
"""
import hashlib
import json
import math

import numpy as np


VERSION = "window-amplitude-qc-v1"
METRICS = ("peak_to_peak_uv", "rms_uv")


def measure_windows(eeg, valid, window_samples):
    """Return [windows, channels, metrics]; invalid windows never enter fitting."""
    eeg, valid = np.asarray(eeg), np.asarray(valid)
    if eeg.ndim != 2 or valid.dtype != np.bool_ or valid.shape != (eeg.shape[1] // window_samples,):
        raise ValueError("Window QC needs EEG [channels,samples] and a matching boolean mask")
    metrics = np.full((len(valid), eeg.shape[0], 2), np.nan)
    for index in np.flatnonzero(valid):
        block = np.asarray(eeg[:, index * window_samples:(index + 1) * window_samples], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Window marked valid contains non-finite samples")
        metrics[index, :, 0] = np.ptp(block, axis=-1)
        metrics[index, :, 1] = np.sqrt(np.mean(block ** 2, axis=-1))
    return metrics


def _weighted_median(values, weights):
    order = np.argsort(values, kind="stable")
    return float(values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]])


def reference_id(reference):
    payload = {k: v for k, v in reference.items() if k != "reference_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def fit_reference(metric_sets, channel_names, sample_rate, window_seconds, preprocessing_id,
                  *, subject_ids=None, score_threshold=6.0, min_log_scale=0.1,
                  min_reference_windows=20, scope="training", fitted_records=None):
    if scope not in {"training", "record_diagnostic"}:
        raise ValueError("Unknown window QC reference scope")
    if not all(math.isfinite(v) and v > 0 for v in (score_threshold, min_log_scale)):
        raise ValueError("Window QC thresholds must be positive and finite")
    if not isinstance(min_reference_windows, int) or min_reference_windows < 3:
        raise ValueError("min_reference_windows must be an integer >= 3")
    subject_ids = list(subject_ids) if subject_ids is not None else [str(i) for i in range(len(metric_sets))]
    if len(subject_ids) != len(metric_sets):
        raise ValueError("One subject ID is required per metric set")
    usable = []
    for subject, values in zip(subject_ids, metric_sets):
        values = np.asarray(values)
        if values.ndim != 3 or values.shape[1:] != (len(channel_names), 2):
            raise ValueError("Window QC metric/channel shape mismatch")
        values = values[np.isfinite(values).all(axis=(1, 2))]
        if len(values):
            usable.append((subject, values))
    count = sum(len(values) for _, values in usable)
    if count < min_reference_windows:
        raise ValueError(f"Need >= {min_reference_windows} valid reference windows; got {count}")
    # Equal identity mass, equal record mass within an identity; long recordings
    # must not dominate the median/MAD simply because they supply more windows.
    record_counts = {subject: sum(s == subject for s, _ in usable) for subject, _ in usable}
    weights = np.concatenate([np.full(len(v), 1 / (len(record_counts) * record_counts[s] * len(v)))
                              for s, v in usable])
    values = np.log(np.maximum(np.concatenate([v for _, v in usable]), 1e-12))
    center, scale = np.zeros(values.shape[1:]), np.zeros(values.shape[1:])
    for c in range(len(channel_names)):
        for m in range(2):
            center[c, m] = _weighted_median(values[:, c, m], weights)
            mad = _weighted_median(np.abs(values[:, c, m] - center[c, m]), weights)
            scale[c, m] = max(1.4826 * mad, min_log_scale)
    result = dict(algorithm=VERSION, scope=scope, channel_names=list(channel_names),
                  sample_rate=float(sample_rate), window_seconds=float(window_seconds),
                  preprocessing_id=preprocessing_id, metrics=list(METRICS),
                  score_threshold=float(score_threshold), min_log_scale=float(min_log_scale),
                  min_reference_windows=min_reference_windows, reference_window_count=count,
                  weighting="equal_identity_then_equal_record", center_log=center.tolist(),
                  scale_log=scale.tolist(), fitted_records=fitted_records or [])
    result["reference_id"] = reference_id(result)
    return result


def validate_reference(reference, channel_names, sample_rate, window_seconds, preprocessing_id,
                       *, training_only=False):
    if reference.get("algorithm") != VERSION or reference.get("metrics") != list(METRICS):
        raise ValueError("Unsupported window QC reference")
    if training_only and reference.get("scope") != "training":
        raise ValueError("Exclusion requires a training-fitted window QC reference")
    if reference.get("scope") not in {"training", "record_diagnostic"}:
        raise ValueError("Unknown window QC reference scope")
    if (reference.get("channel_names") != list(channel_names)
            or reference.get("sample_rate") != sample_rate
            or reference.get("window_seconds") != window_seconds
            or reference.get("preprocessing_id") != preprocessing_id):
        raise ValueError("Window QC reference does not match channels/rate/window/preprocessing")
    for key in ("center_log", "scale_log"):
        values = np.asarray(reference.get(key), dtype=float)
        if values.shape != (len(channel_names), 2) or not np.isfinite(values).all():
            raise ValueError("Invalid window QC reference statistics")
        if key == "scale_log" and not (values > 0).all():
            raise ValueError("Window QC scales must be positive")
    threshold = reference.get("score_threshold", 0)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Window QC score threshold must be positive")
    if reference.get("reference_id") != reference_id(reference):
        raise ValueError("Window QC reference checksum mismatch")


def score_windows(metrics, reference):
    scores = (np.log(np.maximum(metrics, 1e-12)) - np.asarray(reference["center_log"])) / np.asarray(reference["scale_log"])
    # Only unusually HIGH amplitude is flagged; a small amplitude is not itself
    # proof of corruption. Hard flatline detection remains a separate rule.
    suspect = np.any(scores > reference["score_threshold"], axis=(1, 2))
    return scores, suspect


def assert_reference_training_records(reference, records):
    """Prevent accidentally fitting on test records then training on S1 only."""
    if reference is None:
        return
    fields = ("subject_id", "session_id", "record_id", "path")
    current = {tuple(r.metadata()[key] for key in fields) for r in records}
    fitted = reference.get("fitted_records", [])
    if not fitted or any(tuple(row.get(key) for key in fields) not in current for row in fitted):
        raise ValueError("Window QC reference must be fitted only on records in this training manifest")


def _nullable(values):
    return np.where(np.isfinite(values), values, None).tolist()


def diagnose_windows(eeg, valid, channel_names, sample_rate, window_seconds, preprocessing_id,
                     reference=None):
    metrics = measure_windows(eeg, valid, round(sample_rate * window_seconds))
    status = "scored"
    if reference is None:
        if int(valid.sum()) >= 20:
            reference = fit_reference([metrics], channel_names, sample_rate, window_seconds,
                                      preprocessing_id, scope="record_diagnostic")
        else:
            status = "insufficient_reference_windows"
    scores = np.full_like(metrics, np.nan)
    suspect = np.zeros(len(valid), dtype=bool)
    if reference is not None:
        validate_reference(reference, channel_names, sample_rate, window_seconds, preprocessing_id)
        scores, suspect = score_windows(metrics, reference)
    candidates = []
    for index in np.flatnonzero(suspect):
        triggers = []
        for c, m in zip(*np.where(scores[index] > reference["score_threshold"])):
            triggers.append(dict(channel=channel_names[c], metric=METRICS[m],
                                 value_uv=float(metrics[index, c, m]), score=float(scores[index, c, m])))
        candidates.append(dict(window_index=int(index), start_seconds=float(index * window_seconds),
                               end_seconds=float((index + 1) * window_seconds),
                               max_score=float(np.max(scores[index])), triggers=triggers))
    candidates.sort(key=lambda item: (-item["max_score"], item["window_index"]))
    review = []
    if reference is not None:
        indices = np.flatnonzero(valid)
        ranked = sorted(indices, key=lambda i: (-float(np.max(scores[i])), int(i)))
        for index in ranked[:3]:
            c, m = np.unravel_index(np.argmax(scores[index]), scores[index].shape)
            review.append(dict(window_index=int(index), start_seconds=float(index * window_seconds),
                               end_seconds=float((index + 1) * window_seconds),
                               max_score=float(scores[index, c, m]), suspect=bool(suspect[index]),
                               triggers=[dict(channel=channel_names[c], metric=METRICS[m],
                                              value_uv=float(metrics[index, c, m]), score=float(scores[index, c, m]))]))
    fraction = float(suspect.sum() / valid.sum()) if valid.any() else 0.0
    return dict(algorithm=VERSION, status=status, mode="mark_only", reference=reference,
                channel_names=list(channel_names), window_seconds=float(window_seconds),
                valid_windows=valid.tolist(), suspect_windows=suspect.tolist(),
                peak_to_peak_uv=_nullable(metrics[:, :, 0]), rms_uv=_nullable(metrics[:, :, 1]),
                peak_to_peak_scores=_nullable(scores[:, :, 0]), rms_scores=_nullable(scores[:, :, 1]),
                suspect_window_count=int(suspect.sum()), base_valid_window_count=int(valid.sum()),
                suspect_fraction_of_valid=fraction,
                retained_if_excluded_seconds=float((valid & ~suspect).sum() * window_seconds),
                suspect_details=candidates,
                review_candidates=review,
                note="High amplitude is a review flag, not a diagnosed artifact. Record-local scores are descriptive; exclusion requires frozen training statistics.")
