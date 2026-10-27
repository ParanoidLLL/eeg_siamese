import hashlib
import json
import math
from zipfile import BadZipFile
from pathlib import Path

import numpy as np
import torch

from .data import DataConfig, block_is_usable, load_windows, padded_block, pad_quality
from .model import EEGEncoder, EncoderConfig


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_id(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_encoder(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    encoder = EEGEncoder(EncoderConfig(**checkpoint["encoder_config"]))
    data_config = DataConfig(**checkpoint["data_config"])
    if encoder.config.spectral_fusion!='none':
        if not math.isclose(encoder.config.sample_rate,data_config.sample_rate) or not math.isclose(encoder.config.spectral_input_scale_uv,data_config.signal_scale_uv) or not math.isclose(data_config.window_seconds,encoder.config.spectral_window_seconds):
            raise ValueError('Spectral residual physical configuration disagrees with data')
    if encoder.config.channels != len(data_config.channel_names):
        raise ValueError("Checkpoint channel configurations disagree")
    if encoder.config.quality_aware != (data_config.quality is not None):
        raise ValueError('Checkpoint quality data/encoder mode mismatch')
    if encoder.config.quality_aware and encoder.config.patch_samples != data_config.quality['patch_samples']:
        raise ValueError('Checkpoint quality patch size mismatch')
    encoder.load_state_dict(checkpoint["encoder_state"], strict=True)
    return encoder.to(device).eval(), data_config, checkpoint, checkpoint_id(path)


def normalized_mean(vectors, weights=None):
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError("Expected finite nonempty embedding vectors")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 1e-8).any():
        raise ValueError("Cannot normalize zero embeddings")
    vectors = vectors / norms
    mean = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if norm <= 1e-8:
        raise ValueError("Embeddings cancel; cannot build a stable template")
    return mean / norm


@torch.inference_mode()
def encode_record(encoder, config, record, device):
    tasks = None
    if encoder.config.requires_task:
        tasks = torch.tensor([encoder.task_index(record.task_id)], dtype=torch.long, device=device)
    loaded = load_windows(record, config)
    windows, valid = loaded[:2]
    vectors, weights = [], []
    for start in range(0, len(windows), config.block_windows):
        if not block_is_usable(valid[start:start + config.block_windows], config):
            continue
        x, mask = padded_block(windows, valid, start, config.block_windows)
        if not mask.any():
            continue
        if len(loaded) == 3:
            quality = pad_quality(loaded[2], start, config.block_windows)
            if not encoder.config.quality_soft_weights:
                quality = (quality > 0).to(quality.dtype)
            z = encoder(x.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), quality.unsqueeze(0).to(device), tasks=tasks)
            weight = float(quality.sum())
        else:
            z = encoder(x.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
            weight = int(mask.sum())
        vectors.append(z[0].cpu().numpy())
        weights.append(weight)
    if not vectors:
        raise ValueError(f"{record.record_id}: no usable block under duration rules")
    return normalized_mean(vectors, weights)


def predict_records(encoder, config, records, device, gallery, threshold=None):
    """Preserve order and count failed records without inventing an identity."""
    rows = []
    for record in records:
        try:
            vector = encode_record(encoder, config, record, device)
            row = dict(status="ok", error=None, **match([vector], gallery, threshold)[0])
        except (ValueError, OSError, EOFError, BadZipFile) as error:
            row = dict(status="signal_error", error=str(error), predicted_id=None,
                       best_id=None, best_score=None, score_gap=None)
        rows.append(dict(record_id=record.record_id, **row))
    return rows


def make_gallery(encoder, config, records, device, model_id):
    identities = sorted({r.subject_id for r in records})
    vectors = []
    for identity in identities:
        vectors.append(normalized_mean([
            encode_record(encoder, config, r, device)
            for r in records if r.subject_id == identity
        ]).tolist())
    return dict(schema_version=1, model_id=model_id, identities=identities,
                prototypes=vectors, enrollment_records=[r.metadata() for r in records])


def validate_gallery(gallery, model_id, embedding_dim):
    if gallery.get("schema_version") != 1 or gallery.get("model_id") != model_id:
        raise ValueError("Gallery belongs to a different checkpoint; enroll again")
    identities = gallery["identities"]
    prototypes = np.asarray(gallery["prototypes"], dtype=np.float64)
    if not identities or len(set(identities)) != len(identities) or "Unknown" in identities:
        raise ValueError("Invalid gallery identity labels")
    if prototypes.shape != (len(identities), embedding_dim) or not np.isfinite(prototypes).all():
        raise ValueError("Invalid gallery vector shape/values")
    if not np.allclose(np.linalg.norm(prototypes, axis=1), 1, atol=1e-5):
        raise ValueError("Gallery prototypes must be normalized")


def match(vectors, gallery, threshold=None):
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    vectors = np.asarray(vectors, dtype=np.float64)
    prototypes = np.asarray(gallery["prototypes"], dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[1] != prototypes.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("Invalid query embeddings")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if (norms <= 1e-8).any():
        raise ValueError("Zero query embedding")
    scores = vectors / norms @ prototypes.T
    rows = []
    for score in scores:
        best = int(score.argmax())
        maximum = float(score[best])
        ordered = np.sort(score)
        prediction = gallery["identities"][best]
        if threshold is not None and maximum < threshold:
            prediction = "Unknown"
        rows.append(dict(predicted_id=prediction, best_id=gallery["identities"][best],
                         best_score=maximum,
                         score_gap=float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else None))
    return rows


def classification_metrics(rows, truth_ids, gallery_ids):
    if not rows or len(rows) != len(truth_ids):
        raise ValueError("Prediction/label lengths disagree")
    registered = set(gallery_ids)
    correct = known = unknown = wrong_identity = rejected = accepted_unknown = known_correct = failed = 0
    for row, identity in zip(rows, truth_ids):
        prediction = row["predicted_id"]
        target = identity if identity in registered else "Unknown"
        if row.get("status", "ok") != "ok":
            failed += 1
            known += int(identity in registered)
            unknown += int(identity not in registered)
            continue
        correct += int(prediction == target)
        if identity in registered:
            known += 1
            known_correct += int(prediction == identity)
            rejected += int(prediction == "Unknown")
            wrong_identity += int(prediction not in (identity, "Unknown"))
        else:
            unknown += 1
            accepted_unknown += int(prediction != "Unknown")
    return dict(records=len(rows), accuracy=correct / len(rows), failed_records=failed,
                coverage=(len(rows) - failed) / len(rows),
                known_records=known, unknown_records=unknown,
                known_correct=known_correct, known_wrong_identity=wrong_identity,
                known_rejected=rejected, unknown_accepted=accepted_unknown,
                known_identification_rate=known_correct / known if known else None,
                known_rejection_rate=rejected / known if known else None,
                unknown_acceptance_rate=accepted_unknown / unknown if unknown else None)


def identity_breakdown(rows, truth_ids, gallery_ids):
    labels = list(gallery_ids)
    columns = labels + ["Unknown", "SignalError"]
    matrix = [[0] * len(columns) for _ in labels]
    per_identity = {}
    for row, identity in zip(rows, truth_ids):
        if identity not in labels:
            continue
        predicted = row["predicted_id"] if row.get("status", "ok") == "ok" else "SignalError"
        matrix[labels.index(identity)][columns.index(predicted)] += 1
        item = per_identity.setdefault(identity, dict(records=0, correct=0, failed=0))
        item["records"] += 1
        item["correct"] += int(predicted == identity)
        item["failed"] += int(predicted == "SignalError")
    for item in per_identity.values():
        item["accuracy"] = item["correct"] / item["records"]
    return dict(confusion_matrix=dict(truth_labels=labels, predicted_labels=columns, counts=matrix),
                per_identity=per_identity)


def calibrate_threshold(rows, truth_ids, gallery_ids):
    """Optimize record-level development accuracy for this gallery size.

    Equal-accuracy ties select the higher threshold. This score is development
    performance, not held-out evaluation; use separate identities for evaluation.
    """
    registered = set(gallery_ids)
    if not any(s in registered for s in truth_ids) or not any(s not in registered for s in truth_ids):
        raise ValueError("Calibration needs both registered and unregistered query identities")
    scores = np.asarray([r["best_score"] for r in rows], dtype=np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores")
    candidates = sorted(set(scores.tolist()) | {float(np.nextafter(scores.max(), np.inf))})
    best = None
    for threshold in candidates:
        predictions = [dict(r, predicted_id=r["best_id"] if r["best_score"] >= threshold else "Unknown") for r in rows]
        metrics = classification_metrics(predictions, truth_ids, gallery_ids)
        if best is None or metrics["accuracy"] >= best[1]["accuracy"]:
            best = threshold, metrics
    return best
