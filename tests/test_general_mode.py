import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from eeg_siamese.cli import main
from eeg_siamese.data import (DataConfig, Record, SessionBatchSampler,
                             audit_episode, load_windows, read_manifest)
from eeg_siamese.inference import (calibrate_threshold, classification_metrics,
                                  match, read_json, validate_gallery)
from eeg_siamese.losses import pairwise_contrastive_loss, supervised_contrastive_loss
from eeg_siamese.model import EEGEncoder, EncoderConfig


torch.set_num_threads(1)


def write_record(path, identity="S01", session="s1", **overrides):
    values = dict(eeg=np.random.default_rng(1).normal(0, 10, (2, 192)).astype(np.float32),
                  sample_rate=np.array(32), channel_names=np.array(["C3", "C4"]), unit=np.array("uV"), valid_channel_samples=np.ones((2,192),dtype=bool))
    values.update(overrides)
    if "valid_channel_samples" not in overrides:
        values["valid_channel_samples"] = np.ones(values["eeg"].shape,dtype=bool)
    np.savez(path, **values)
    return Record(identity, session, path.stem, path.resolve())


def write_manifest(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "session_id", "record_id", "path"])
        writer.writeheader()
        for record in records:
            writer.writerow(dict(record.metadata(), path=str(record.path.relative_to(path.parent))))
    return str(path)


@pytest.mark.parametrize("layers", [1, 2])
def test_masked_padding_cannot_change_embedding(layers):
    torch.manual_seed(8)
    encoder = EEGEncoder(EncoderConfig(2, hidden_dim=16, embedding_dim=8,
                                       transformer_layers=layers, attention_heads=2, dropout=0, patch_samples=4, token_stride=2)).eval()
    x = torch.randn(2, 3, 2, 16)
    mask = torch.ones(2, 3, dtype=torch.bool)
    quality=torch.ones(2,3,2,4)
    expected = encoder(x, mask, quality)
    padded = torch.cat([x, torch.full((2, 2, 2, 16), float("nan"))], dim=1)
    padded_mask = torch.cat([mask, torch.zeros(2, 2, dtype=torch.bool)], dim=1)
    actual = encoder(padded, padded_mask, torch.cat([quality,torch.zeros(2,2,2,4)],dim=1))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual.norm(dim=1), torch.ones(2))


def test_unusable_records_fail_instead_of_returning_identity():
    encoder = EEGEncoder(EncoderConfig(2, patch_samples=4, token_stride=2))
    with pytest.raises(ValueError, match="valid window"):
        encoder(torch.zeros(1, 2, 2, 16), torch.zeros(1, 2, dtype=torch.bool), torch.ones(1,2,2,4))
    with pytest.raises(ValueError, match="Non-finite"):
        encoder(torch.full((1, 1, 2, 16), float("nan")), torch.ones(1, 1, dtype=torch.bool), torch.ones(1,1,2,4))


@pytest.mark.parametrize("loss_function", [supervised_contrastive_loss, pairwise_contrastive_loss])
def test_identity_loss_rewards_correct_pairs(loss_function):
    identities = torch.tensor([0, 0, 1, 1])
    sessions = torch.tensor([0, 1, 0, 1])
    good = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]], requires_grad=True)
    bad = good.detach()[[0, 2, 1, 3]].requires_grad_()
    low, high = loss_function(good, identities, sessions), loss_function(bad, identities, sessions)
    assert low < high
    high.backward()
    assert torch.isfinite(bad.grad).all()
    assert bad.grad.abs().sum() > 0


def test_cross_session_supcon_requires_independent_positives():
    z = torch.randn(4, 8)
    with pytest.raises(ValueError, match="positive"):
        supervised_contrastive_loss(z, torch.tensor([0, 0, 1, 1]), torch.zeros(4, dtype=torch.long))
    z = torch.randn(6, 8, requires_grad=True)
    loss = supervised_contrastive_loss(z, torch.tensor([0, 0, 0, 1, 1, 1]), torch.tensor([0, 0, 1, 0, 0, 1]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


def test_sampler_provides_cross_session_identity_pairs(tmp_path):
    records = [Record(f"S{i}", f"s{j}", f"r{i}{j}", tmp_path / f"r{i}{j}") for i in range(4) for j in range(2)]
    sampler = SessionBatchSampler(records, 3, 2, steps=4)
    for batch in sampler:
        chosen = [records[i] for i in batch]
        assert len({r.subject_id for r in chosen}) == 3
        for identity in {r.subject_id for r in chosen}:
            assert len({r.session_id for r in chosen if r.subject_id == identity}) == 2
    with pytest.raises(ValueError, match="sessions"):
        SessionBatchSampler(records[::2], 2, 2, steps=1)


@pytest.mark.parametrize("overrides,error", [
    ({"sample_rate": np.array(64)}, "sampling rate"),
    ({"channel_names": np.array(["C4", "C3"])}, "channel names"),
    ({"unit": np.array("V")}, "units"),
    ({"eeg": np.ones((2, 192), dtype=np.complex64)}, "real numeric"),
])
def test_source_metadata_is_not_silently_reinterpreted(tmp_path, overrides, error):
    record = write_record(tmp_path / "record.npz", **overrides)
    with pytest.raises(ValueError, match=error):
        load_windows(record, DataConfig(32, ("C3", "C4"), window_seconds=0.5))


def test_bad_windows_are_masked_and_all_bad_record_rejected(tmp_path):
    eeg = np.ones((2, 64), dtype=np.float32)
    eeg[0, 0] = np.nan
    record = write_record(tmp_path / "record.npz", eeg=eeg)
    config = DataConfig(32, ("C3", "C4"), window_seconds=0.5)
    x, mask = load_windows(record, config)
    assert mask.tolist() == [False, True, True, True]
    assert torch.isfinite(x).all()
    assert x[0].abs().sum() == 0
    record = write_record(tmp_path / "all_bad.npz", eeg=np.full((2, 64), np.nan))
    with pytest.raises(ValueError, match="no valid windows"):
        load_windows(record, config)


def test_split_audit_rejects_identity_and_record_leakage(tmp_path):
    train = [Record("seen", "s1", "train", tmp_path / "train")]
    enrollment = [Record("new", "s1", "enroll", tmp_path / "enroll")]
    queries = [Record("new", "s2", "query", tmp_path / "query")]
    audit_episode(train, enrollment, queries)
    with pytest.raises(ValueError, match="training identities"):
        audit_episode(train, [Record("seen", "s2", "enroll", tmp_path / "enroll")], queries)
    with pytest.raises(ValueError, match="sessions overlap"):
        audit_episode(train, enrollment, [Record("new", "s1", "query", tmp_path / "query")])
    with pytest.raises(ValueError, match="paths overlap"):
        audit_episode(train, enrollment, [Record("new", "s2", "query", tmp_path / "enroll")])


def test_manifest_rejects_duplicate_physical_file(tmp_path):
    record = write_record(tmp_path / "record.npz")
    duplicate = Record("S02", "s2", "different_name", record.path)
    path = write_manifest(tmp_path / "duplicates.csv", [record, duplicate])
    with pytest.raises(ValueError, match="physical EEG file"):
        read_manifest(path)


def test_open_set_decision_and_threshold_boundary():
    gallery = dict(identities=["A", "B", "C"], prototypes=np.eye(3).tolist())
    vectors = np.array([[1, 0, 0], [-1, -1, -1]], dtype=np.float64)
    rows = match(vectors, gallery, threshold=1.0)
    assert [r["predicted_id"] for r in rows] == ["A", "Unknown"]
    assert match(vectors[1:], gallery)[0]["predicted_id"] != "Unknown"
    rows = [dict(best_id="A", best_score=.8), dict(best_id="B", best_score=.9),
            dict(best_id="A", best_score=.4), dict(best_id="B", best_score=.3)]
    threshold, metrics = calibrate_threshold(rows, ["A", "B", "D", "E"], ["A", "B", "C"])
    assert threshold == .8
    assert metrics["accuracy"] == 1
    with pytest.raises(ValueError, match="both registered"):
        calibrate_threshold(rows[:2], ["A", "B"], ["A", "B", "C"])


def test_metrics_separate_known_misidentification_rejection_and_unknown_acceptance():
    rows = [dict(predicted_id=s) for s in ["A", "B", "Unknown", "A", "Unknown"]]
    metrics = classification_metrics(rows, ["A", "A", "C", "D", "E"], ["A", "B", "C"])
    assert metrics["accuracy"] == 2 / 5
    assert metrics["known_wrong_identity"] == 1
    assert metrics["known_rejected"] == 1
    assert metrics["unknown_accepted"] == 1
    assert metrics["unknown_acceptance_rate"] == .5


def test_gallery_is_bound_to_model():
    gallery = dict(schema_version=1, model_id="old", identities=["A"], prototypes=[[1., 0.]])
    with pytest.raises(ValueError, match="different checkpoint"):
        validate_gallery(gallery, "new", 2)


def test_complete_general_pipeline_and_pretrained_initialization(tmp_path, capsys):
    # Synthetic fixtures check plumbing only; no assertion of biometric accuracy.
    groups = {}
    for role, count in [("train", 10), ("dev", 5), ("eval", 5)]:
        groups[role] = []
        for identity in range(count):
            for session in ("s1", "s2"):
                groups[role].append(write_record(tmp_path / f"{role}_{identity}_{session}.npz",
                                                 identity=f"{role}_{identity}", session=session))
    manifests = {"train": write_manifest(tmp_path / "train.csv", groups["train"])}
    for role in ("dev", "eval"):
        registered = {f"{role}_{i}" for i in range(3)}
        manifests[role + "_enroll"] = write_manifest(tmp_path / f"{role}_enroll.csv",
            [r for r in groups[role] if r.session_id == "s1" and r.subject_id in registered])
        manifests[role + "_query"] = write_manifest(tmp_path / f"{role}_query.csv",
            [r for r in groups[role] if r.session_id == "s2"])
    audit_args = ["audit"]
    for role, path in manifests.items():
        audit_args.extend(["--" + role.replace("_", "-"), path])
    main(audit_args)
    config = dict(data=dict(sample_rate=32, channel_names=["C3", "C4"], window_seconds=.5, block_windows=3,quality=dict(patch_samples=4)),
                  encoder=dict(token_stride=2, hidden_dim=16, embedding_dim=8, transformer_layers=1, attention_heads=2, dropout=0,patch_samples=4),
                  training=dict(epochs=1, steps_per_epoch=2, identities_per_batch=2, samples_per_identity=2))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    checkpoint = str(tmp_path / "model.pt")
    training_args = ["train", "--manifest", manifests["train"], "--config", str(config_path), "--device", "cpu"]
    main(training_args + ["--output", checkpoint])
    initialized = str(tmp_path / "initialized.pt")
    adaptation_manifest = write_manifest(tmp_path / "adaptation.csv", groups["dev"])
    adaptation_args = training_args.copy()
    adaptation_args[adaptation_args.index("--manifest") + 1] = adaptation_manifest
    main(adaptation_args + ["--output", initialized, "--init", checkpoint])
    initial = torch.load(initialized, weights_only=True)
    assert initial["initialization_id"] is not None
    assert len(initial["training_records"]) == 30
    assert len(initial["identity_labels"]) == 5
    assert {r["subject_id"] for r in initial["training_records"]} == {
        r.subject_id for r in groups["train"] + groups["dev"]
    }
    conflicting = [Record(groups["dev"][0].subject_id, groups["dev"][0].session_id,
                          groups["train"][0].record_id, groups["dev"][0].path)] + groups["dev"][1:]
    conflicting_manifest = write_manifest(tmp_path / "conflicting.csv", conflicting)
    conflicting_args = adaptation_args.copy()
    conflicting_args[conflicting_args.index("--manifest") + 1] = conflicting_manifest
    with pytest.raises(SystemExit):
        main(conflicting_args + ["--output", str(tmp_path / "invalid.pt"), "--init", checkpoint])
    galleries = {}
    for role in ("dev", "eval"):
        galleries[role] = str(tmp_path / f"{role}_gallery.json")
        main(["enroll", "--checkpoint", checkpoint, "--manifest", manifests[role + "_enroll"],
              "--output", galleries[role], "--device", "cpu"])
    calibration = str(tmp_path / "threshold.json")
    main(["calibrate", "--checkpoint", checkpoint, "--gallery", galleries["dev"],
          "--manifest", manifests["dev_query"], "--output", calibration, "--device", "cpu"])
    report = str(tmp_path / "evaluation.json")
    main(["evaluate", "--checkpoint", checkpoint, "--gallery", galleries["eval"],
          "--manifest", manifests["eval_query"], "--calibration", calibration, "--output", report, "--device", "cpu"])
    assert read_json(report)["metrics"]["records"] == 5
    predictions = str(tmp_path / "predictions.csv")
    main(["predict", "--checkpoint", checkpoint, "--gallery", galleries["eval"],
          "--manifest", manifests["eval_query"], "--calibration", calibration, "--output", predictions, "--device", "cpu"])
    with Path(predictions).open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert all(np.isfinite(float(row["best_score"])) for row in rows)
    closed_gallery = str(tmp_path / "closed_gallery.json")
    main(["enroll", "--checkpoint", checkpoint, "--manifest", manifests["train"],
          "--allow-seen-identities", "--output", closed_gallery, "--device", "cpu"])
    closed_queries = [write_record(tmp_path / f"closed_{i}_s3.npz", identity=f"train_{i}", session="s3")
                      for i in range(10)]
    closed_manifest = write_manifest(tmp_path / "closed_query.csv", closed_queries)
    main(["predict", "--checkpoint", checkpoint, "--gallery", closed_gallery,
          "--manifest", closed_manifest, "--closed-set", "--output", predictions, "--device", "cpu"])
    with Path(predictions).open() as stream:
        closed_rows = list(csv.DictReader(stream))
    assert len(closed_rows) == 10
    assert all(row["predicted_id"] in {f"train_{i}" for i in range(10)} for row in closed_rows)
    main(["evaluate", "--checkpoint", checkpoint, "--gallery", closed_gallery,
          "--manifest", closed_manifest, "--closed-set", "--output", report, "--device", "cpu"])
    assert read_json(report)["metrics"]["records"] == 10
    with pytest.raises(SystemExit):
        main(["evaluate", "--checkpoint", checkpoint, "--gallery", galleries["dev"],
              "--manifest", manifests["dev_query"], "--calibration", calibration, "--output", report, "--device", "cpu"])
    with pytest.raises(SystemExit):
        main(["predict", "--checkpoint", initialized, "--gallery", galleries["eval"],
              "--manifest", manifests["eval_query"], "--calibration", calibration, "--output", predictions, "--device", "cpu"])
    two_person_gallery = read_json(galleries["eval"])
    two_person_gallery["identities"] = two_person_gallery["identities"][:2]
    two_person_gallery["prototypes"] = two_person_gallery["prototypes"][:2]
    two_person_path = tmp_path / "two_person_gallery.json"
    two_person_path.write_text(json.dumps(two_person_gallery))
    with pytest.raises(SystemExit):
        main(["predict", "--checkpoint", checkpoint, "--gallery", str(two_person_path),
              "--manifest", manifests["eval_query"], "--calibration", calibration, "--output", predictions, "--device", "cpu"])
    assert "training_loss" in capsys.readouterr().out
