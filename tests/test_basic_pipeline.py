import csv
import json
from dataclasses import asdict

import numpy as np
import pytest
import torch
from scipy.io import savemat

from eeg_siamese.cli import main
from eeg_siamese.data import DataConfig, Record, RecordingDataset, block_is_usable, load_windows
from eeg_siamese.inference import classification_metrics, identity_breakdown, predict_records, read_json
from eeg_siamese.model import EEGEncoder, EncoderConfig
from eeg_siamese.preprocessing import PreprocessingConfig, preprocess, save_processed
from eeg_siamese.protocol import split_closed


torch.set_num_threads(1)


def manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_data_binds_preprocessing_and_uses_real_short_block_duration(tmp_path):
    path = tmp_path / "record.npz"
    np.savez(path, eeg=np.random.default_rng(1).normal(size=(2, 800)), sample_rate=100,
             channel_names=["C3", "C4"], unit="uV", preprocessing_id="expected")
    record = Record("person", "session_1", "record", path)
    config = DataConfig(100, ("C3", "C4"), window_seconds=1, block_windows=6,
                        preprocessing_id="expected", min_valid_windows=4, min_valid_fraction=.6)
    windows, valid = load_windows(record, config)
    assert block_is_usable(valid[-2:], config)  # all real windows; padding doesn't count
    assert not block_is_usable(torch.tensor([True, False]), config)
    assert RecordingDataset([record], config, cache=True).load(0)[0].shape == windows.shape
    with pytest.raises(ValueError, match="preprocessing_id"):
        load_windows(record, DataConfig(100, ("C3", "C4"), window_seconds=1))


def test_corrupt_missing_and_unusable_queries_preserve_order_and_denominator(tmp_path):
    valid = tmp_path / "valid.npz"
    np.savez(valid, eeg=np.random.default_rng(1).normal(size=(2, 320)), sample_rate=32,
             channel_names=["C3", "C4"], unit="uV",valid_channel_samples=np.ones((2,320),dtype=bool))
    corrupt = tmp_path / "broken.npz"
    corrupt.write_bytes(b"PK\x03\x04broken zip without directory")
    bad = tmp_path / "bad.npz"
    np.savez(bad, eeg=np.full((2, 320), np.nan), sample_rate=32,
             channel_names=["C3", "C4"], unit="uV",valid_channel_samples=np.ones((2,320),dtype=bool))
    records = [Record("A", "s2", name, path) for name, path in
               [("ok1", valid), ("corrupt", corrupt), ("missing", tmp_path / "absent.npz"),
                ("allbad", bad), ("ok2", valid)]]
    config = DataConfig(32, ("C3", "C4"), window_seconds=1,quality=dict(patch_samples=8))
    encoder = EEGEncoder(EncoderConfig(2, hidden_dim=16, embedding_dim=8, transformer_layers=1,patch_samples=8, token_stride=2)).eval()
    gallery = dict(identities=["A"], prototypes=[[1.] + [0.] * 7])
    rows = predict_records(encoder, config, records, "cpu", gallery)
    assert [r["record_id"] for r in rows] == [r.record_id for r in records]
    assert [r["status"] for r in rows] == ["ok", "signal_error", "signal_error", "signal_error", "ok"]
    metrics = classification_metrics(rows, ["A"] * 5, ["A"])
    assert metrics["accuracy"] == .4 and metrics["failed_records"] == 3
    assert identity_breakdown(rows, ["A"] * 5, ["A"])["confusion_matrix"]["counts"] == [[2, 0, 3]]


def test_complete_basic_closed_set_pipeline_from_mat(tmp_path):
    # Fixtures verify preprocessing/training/report contracts, NOT real EEG accuracy.
    index = []
    rate, duration = 128, 30
    t = np.arange(rate * duration) / rate
    for person in range(1, 4):
        for session in (1, 2):
            rng = np.random.default_rng(person * 100 + session)
            eeg = np.column_stack([20 * np.sin(2 * np.pi * (6 + 3 * person) * t) + rng.normal(size=len(t)),
                                   15 * np.cos(2 * np.pi * (8 + 2 * person) * t) + rng.normal(size=len(t))])
            fields = dict(x=eeg, fs=rate, chan=np.array(["C3", "C4"], dtype=object),
                          t=np.array([512, 1024]), y_dec=np.array([1, 2]))
            path = tmp_path / f"sess{session:02d}_subj{person:02d}_EEG_MI.mat"
            savemat(path, dict(EEG_MI_train=fields, EEG_MI_test=fields))
            index.append(dict(path=str(path), subject_id=person, session_id=session))
    index_path = tmp_path / "index.csv"
    manifest(index_path, index)
    prep_path = tmp_path / "prep.json"
    prep_path.write_text(json.dumps(PreprocessingConfig(mask_scope="channel_sample").to_dict()))
    converted = tmp_path / "converted"
    main(["convert-openbmi", "--index", str(index_path), "--input-unit", "uV",
          "--config", str(prep_path), "--output-dir", str(converted)])
    assert len(list((converted / "records").glob("*.qc.html"))) == 12
    qc = read_json(next((converted / "records").glob("*.qc.json")))
    assert qc["valid_window_count"] == 6  # Channel masks retain partially usable boundary windows.
    with np.load(next((converted / "records").glob("*.npz"))) as archive:
        assert (~archive["valid_channel_samples"]).any()
    assert qc["source_sha256"] and qc["psd_before"]["frequency_hz"]
    split = tmp_path / "split"
    main(["split-closed", "--manifest", str(converted / "manifest.csv"),
          "--config", str(converted / "general.json"), "--output-dir", str(split), "--subjects", "3"])
    split2 = tmp_path / "split2"
    repeat = split_closed(converted / "manifest.csv", converted / "general.json", split2, 3)
    assert repeat["identities"] == read_json(split / "split.json")["identities"]
    config = read_json(split / "general.json")
    assert config['data']['quality']
    assert not config['training']['cross_session']
    config["encoder"].update(hidden_dim=16, embedding_dim=8, transformer_layers=1, attention_heads=2, dropout=0)
    config["training"].update(epochs=1, steps_per_epoch=1, identities_per_batch=3)
    config_path = split / "quick.json"
    config_path.write_text(json.dumps(config))
    checkpoint, gallery, report = [str(tmp_path / name) for name in ("basic.pt", "gallery.json", "accuracy.json")]
    main(["train", "--manifest", str(split / "train.csv"), "--config", str(config_path),
          "--output", checkpoint, "--device", "cpu"])
    main(["enroll", "--checkpoint", checkpoint, "--manifest", str(split / "train.csv"),
          "--allow-seen-identities", "--output", gallery, "--device", "cpu"])
    main(["evaluate", "--checkpoint", checkpoint, "--gallery", gallery,
          "--manifest", str(split / "test.csv"), "--closed-set", "--output", report, "--device", "cpu"])
    result = read_json(report)
    assert result["metrics"]["records"] == 6 and result["metrics"]["failed_records"] == 0
    assert set(result["per_identity"]) == set(repeat["identities"])
    assert all(r["predicted_id"] != "Unknown" for r in result["predictions"])
    # Frozen test protocol is protected from accidental regeneration.
    with pytest.raises(ValueError, match="frozen"):
        split_closed(converted / "manifest.csv", converted / "general.json", split, 3)


def test_raw_preprocessing_does_not_silently_discard_existing_masks(tmp_path):
    raw = tmp_path / "raw.npz"
    np.savez(raw, eeg=np.ones((2, 4000)), sample_rate=128, channel_names=["C3", "C4"],
             unit="uV", valid_windows=np.ones(6, dtype=bool))
    source = tmp_path / "raw.csv"
    manifest(source, [dict(subject_id="A", session_id="s1", record_id="raw", path=str(raw))])
    config = tmp_path / "prep.json"
    config.write_text(json.dumps(PreprocessingConfig(mask_scope="channel_sample").to_dict()))
    with pytest.raises(SystemExit):
        main(["preprocess", "--manifest", str(source), "--config", str(config), "--output-dir", str(tmp_path / "out")])
