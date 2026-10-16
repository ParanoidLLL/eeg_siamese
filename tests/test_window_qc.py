import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from eeg_siamese.cli import main
from eeg_siamese.data import DataConfig, Record, load_windows
from eeg_siamese.preprocessing import preprocess, save_processed
from eeg_siamese.report import _trace, render_qc_report
from eeg_siamese.window_qc import (assert_reference_training_records, diagnose_windows,
                                 fit_reference, measure_windows, score_windows)


def eeg_windows(count=40, samples=1250):
    rng = np.random.default_rng(17)
    t = np.arange(samples) / 250
    return np.concatenate([np.stack([20 * rng.uniform(.9, 1.1) * np.sin(2 * np.pi * 10 * t),
                                     15 * rng.uniform(.9, 1.1) * np.sin(2 * np.pi * 18 * t)])
                           for _ in range(count)], axis=1)


def reference(eeg, valid):
    return fit_reference([measure_windows(eeg, valid, 1250)], ["C3", "C4"], 250, 5, None)


def test_slow_large_wave_is_marked_without_removing_or_modifying_signal(tmp_path):
    eeg = eeg_windows()
    t = np.arange(1250) / 250
    eeg[0, 25 * 1250:26 * 1250] += 1000 * np.sin(2 * np.pi * t) * np.sin(np.pi * t / 5) ** 2
    result = preprocess(eeg, 250, ["C3", "C4"])
    assert not any(row["reason"] == "strong_derivative_jump" for row in result["qc"]["raw_invalid_intervals"])
    assert result["valid_windows"][25]
    assert result["suspect_windows"][25]
    assert result["qc"]["window_qc"]["mode"] == "mark_only"
    assert any(row["window_index"] == 25 for row in result["qc"]["window_qc"]["suspect_details"])
    save_processed(result, tmp_path / "slow.npz")
    with np.load(tmp_path / "slow.npz") as saved:
        assert saved["valid_windows"][25] and saved["suspect_windows"][25]
        assert np.max(saved["eeg"][0, 25 * 1250:26 * 1250]) > 100
    render_qc_report(eeg, 250, result, tmp_path / "slow.html")
    page = (tmp_path / "slow.html").read_text(encoding="utf-8")
    assert "125.0–130.0" in page and "原始可疑窗口 25" in page and "处理后可疑窗口 25" in page
    assert 'fill="#fff0c2"' in page


def test_metrics_have_physical_units_and_ignore_invalid_and_partial_windows():
    eeg = np.array([[3., -3., 3., -3., np.nan, np.nan, np.nan, np.nan, 999.]])
    valid = np.array([True, False])
    metrics = measure_windows(eeg, valid, 4)
    np.testing.assert_allclose(metrics[0, 0], [6, 3])
    assert np.isnan(metrics[1]).all()
    diagnostic = diagnose_windows(eeg, valid, ["A"], 4, 1, None)
    assert diagnostic["status"] == "insufficient_reference_windows"
    assert diagnostic["rms_uv"] == [[3.], [None]]
    json.dumps(diagnostic, allow_nan=False)


def test_zero_mad_scale_floor_and_high_only_scoring():
    metrics = np.full((30, 1, 2), 10.)
    ref = fit_reference([metrics], ["A"], 250, 5, None)
    np.testing.assert_allclose(ref["scale_log"], [[.1, .1]])
    scores, suspect = score_windows(np.array([[[10., 10.]], [[100., 100.]], [[1., 1.]]]), ref)
    assert suspect.tolist() == [False, True, False]
    assert np.isfinite(scores).all()


def test_reference_balances_identities_and_records_instead_of_raw_duration():
    small = np.full((20, 1, 2), 10.)
    long = np.full((1000, 1, 2), 1000.)
    ref = fit_reference([small, small, long], ["A"], 250, 5, None, subject_ids=["s1", "s2", "s3"])
    np.testing.assert_allclose(np.exp(ref["center_log"]), 10.)
    ref = fit_reference([small, small, long], ["A"], 250, 5, None, subject_ids=["s1", "s1", "s2"])
    assert ref["weighting"] == "equal_identity_then_equal_record"


def test_optional_exclusion_uses_frozen_training_reference_before_model_scaling(tmp_path):
    clean = eeg_windows()
    valid = np.ones(40, dtype=bool)
    ref = reference(clean, valid)
    query = clean.copy()
    query[0, 5 * 1250:6 * 1250] *= 100
    path = tmp_path / "query.npz"
    np.savez(path, eeg=query, sample_rate=250, channel_names=["C3", "C4"], unit="uV", valid_windows=valid,
             suspect_windows=np.zeros(40, dtype=bool))  # advisory old flags must not control exclusion
    record = Record("person", "s2", "query", path)
    base = DataConfig(250, ("C3", "C4"), window_qc_reference=ref)
    windows, baseline = load_windows(record, base)
    assert baseline.all() and windows[5].abs().max() > 1
    excluded, mask = load_windows(record, replace(base, exclude_suspect_windows=True, signal_scale_uv=1, demean=False))
    assert not mask[5] and int(mask.sum()) == 39
    assert not excluded[5].any()
    # Query amplitude cannot alter the frozen reference.
    scores, suspect = score_windows(measure_windows(query, valid, 1250), ref)
    assert suspect[5] and ref == reference(clean, valid)
    with pytest.raises(ValueError, match="training-fitted"):
        DataConfig(250, ("C3", "C4"), exclude_suspect_windows=True)
    with pytest.raises(ValueError, match="match channels"):
        DataConfig(250, ("C4", "C3"), window_qc_reference=ref)
    local = diagnose_windows(clean, valid, ["C3", "C4"], 250, 5, None)["reference"]
    with pytest.raises(ValueError, match="training-fitted"):
        DataConfig(250, ("C3", "C4"), window_qc_reference=local)
    np.savez(path, eeg=clean * 100, sample_rate=250, channel_names=["C3", "C4"], unit="uV", valid_windows=valid)
    with pytest.raises(ValueError, match="no valid windows"):
        load_windows(record, replace(base, exclude_suspect_windows=True))


def test_display_reduction_does_not_miss_a_single_peak():
    signal = np.zeros((1, 10000))
    signal[0, 4321] = 1234
    x, y = _trace(signal, 0, 1000, 0, 10)
    assert y.max() == 1234 and 4.321 in x
    assert len(x) <= 2000


def test_check_old_npz_and_fit_cli_without_overwriting_source(tmp_path):
    path = tmp_path / "source.npz"
    np.savez(path, eeg=eeg_windows(), sample_rate=250, channel_names=["C3", "C4"], unit="uV",
             valid_windows=np.ones(40, dtype=bool))
    before = path.read_bytes()
    manifest = tmp_path / "train.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "session_id", "record_id", "path"])
        writer.writeheader()
        writer.writerow(dict(subject_id="A", session_id="session_1", record_id="source", path=str(path)))
    config = tmp_path / "general.json"
    config.write_text(json.dumps(dict(data=dict(sample_rate=250, channel_names=["C3", "C4"]))))
    out = tmp_path / "review"
    main(["check-windows", "--manifest", str(manifest), "--config", str(config), "--output-dir", str(out)])
    assert path.read_bytes() == before
    assert (out / "source.window_qc.html").exists()
    with pytest.raises(SystemExit):
        main(["check-windows", "--manifest", str(manifest), "--config", str(config), "--output-dir", str(out)])
    fitted = tmp_path / "fit"
    main(["fit-window-qc", "--manifest", str(manifest), "--config", str(config), "--output-dir", str(fitted)])
    ref = json.loads((fitted / "reference.json").read_text())
    assert ref["scope"] == "training" and ref["fitted_records"][0]["source_sha256"]
    mark = json.loads((fitted / "general.mark.json").read_text())["data"]
    exclude = json.loads((fitted / "general.exclude.json").read_text())["data"]
    assert not DataConfig(**mark).exclude_suspect_windows and DataConfig(**exclude).exclude_suspect_windows
    assert_reference_training_records(ref, [Record("A", "session_1", "source", path)])
    with pytest.raises(ValueError, match="training manifest"):
        assert_reference_training_records(ref, [Record("A", "session_2", "other", path)])


@pytest.mark.parametrize("valid", [np.zeros(0, dtype=bool), np.zeros(20, dtype=bool)])
def test_all_invalid_reports_remain_json_safe(valid):
    result = diagnose_windows(np.zeros((2, len(valid) * 1250)), valid, ["C3", "C4"], 250, 5, None)
    assert result["suspect_window_count"] == 0 and result["retained_if_excluded_seconds"] == 0
    json.dumps(result, allow_nan=False)
