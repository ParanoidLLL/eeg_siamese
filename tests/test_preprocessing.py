import json
from dataclasses import replace

import numpy as np
import pytest

from eeg_siamese.preprocessing import PreprocessingConfig, preprocess, save_processed


def signal(seconds=40, rate=1000):
    t = np.arange(round(seconds * rate)) / rate
    return np.stack([20 * np.sin(2 * np.pi * 10 * t),
                     15 * np.sin(2 * np.pi * 18 * t + 0.2)])


def spectral_amplitude(x, rate, hz):
    t = np.arange(len(x)) / rate
    return 2 * abs(np.mean(x * np.exp(-2j * np.pi * hz * t)))


def test_filter_retains_rhythm_suppresses_out_of_band_and_alias():
    eeg = signal()
    t = np.arange(eeg.shape[1]) / 1000
    eeg += 30 * np.sin(2 * np.pi * 60 * t) + 20 * np.sin(2 * np.pi * 300 * t)
    result = preprocess(eeg, 1000, ["C3", "C4"])
    x = result["eeg"][0, 2500:7500]
    assert spectral_amplitude(x, 250, 10) == pytest.approx(20, rel=0.01)
    assert spectral_amplitude(x, 250, 60) < 0.05
    assert spectral_amplitude(x, 250, 50) < 0.05
    assert result["qc"]["filter"]["notch_applied"] is False
    assert result["valid_windows"].tolist() == [False, True, True, True, True, True, True, False]


def test_nonfinite_gap_is_local_and_cannot_contaminate_distant_signal():
    baseline = signal(seconds=60)
    contaminated = baseline.copy()
    contaminated[0, 29500:30000] = np.nan
    contaminated[1, 30000] = np.inf
    clean = preprocess(baseline, 1000, ["C3", "C4"])
    damaged = preprocess(contaminated, 1000, ["C3", "C4"])
    assert np.isfinite(damaged["eeg"]).all()
    np.testing.assert_allclose(damaged["eeg"][:, 2500:5000], clean["eeg"][:, 2500:5000], atol=1e-5)
    np.testing.assert_allclose(damaged["eeg"][:, 10000:12500], clean["eeg"][:, 10000:12500], atol=1e-5)
    assert not damaged["valid_windows"][5:7].any()
    assert damaged["valid_windows"][2] and damaged["valid_windows"][9]
    assert any(i["reason"] == "nonfinite" for i in damaged["qc"]["raw_invalid_intervals"])


def test_preserves_length_event_time_and_original_metadata(tmp_path):
    eeg = signal(seconds=30.003)
    events = np.array([[4001, 11], [14003, 21], [30002, 7]], dtype=np.int64)
    result = preprocess(eeg, 1000, ["C3", "C4"], events=events)
    assert result["eeg"].shape == (2, 7501)
    assert result["qc"]["tail_unused_samples"] == 1
    assert np.max(np.abs(result["events"][:, 0] / 250 - events[:, 0] / 1000)) <= 1 / 250
    assert result["qc"]["original_events"] == events.tolist()
    output, qc = save_processed(result, tmp_path / "example.npz")
    with np.load(output, allow_pickle=False) as saved:
        assert saved["unit"].item() == "uV"
        assert saved["preprocessing_id"].item() == result["preprocessing_id"]
        assert saved["valid_windows"].dtype == np.bool_
        np.testing.assert_array_equal(saved["events"], result["events"])
    assert json.loads(qc.read_text())["quality_status"] == "usable"
    with pytest.raises(FileExistsError):
        save_processed(result, output)


def test_all_bad_and_too_short_emit_explicit_qc_without_fake_validity():
    result = preprocess(np.zeros((2, 30000)), 1000, ["C3", "C4"])
    assert not result["valid_windows"].any()
    assert result["qc"]["quality_status"] == "insufficient_signal"
    assert result["qc"]["valid_window_seconds"] == 0
    assert result["qc"]["persistent_bad_channels"] == ["C3", "C4"]
    car = preprocess(np.zeros((16, 30000)), 1000, [f"E{i}" for i in range(16)],
                     PreprocessingConfig(reference="car", car_full_head=True))
    assert car["qc"]["quality_status"] == "insufficient_signal"
    assert not car["valid_windows"].any()
    short = preprocess(signal(seconds=4), 1000, ["C3", "C4"])
    assert short["valid_windows"].shape == (0,)
    assert short["qc"]["quality_status"] == "insufficient_signal"


def test_car_excludes_bad_channel_and_does_not_treat_missing_spatial_input_as_valid():
    common = signal(seconds=30)[0]
    eeg = np.stack([common, common + 5 * np.sin(2 * np.pi * 20 * np.arange(len(common)) / 1000),
                    np.zeros_like(common)])
    config = PreprocessingConfig(reference="car", car_full_head=True, car_min_channels=2)
    result = preprocess(eeg, 1000, ["A", "B", "BAD"], config)
    assert result["qc"]["reference_channels"] == ["A", "B"]
    assert not result["valid_windows"].any()
    assert result["qc"]["persistent_bad_channels"] == ["BAD"]
    # With full valid input the common reference disappears on trusted windows.
    result = preprocess(eeg[:2], 1000, ["A", "B"], config)
    np.testing.assert_allclose(result["eeg"].sum(axis=0), 0, atol=1e-5)
    assert result["valid_windows"].any()
    with pytest.raises(ValueError, match="spatial coverage"):
        PreprocessingConfig(reference="car")


def test_config_hash_stable_roundtrip_and_tracks_effective_policy():
    config = PreprocessingConfig(known_bad_channels=("C3",))
    rebuilt = PreprocessingConfig(**json.loads(json.dumps(config.to_dict())))
    assert config.preprocessing_id == rebuilt.preprocessing_id
    assert config.preprocessing_id != replace(config, high_cut_hz=40).preprocessing_id
    assert config.preprocessing_id != replace(config, detect_jumps=False).preprocessing_id
    assert PreprocessingConfig(target_sample_rate=250).preprocessing_id == PreprocessingConfig(target_sample_rate=250.0).preprocessing_id
    assert PreprocessingConfig(known_bad_channels=("B", "A")).preprocessing_id == PreprocessingConfig(known_bad_channels=("A", "B")).preprocessing_id


def test_units_channels_event_indices_and_adc_limits_are_validated():
    eeg = signal(seconds=15)
    with pytest.raises(ValueError, match="physical uV"):
        preprocess(eeg, 1000, ["C3", "C4"], unit="V")
    with pytest.raises(ValueError, match="unique"):
        preprocess(eeg, 1000, ["C3", "C3"])
    with pytest.raises(ValueError, match="Event index"):
        preprocess(eeg, 1000, ["C3", "C4"], events=np.array([[15000, 1]]))
    with pytest.raises(ValueError, match="both"):
        PreprocessingConfig(adc_max_uv=500)


def test_documented_clipping_and_strong_jump_receive_explicit_protection():
    eeg = signal(seconds=60)
    eeg[0, 29000:30000] = 2000
    config = PreprocessingConfig(adc_min_uv=-2000, adc_max_uv=2000)
    result = preprocess(eeg, 1000, ["C3", "C4"], config)
    reasons = {i["reason"] for i in result["qc"]["raw_invalid_intervals"]}
    assert "documented_adc_limit" in reasons
    assert "strong_derivative_jump" in reasons
    assert not result["valid_windows"][5:7].any()
    assert result["valid_windows"][2]


def test_conditional_inband_notch_and_global_grid_after_gap():
    eeg = signal(seconds=60)
    t = np.arange(eeg.shape[1]) / 1000
    eeg += 40 * np.sin(2 * np.pi * 60 * t)
    config = PreprocessingConfig(high_cut_hz=90, lowpass_transition_hz=10)
    result = preprocess(eeg, 1000, ["C3", "C4"], config)
    assert result["qc"]["filter"]["notch_applied"]
    assert spectral_amplitude(result["eeg"][0, 2500:5000], 250, 60) < 0.5
    # A gap that does NOT align to downsampling groups must not shift the phase
    # of subsequent signal; polyphase output uses the original global grid.
    damaged = preprocess(eeg, 1000, ["C3", "C4"], config,
                         invalid_intervals=[(28003, 30003, "acquisition_gap")])
    np.testing.assert_allclose(damaged["eeg"][:, 10000:12500], result["eeg"][:, 10000:12500], atol=1e-5)


@pytest.mark.parametrize("channel_count", [16, 62])
def test_dense_inputs_keep_exact_order_and_correct_valid_coverage(channel_count):
    t = np.arange(30000) / 1000
    eeg = np.stack([10 * np.sin(2 * np.pi * (8 + c % 20) * t + c / 20)
                    for c in range(channel_count)])
    names = [f"EEG_{c:02d}" for c in range(channel_count)]
    result = preprocess(eeg, 1000, names)
    assert result["eeg"].shape == (channel_count, 7500)
    assert result["channel_names"] == tuple(names)
    assert result["valid_windows"].sum() == 4
    assert result["qc"]["valid_window_seconds"] == 20
    assert len(result["qc"]["channel_metrics"]) == channel_count
    assert result["qc"]["psd_before"] is not None
    assert result["qc"]["psd_after"] is not None
    assert all(m["high_frequency_power_ratio_40_100_over_1_100"] < 0.01
               for m in result["qc"]["channel_metrics"])


def test_noninteger_resampling_ratio_keeps_timeline_and_safe_alias_policy():
    rate = 512
    eeg = signal(seconds=40, rate=rate)
    result = preprocess(eeg, rate, ["C3", "C4"], events=np.array([[5123, 4]]))
    assert result["eeg"].shape[1] == 10000
    assert result["qc"]["resampling"]["up"] == 125
    assert result["qc"]["resampling"]["down"] == 256
    assert result["events"][0, 0] / 250 == pytest.approx(5123 / 512, abs=0.002)
    assert spectral_amplitude(result["eeg"][0, 2500:5000], 250, 10) == pytest.approx(20, rel=0.01)
