import csv
import json

import numpy as np
import pytest
from scipy.io import savemat

from eeg_siamese.data import read_manifest
from eeg_siamese.datasets import discover_openbmi, inject_noise, read_openbmi, split_manifest


def make_mat(path, *, units_factor=1.0, transpose=False, labels=True):
    phase = {
        "x": np.arange(2000, dtype=float).reshape(1000, 2) * units_factor,
        "chan": np.array(["C3", "C4"], dtype=object), "fs": 1000,
        "t": np.array([10, 900]), "y_dec": np.array([1, 2]) if labels else np.array([]),
        "EMG": np.zeros((1000, 4)),
    }
    if transpose:
        phase["x"] = phase["x"].T
    savemat(path, {"EEG_MI_train": phase, "EEG_MI_test": phase})


def test_read_openbmi_converts_continuous_shape_units_and_preserves_session(tmp_path):
    path = tmp_path / "sess01_subj01_EEG_MI.mat"
    make_mat(path, units_factor=1e-6)
    records = read_openbmi(path, 1, 1, "V")
    assert len(records) == 2
    assert {r["session_id"] for r in records} == {"session_1"}
    assert {r["subject_id"] for r in records} == {"openbmi_S01"}
    assert {r["phase"] for r in records} == {"train", "test"}
    assert records[0]["eeg_uv"].shape == (2, 1000)
    np.testing.assert_allclose(records[0]["eeg_uv"][:, 1], [2, 3])
    assert records[0]["channel_names"] == ("C3", "C4")
    np.testing.assert_array_equal(records[0]["events"], [[10, 1], [900, 2]])
    assert len(records[0]["provenance"]["source_sha256"]) == 64
    assert records[1]["provenance"]["phase_condition"] == "online_feedback"
    assert discover_openbmi(tmp_path) == [(path.resolve(), 1, 1)]


def test_events_explicit_index_conversion_and_missing_task_labels(tmp_path):
    path = tmp_path / "one.mat"
    make_mat(path, labels=False)
    record = read_openbmi(path, 1, 1, "uV", event_index_base=1)[0]
    np.testing.assert_array_equal(record["events"], [[9, 0], [899, 0]])
    assert record["provenance"]["original_event_samples"] == [10, 900]
    assert not record["provenance"]["event_labels_available"]


def test_reader_does_not_guess_units_orientation_or_subject(tmp_path):
    path = tmp_path / "sess01_subj01_EEG_MI.mat"
    make_mat(path)
    with pytest.raises(ValueError, match="input_unit"):
        read_openbmi(path, 1, 1, "unknown")
    with pytest.raises(ValueError, match="filename"):
        read_openbmi(path, 2, 1, "uV")
    make_mat(path, transpose=True)
    with pytest.raises(ValueError, match="continuous"):
        read_openbmi(path, 1, 1, "uV")


def test_v73_has_actionable_failure(tmp_path):
    path = tmp_path / "data.mat"
    path.write_bytes(b"MATLAB 7.3 MAT-file" + b" " * 200)
    with pytest.raises(ValueError, match="classic MAT"):
        read_openbmi(path, 1, 1, "uV")


def make_manifest(tmp_path, count=21, sessions=("session_1", "session_2")):
    manifest = tmp_path / "all.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["subject_id", "session_id", "record_id", "path"])
        for subject in range(count):
            for session in sessions:
                for phase in ("train", "test"):
                    record = f"S{subject:02d}_{session}_{phase}"
                    path = tmp_path / f"{record}.npz"
                    np.savez(path, eeg=np.zeros((2, 10)))
                    writer.writerow([f"S{subject:02d}", session, record, path.name])
    return manifest


def test_split_fixed_identity_roles_and_cross_session_open_set_episodes(tmp_path):
    manifest = make_manifest(tmp_path)
    first = split_manifest(manifest, tmp_path / "first", 10, 5, 5, seed=19, episodes=4)
    second = split_manifest(manifest, tmp_path / "second", 10, 5, 5, seed=19, episodes=4)
    assert first["subjects"] == second["subjects"]
    assert len(first["unused_subject_ids"]) == 1
    roles = first["subjects"]
    assert not set(roles["train"]) & (set(roles["dev"]) | set(roles["eval"]))
    assert not set(roles["dev"]) & set(roles["eval"])
    assert len(read_manifest(tmp_path / "first" / "train.csv")) == 40
    episodes = json.loads((tmp_path / "first" / "dev" / "episodes.json").read_text())
    repeat = json.loads((tmp_path / "second" / "dev" / "episodes.json").read_text())
    for episode, repeated in zip(episodes, repeat):
        assert episode["registered_subject_ids"] == repeated["registered_subject_ids"]
        enrollment, query = read_manifest(episode["enrollment"]), read_manifest(episode["query"])
        assert {r.session_id for r in enrollment} == {"session_1"}
        assert {r.session_id for r in query} == {"session_2"}
        assert len({r.subject_id for r in enrollment}) == 3
        assert len({r.subject_id for r in query}) == 5
        assert not set(episode["registered_subject_ids"]) & set(episode["unknown_subject_ids"])
        assert not {r.path for r in enrollment} & {r.path for r in query}
    with pytest.raises(ValueError, match="frozen"):
        split_manifest(manifest, tmp_path / "first", 10, 5, 5)


def test_phases_are_not_accepted_as_independent_sessions(tmp_path):
    manifest = make_manifest(tmp_path, 20, sessions=("session_1",))
    with pytest.raises(ValueError, match="phases are not sessions"):
        split_manifest(manifest, tmp_path / "split", 10, 5, 5)
    assert not (tmp_path / "split").exists()


@pytest.mark.parametrize("kind", ["broadband", "line", "burst", "drift", "jump"])
def test_nominal_noise_strength_reproducibility_and_invalid_samples(kind):
    rng = np.random.default_rng(1)
    eeg = rng.normal(0, 10, (3, 2000))
    eeg[0, 10] = np.nan
    noisy, metadata = inject_noise(eeg, 250, kind=kind, snr_db=10, seed=6)
    again, _ = inject_noise(eeg, 250, kind=kind, snr_db=10, seed=6)
    np.testing.assert_array_equal(noisy, again)
    assert np.isnan(noisy[0, 10])
    assert metadata["nominal_snr_db"] == 10
    assert "not true neural SNR" in metadata["snr_definition"]
    for observed, added in zip(metadata["observed_power_uv2"], metadata["added_noise_power_uv2"]):
        assert observed / added == pytest.approx(10)
    np.testing.assert_array_equal(eeg[1], rng_state_channel(seed=1, channels=3, samples=2000)[1])


def rng_state_channel(seed, channels, samples):
    return np.random.default_rng(seed).normal(0, 10, (channels, samples))


def test_dropout_is_explicit_missing_interval_not_fake_snr():
    signal = np.ones((3, 1000))
    noisy, metadata = inject_noise(signal, 250, kind="dropout", dropout_fraction=0.2, seed=6)
    assert np.count_nonzero(~np.isfinite(noisy)) == 200
    assert metadata["nominal_snr_db"] is None
    assert np.isfinite(signal).all()


def test_noise_refuses_undefined_power_and_aliasing():
    with pytest.raises(ValueError, match="constant"):
        inject_noise(np.ones((2, 100)), 250)
    with pytest.raises(ValueError, match="Nyquist"):
        inject_noise(np.arange(100).reshape(1, -1), 100, kind="line", line_hz=60)
