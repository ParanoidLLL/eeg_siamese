"""Conservative offline EEG processing, with explicit unusable-sample masks.

Input amplitudes MUST already be physical microvolts. No unit inference, spatial
interpolation, ICA, or learned reconstruction is performed. Invalid samples are
zero placeholders only; they and FIR boundary guards never enter valid windows.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import fftconvolve, firwin, resample_poly, welch

from .window_qc import diagnose_windows


ALGORITHM_VERSION = "conservative-fir-qc-v1"


@dataclass(frozen=True)
class PreprocessingConfig:
    target_sample_rate: float = 250.0
    low_cut_hz: float = 0.5
    high_cut_hz: float = 45.0
    highpass_transition_hz: float = 0.5
    lowpass_transition_hz: float = 10.0
    window_seconds: float = 5.0
    line_frequency_hz: float | None = 60.0
    line_peak_ratio: float = 10.0
    notch_half_width_hz: float = 1.0
    flatline_seconds: float = 1.0
    flatline_tolerance_uv: float = 1e-6
    detect_flatline: bool = True
    detect_jumps: bool = True
    jump_min_uv: float = 1000.0
    jump_mad_multiplier: float = 50.0
    adc_min_uv: float | None = None
    adc_max_uv: float | None = None
    adc_tolerance_uv: float = 0.0
    known_bad_channels: tuple[str, ...] = ()
    reference: str = "original"
    car_full_head: bool = False
    car_min_channels: int = 16
    mask_scope: str = "window"

    def __post_init__(self):
        object.__setattr__(self, "known_bad_channels", tuple(self.known_bad_channels))
        positive = (self.target_sample_rate, self.low_cut_hz, self.high_cut_hz,
                    self.highpass_transition_hz, self.lowpass_transition_hz,
                    self.window_seconds, self.line_peak_ratio,
                    self.notch_half_width_hz, self.flatline_seconds,
                    self.jump_min_uv, self.jump_mad_multiplier)
        if not all(math.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Sampling, frequency, duration and detection thresholds must be positive")
        if not 0 < self.low_cut_hz < self.high_cut_hz < self.target_sample_rate / 2:
            raise ValueError("Require 0 < low_cut_hz < high_cut_hz < output Nyquist")
        if self.high_cut_hz + self.lowpass_transition_hz / 2 >= self.target_sample_rate / 2:
            raise ValueError("Low-pass transition must fit below output Nyquist")
        if self.line_frequency_hz is not None and (
                not math.isfinite(self.line_frequency_hz) or self.line_frequency_hz <= 0):
            raise ValueError("line_frequency_hz must be positive or None")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.flatline_tolerance_uv, self.adc_tolerance_uv)):
            raise ValueError("Tolerances must be finite and nonnegative")
        if (self.adc_min_uv is None) != (self.adc_max_uv is None):
            raise ValueError("Supply both documented ADC limits, or neither")
        if self.adc_min_uv is not None and not (
                math.isfinite(self.adc_min_uv) and math.isfinite(self.adc_max_uv)
                and self.adc_min_uv < self.adc_max_uv):
            raise ValueError("ADC limits must be finite and ordered")
        if self.reference not in {"original", "car"}:
            raise ValueError("reference must be original or car")
        if self.mask_scope not in {"window", "channel_sample"}:
            raise ValueError("mask_scope must be window or channel_sample")
        if self.mask_scope == "channel_sample" and self.reference != "original":
            raise ValueError("Channel-local masks currently require original reference")
        if self.reference == "car" and not self.car_full_head:
            raise ValueError("CAR requires explicit car_full_head=True confirmation of spatial coverage")
        if not isinstance(self.car_min_channels, int) or self.car_min_channels < 2:
            raise ValueError("car_min_channels must be an integer >= 2")
        if len(set(self.known_bad_channels)) != len(self.known_bad_channels):
            raise ValueError("known_bad_channels must be unique")
        if any(not isinstance(c, str) or not c for c in self.known_bad_channels):
            raise ValueError("Bad channel names must be nonempty strings")
        object.__setattr__(self, "known_bad_channels", tuple(sorted(self.known_bad_channels)))
        samples = self.target_sample_rate * self.window_seconds
        if not math.isclose(samples, round(samples), abs_tol=1e-6) or samples < 4:
            raise ValueError("Output windows must contain an integer number of >= 4 samples")

    def to_dict(self):
        values = asdict(self)
        if self.mask_scope == "window":
            values.pop("mask_scope")  # Preserve existing default preprocessing IDs.
        # Numeric spellings such as 250 and 250.0 describe the SAME pipeline.
        for key in ("target_sample_rate", "low_cut_hz", "high_cut_hz", "highpass_transition_hz",
                    "lowpass_transition_hz", "window_seconds", "line_frequency_hz", "line_peak_ratio",
                    "notch_half_width_hz", "flatline_seconds", "flatline_tolerance_uv", "jump_min_uv",
                    "jump_mad_multiplier", "adc_min_uv", "adc_max_uv", "adc_tolerance_uv"):
            if values[key] is not None:
                values[key] = float(values[key])
        values["known_bad_channels"] = list(self.known_bad_channels)
        return values

    @property
    def preprocessing_id(self):
        payload = {"algorithm": ALGORITHM_VERSION, "config": self.to_dict()}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runs(mask):
    """Return half-open runs of True without joining gaps."""
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _line_ratio(eeg, good, rate, frequency):
    if frequency is None or not 2 < frequency < rate / 2 - 2:
        return None
    # A contiguous clean sample interval avoids introducing spectral jumps by
    # concatenating disjoint stretches. This is a QC proxy, not a neural SNR.
    eligible = [(a, b) for a, b in _runs(good) if b - a >= math.ceil(4 * rate)]
    if not eligible:
        return None
    a, b = max(eligible, key=lambda pair: pair[1] - pair[0])
    b = min(b, a + math.ceil(20 * rate))
    frequencies, power = welch(eeg[:, a:b], fs=rate, nperseg=min(b - a, round(4 * rate)), axis=-1)
    peak = np.abs(frequencies - frequency) <= 0.5
    neighbors = (np.abs(frequencies - frequency) >= 2) & (np.abs(frequencies - frequency) <= 4)
    if not peak.any() or not neighbors.any():
        return None
    floor = np.median(power[:, neighbors], axis=-1)
    ratios = np.max(power[:, peak], axis=-1) / np.maximum(floor, np.finfo(float).tiny)
    # Preserve finite JSON even for a mathematical single-frequency signal.
    return float(np.median(np.minimum(ratios, 1e300)))


def _spectral_summary(eeg, good, rate):
    eligible = [(a, b) for a, b in _runs(good) if b - a >= math.ceil(4 * rate)]
    if not eligible:
        return None, None
    a, b = max(eligible, key=lambda pair: pair[1] - pair[0])
    b = min(b, a + math.ceil(20 * rate))
    frequencies, power = welch(eeg[:, a:b], fs=rate, nperseg=round(4 * rate), axis=-1)
    ratios = None
    if rate / 2 >= 100:
        whole = (frequencies >= 1) & (frequencies <= 100)
        high = (frequencies >= 40) & (frequencies <= 100)
        denominator = power[:, whole].sum(axis=-1)
        ratios = np.divide(power[:, high].sum(axis=-1), denominator,
                           out=np.zeros(len(power)), where=denominator > 0)
    summary = {"frequency_hz": frequencies.tolist(),
               "median_power_uv2_per_hz": np.median(power, axis=0).tolist(),
               "start_sample": int(a), "end_sample": int(b),
               "note": "PSD from a contiguous QC-usable interval; frequency bands are diagnostic only"}
    return summary, ratios


def preprocess(eeg_uv, sample_rate, channel_names, config=None, events=None,
               *, unit="uV", invalid_intervals=None):
    """Process a separate continuous record and retain its complete time axis.

    ``events`` is an integer [N, 2] array: original zero-based sample index,
    arbitrary integer event code. It is mapped to the nearest output sample;
    originals and the exact rational mapping are retained in QC. Duplicate
    output events remain distinct. ``invalid_intervals`` is an optional list
    of (start, exclusive_end, reason) in ORIGINAL sample coordinates, e.g.
    acquisition gaps detected by a reader. All channels remain in fixed order.
    All-invalid recordings produce explicit QC and all-False valid_windows.
    """
    config = config or PreprocessingConfig()
    if not isinstance(config, PreprocessingConfig):
        raise TypeError("config must be PreprocessingConfig")
    if unit != "uV":
        raise ValueError("Input must be explicitly converted to physical uV; ADC counts or inferred units are forbidden")
    source = np.asarray(eeg_uv)
    if source.dtype.kind not in "fiu" or source.ndim != 2 or not source.shape[1]:
        raise ValueError("EEG must be a nonempty real numeric [channels, samples] array")
    names = tuple(channel_names)
    if source.shape[0] != len(names) or not names or len(set(names)) != len(names):
        raise ValueError("Provide unique channel names in the exact input order")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Channel names must be nonempty strings")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if config.high_cut_hz + config.lowpass_transition_hz / 2 >= sample_rate / 2:
        raise ValueError("Filter passband and transition must fit below input Nyquist")
    unknown = set(config.known_bad_channels) - set(names)
    if unknown:
        raise ValueError(f"Unknown bad channels: {sorted(unknown)}")
    eeg = np.asarray(source, dtype=np.float64)
    finite_values = eeg[np.isfinite(eeg)]
    if finite_values.size and np.max(np.abs(finite_values)) > np.finfo(np.float32).max:
        raise ValueError("Physical uV amplitudes exceed supported numeric range")
    del finite_values
    channels, length = eeg.shape
    bad = np.zeros(length, dtype=bool)
    channel_bad = np.zeros((channels, length), dtype=bool) if config.mask_scope == "channel_sample" else None
    reasons = []
    metrics = []

    def mark(a, b, reason, channel=None):
        a, b = int(a), int(b)
        bad[a:b] = True
        if channel_bad is not None:
            if channel is None:
                channel_bad[:, a:b] = True
            else:
                channel_bad[names.index(channel), a:b] = True
        reasons.append({"start_sample": a, "end_sample": b, "reason": reason, "channel": channel})

    persistent_bad = set(config.known_bad_channels)
    for index, name in enumerate(names):
        x = eeg[index]
        finite = np.isfinite(x)
        for a, b in _runs(~finite):
            mark(a, b, "nonfinite", name)
        finite_x = x[finite]
        if not finite_x.size:
            persistent_bad.add(name)
        metric = {"channel": name, "nonfinite_fraction": float(1 - finite.mean()),
                  "peak_to_peak_uv": float(np.ptp(finite_x)) if finite_x.size else None,
                  "rms_uv": float(np.sqrt(np.mean(np.square(finite_x)))) if finite_x.size else None,
                  "flatline_samples": 0, "clipped_samples": 0, "jump_count": 0}
        if name in persistent_bad:
            mark(0, length, "known_bad_channel", name)
        if config.detect_flatline:
            with np.errstate(invalid="ignore"):
                same = finite[1:] & finite[:-1] & (np.abs(np.diff(x)) <= config.flatline_tolerance_uv)
            for a, b in _runs(same):
                if b - a + 1 >= math.ceil(config.flatline_seconds * sample_rate):
                    mark(a, b + 1, "flatline", name)
                    metric["flatline_samples"] += int(b - a + 1)
                    if a == 0 and b + 1 == length:
                        persistent_bad.add(name)
        if config.adc_min_uv is not None:
            clipped = finite & ((x <= config.adc_min_uv + config.adc_tolerance_uv)
                                | (x >= config.adc_max_uv - config.adc_tolerance_uv))
            metric["clipped_samples"] = int(clipped.sum())
            for a, b in _runs(clipped):
                mark(a, b, "documented_adc_limit", name)
        if config.detect_jumps:
            with np.errstate(invalid="ignore"):
                dx = np.diff(x)
            finite_dx = dx[np.isfinite(dx)]
            if finite_dx.size:
                center = np.median(finite_dx)
                robust_scale = 1.4826 * np.median(np.abs(finite_dx - center))
                threshold = max(config.jump_min_uv, config.jump_mad_multiplier * robust_scale)
                jumps = np.flatnonzero(np.isfinite(dx) & (np.abs(dx - center) > threshold))
                metric["jump_count"] = int(len(jumps))
                metric["jump_threshold_uv"] = float(threshold)
                for i in jumps:
                    mark(i, i + 2, "strong_derivative_jump", name)
        metrics.append(metric)
    for interval in invalid_intervals or ():
        if len(interval) != 3:
            raise ValueError("invalid_intervals requires (start, exclusive_end, reason)")
        a, b, reason = interval
        if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)) or not 0 <= a < b <= length:
            raise ValueError("Invalid source interval")
        if not isinstance(reason, str) or not reason:
            raise ValueError("Invalid interval reason")
        mark(a, b, reason)

    good = ~bad
    psd_before, high_frequency_ratios = _spectral_summary(eeg, good, sample_rate)
    for index, metric in enumerate(metrics):
        metric["high_frequency_power_ratio_40_100_over_1_100"] = (
            float(high_frequency_ratios[index]) if high_frequency_ratios is not None else None)
    line_ratio = _line_ratio(eeg, good, sample_rate, config.line_frequency_hz)
    notch = False
    cutoff = [config.low_cut_hz, config.high_cut_hz]
    notch_reason = "not_requested" if config.line_frequency_hz is None else "outside_retained_band"
    if config.line_frequency_hz is not None and (
            config.low_cut_hz < config.line_frequency_hz - config.notch_half_width_hz
            < config.line_frequency_hz + config.notch_half_width_hz < config.high_cut_hz):
        if line_ratio is not None and line_ratio >= config.line_peak_ratio:
            cutoff = [config.low_cut_hz, config.line_frequency_hz - config.notch_half_width_hz,
                      config.line_frequency_hz + config.notch_half_width_hz, config.high_cut_hz]
            notch, notch_reason = True, "in_band_peak_exceeds_threshold"
        else:
            notch_reason = "no_confirmed_in_band_peak"
    transition = min(config.highpass_transition_hz, config.lowpass_transition_hz,
                     config.notch_half_width_hz if notch else math.inf)
    taps_count = math.ceil(3.3 * sample_rate / transition)
    taps_count += 1 - taps_count % 2
    taps = firwin(taps_count, cutoff, pass_zero=False, window="hamming", fs=sample_rate)

    ratio = (Fraction(str(config.target_sample_rate)) / Fraction(str(sample_rate))).limit_denominator(10000)
    up, down = ratio.numerator, ratio.denominator
    effective_rate = sample_rate * up / down
    if not math.isclose(effective_rate, config.target_sample_rate, rel_tol=1e-10):
        raise ValueError("Sampling ratio cannot be represented accurately with bounded polyphase factors")
    if max(up, down) > 10000:
        raise ValueError("Sampling ratio requires excessive polyphase factors")
    if up == down == 1:
        poly = np.ones(1)
    else:
        half = 10 * max(up, down)
        poly = firwin(2 * half + 1, 1 / max(up, down), window=("kaiser", 5.0))
    guard = (len(taps) - 1) // 2 + math.ceil(((len(poly) - 1) / 2) / up)
    # Filter separate trusted runs. FFT convolutions NEVER ingest a NaN,
    # saturation, contact discontinuity, or synthetic placeholder from a gap.
    filtered = np.zeros_like(eeg)
    reference_channels = [i for i, name in enumerate(names) if name not in persistent_bad]
    if config.reference == "car" and len(reference_channels) < config.car_min_channels and good.any():
        raise ValueError("Too few trusted channels for the configured CAR policy")
    output_length = (length * up + down - 1) // down
    sample_valid = np.zeros(output_length, dtype=bool)
    channel_sample_valid = np.zeros((channels, output_length), dtype=bool) if channel_bad is not None else None
    trusted_intervals = []
    runs = ((None, a, b) for a, b in _runs(good)) if channel_bad is None else (
        (i, a, b) for i in range(channels) for a, b in _runs(~channel_bad[i]))
    for channel_index, a, b in runs:
        if b - a <= 2 * guard:
            continue
        channel_slice = slice(None) if channel_index is None else slice(channel_index, channel_index + 1)
        block = eeg[channel_slice, a:b].copy()
        if config.reference == "car":
            block -= block[reference_channels].mean(axis=0, keepdims=True)
        filtered[channel_slice, a:b] = fftconvolve(block, taps[None, :], mode="same", axes=-1)
        start = ((int(a) + guard) * up + down - 1) // down
        end = ((int(b) - guard) * up + down - 1) // down
        if channel_sample_valid is None:
            sample_valid[start:end] = True
        else:
            channel_sample_valid[channel_index, start:end] = True
        trusted_intervals.append({"start_sample": int(a), "end_sample": int(b),
                                  "output_valid_start": start, "output_valid_end": end,
                                  "channel": names[channel_index] if channel_index is not None else None})
    # Bounded-support antialiasing preserves the GLOBAL sample grid. Any output
    # possibly touched by a zero placeholder is already excluded by the guard.
    output = filtered if up == down == 1 else resample_poly(filtered, up, down, axis=-1, window=poly)
    if channel_sample_valid is None:
        output[:, ~sample_valid] = 0
    else:
        sample_valid = channel_sample_valid.any(axis=0)
        output[~channel_sample_valid] = 0
    output = output.astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("Processing overflow: verify physical units and dynamic range")
    psd_after, _ = _spectral_summary(output, sample_valid, config.target_sample_rate)
    window_samples = round(config.window_seconds * config.target_sample_rate)
    windows = output_length // window_samples
    valid_windows = sample_valid[:windows * window_samples].reshape(windows, window_samples).all(axis=1)
    if channel_sample_valid is not None:
        # Exact cell validity is determined by the loader's patch layout.
        valid_windows = sample_valid[:windows * window_samples].reshape(windows, window_samples).any(axis=1)
    invalid_output = [{"start_sample": int(a), "end_sample": int(b),
                       "reason": "raw_invalid_or_filter_boundary_guard"} for a, b in _runs(~sample_valid)]
    qc = {"algorithm": ALGORITHM_VERSION, "config": config.to_dict(),
          "preprocessing_id": config.preprocessing_id,
          "software": {"numpy": np.__version__, "scipy": scipy.__version__},
          "input_sample_rate": float(sample_rate), "input_samples": length,
          "output_sample_rate": float(config.target_sample_rate), "output_samples": output_length,
          "input_unit": unit, "channel_names": list(names), "channel_metrics": metrics,
          "persistent_bad_channels": sorted(persistent_bad), "interpolated_channels": [],
          "raw_invalid_intervals": reasons, "output_invalid_intervals": invalid_output,
          "trusted_runs": trusted_intervals,
          "filter": {"kind": "symmetric_hamming_fir", "phase": "zero_phase_single_convolution",
                     "cutoff_half_amplitude_hz": cutoff, "numtaps": len(taps),
                     "requested_min_transition_hz": transition, "padding": "zero_excluded_by_guard",
                     "notch_applied": notch, "notch_reason": notch_reason},
          "resampling": {"up": up, "down": down, "polyphase_numtaps": len(poly),
                         "sample_mapping": "output_index = round(input_index * up / down)"},
          "reference": config.reference, "reference_channels": [names[i] for i in reference_channels],
          "reference_status": "applied_or_preserved" if trusted_intervals else "no_trusted_input_run",
          "guard_input_samples": guard, "guard_seconds": guard / sample_rate,
          "raw_duration_seconds": length / sample_rate,
          "valid_sample_seconds": float(sample_valid.sum() / config.target_sample_rate),
          "valid_window_seconds": float(valid_windows.sum() * config.window_seconds),
          "valid_window_count": int(valid_windows.sum()), "total_window_count": windows,
          "valid_window_fraction": float(valid_windows.mean()) if windows else 0.0,
          "tail_unused_samples": output_length - windows * window_samples,
          "tail_unused_seconds": (output_length - windows * window_samples) / config.target_sample_rate,
          "line_peak_ratio_before": line_ratio,
          "psd_before": psd_before, "psd_after": psd_after,
          "quality_status": "usable" if valid_windows.any() else "insufficient_signal",
          "quality_note": "QC proxies are not estimates of true neural SNR"}
    result = {"eeg": output, "sample_rate": float(config.target_sample_rate),
              "channel_names": names, "unit": "uV", "valid_windows": valid_windows,
              "preprocessing_id": config.preprocessing_id, "qc": qc}
    if channel_sample_valid is not None:
        result["valid_channel_samples"] = channel_sample_valid
        qc["quality_mask"] = {"scope": "channel_sample", "shape": list(channel_sample_valid.shape),
                              "valid_fraction": float(channel_sample_valid.mean()),
                              "note": "Per-channel FIR/polyphase guards; no interpolation. Window validity means some trusted samples; loader refines to complete electrode-time patches."}
    # Advisory annotations do not alter the physical signal, preprocessing ID,
    # or the hard validity mask. Statistical exclusion is a DataConfig policy.
    qc["window_qc"] = diagnose_windows(output, valid_windows, names, config.target_sample_rate,
                                       config.window_seconds, config.preprocessing_id)
    result["suspect_windows"] = np.asarray(qc["window_qc"]["suspect_windows"], dtype=bool)
    if events is not None:
        original = np.asarray(events)
        if original.dtype.kind not in "iu" or original.ndim != 2 or original.shape[1] != 2:
            raise ValueError("events must be integer [N,2]: zero-based sample index, event code")
        if np.any(original[:, 0] < 0) or np.any(original[:, 0] >= length):
            raise ValueError("Event index outside the original record")
        mapped = original.astype(np.int64, copy=True)
        mapped[:, 0] = np.minimum(np.rint(original[:, 0].astype(float) * up / down).astype(np.int64), output_length - 1)
        result["events"] = mapped
        qc["original_events"] = original.tolist()
        qc["event_valid_samples"] = sample_valid[mapped[:, 0]].tolist()
        qc["max_event_time_error_seconds"] = float(np.max(np.abs(
            mapped[:, 0] / config.target_sample_rate - original[:, 0] / sample_rate))) if len(original) else 0.0
    return result


def save_processed(result, path, *, overwrite=False):
    """Save NPZ plus '<stem>.qc.json'; callers may add source provenance to QC.

    The preprocessing_id belongs in DataConfig and checkpoints. Source SHA256,
    participant/session/phase and original reference are reader provenance,
    NOT fitted global statistics and NOT part of the shared configuration ID.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Processed output must have .npz suffix")
    qc_path = path.with_suffix(".qc.json")
    if not overwrite and (path.exists() or qc_path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing processed output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    qc_text = json.dumps(result["qc"], ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    arrays = {key: result[key] for key in
              ("eeg", "sample_rate", "channel_names", "unit", "valid_windows", "preprocessing_id")}
    if "events" in result:
        arrays["events"] = result["events"]
    if "suspect_windows" in result:
        arrays["suspect_windows"] = result["suspect_windows"]
    if "valid_channel_samples" in result:
        arrays["valid_channel_samples"] = result["valid_channel_samples"]
    np.savez_compressed(path, **arrays)
    qc_path.write_text(qc_text, encoding="utf-8")
    return path, qc_path
