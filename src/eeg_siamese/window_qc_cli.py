"""Report existing processed files and fit frozen training-only QC standards."""
import copy
import json
from pathlib import Path

import numpy as np

from .data import DataConfig, read_manifest
from .inference import checkpoint_id, read_json, write_json
from .report import render_window_report
from .window_qc import diagnose_windows, fit_reference, measure_windows


def _read_processed(record, config):
    with np.load(record.path, allow_pickle=False) as archive:
        required = {"eeg", "sample_rate", "channel_names", "unit", "valid_windows"}
        if not required.issubset(archive.files):
            raise ValueError(f"{record.record_id}: window QC requires a processed NPZ with validity metadata")
        rate = float(archive["sample_rate"].item())
        names = tuple(archive["channel_names"].tolist())
        actual_id = str(archive["preprocessing_id"].item()) if "preprocessing_id" in archive else None
        if (rate != config.sample_rate or names != config.channel_names
                or actual_id != config.preprocessing_id or str(archive["unit"].item()) != "uV"):
            raise ValueError(f"{record.record_id}: QC input metadata does not match DataConfig")
        eeg, valid = archive["eeg"], archive["valid_windows"]
    return dict(eeg=eeg, valid_windows=valid, channel_names=names,
                sample_rate=rate, preprocessing_id=actual_id, unit="uV")


def check_windows_command(args):
    config = DataConfig(**read_json(args.config)["data"])
    reference = read_json(args.reference) if args.reference else config.window_qc_reference
    output = Path(args.output_dir)
    records = read_manifest(args.manifest)
    targets = []
    for record in records:
        if Path(record.record_id).name != record.record_id or record.record_id in (".", ".."):
            raise ValueError("Record IDs must be safe filename components")
        targets.extend([output / f"{record.record_id}.window_qc.json",
                        output / f"{record.record_id}.window_qc.html"])
    if not args.overwrite and any(p.exists() for p in targets):
        raise ValueError("Window report exists; choose another output directory or use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for record in records:
        result = _read_processed(record, config)
        diagnostic = diagnose_windows(result["eeg"], result["valid_windows"], result["channel_names"],
                                      config.sample_rate, config.window_seconds, config.preprocessing_id, reference)
        diagnostic.update(record.metadata(), source_sha256=checkpoint_id(record.path))
        write_json(output / f"{record.record_id}.window_qc.json", diagnostic)
        result["qc"] = dict(record_id=record.record_id, window_qc=diagnostic)
        render_window_report(result, output / f"{record.record_id}.window_qc.html")
        summary.append(dict(record_id=record.record_id, valid_windows=diagnostic["base_valid_window_count"],
                            suspect_windows=diagnostic["suspect_window_count"]))
    print(json.dumps(dict(output_dir=str(output), records=summary), ensure_ascii=False))


def fit_window_qc_command(args):
    general = read_json(args.config)
    config = DataConfig(**general["data"])
    records = read_manifest(args.manifest)
    output = Path(args.output_dir)
    paths = [output / name for name in ("reference.json", "general.mark.json", "general.exclude.json")]
    if not args.overwrite and any(p.exists() for p in paths):
        raise ValueError("Window QC reference/config exists; choose another output directory or use --overwrite")
    metrics, fitted = [], []
    for record in records:
        result = _read_processed(record, config)
        metrics.append(measure_windows(result["eeg"], result["valid_windows"], config.window_samples))
        fitted.append(dict(record.metadata(), source_sha256=checkpoint_id(record.path)))
    reference = fit_reference(metrics, config.channel_names, config.sample_rate, config.window_seconds,
                              config.preprocessing_id, subject_ids=[r.subject_id for r in records],
                              score_threshold=args.score_threshold, fitted_records=fitted)
    output.mkdir(parents=True, exist_ok=True)
    write_json(paths[0], reference)
    for path, exclude in zip(paths[1:], (False, True)):
        updated = copy.deepcopy(general)
        updated["data"].update(window_qc_reference=reference, exclude_suspect_windows=exclude)
        write_json(path, updated)
    print(json.dumps(dict(reference=str(paths[0]), mark_config=str(paths[1]),
                          exclude_config=str(paths[2]), reference_windows=reference["reference_window_count"]),
                     ensure_ascii=False))
