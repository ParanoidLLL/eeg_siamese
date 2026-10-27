import argparse
import csv
import json
from pathlib import Path

from .data import Record, assert_records_disjoint, audit_episode, read_manifest
from .inference import (calibrate_threshold, classification_metrics, encode_record,
                        load_encoder, make_gallery, match, read_json,
                        validate_gallery, write_json, predict_records, checkpoint_id, identity_breakdown)
from .evaluation import load_episodes, evaluate_episodes, evaluate_development, select_threshold
from .training import select_device, train


def metadata_records(items):
    return [Record(r["subject_id"], r["session_id"], r["record_id"], Path(r["path"]).resolve()) for r in items]


def enrollment_command(args):
    device = select_device(args.device)
    encoder, config, checkpoint, model_id = load_encoder(args.checkpoint, device)
    records = read_manifest(args.manifest)
    if not args.allow_seen_identities:
        training = metadata_records(checkpoint["training_records"])
        assert_records_disjoint(training, records)
        if {r.subject_id for r in training} & {r.subject_id for r in records}:
            raise ValueError("Enrollment contains seen identities; use --allow-seen-identities only for closed-set mode")
    gallery = make_gallery(encoder, config, records, device, model_id)
    write_json(args.output, gallery)
    print(json.dumps(dict(registered=gallery["identities"], output=str(args.output)), ensure_ascii=False))


def episode_context(args):
    device = select_device(args.device)
    encoder, config, checkpoint, model_id = load_encoder(args.checkpoint, device)
    gallery = read_json(args.gallery)
    validate_gallery(gallery, model_id, encoder.config.embedding_dim)
    queries = read_manifest(args.manifest, allow_missing=True)
    enrollment = metadata_records(gallery["enrollment_records"])
    return device, encoder, config, checkpoint, model_id, gallery, enrollment, queries


def calibration_command(args):
    device, encoder, config, checkpoint, model_id, gallery, enrollment, queries = episode_context(args)
    audit_episode(metadata_records(checkpoint["training_records"]), enrollment, queries)
    rows = predict_records(encoder, config, queries, device, gallery)
    scored = [dict(row, target_id=r.subject_id if r.subject_id in gallery["identities"] else "Unknown")
              for r, row in zip(queries, rows)]
    threshold, metrics = select_threshold(scored)
    artifact = dict(
        schema_version=1, model_id=model_id, gallery_size=len(gallery["identities"]),
        threshold=threshold, objective="development_record_accuracy", tie_break="higher_threshold",
        development_metrics=metrics,
        development_subject_ids=sorted({r.subject_id for r in enrollment + queries}),
        development_records=[r.metadata() for r in enrollment + queries],
    )
    write_json(args.output, artifact)
    print(json.dumps(dict(threshold=threshold, development_metrics=metrics), ensure_ascii=False))


def prediction_command(args, evaluate=False):
    device, encoder, config, checkpoint, model_id, gallery, enrollment, queries = episode_context(args)
    assert_records_disjoint(enrollment, queries)
    calibration = None
    threshold = None
    if not args.closed_set:
        calibration = read_json(args.calibration)
        if calibration.get("schema_version") != 1 or calibration["model_id"] != model_id:
            raise ValueError("Threshold belongs to a different checkpoint; recalibrate")
        if calibration["gallery_size"] != len(gallery["identities"]):
            raise ValueError("Calibration gallery size differs from prediction gallery size")
        threshold = calibration["threshold"]
    if evaluate:
        training = metadata_records(checkpoint["training_records"])
        if calibration is not None:
            audit_episode(training, enrollment, queries)
            development = metadata_records(calibration["development_records"])
            assert_records_disjoint(development, enrollment, queries)
            if set(calibration["development_subject_ids"]) & {r.subject_id for r in enrollment + queries}:
                raise ValueError("Evaluation identities overlap threshold development identities")
        else:
            assert_records_disjoint(training, queries)
            audit_episode([], enrollment, queries, unseen=False)
            if {r.subject_id for r in queries} - set(gallery["identities"]):
                raise ValueError("Closed-set evaluation includes unregistered identities")
        if set(checkpoint.get("development_subject_ids", [])) & {r.subject_id for r in enrollment + queries}:
            raise ValueError("Evaluation identities overlap model-selection development identities")
    rows = predict_records(encoder, config, queries, device, gallery, threshold)
    if evaluate:
        metrics = classification_metrics(rows, [r.subject_id for r in queries], gallery["identities"])
        write_json(args.output, dict(protocol="closed_set" if args.closed_set else "unseen_identity_open_set",
                                     model_id=model_id, gallery_size=len(gallery["identities"]),
                                     threshold=threshold, metrics=metrics, predictions=rows,
                                     **identity_breakdown(rows, [r.subject_id for r in queries], gallery["identities"])))
        print(json.dumps(metrics, ensure_ascii=False))
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["record_id", "predicted_id", "best_id", "best_score", "score_gap", "status", "error"])
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(dict(records=len(rows), output=str(output)), ensure_ascii=False))


def episodes_command(args):
    device = select_device(args.device)
    encoder, config, checkpoint, model_id = load_encoder(args.checkpoint, device)
    episodes = load_episodes(args.episodes)
    records = [r for e in episodes for r in e.enrollment + e.queries]
    training = metadata_records(checkpoint["training_records"])
    for episode in episodes:
        audit_episode(training, episode.enrollment, episode.queries)
    ids = sorted({r.subject_id for r in records})
    size = len({r.subject_id for r in episodes[0].enrollment})
    if args.command == "calibrate-episodes":
        report = evaluate_development(encoder, config, episodes, device)
        artifact = dict(schema_version=1, model_id=model_id, gallery_size=size,
                        threshold=report["threshold"], objective="development_record_accuracy",
                        tie_break="higher_threshold", development_metrics=report["metrics"],
                        development_subject_ids=ids,
                        development_records=list({r.record_id: r.metadata() for r in records}.values()),
                        episodes=len(episodes), note=report["note"])
        write_json(args.output, artifact)
    else:
        calibration = read_json(args.calibration)
        if calibration["model_id"] != model_id or calibration["gallery_size"] != size:
            raise ValueError("Calibration model/gallery size mismatch; recalibrate")
        exposed = set(checkpoint.get("development_subject_ids", [])) | set(calibration["development_subject_ids"])
        if exposed & set(ids):
            raise ValueError("Evaluation identities overlap model-selection or threshold development identities")
        assert_records_disjoint(metadata_records(calibration["development_records"]), records)
        report = evaluate_episodes(encoder, config, episodes, device, calibration["threshold"])
        write_json(args.output, dict(report, model_id=model_id, gallery_size=size))
    print(json.dumps(dict(metrics=report["metrics"], threshold=report["threshold"]), ensure_ascii=False))


def conversion_command(args):
    import numpy as np
    from .preprocessing import PreprocessingConfig, preprocess, save_processed
    from .datasets import read_openbmi
    config = PreprocessingConfig(**read_json(args.config))
    if config.mask_scope != "channel_sample":
        raise ValueError("Local axial-only conversion requires preprocessing.quality.json with channel_sample masks")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    source_manifest = Path(args.index if args.command == "convert-openbmi" else args.manifest).resolve()
    def iter_sources():
        if args.command == "convert-openbmi":
            with source_manifest.open(newline="", encoding="utf-8-sig") as stream:
                index = list(csv.DictReader(stream))
            if not index or not {"path", "subject_id", "session_id"} <= set(index[0]):
                raise ValueError("OpenBMI index requires path,subject_id,session_id (numeric IDs)")
            for item in index:
                path = Path(item["path"])
                path = path if path.is_absolute() else source_manifest.parent / path
                for raw in read_openbmi(path, int(item["subject_id"]), int(item["session_id"]),
                                        args.input_unit, event_index_base=args.event_index_base):
                    yield raw, path, raw["provenance"]["source_sha256"]
        else:
            for record in read_manifest(source_manifest):
                with np.load(record.path, allow_pickle=False) as archive:
                    if "preprocessing_id" in archive:
                        raise ValueError("Use raw NPZ inputs; refusing to preprocess processed data twice")
                    if "valid_windows" in archive:
                        raise ValueError("Raw NPZ valid_windows needs explicit original-time interval metadata; refusing to ignore a supplied quality mask")
                    unit = str(archive["unit"].item())
                    if unit not in ("V", "uV"):
                        raise ValueError("Raw NPZ requires explicit V or uV unit")
                    if archive["eeg"].dtype.kind not in "fiu":
                        raise ValueError("Raw EEG must be real numeric data")
                    intervals = None
                    if 'invalid_intervals' in archive:
                        bounds = archive['invalid_intervals']
                        if bounds.dtype.kind not in 'iu' or bounds.ndim != 2 or bounds.shape[1] != 2:
                            raise ValueError('Raw invalid_intervals must be integer [N,2] source-sample bounds')
                        reasons = archive['invalid_interval_reasons'] if 'invalid_interval_reasons' in archive else np.full(len(bounds),'declared_acquisition_gap')
                        if reasons.shape != (len(bounds),) or reasons.dtype.kind != 'U':
                            raise ValueError('Raw interval reasons must be Unicode [N]')
                        intervals = [(int(a),int(b),str(reason)) for (a,b),reason in zip(bounds,reasons)]
                    raw = dict(record.metadata(), eeg_uv=archive["eeg"].astype(np.float64) * (1e6 if unit == "V" else 1),
                               sample_rate=float(archive["sample_rate"].item()),
                               channel_names=archive["channel_names"].tolist(),
                               events=archive["events"] if "events" in archive else None,
                               invalid_intervals=intervals)
                yield raw, record.path, checkpoint_id(record.path)
    names = set()
    input_signature = None
    for raw, path, source_id in iter_sources():
        record_id = raw["record_id"]
        if record_id in names or Path(record_id).name != record_id or record_id in (".", ".."):
            raise ValueError("Record IDs must be unique safe filename components")
        names.add(record_id)
        signature = tuple(raw["channel_names"])
        if input_signature is not None and signature != input_signature:
            raise ValueError("Mixed channel layouts: convert into separate model configurations")
        input_signature = signature
        target = output / "records" / f"{record_id}.npz"
        if target.exists() and not args.overwrite:
            raise ValueError(f"Output exists: {target}; use --overwrite to regenerate")
        noise_metadata = None
        if args.noise_kind:
            from .datasets import inject_noise
            import hashlib
            record_seed = (args.noise_seed + int(hashlib.sha256(record_id.encode()).hexdigest()[:8], 16)) % (2**32)
            raw["eeg_uv"], noise_metadata = inject_noise(raw["eeg_uv"], raw["sample_rate"], kind=args.noise_kind,
                                                       snr_db=args.noise_snr, seed=record_seed,
                                                       line_hz=args.noise_line_hz)
        processed = preprocess(raw["eeg_uv"], raw["sample_rate"], raw["channel_names"], config,
                               events=raw.get("events"), invalid_intervals=raw.get("invalid_intervals"))
        processed["qc"].update(source_path=str(path.resolve()), source_sha256=source_id,
                               subject_id=raw["subject_id"], session_id=raw["session_id"],
                               record_id=record_id, source_provenance=raw.get("provenance", {}),
                               injected_noise=noise_metadata)
        save_processed(processed, target, overwrite=args.overwrite)
        from .report import render_qc_report
        render_qc_report(raw["eeg_uv"], raw["sample_rate"], processed, target.with_suffix(".qc.html"))
        results.append(dict(subject_id=raw["subject_id"], session_id=raw["session_id"],
                            record_id=record_id, path=str(target)))
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["subject_id", "session_id", "record_id", "path"])
        writer.writeheader()
        writer.writerows(results)
    from importlib.resources import files
    template = files("eeg_siamese").joinpath("configs/axial_transformer.json")
    general = json.loads(template.read_text(encoding="utf-8"))
    general["data"].update(sample_rate=float(processed["sample_rate"]),
                           channel_names=list(input_signature), window_seconds=config.window_seconds,
                           preprocessing_id=config.preprocessing_id)
    if config.mask_scope == 'channel_sample':
        window_samples = round(config.target_sample_rate * config.window_seconds)
        patch_samples = max(2, round(config.target_sample_rate * .5))
        if window_samples % patch_samples:
            patch_samples = window_samples
        general['data']['quality'] = dict(patch_samples=patch_samples, rms_uv=100.0, derivative_uv=20.0)
        general['encoder'].update(attention_scope='axial_patches', patch_samples=patch_samples,
                                  quality_aware=True, position_encoding='sinusoidal')
    write_json(output / "general.json", general)
    print(json.dumps(dict(records=len(results), manifest=str(output / "manifest.csv"),
                          config=str(output / "general.json")), ensure_ascii=False))


def split_command(args):
    from .datasets import split_manifest
    result = split_manifest(args.manifest, args.output_dir, args.train_count, args.dev_count,
                            args.eval_count, args.seed, registered_count=3, episodes=args.episodes)
    print(json.dumps(result, ensure_ascii=False, default=str))


def audit_command(args):
    training = read_manifest(args.train)
    dev_enrollment, dev_queries = read_manifest(args.dev_enroll), read_manifest(args.dev_query)
    eval_enrollment, eval_queries = read_manifest(args.eval_enroll), read_manifest(args.eval_query)
    audit_episode(training, dev_enrollment, dev_queries)
    audit_episode(training, eval_enrollment, eval_queries)
    assert_records_disjoint(training, dev_enrollment, dev_queries, eval_enrollment, eval_queries)
    dev_ids = {r.subject_id for r in dev_enrollment + dev_queries}
    eval_ids = {r.subject_id for r in eval_enrollment + eval_queries}
    if dev_ids & eval_ids:
        raise ValueError("Development and evaluation identities overlap")
    print(json.dumps(dict(training_identities=len({r.subject_id for r in training}),
                          development_identities=len(dev_ids), evaluation_identities=len(eval_ids),
                          status="passed"), ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="EEG Siamese training, registration and matching")
    commands = parser.add_subparsers(dest="command", required=True)
    w = commands.add_parser('train-w', help='Training-only Q warmup and W log-residual continuation')
    for key in ('manifest','config','output-dir'):
        w.add_argument('--'+key, required=True)
    w.add_argument('--device', default='auto')
    w.add_argument('--warmup-epochs', type=int, help='Defaults to the continuation epoch budget in config')
    w.add_argument('--control', action='store_true', help='Also train equal-budget waveform continuation U')
    export = commands.add_parser('export-w', help='Export an existing W checkpoint as a single encoder')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--output', required=True)
    training = commands.add_parser("train")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--init", help="Compatible project checkpoint; reinitializes the auxiliary classification head")
    training.add_argument("--init-mode", choices=["strict"], default="strict")
    training.add_argument("--resume", help="Full .last.pt checkpoint, including optimizer and RNG")
    training.add_argument("--dev-episodes", help="Fixed unseen-identity development episodes JSON")
    training.add_argument("--tensorboard-dir", help="Optional TensorBoard log directory (install .[monitor])")
    training.add_argument("--dev-closed-set", action="store_true", help="Select epochs on registered development queries without rejection threshold")
    training.add_argument("--device", default="auto")
    for name in ("enroll", "calibrate", "predict", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--device", default="auto")
        if name == "enroll":
            command.add_argument("--allow-seen-identities", action="store_true")
        else:
            command.add_argument("--gallery", required=True)
        if name in ("predict", "evaluate"):
            mode = command.add_mutually_exclusive_group(required=True)
            mode.add_argument("--closed-set", action="store_true")
            mode.add_argument("--calibration", help="Open-set threshold artifact")
    audit = commands.add_parser("audit")
    for name in ("train", "dev-enroll", "dev-query", "eval-enroll", "eval-query"):
        audit.add_argument("--" + name, required=True)
    for name in ("calibrate-episodes", "evaluate-episodes"):
        command = commands.add_parser(name)
        for key in ("checkpoint", "episodes", "output"):
            command.add_argument("--" + key, required=True)
        command.add_argument("--device", default="auto")
        if name == "evaluate-episodes":
            command.add_argument("--calibration", required=True)
    for name in ("convert-openbmi", "preprocess"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, help="Preprocessing configuration JSON")
        command.add_argument("--output-dir", required=True)
        command.add_argument("--overwrite", action="store_true")
        command.add_argument("--noise-kind", choices=["broadband", "line", "burst", "drift", "dropout", "jump"])
        command.add_argument("--noise-snr", type=float, default=10.0)
        command.add_argument("--noise-seed", type=int, default=42)
        command.add_argument("--noise-line-hz", type=float, default=60.0)
        if name == "convert-openbmi":
            command.add_argument("--index", required=True)
            command.add_argument("--input-unit", choices=["V", "uV"], required=True)
            command.add_argument("--event-index-base", type=int, choices=[0, 1], default=0)
        else:
            command.add_argument("--manifest", required=True)
    split = commands.add_parser("split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--output-dir", required=True)
    split.add_argument("--train-count", type=int, default=10)
    split.add_argument("--dev-count", type=int, default=5)
    split.add_argument("--eval-count", type=int, default=5)
    split.add_argument("--episodes", type=int, default=5)
    split.add_argument("--seed", type=int, default=42)
    basic = commands.add_parser("split-closed", help="Same identities: session 1 training, session 2 testing")
    for name in ("manifest", "config", "output-dir"):
        basic.add_argument("--" + name, required=True)
    basic.add_argument("--subjects", type=int, default=20)
    basic.add_argument("--seed", type=int, default=42)
    basic.add_argument("--train-session", default="session_1")
    basic.add_argument("--test-session", default="session_2")
    check = commands.add_parser("check-windows", help="Mark suspect windows in existing NPZ files; do not change EEG")
    fit = commands.add_parser("fit-window-qc", help="Fit frozen amplitude QC using a training manifest only")
    for command in (check, fit):
        for key in ("manifest", "config", "output-dir"):
            command.add_argument("--" + key, required=True)
        command.add_argument("--overwrite", action="store_true")
    check.add_argument("--reference", help="Optional frozen training reference JSON")
    fit.add_argument("--score-threshold", type=float, default=6.0)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train-w':
            from .w_model import train_w
            print(json.dumps(train_w(args.manifest,args.config,args.output_dir,args.device,args.warmup_epochs,args.control)))
        elif args.command == 'export-w':
            from .w_model import export_w_checkpoint
            print(json.dumps(export_w_checkpoint(args.checkpoint,args.output)))
        elif args.command == "train":
            train(read_manifest(args.manifest), args.config, args.output, select_device(args.device),
                  args.init, resume=args.resume,
                  dev_episodes=load_episodes(args.dev_episodes, closed_set=args.dev_closed_set) if args.dev_episodes else None,
                  init_mode=args.init_mode, tensorboard_dir=args.tensorboard_dir, dev_closed_set=args.dev_closed_set)
        elif args.command == "enroll":
            enrollment_command(args)
        elif args.command == "calibrate":
            calibration_command(args)
        elif args.command == "audit":
            audit_command(args)
        elif args.command in ("calibrate-episodes", "evaluate-episodes"):
            episodes_command(args)
        elif args.command in ("convert-openbmi", "preprocess"):
            conversion_command(args)
        elif args.command == "split":
            split_command(args)
        elif args.command == "split-closed":
            from .protocol import split_closed
            print(json.dumps(split_closed(args.manifest, args.config, args.output_dir,
                                          args.subjects, args.seed, args.train_session, args.test_session),
                             ensure_ascii=False))
        elif args.command == "check-windows":
            from .window_qc_cli import check_windows_command
            check_windows_command(args)
        elif args.command == "fit-window-qc":
            from .window_qc_cli import fit_window_qc_command
            fit_window_qc_command(args)
        else:
            prediction_command(args, evaluate=args.command == "evaluate")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

