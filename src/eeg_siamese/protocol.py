"""Basic closed-set identity recognition across independent sessions."""
import csv
from pathlib import Path

from .data import audit_episode, read_manifest
from .inference import checkpoint_id, read_json, write_json


def split_closed(manifest, config, output_dir, subjects=20, seed=42,
                 train_session="session_1", test_session="session_2"):
    import numpy as np
    records = read_manifest(manifest)
    ids = sorted({r.subject_id for r in records})
    if not isinstance(subjects, int) or not 2 <= subjects <= len(ids):
        raise ValueError("Choose 2..available identity count for closed-set recognition")
    if train_session == test_session:
        raise ValueError("Training and testing must use independent sessions")
    chosen = sorted(str(s) for s in np.random.default_rng(seed).choice(ids, subjects, replace=False))
    train = [r for r in records if r.subject_id in chosen and r.session_id == train_session]
    test = [r for r in records if r.subject_id in chosen and r.session_id == test_session]
    if {r.subject_id for r in train} != set(chosen) or {r.subject_id for r in test} != set(chosen):
        raise ValueError("Every selected identity needs both independent sessions")
    audit_episode([], train, test, unseen=False)
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty split directory to preserve the frozen experiment")
    output.mkdir(parents=True, exist_ok=True)
    for name, group in [("train", train), ("test", test)]:
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["subject_id", "session_id", "record_id", "path"])
            writer.writeheader()
            writer.writerows(r.metadata() for r in group)
    general = read_json(config)
    general.setdefault("training", {})["cross_session"] = False
    general["training"]["identities_per_batch"] = min(subjects, general["training"].get("identities_per_batch", 4))
    write_json(output / "general.json", general)
    result = dict(protocol="closed_set_cross_session", seed=seed, identities=chosen,
                  train_session=train_session, test_session=test_session,
                  training_records=len(train), test_records=len(test),
                  source_manifest_sha256=checkpoint_id(manifest),
                  unused_identities=sorted(set(ids) - set(chosen)),
                  note="Test labels must not be used to choose epochs, preprocessing or models")
    write_json(output / "split.json", result)
    return result
