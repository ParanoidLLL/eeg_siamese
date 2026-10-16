"""Fixed enrollment/query episodes; failed signals remain in the denominator."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Record, audit_episode, read_manifest
from .inference import make_gallery, predict_records, read_json


@dataclass(frozen=True)
class Episode:
    enrollment: list[Record]
    queries: list[Record]
    name: str = "episode"


def load_episodes(path, closed_set=False):
    path = Path(path).resolve()
    items = read_json(path)
    if not isinstance(items, list) or not items:
        raise ValueError("Episodes must be a nonempty JSON list")
    result = []
    for index, item in enumerate(items):
        def resolve(key):
            source = Path(item[key])
            return source if source.is_absolute() else path.parent / source
        episode = Episode(read_manifest(resolve("enrollment")),
                          read_manifest(resolve("query"), allow_missing=True),
                          str(item.get("name", f"episode_{index:03d}")))
        audit_episode([], episode.enrollment, episode.queries)
        enrolled = {r.subject_id for r in episode.enrollment}
        query_ids = {r.subject_id for r in episode.queries}
        if closed_set and (not enrolled or query_ids != enrolled):
            raise ValueError("Closed-set development requires exactly the registered query identities")
        if not closed_set and (not enrolled <= query_ids or not query_ids - enrolled):
            raise ValueError("Every episode needs registered queries and unregistered identities")
        result.append(episode)
    if len({e.name for e in result}) != len(result):
        raise ValueError("Episode names must be unique")
    sizes = {len({r.subject_id for r in e.enrollment}) for e in result}
    if len(sizes) != 1:
        raise ValueError("All episodes must use the same gallery size")
    return result


def summarize_rows(rows, threshold=None):
    if not rows:
        raise ValueError("No query records")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    counts = dict(records=len(rows), known_records=0, unknown_records=0,
                  known_correct=0, known_wrong_identity=0, known_rejected=0,
                  unknown_accepted=0, failed_records=0)
    correct = 0
    for row in rows:
        known = row["target_id"] != "Unknown"
        counts["known_records" if known else "unknown_records"] += 1
        if row["status"] != "ok":
            counts["failed_records"] += 1
            continue
        prediction = row["best_id"]
        if threshold is not None and row["best_score"] < threshold:
            prediction = "Unknown"
        correct += int(prediction == row["target_id"])
        if known:
            counts["known_correct"] += int(prediction == row["target_id"])
            counts["known_wrong_identity"] += int(prediction not in (row["target_id"], "Unknown"))
            counts["known_rejected"] += int(prediction == "Unknown")
        else:
            counts["unknown_accepted"] += int(prediction != "Unknown")
    counts.update(accuracy=correct / len(rows), coverage=1 - counts["failed_records"] / len(rows))
    for metric, numerator, denominator in [
        ("known_identification_rate", "known_correct", "known_records"),
        ("known_rejection_rate", "known_rejected", "known_records"),
        ("unknown_acceptance_rate", "unknown_accepted", "unknown_records"),
    ]:
        counts[metric] = counts[numerator] / counts[denominator] if counts[denominator] else None
    return counts


def select_threshold(rows):
    if not any(r["target_id"] == "Unknown" for r in rows) or not any(r["target_id"] != "Unknown" for r in rows):
        raise ValueError("Calibration needs registered and unregistered queries")
    scores = [r["best_score"] for r in rows if r["status"] == "ok"]
    if not scores:
        raise ValueError("No usable query scores for calibration")
    candidates = sorted(set(scores) | {float(np.nextafter(max(scores), np.inf))})
    best = None
    for threshold in candidates:
        metrics = summarize_rows(rows, threshold)
        if best is None or metrics["accuracy"] >= best[1]["accuracy"]:
            best = threshold, metrics
    return best


def evaluate_episodes(encoder, config, episodes, device, threshold=None):
    rows = []
    for episode in episodes:
        gallery = make_gallery(encoder, config, episode.enrollment, device, "development")
        for record, row in zip(episode.queries, predict_records(encoder, config, episode.queries, device, gallery, threshold)):
            rows.append(dict(row, episode=episode.name, subject_id=record.subject_id,
                             target_id=record.subject_id if record.subject_id in gallery["identities"] else "Unknown"))
    return dict(rows=rows, metrics=summarize_rows(rows, threshold), threshold=threshold,
                episodes=len(episodes), note="Repeated queries across episodes are not independent observations")


def evaluate_development(encoder, config, episodes, device, closed_set=False):
    report = evaluate_episodes(encoder, config, episodes, device)
    if closed_set:
        if any(row["target_id"] == "Unknown" for row in report["rows"]):
            raise ValueError("Closed-set development contains unregistered queries")
        report.update(protocol="closed_set_development", threshold=None)
        return report
    threshold, metrics = select_threshold(report["rows"])
    report.update(threshold=threshold, metrics=metrics)
    return report
