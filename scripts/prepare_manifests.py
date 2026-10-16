"""Prepare a disjoint source-training and new-identity registration/query episode."""
import argparse
import csv
from pathlib import Path
from eeg_siamese.data import read_manifest, audit_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--enroll-session', default='session_1')
    parser.add_argument('--query-session', default='session_2')
    args = parser.parse_args()
    source, target = read_manifest(args.source), read_manifest(args.target)
    if args.enroll_session == args.query_session:
        parser.error('Enrollment and query sessions must differ')
    if {r.session_id for r in target} != {args.enroll_session, args.query_session}:
        parser.error('Target manifest must contain exactly the enrollment and query sessions')
    enroll = [r for r in target if r.session_id == args.enroll_session]
    query = [r for r in target if r.session_id == args.query_session]
    if {r.subject_id for r in enroll} != {r.subject_id for r in query}:
        parser.error('Every target identity needs both enrollment and query data')
    audit_episode(source, enroll, query)
    for identity in {r.subject_id for r in source}:
        if len({r.session_id for r in source if r.subject_id == identity}) < 2:
            parser.error('Cross-session source training needs two sessions for every identity')
    output = args.output_dir.resolve()
    if any((output / name).exists() for name in ('train.csv', 'enroll.csv', 'query.csv')):
        parser.error('Split files already exist; use a new output directory')
    output.mkdir(parents=True, exist_ok=True)
    for name, records in [('train', source), ('enroll', enroll), ('query', query)]:
        with (output / (name + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['subject_id', 'session_id', 'record_id', 'path'])
            writer.writeheader()
            writer.writerows(r.metadata() for r in sorted(records, key=lambda r: r.record_id))
    print(output)


if __name__ == '__main__':
    main()
