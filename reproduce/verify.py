"""Verify package hashes, fixed partitions and dataset separation offline."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'manifests/input_files.json').read_text())
    for name, expected in manifest.items():
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert digest(path) == expected, ('File changed', name)
    result = {}
    for dataset, name in [('AI4I', 'tasks_ai4i.csv'), ('PdM', 'tasks_azure.csv')]:
        path = ROOT / 'Dataset/Preprocessed' / dataset / name
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == len({r['task_id'] for r in rows}) == 300
        assert Counter(r['split'] for r in rows) == {'search': 100, 'validation': 50, 'test': 150}
        test = {r['task_id'] for r in rows if r['split'] == 'test'}
        planned = json.loads((ROOT / 'manifests' / (dataset + '_test.json')).read_text())
        assert test == set(planned['task_ids'])
        unit = 'udi' if dataset == 'AI4I' else 'machineID'
        units = {s: {r[unit] for r in rows if r['split'] == s} for s in ['search', 'validation', 'test']}
        assert not units['search'] & units['validation']
        assert not units['search'] & units['test']
        assert not units['validation'] & units['test']
        if dataset == 'AI4I':
            quotas = {'No Failure': (26, 13, 39), 'TWF': (14, 7, 21),
                      'HDF': (20, 10, 30), 'PWF': (20, 10, 30), 'OSF': (20, 10, 30)}
            assert len({r['udi'] for r in rows}) == 300
            for label, counts in quotas.items():
                for split, n in zip(['search', 'validation', 'test'], counts):
                    assert sum(r['gold_label'] == label and r['split'] == split for r in rows) == n
        else:
            assert len(units['test']) == 45
        result[dataset] = dict(cases=300, unique_test_cases=150, test_units=len(units['test']))
    if args.artifacts:
        assets = json.loads((ROOT / 'artifacts/index.json').read_text())['assets']
        for asset in assets:
            assert digest(args.artifacts / asset['file']) == asset['sha256'], asset['file']
    print(json.dumps(dict(status='PASS', input_files=len(manifest), datasets=result)))


if __name__ == '__main__':
    main()
