"""Rebuild searchable catalogs for the saved runs without modifying their data."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, fields, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    entries, files = [], []
    for collection in sorted((ROOT / 'runs').iterdir()):
        if not collection.is_dir():
            continue
        for run in sorted(collection.iterdir()):
            if not run.is_dir():
                continue
            candidates = [run] + [p for p in sorted(run.iterdir())
                                  if p.is_dir() and p.name.startswith(('evaluation_', 'replays'))]
            for item in candidates:
                kind = ('checkpoint' if 'checkpoint' in item.name else 'replay' if item.name.startswith('replays')
                        else 'frozen_evaluation' if item.name.startswith('evaluation_') else collection.name)
                entries.append({'collection': collection.name, 'run_id': item.name,
                                'kind': kind, 'path': item.relative_to(ROOT).as_posix()})
            for path in sorted(run.rglob('*')):
                if not path.is_file() or path.name == '.DS_Store':
                    continue
                suffix = path.suffix.lower()
                kind = {'.csv': 'data', '.json': 'configuration_or_metadata', '.pt': 'model',
                        '.npz': 'array_or_model', '.png': 'figure', '.gif': 'replay',
                        '.html': 'viewer', '.md': 'report', '.log': 'log'}.get(suffix, 'other')
                files.append({'collection': collection.name, 'run_id': run.name, 'kind': kind,
                              'path': path.relative_to(ROOT).as_posix(), 'format': suffix.lstrip('.'),
                              'bytes': path.stat().st_size,
                              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    (ROOT / 'data').mkdir(exist_ok=True)
    write_csv(ROOT / 'runs/index.csv', ['collection', 'run_id', 'kind', 'path'], entries)
    write_csv(ROOT / 'data/catalog.csv', ['collection', 'run_id', 'kind', 'path', 'format', 'bytes', 'sha256'], files)
    lines = ['# Saved runs', '',
             'Recorded models, data, configuration, charts, and replays are grouped by experiment.',
             'Run IDs are preserved. A checkpoint folder is not a separate completed experiment.', '',
             '| Collection | Run | Type |', '| --- | --- | --- |']
    for row in entries:
        relative = Path(row['path']).relative_to('runs').as_posix()
        lines.append(f"| {row['collection']} | [{row['run_id']}]({relative}/) | {row['kind']} |")
    lines += ['', 'Search [index.csv](index.csv) for runs or [the data catalog](../data/catalog.csv) for individual files.',
              'Regenerate both catalogs from the repository root with `python scripts/catalog_runs.py`.', '']
    (ROOT / 'runs/README.md').write_text('\n'.join(lines))
    print(f'Indexed {len(entries)} run entries and {len(files)} recorded files.')


if __name__ == '__main__':
    main()
