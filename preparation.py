"""A real cached data validation stage; no tensors cross flow task boundaries."""
import json
from pathlib import Path

import journal
import release


def configuration(manifest):
    return dict(tokenizer_revision=manifest['runtime']['base_revision'],
                tokenizer_sha256=manifest['runtime']['tokenizer_sha256'],
                template_sha256=manifest['runtime']['template_sha256'],
                max_length=manifest['runtime'].get('max_input_tokens', 128),
                code_revision=code_revision(),
                validation_contract='nonempty-unique-id-input-v1')


def code_revision():
    return release.digest(release.canonical({
        name: release.digest(Path(path).read_bytes()) for name, path in
        [('preparation.py', __file__), ('release.py', release.__file__)]}))


def prepare(root, data_path, config):
    root, data_path = Path(root), Path(data_path)
    key = release.cache_key(data_path, config)
    entry = root / 'cache' / (key + '.json')
    output = root / 'prepared' / (key + '.jsonl')
    if entry.exists() and output.exists():
        cached = journal.read(entry)
        if release.digest(output.read_bytes()) == cached['output_sha256']:
            journal.attempt(root, 'CACHE_HIT', cache_key=key, output_uri=str(output.resolve()))
            print('CACHE_HIT', key, flush=True)
            return cached
    lines = [line for line in data_path.read_text().splitlines() if line.strip()]
    rows = ([{'id': str(i), 'input': line} for i, line in enumerate(lines)]
            if config.get('format') == 'fixture-text' else [json.loads(line) for line in lines])
    ids = [row.get('id') for row in rows]
    if not rows or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Input requires unique nonempty IDs')
    if any(not isinstance(r.get('input'), str) or not r['input'].strip() for r in rows):
        raise ValueError('Input requires nonempty text')
    payload = b''.join(release.canonical(row) + b'\n' for row in rows)
    release.atomic_bytes(output, payload)
    result = dict(cache_key=key, input_sha256=release.digest(data_path.read_bytes()),
                  output_sha256=release.digest(payload), output_uri=str(output.resolve()),
                  rows=len(rows), config=config)
    release.write_json(entry, result)
    journal.attempt(root, 'PREPARED', cache_key=key, rows=len(rows))
    print('PREPARED', key, flush=True)
    return result
