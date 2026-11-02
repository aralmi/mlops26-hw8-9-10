"""Recoverable local publication journal; explicitly supports one writer."""
import json
import time
import uuid
from pathlib import Path

import release


def read(path):
    return json.loads(Path(path).read_text())


def attempt(root, event, **fields):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = dict(attempt_id=str(uuid.uuid4()), time_ns=time.time_ns(), event=event, **fields)
    with (root / 'attempts.jsonl').open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
    return record


def identity(manifest, report):
    return release.digest(release.canonical({'manifest': manifest, 'decision': report}))


def _finish(root, pending):
    history = root / 'release-history'
    history.mkdir(exist_ok=True)
    release.write_json(history / (pending['release_id'] + '.json'), pending)
    events_path = root / 'publication-events.json'
    events = read(events_path) if events_path.exists() else {}
    events.setdefault(pending['event_id'], pending)
    release.write_json(events_path, events)
    release.atomic_bytes(root / 'publications.jsonl', b''.join(
        release.canonical(value) + b'\n' for value in events.values()))
    (root / 'release-intent.json').unlink(missing_ok=True)


def recover(root):
    root = Path(root)
    intent = root / 'release-intent.json'
    if not intent.exists():
        return
    pending = read(intent)
    pointer = root / 'accepted.json'
    if pointer.exists() and read(pointer) == pending['manifest']:
        _finish(root, pending)
        attempt(root, 'RECOVERED_PUBLICATION', release_id=pending['release_id'])
    elif (release.digest(pointer.read_bytes()) if pointer.exists() else None) == pending['previous_sha256']:
        intent.unlink()
        attempt(root, 'DISCARDED_UNCOMMITTED_INTENT')
    else:
        raise ValueError('Pointer/journal conflict: manual recovery required')


def commit(root, manifest, report, *, fail_after_pointer=False, action='release', verify=True):
    root = Path(root)
    release.validate_release(manifest, report)
    if verify:
        release.verify_artifact(manifest, artifact_root(root, manifest))
    recover(root)
    target = root / 'accepted.json'
    release_id = identity(manifest, report)
    events = read(root / 'publication-events.json') if (root / 'publication-events.json').exists() else {}
    if target.exists() and read(target) == manifest and any(
            e['release_id'] == release_id for e in events.values()):
        attempt(root, 'ALREADY_PUBLISHED', release_id=release_id)
        return release_id
    previous = release.digest(target.read_bytes()) if target.exists() else None
    event_id = release_id if action == 'release' else release.digest(release.canonical(
        {'rollback_to': release_id, 'from_pointer_sha256': previous}))
    pending = dict(release_id=release_id, event_id=event_id, model_id=manifest['model_id'],
                   manifest=manifest, report=report, action=action, previous_sha256=previous)
    release.write_json(root / 'release-intent.json', pending)
    try:
        release.publish(manifest, report, target)
    except Exception:
        (root / 'release-intent.json').unlink(missing_ok=True)
        raise
    if fail_after_pointer:
        raise RuntimeError('Injected failure between pointer and journal')
    _finish(root, pending)
    attempt(root, 'PUBLISHED', release_id=release_id, action=action)
    return release_id


def artifact_root(root, manifest):
    uri = manifest['artifact_uri']
    if uri.startswith('file://'):
        return Path(uri[7:])
    path = Path(uri)
    if path.is_absolute():
        return path
    path = Path(root) / path
    return path.parent if path.is_file() else path


def rollback(root, release_id):
    record = read(Path(root) / 'release-history' / (release_id + '.json'))
    if record['release_id'] != release_id or identity(record['manifest'], record['report']) != release_id:
        raise ValueError('Corrupt release history')
    return commit(root, record['manifest'], record['report'], action='rollback')
