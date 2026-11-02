"""Contracts shared by the lightweight lab, Prefect flow and API."""
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

HASH = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def cache_key(data_path: Path, config: dict) -> str:
    return digest(canonical({'data_sha256': digest(Path(data_path).read_bytes()), 'config': config}))


def _string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field}: expected nonempty string')


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1:
        raise ValueError('schema_version must be integer 1')
    for field in ('model_id', 'artifact_uri'):
        _string(manifest.get(field), field)
    for field in ('model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256'):
        if not isinstance(manifest.get(field), str) or not HASH.fullmatch(manifest[field]):
            raise ValueError(f'{field}: expected lowercase SHA-256')
    runtime = manifest.get('runtime')
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('runtime must be a nonempty mapping')
    try:
        runtime_hash = digest(canonical(runtime))
    except (TypeError, ValueError) as exc:
        raise ValueError('runtime is not finite JSON') from exc
    if runtime_hash != manifest['runtime_sha256']:
        raise ValueError('runtime_sha256 mismatch')


def validate_release(manifest: dict, report: dict) -> None:
    validate_manifest(manifest)
    if not isinstance(report, dict) or report.get('status') != 'complete' or report.get('passed') is not True:
        raise ValueError('Complete boolean PASS required')
    _string(report.get('evaluation_run_id'), 'evaluation_run_id')
    for field in ('model_id', 'model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256'):
        if report.get(field) != manifest[field]:
            raise ValueError(f'Unrelated report: {field}')
    metrics = report.get('metrics')
    if not isinstance(metrics, dict):
        raise ValueError('metrics missing')
    for field in ('accuracy', 'format_rate'):
        value = metrics.get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid metric {field}')


def atomic_bytes(target, payload):
    """One writer. A failed replace never truncates the previous pointer."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() == payload:
        return
    fd, name = tempfile.mkstemp(prefix='.' + target.name + '.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, canonical(value) + b'\n')


def publish(manifest: dict, report: dict, target: Path) -> None:
    validate_release(manifest, report)
    write_json(target, manifest)


def verify_artifact(manifest, root):
    """Hash the pinned files, not just a URI or a manifest field."""
    validate_manifest(manifest)
    root = Path(root).resolve()
    files = manifest.get('file_hashes')
    if not isinstance(files, dict) or not files:
        raise ValueError('Pinned artifact file list missing')
    if digest(canonical(files)) != manifest['model_sha256']:
        raise ValueError('model_sha256 does not bind file list')
    for name, expected in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or not isinstance(expected, str) or not HASH.fullmatch(expected):
            raise ValueError('Unsafe artifact entry')
        # HF snapshots contain symlinks into a content-addressed blob store.
        hasher = hashlib.sha256()
        with (root / rel).open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        if hasher.hexdigest() != expected:
            raise ValueError(f'Artifact changed: {name}')
    return root
