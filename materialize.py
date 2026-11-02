"""Prepare local descriptors from existing artifacts. Never trains a model."""
import argparse
import hashlib
import os
from pathlib import Path

import journal
import lineage
import release


def hashes(root):
    files = {}
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and not any(p.startswith('.') for p in path.relative_to(root).parts):
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    h.update(chunk)
            files[str(path.relative_to(root))] = h.hexdigest()
    return files


def base_snapshot():
    spec = journal.read(lineage.PREVIOUS / 'evaluation/baseline.json')
    cache = Path(os.environ.get('HF_HUB_CACHE', str(Path.home() / '.cache/huggingface/hub')))
    root = cache / ('models--' + spec['name'].replace('/', '--')) / 'snapshots' / spec['revision']
    if not root.is_dir():
        raise ValueError('Pinned base snapshot absent. Download this exact revision explicitly before materialization.')
    return root


def build_manifest(kind, root, model_id, *, base_subdir='.', adapter_subdir=None):
    root = Path(root).absolute()
    _, reports, _ = lineage.evidence()
    tokenizer_subdir = adapter_subdir if kind == 'adapter' else '.'
    runtime = lineage.runtime_for(root / tokenizer_subdir)
    if kind == 'adapter':
        source = journal.read(root/'source-run.json')
        adapter_files = hashes(root/adapter_subdir)
        if source.get('run_id') != model_id or source.get('adapter_files') != adapter_files:
            raise ValueError('Candidate export/run binding missing or changed')
        files = {base_subdir+'/'+k:v for k,v in hashes(root/base_subdir).items()}
        files.update({adapter_subdir+'/'+k:v for k,v in adapter_files.items()})
        files['source-run.json'] = release.digest((root/'source-run.json').read_bytes())
    else:
        files = hashes(root)
    if not files or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Actual weight files required')
    source = reports['candidate' if kind == 'adapter' else 'baseline']
    manifest = dict(schema_version=1, model_id=model_id, model_kind=kind,
                    artifact_uri=root.as_uri(), file_hashes=files,
                    model_sha256=release.digest(release.canonical(files)),
                    protocol_sha256=source['protocol_sha256'], golden_sha256=source['golden_sha256'],
                    runtime=runtime, runtime_sha256=release.digest(release.canonical(runtime)),
                    base_subdir=base_subdir, tokenizer_subdir=tokenizer_subdir)
    if adapter_subdir:
        manifest['adapter_subdir'] = adapter_subdir
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.runtime'))
    parser.add_argument('--base-snapshot', type=Path)
    parser.add_argument('--candidate-root', type=Path,
                        help='Existing bundle containing base/ and adapter/; never downloaded implicitly')
    args = parser.parse_args()
    root = args.base_snapshot or base_snapshot()
    spec = journal.read(lineage.PREVIOUS / 'evaluation/baseline.json')
    manifest = build_manifest('base', root, 'base:' + spec['name'] + '@' + spec['revision'])
    release.write_json(args.workspace / 'manifest-baseline.json', manifest)
    lineage.bootstrap_baseline(args.workspace, manifest)
    release.write_json(args.workspace / 'prepare-config.json', __import__('preparation').configuration(manifest))
    if args.candidate_root:
        spec = journal.read(lineage.PREVIOUS / 'docs/results/candidate-model.json')
        candidate = build_manifest('adapter', args.candidate_root, spec['run_id'],
                                   base_subdir='base', adapter_subdir='adapter')
        release.write_json(args.workspace / 'manifest-candidate.json', candidate)
    print('Local descriptors written; no training, evaluation or service startup performed.')


if __name__ == '__main__':
    main()
