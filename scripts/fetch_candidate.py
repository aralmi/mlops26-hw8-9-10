"""Explicit optional export from the existing MLflow run; never trains."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import journal
import lineage
import materialize
import release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracking-uri', default='http://127.0.0.1:5006')
    parser.add_argument('--output', type=Path, default=ROOT/'.runtime/artifacts/candidate')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Destination exists; select a new path to avoid mixing artifact versions')
    import mlflow
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException
    import yaml
    # mlflow-artifacts resolves its HTTP proxy through the global tracking URI.
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    spec = journal.read(lineage.PREVIOUS/'docs/results/candidate-model.json')
    version = client.get_model_version(spec['name'], spec['version'])
    if version.run_id != spec['run_id']:
        raise ValueError('Registry version/run mismatch')
    run = client.get_run(spec['run_id'])
    if run.info.status != 'FINISHED':
        raise ValueError('Training run not finished')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Retry only explicitly transient network errors. A bad schema/weights/gate never retries.
    for attempt in range(3):
        try:
            downloaded = Path(client.download_artifacts(spec['run_id'], 'model', str(args.output.parent)))
            break
        except MlflowException as exc:
            if exc.error_code not in ('TEMPORARILY_UNAVAILABLE', 'INTERNAL_ERROR') or attempt == 2:
                raise
            time.sleep(attempt+1)
    meta = yaml.safe_load((downloaded/'MLmodel').read_text())
    if meta.get('run_id') != spec['run_id']:
        raise ValueError('Downloaded MLmodel belongs to another training run')
    adapter = (downloaded/meta['flavors']['python_function']['artifacts']['adapter']['path']).resolve()
    if not adapter.is_relative_to(downloaded.resolve()):
        raise ValueError('Adapter path escapes downloaded model')
    saved = journal.read(adapter/'manifest.json')
    baseline = journal.read(lineage.PREVIOUS/'evaluation/baseline.json')
    if saved['model']['name'] != baseline['name'] or saved['model']['revision'] != baseline['revision']:
        raise ValueError('Adapter uses another base')
    args.output.mkdir()
    shutil.copytree(adapter, args.output/'adapter', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Reuse the exact cached public snapshot through a local link, no 1.2GB duplicate.
    (args.output/'base').symlink_to(materialize.base_snapshot(), target_is_directory=True)
    release.write_json(args.output/'source-run.json', dict(run_id=spec['run_id'],
        model_uri=spec['uri'], tracking_uri=args.tracking_uri,
        adapter_files=materialize.hashes(args.output/'adapter'), base_revision=baseline['revision']))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
