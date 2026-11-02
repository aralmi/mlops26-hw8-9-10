"""Local Prefect release flow; operates on descriptors, never model tensors."""
import argparse
import os
from pathlib import Path

# Set before importing Prefect; all persistent state stays in the ignored workspace.
os.environ.setdefault('PREFECT_HOME', str(Path('.prefect').resolve()))
os.environ.setdefault('PREFECT_LOCAL_STORAGE_PATH', str(Path('.prefect/results').resolve()))
os.environ.setdefault('PREFECT_SERVER_ANALYTICS_ENABLED', 'false')

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

import journal
import preparation
import release


def preparation_key(context, parameters):
    return release.digest(release.canonical({
        'workspace': str(Path(parameters['root']).resolve()),
        'key': release.cache_key(Path(parameters['data_path']), parameters['config'])}))


@task(name='prepare-input', cache_key_fn=preparation_key, persist_result=True,
      result_serializer='json', retries=0)
def prepare_task(root, data_path, config):
    return preparation.prepare(root, data_path, config)


@task(name='obtain-existing-artifact', cache_policy=NO_CACHE, retries=0)
def obtain_task(root, manifest_path):
    manifest = journal.read(manifest_path)
    release.verify_artifact(manifest, journal.artifact_root(root, manifest))
    return dict(manifest_uri=str(Path(manifest_path).resolve()),
                manifest_sha256=release.digest(release.canonical(manifest)),
                model_id=manifest['model_id'], note='Read existing artifact; no training')


@task(name='evaluate-admission', cache_policy=NO_CACHE, retries=0)
def admission_task(root, artifact, synthetic_report=None):
    manifest = journal.read(artifact['manifest_uri'])
    if release.digest(release.canonical(manifest)) != artifact['manifest_sha256']:
        raise ValueError('Manifest changed between tasks')
    if synthetic_report:
        report = journal.read(synthetic_report)
        if manifest['runtime'].get('base_revision') != 'fixture-only' or report.get('synthetic') is not True:
            raise ValueError('Synthetic route requires explicitly marked fixture data')
    else:
        import lineage
        report = lineage.release_report(manifest)
    path = Path(root) / 'release-report.json'
    release.write_json(path, report)  # Preserve REJECT as well as PASS.
    return dict(report_uri=str(path.resolve()), report_sha256=release.digest(release.canonical(report)),
                passed=report['passed'])


@task(name='publish-accepted-pointer', cache_policy=NO_CACHE, retries=0)
def publish_task(root, artifact, admission, fail_after_pointer, synthetic_report=None):
    manifest, report = journal.read(artifact['manifest_uri']), journal.read(admission['report_uri'])
    if release.digest(release.canonical(manifest)) != artifact['manifest_sha256']:
        raise ValueError('Manifest changed before publication')
    if release.digest(release.canonical(report)) != admission['report_sha256']:
        raise ValueError('Report changed before publication')
    # A saved passed=True is insufficient: recompute the original full gate here.
    if synthetic_report:
        if report != journal.read(synthetic_report) or manifest['runtime'].get('base_revision') != 'fixture-only':
            raise ValueError('Changed fixture report')
    else:
        import lineage
        if report != lineage.release_report(manifest):
            raise ValueError('Report is not the bound full quality decision')
    return journal.commit(root, manifest, report, fail_after_pointer=fail_after_pointer)


@flow(name='hw8910-release', persist_result=True, retries=0)
def release_flow(root, data_path, manifest_path, config_path,
                 fail_after_prepare=False, fail_after_pointer=False, synthetic_report=None):
    before = release.digest((Path(root)/'accepted.json').read_bytes()) if (Path(root)/'accepted.json').exists() else None
    journal.attempt(root, 'FLOW_START', accepted_before_sha256=before)
    config = {**journal.read(config_path), 'code_revision': preparation.code_revision()}
    if synthetic_report:
        config = {**config, 'format': 'fixture-text',
                  'preparation_code_sha256': release.digest(Path(preparation.__file__).read_bytes())}
    prepared = prepare_task(str(root), str(data_path), config, return_state=True)
    descriptor = prepared.result()
    journal.attempt(root, 'PREPARATION_STATE', state=prepared.name, cache_key=descriptor['cache_key'])
    print('CACHE_HIT' if prepared.name == 'Cached' else 'PREPARATION_EXECUTED', descriptor['cache_key'], flush=True)
    # Validate that a persisted Prefect descriptor still points to a real intact result.
    if release.digest(Path(descriptor['output_uri']).read_bytes()) != descriptor['output_sha256']:
        raise ValueError('Persisted prepared output changed')
    if fail_after_prepare:
        raise RuntimeError('Injected failure after preparation')
    if not synthetic_report:
        import lineage
        expected_input = lineage.ROOT/'evidence/hw67/data/eval/golden.jsonl'
        if descriptor['input_sha256'] != release.digest(expected_input.read_bytes()):
            raise ValueError('Prepared input differs from the evaluated golden set; evaluate again')
    artifact = obtain_task(str(root), str(manifest_path))
    admission = admission_task(str(root), artifact, synthetic_report)
    if admission['passed'] is not True:
        after = release.digest((Path(root)/'accepted.json').read_bytes()) if (Path(root)/'accepted.json').exists() else None
        journal.attempt(root, 'REJECTED', accepted_before_sha256=before, accepted_after_sha256=after)
        if before != after:
            raise RuntimeError('REJECT changed accepted pointer')
        raise ValueError('Quality gate rejected candidate; accepted baseline unchanged')
    return publish_task(str(root), artifact, admission, fail_after_pointer, synthetic_report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, default=Path('.runtime'))
    p.add_argument('--input', type=Path, default=Path('evidence/hw67/data/eval/golden.jsonl'))
    p.add_argument('--manifest', type=Path)
    p.add_argument('--config', type=Path)
    p.add_argument('--fail-after-prepare', action='store_true')
    p.add_argument('--fail-after-pointer', action='store_true')
    a = p.parse_args()
    release_flow(a.workspace, a.input, a.manifest or a.workspace/'manifest-candidate.json',
                 a.config or a.workspace/'prepare-config.json', a.fail_after_prepare, a.fail_after_pointer)


if __name__ == '__main__':
    main()
