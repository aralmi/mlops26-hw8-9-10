"""Recompute the HW6-7 decision from the unchanged raw evidence.

The old baseline is an existing accepted version, not a newly passing candidate.
No model is loaded by this module.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import journal
import release

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT / 'previous' / 'hw67'
sys.path.insert(0, str(PREVIOUS))
from evaluation.contract import fingerprint, read_rows, validate_golden
from evaluation.solution import quality_gate, score


def checked_previous():
    source = journal.read(ROOT / 'previous/source.json')
    for name, expected in source['files'].items():
        if release.digest((PREVIOUS / name).read_bytes()) != expected:
            raise ValueError('Changed HW6-7 source/evidence: ' + name)
    return source


def evidence():
    checked_previous()
    config = journal.read(PREVIOUS / 'evaluation/params.json')
    folder = ROOT / 'evidence/hw67'
    rows = read_rows(folder / config['golden'])
    training = read_rows(folder / config['training_index'])
    behavioral = read_rows(folder / config['behavioral'])
    proof = validate_golden(rows, training, behavioral)
    protocol = dict(config=config, golden=rows, behavioral=behavioral, training_index=training,
                    evaluation_code={p.name: p.read_text() for p in sorted((PREVIOUS / 'evaluation').glob('*.py'))})
    baseline_spec = journal.read(PREVIOUS / 'evaluation/baseline.json')
    candidate_spec = journal.read(PREVIOUS / 'docs/results/candidate-model.json')
    reports = {}
    refs = {}
    for role, spec in [('baseline', baseline_spec), ('candidate', candidate_spec)]:
        report_path = PREVIOUS / f'docs/results/{role}-eval.json'
        output_path = PREVIOUS / f'docs/results/{role}-outputs.json'
        report, outputs = journal.read(report_path), journal.read(output_path)
        by_id = {item['id']: item for item in outputs}
        if len(by_id) != len(outputs) or set(by_id) != {r['id'] for r in rows + behavioral}:
            raise ValueError('Output ID mismatch')
        expected = dict(model=spec, protocol_sha256=fingerprint(protocol),
                        golden_sha256=fingerprint(rows), baseline_spec_sha256=fingerprint(baseline_spec),
                        outputs_sha256=fingerprint(outputs), leakage=proof, status='complete')
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Stale {role} report')
        recalculated = score(rows, [by_id[r['id']] for r in rows])
        if any(report.get(k) != recalculated[k] for k in ('overall', 'slices', 'per_item')):
            raise ValueError('Metrics/raw output mismatch')
        if report['behavioral'] != score(behavioral, [by_id[r['id']] for r in behavioral]):
            raise ValueError('Behavioral/raw output mismatch')
        for path in (report_path, output_path):
            refs[str(path.relative_to(ROOT))] = release.digest(path.read_bytes())
        reports[role] = report
    return config, reports, refs


def runtime_for(tokenizer_root):
    config = journal.read(PREVIOUS / 'evaluation/params.json')
    baseline = journal.read(PREVIOUS / 'evaluation/baseline.json')
    token_files = {}
    for name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                 'added_tokens.json', 'special_tokens_map.json', 'chat_template.jinja'):
        path = Path(tokenizer_root) / name
        if path.exists():
            token_files[name] = release.digest(path.read_bytes())
    if not {'tokenizer.json', 'tokenizer_config.json'} <= token_files.keys():
        raise ValueError('Tokenizer files missing')
    cfg = journal.read(Path(tokenizer_root) / 'tokenizer_config.json')
    template_file = Path(tokenizer_root) / 'chat_template.jinja'
    template = template_file.read_text() if template_file.exists() else cfg.get('chat_template')
    if not isinstance(template, str) or not template:
        raise ValueError('Chat template missing')
    return dict(base_name=baseline['name'], base_revision=baseline['revision'],
                tokenizer_sha256=release.digest(release.canonical(token_files)),
                template_sha256=release.digest(template.encode()), device='cpu', dtype='float32',
                threads=4, max_input_tokens=4096,
                packages={'torch': '2.9.1', 'transformers': '4.57.1', 'peft': '0.18.1', 'numpy': '2.3.2'},
                generation=dict(system=config['system'], seed=config['seed'], do_sample=False,
                                enable_thinking=config['enable_thinking'], max_new_tokens=config['max_new_tokens']))


def verify_base_inventory(manifest):
    inventory = journal.read(ROOT/'config/base-files.json')
    prefix = '' if manifest['model_kind']=='base' else manifest['base_subdir'].rstrip('/')+'/'
    expected = {prefix+name:value for name,value in inventory['files'].items()}
    if any(manifest['file_hashes'].get(name)!=value for name,value in expected.items()):
        raise ValueError('Base artifact differs from the pinned evaluated HF snapshot')
    if manifest['runtime']['base_revision'] != inventory['revision']:
        raise ValueError('Base revision mismatch')


def release_report(manifest):
    config, reports, refs = evidence()
    candidate = reports['candidate']
    verify_base_inventory(manifest)
    expected_id = candidate['model']['run_id']
    if manifest['model_id'] != expected_id or manifest['model_kind'] != 'adapter':
        raise ValueError('Saved candidate evaluation belongs to another artifact')
    if any(manifest[k] != candidate[k] for k in ('protocol_sha256', 'golden_sha256')):
        raise ValueError('Manifest/evaluation protocol mismatch')
    artifact = journal.artifact_root('.', manifest)
    release.verify_artifact(manifest, artifact)
    source_run = journal.read(artifact/'source-run.json')
    if source_run.get('run_id') != expected_id or source_run.get('model_uri') != candidate['model']['uri']:
        raise ValueError('Artifact export is not bound to the evaluated training run')
    for name, expected in source_run['adapter_files'].items():
        if manifest['file_hashes'].get(manifest['adapter_subdir']+'/'+name) != expected:
            raise ValueError('Adapter export file hashes changed')
    expected_runtime = runtime_for(artifact / manifest.get('tokenizer_subdir', '.'))
    if manifest['runtime'] != expected_runtime:
        raise ValueError('Runtime changed; a new full evaluation is required')
    try:
        decision = quality_gate(reports['baseline'], candidate, config['critical_slices'])
        passed, reason = decision.get('passed') is True, None
    except ValueError as exc:
        passed, reason = False, str(exc)
        decision = {'passed': False, 'reason': reason}
    return dict(status='complete', passed=passed, reason=reason,
                evaluation_run_id=candidate['evaluation_run_id'],
                **{k: manifest[k] for k in ('model_id', 'model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256')},
                metrics={k: candidate['overall'][k] for k in ('accuracy', 'format_rate')},
                decision=decision, evidence_sha256=refs, synthetic=False,
                training_run_id=expected_id, source='HW6-7 saved full eval; no retraining')


def bootstrap_baseline(root, manifest):
    """Import the previous baseline explicitly; never manufacture a quality PASS."""
    root = Path(root)
    _, reports, refs = evidence()
    baseline = reports['baseline']
    verify_base_inventory(manifest)
    expected_id = 'base:' + baseline['model']['name'] + '@' + baseline['model']['revision']
    if manifest['model_kind'] != 'base' or manifest['model_id'] != expected_id:
        raise ValueError('Only the pinned pre-existing baseline can be bootstrapped')
    if any(manifest[k] != baseline[k] for k in ('protocol_sha256', 'golden_sha256')):
        raise ValueError('Baseline manifest/eval mismatch')
    path = journal.artifact_root(root, manifest)
    release.verify_artifact(manifest, path)
    if manifest['runtime'] != runtime_for(path):
        raise ValueError('Baseline runtime changed; full evaluation required')
    proof = dict(kind='preexisting-baseline', quality_passed=False,
                 model_id=expected_id, manifest_sha256=release.digest(release.canonical(manifest)),
                 evaluation_run_id=baseline['evaluation_run_id'], evidence_sha256=refs,
                 source_baseline='previous/hw67/evaluation/baseline.json',
                 note='Original accepted base retained after candidate rejection; not a new gate PASS.')
    target = root / 'accepted.json'
    if target.exists() and journal.read(target) != manifest:
        raise ValueError('Bootstrap cannot overwrite another accepted version; use a managed release/rollback')
    release.write_json(root / 'baseline-import.json', proof)
    release.write_json(target, manifest)
    journal.attempt(root, 'BASELINE_IMPORTED', model_id=expected_id, quality_passed=False)


def verify_accepted(root, manifest):
    root = Path(root)
    verify_base_inventory(manifest)
    if manifest.get('model_kind') == 'base':
        proof = journal.read(root / 'baseline-import.json')
        if proof.get('kind') != 'preexisting-baseline' or proof.get('quality_passed') is not False:
            raise ValueError('Missing baseline import provenance')
        if proof['manifest_sha256'] != release.digest(release.canonical(manifest)):
            raise ValueError('Baseline pointer changed')
        _, reports, refs = evidence()
        expected = reports['baseline']['model']
        if any(manifest[k] != reports['baseline'][k] for k in ('protocol_sha256','golden_sha256')):
            raise ValueError('Baseline protocol/golden mismatch')
        if (manifest['model_id'] != 'base:' + expected['name'] + '@' + expected['revision'] or
                proof['evidence_sha256'] != refs or manifest['runtime'] != runtime_for(journal.artifact_root(root, manifest))):
            raise ValueError('Baseline lineage/runtime mismatch')
        return
    record = next((journal.read(p) for p in (root / 'release-history').glob('*.json')
                   if journal.read(p)['manifest'] == manifest), None)
    if record is None or record['report'] != release_report(manifest):
        raise ValueError('No bound acceptance decision for loaded model')
    release.validate_release(manifest, record['report'])
