"""Validate saved results against raw responses, checksums and current source."""
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import lineage
import release
from source_snapshot import snapshot


def read(name):return json.loads((ROOT/name).read_text())


def main():
    index=read('evidence/manifest.json')
    assert index['code_sha256']==snapshot(), 'Source file set changed'
    for name,expected in {**index['files'],**index['code_sha256']}.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Source/evidence changed: '+name)
    _,reports,_=lineage.evidence()
    full=(ROOT/'evidence/checks/full.txt').read_text()
    count=sum(map(int,re.findall(r'Ran (\d+) tests',full)))
    check=read('evidence/checks/full.txt.json')
    assert count==57 and check['exit_code']==0
    assert check['source_sha256_before']==index['code_sha256']==check['source_sha256_after']
    assert len(re.findall(r'^OK$',full,re.M))==5 and 'FAILED (' not in full
    before=read('evidence/regressions/before.json')
    assert before['tests']==7 and len({r['test'] for r in before['failures']})==7 and not before['errors']
    for name in ('after','target'):
        result=read('evidence/regressions/'+name+'.json')
        assert result['tests']==7 and result['successful'] is True and not result['failures'] and not result['errors']
    accepted=read('evidence/manifests/accepted.json')
    release.validate_manifest(accepted)
    release_report=read('evidence/manifests/release-report.json')
    try:
        lineage.quality_gate(reports['baseline'],reports['candidate'],
                             read('previous/hw67/evaluation/params.json')['critical_slices'])
    except ValueError as exc:
        assert release_report['passed'] is False and release_report['reason']==str(exc)
    else:raise ValueError('Saved rejection no longer matches quality gate')
    expected={r['id']:r['raw'] for r in read('previous/hw67/docs/results/baseline-outputs.json')}
    replay=[json.loads(line) for line in (ROOT/'evidence/baseline-replay.jsonl').read_text().splitlines()]
    assert len(replay)==len(expected)==53 and {r['id'] for r in replay}==expected.keys()
    for r in replay:
        assert r['actual']['status']==200 and r['actual']['body']['answer']==expected[r['id']]
        assert r['actual']['body']['model_id']==accepted['model_id']
        assert r['actual']['body']['runtime_sha256']==accepted['runtime_sha256']
    raw=[json.loads(line) for line in (ROOT/'evidence/benchmark/requests.jsonl').read_text().splitlines()]
    summary=read('evidence/benchmark/summary.json')
    assert len(raw)==16
    for mode in summary['modes']:
        rows=[r for r in raw if r['concurrency']==mode['concurrency']]
        assert len(rows)==mode['n']==8 and {r['id'] for r in rows}==set(summary['request_ids'])
        good=[r for r in rows if r['status']==200 and r['error'] is None]
        assert len(good)==mode['successes'] and len(rows)-len(good)==mode['errors']
        assert all(r['model_id']==accepted['model_id'] and r['runtime_sha256']==accepted['runtime_sha256'] for r in good)
        assert math.isclose(statistics.mean(r['client_seconds'] for r in good),mode['mean_success_seconds'])
        assert math.isclose(max(r['client_seconds'] for r in good),mode['max_success_seconds'])
        assert sum(r['output_tokens'] for r in good)==mode['actual_output_tokens']
        assert math.isclose(len(good)/mode['batch_seconds'],mode['successful_requests_per_second'])
    for r in read('evidence/real-process/summary.json')['experiments']:
        assert r['accepted_before_sha256']==r['accepted_after_sha256']
    versions=read('evidence/versions/responses.json')['observations']
    assert [r['body']['model_id'] for r in versions]==['fixture-A','fixture-A','fixture-B','fixture-B','fixture-A']
    restart=read('evidence/restart/summary.json')
    assert restart['restarted'] is True and len(restart['phases'])==2
    assert all(r['answer']['body']['model_id']==accepted['model_id'] for r in restart['phases'])
    fingerprint=hashlib.sha256(b'test_release.py'+(ROOT/'tests/test_release.py').read_bytes()).hexdigest()[:12]
    assert fingerprint=='09bd65e6b886'
    print('Saved checks verified: 57/57; original defects: 7/7 reproduced and fixed')
    print('Real API replay: 53/53; benchmark: 8+8; real restart: verified')
    print('Candidate: REJECT; accepted baseline unchanged')
    print('Original test fingerprint:',fingerprint)
    print('Source and evidence SHA-256: match')


if __name__=='__main__':main()
