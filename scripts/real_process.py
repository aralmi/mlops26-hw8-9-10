"""Cache recovery on the real golden input, with the unchanged full quality gate."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=ROOT/'.runtime/real-process')
    parser.add_argument('--output', type=Path, default=ROOT/'evidence/real-process')
    args = parser.parse_args()
    if args.workspace.exists() or args.output.exists():
        raise SystemExit('Choose new workspace/output paths to preserve earlier evidence')
    args.workspace.mkdir(parents=True)
    args.output.mkdir(parents=True)
    for name in ('accepted.json','manifest-candidate.json','manifest-baseline.json',
                 'prepare-config.json','baseline-import.json'):
        shutil.copy2(ROOT/'.runtime'/name, args.workspace/name)
    input_path = args.workspace/'input.jsonl'
    payload = (ROOT/'evidence/hw67/data/eval/golden.jsonl').read_bytes()
    input_path.write_bytes(payload)
    pointer = (args.workspace/'accepted.json').read_bytes()
    records = []

    def run(label, *flags, reason):
        command = [sys.executable, str(ROOT/'flow.py'), '--workspace', str(args.workspace),
                   '--input', str(input_path), *flags]
        result = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
        (args.output/(label+'.txt')).write_text(result.stdout+result.stderr+f'\nExit code: {result.returncode}\n')
        after = (args.workspace/'accepted.json').read_bytes()
        record = dict(label=label,command=command,exit_code=result.returncode,
                      input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                      accepted_before_sha256=hashlib.sha256(pointer).hexdigest(),
                      accepted_after_sha256=hashlib.sha256(after).hexdigest(),
                      preparation='CACHE_HIT' if 'CACHE_HIT' in result.stdout else 'PREPARED')
        records.append(record)
        assert result.returncode != 0 and reason in result.stdout+result.stderr, label
        assert after == pointer, 'Failed/rejected attempt changed accepted baseline'
        print(label,record['preparation'],'exit='+str(result.returncode),flush=True)
        return record

    run('01-fail-after-prepare','--fail-after-prepare',reason='Injected failure after preparation')
    recovered=run('02-recovered-rejected',reason='Quality gate rejected candidate')
    assert recovered['preparation']=='CACHE_HIT'
    run('03-repeat-rejected',reason='Quality gate rejected candidate')
    rows=[json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    rows[0]['input'] += ' Additional wording for the cache invalidation experiment.'
    input_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    changed=run('04-changed-input','--fail-after-prepare',reason='Injected failure after preparation')
    assert changed['preparation']=='PREPARED'
    # The changed golden set cannot use the old eval after the preparation stage.
    run('05-changed-input-closed',reason='Prepared input differs from the evaluated golden set')
    input_path.write_bytes(payload)
    restored=run('06-restored-rejected',reason='Quality gate rejected candidate')
    assert restored['preparation']=='CACHE_HIT'
    for name in ('attempts.jsonl','release-report.json'):
        shutil.copy2(args.workspace/name,args.output/name)
    (args.output/'summary.json').write_text(json.dumps({'synthetic':False,'experiments':records},indent=2)+'\n')


if __name__=='__main__':
    main()
