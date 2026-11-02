"""Collect actual Prefect/cache/crash/rejection evidence in isolated workspaces."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/process')
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Choose a new output directory to preserve previous experiments')
    args.output.mkdir(parents=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix='hw8910-process-') as temporary:
        root=Path(temporary)
        def run(workspace, label, *command, ok=True):
            result=subprocess.run([sys.executable,str(ROOT/'flow_lab.py'),'--workspace',str(workspace),*command],
                                  cwd=ROOT,capture_output=True,text=True,timeout=180)
            (args.output/(label+'.txt')).write_text(result.stdout+result.stderr+f'\nExit code: {result.returncode}\n')
            records.append(dict(label=label,command=list(command),exit_code=result.returncode))
            assert (result.returncode==0)==ok, label+': unexpected exit code; inspect log'
            return result
        workspace=root/'recovery'
        run(workspace,'01-init','init')
        before=hashlib.sha256((workspace/'accepted.json').read_bytes()).hexdigest()
        run(workspace,'02-fail-after-prepare','run','--fail-after-prepare',ok=False)
        recovered=run(workspace,'03-recovered','run')
        assert 'CACHE_HIT' in recovered.stdout
        run(workspace,'04-repeat','run')
        with (workspace/'input.txt').open('a') as stream:stream.write('Дополнительная строка\n')
        changed=run(workspace,'05-changed-input','run')
        assert 'PREPARED' in changed.stdout
        for label, filename in [('06-incomplete','report-incomplete.json'),('07-foreign','report-A.json'),('08-rejected','report-reject.json')]:
            pointer=(workspace/'accepted.json').read_bytes()
            run(workspace,label,'run','--report',filename,ok=False)
            after=(workspace/'accepted.json').read_bytes()
            assert after==pointer
            records[-1].update(accepted_before_sha256=hashlib.sha256(pointer).hexdigest(),
                               accepted_after_sha256=hashlib.sha256(after).hexdigest())
        crash=root/'crash'
        run(crash,'09-crash-init','init')
        run(crash,'10-crash-between-pointer-and-journal','run','--fail-after-pointer',ok=False)
        run(crash,'11-recover-journal','run')
        run(crash,'12-repeat-recovered','run')
        events=json.loads((crash/'publication-events.json').read_text())
        assert sum(e['model_id']=='fixture-B' for e in events.values())==1
        for name, workspace in [('recovery',workspace),('crash',crash)]:
            for filename in ('attempts.jsonl','publications.jsonl','publication-events.json'):
                (args.output/(name+'-'+filename)).write_bytes((workspace/filename).read_bytes())
        (args.output/'summary.json').write_text(json.dumps({'experiments':records,'synthetic':True},ensure_ascii=False,indent=2)+'\n')
    print('Saved process experiments:',args.output)


if __name__=='__main__':main()
