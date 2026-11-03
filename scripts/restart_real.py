"""Load the same accepted real manifest in two consecutive fresh API processes."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/restart')
    output=parser.parse_args().output
    if output.exists():raise SystemExit('Preserve the previous restart experiment before repeating')
    output.mkdir(parents=True)
    records=[]
    for name in ('before','after-restart'):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/real_smoke.py'),'--output',str(output/name)],
                              cwd=ROOT,capture_output=True,text=True,timeout=360)
        (output/(name+'.txt')).write_text(result.stdout+result.stderr+f'\nExit code: {result.returncode}\n')
        if result.returncode:raise SystemExit('Real restart smoke failed: '+name)
        smoke=json.loads((output/name/'smoke.json').read_text())
        records.append(dict(phase=name,exit_code=result.returncode,answer=smoke['answer'],
                            invalid_request=smoke['invalid_request'],health=smoke['health'],
                            ready_statuses=sorted({r['status'] for r in smoke['readiness']})))
    first,second=[r['answer']['body'] for r in records]
    assert all(first[k]==second[k] for k in ('answer','model_id','runtime_sha256'))
    assert first['request_id']!=second['request_id']
    (output/'summary.json').write_text(json.dumps({'restarted':True,'phases':records},ensure_ascii=False,indent=2)+'\n')
    print('Two fresh real API processes loaded the accepted manifest and returned the same evaluated answer.')


if __name__=='__main__':main()
