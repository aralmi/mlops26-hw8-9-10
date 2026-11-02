"""Start the actual model API, wait for warmup, record responses, then stop it."""
import argparse
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def call(url, path, payload=None):
    request = urllib.request.Request(url+path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type':'application/json'})
    try:
        response = urllib.request.urlopen(request,timeout=120)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        return {'status':response.status, 'body':json.load(response)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=ROOT/'.runtime')
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/real-smoke')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output exists; select a new path to preserve previous evidence')
    args.output.mkdir(parents=True)
    manifest = json.loads((args.workspace/'accepted.json').read_text())
    row = json.loads((ROOT/'evidence/hw67/data/eval/golden.jsonl').read_text().splitlines()[0])
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1',0))
        port = reservation.getsockname()[1]
    url=f'http://127.0.0.1:{port}'
    observations=[]
    with (args.output/'server.txt').open('w') as log:
        process=subprocess.Popen([sys.executable,str(ROOT/'api.py'),'--workspace',str(args.workspace),
                                  '--backend','transformers','--port',str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+300
            while time.monotonic()<deadline:
                if process.poll() is not None:
                    raise RuntimeError('API exited; see server.txt')
                try:
                    ready=call(url,'/ready')
                    observations.append(ready)
                    if ready['status']==200:
                        break
                    if ready['body'].get('load_failed'):
                        raise RuntimeError('Model load failed; see server.txt')
                except (OSError,urllib.error.URLError):
                    pass
                time.sleep(.2)
            else:
                raise RuntimeError('Model load timeout')
            health=call(url,'/health')
            actual=call(url,'/generate',{'prompt':row['input'],'max_new_tokens':32})
            invalid=call(url,'/generate',{'prompt':row['input'],'max_new_tokens':100000})
            payload=dict(health=health,readiness=observations,answer=actual,invalid_request=invalid,
                         golden_id=row['id'],model_id=manifest['model_id'],runtime_sha256=manifest['runtime_sha256'])
            (args.output/'smoke.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
            assert actual['status']==200, actual
            assert actual['body']['model_id']==manifest['model_id']
            assert actual['body']['runtime_sha256']==manifest['runtime_sha256']
            assert 'STUB ONLY' not in actual['body']['answer']
            assert invalid['status']==422
            baseline=json.loads((ROOT/'previous/hw67/docs/results/baseline-outputs.json').read_text())
            expected=next(r['raw'] for r in baseline if r['id']==row['id'])
            assert actual['body']['answer']==expected, 'Output differs from old evaluated baseline: investigate runtime before release'
            print(json.dumps(payload,ensure_ascii=False,indent=2))
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':main()
