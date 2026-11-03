"""Real HTTP/process experiment A -> B -> A; both versions are explicit stubs."""
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/versions')
    output=parser.parse_args().output
    if output.exists():raise SystemExit('Evidence exists; preserve it before repeating the experiment')
    output.mkdir(parents=True)
    observations=[]
    process=None
    with tempfile.TemporaryDirectory(prefix='hw8910-versions-') as temp, (output/'processes.txt').open('w') as log:
        workspace=Path(temp)/'lab'
        def command(*args):
            done=subprocess.run([sys.executable,str(ROOT/'lab.py'),'--workspace',str(workspace),*args],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            assert done.returncode==0,args
        def request(path, payload=None):
            req=urllib.request.Request(url+path,data=json.dumps(payload).encode() if payload else None,
                                       headers={'Content-Type':'application/json'})
            try:r=urllib.request.urlopen(req,timeout=10)
            except urllib.error.HTTPError as exc:r=exc
            with r:return {'status':r.status,'body':json.load(r)}
        def start():
            nonlocal process,url
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1',0));port=reservation.getsockname()[1]
            url=f'http://127.0.0.1:{port}'
            process=subprocess.Popen([sys.executable,str(ROOT/'api.py'),'--workspace',str(workspace),
                '--backend','stub','--port',str(port)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('Server exited')
                try:
                    if request('/ready')['status']==200:return
                except OSError:pass
                time.sleep(.05)
            raise RuntimeError('Server startup timeout')
        def stop():
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        url=''
        def observe(label, expected):
            answer=request('/generate',{'prompt':'test','max_new_tokens':32})
            observations.append(dict(stage=label,**answer))
            assert answer['status']==200
            assert answer['body']['model_id']==expected
            assert expected in answer['body']['answer']
        try:
            command('init')
            a_id=next(iter(json.loads((workspace/'publication-events.json').read_text())))
            start();observe('loaded-A','fixture-A')
            command('run');observe('published-B-memory-still-A','fixture-A')
            stop();start();observe('restarted-on-B','fixture-B')
            command('rollback',a_id);observe('rollback-pointer-A-memory-still-B','fixture-B')
            stop();start();observe('restarted-on-A','fixture-A')
        finally:
            stop()
            (output/'responses.json').write_text(json.dumps({'synthetic':True,'observations':observations},indent=2)+'\n')
    print('STUB ONLY: A -> B -> A verified through real HTTP restarts.')


if __name__=='__main__':main()
