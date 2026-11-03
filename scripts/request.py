"""A live HTTP request to the running API, never a replay of saved output."""
import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8091')
    p.add_argument('--invalid',action='store_true')
    p.add_argument('--health',action='store_true')
    args=p.parse_args()
    paths=['/health','/ready'] if args.health else ['/generate']
    for path in paths:
        data=None
        if path=='/generate':
            row=json.loads((ROOT/'evidence/hw67/data/eval/golden.jsonl').read_text().splitlines()[0])
            data=json.dumps({'prompt':row['input'],'max_new_tokens':100000 if args.invalid else 32}).encode()
        request=urllib.request.Request(args.url+path,data=data,headers={'Content-Type':'application/json'})
        try:response=urllib.request.urlopen(request,timeout=120)
        except urllib.error.HTTPError as exc:response=exc
        with response:
            status=response.status
            payload=json.load(response)
        print(path,'HTTP',status)
        print(json.dumps(payload,ensure_ascii=False,indent=2))
        expected=422 if args.invalid else 200
        if status!=expected:raise SystemExit(f'Expected HTTP {expected}')


if __name__=='__main__':main()
