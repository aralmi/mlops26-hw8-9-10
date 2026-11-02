"""Check all 50+3 saved baseline answers against the actual serving backend."""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.real_smoke import call
import lineage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8091')
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/baseline-replay.jsonl')
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Output exists; preserve previous replay')
    lineage.evidence()
    rows=[]
    for name in ('golden.jsonl','behavioral.jsonl'):
        rows.extend(json.loads(line) for line in (ROOT/'evidence/hw67/data/eval'/name).read_text().splitlines())
    expected={r['id']:r['raw'] for r in json.loads((lineage.PREVIOUS/'docs/results/baseline-outputs.json').read_text())}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    mismatches=[]
    with args.output.open('w') as stream:
        for row in rows:
            try:
                result=call(args.url,'/generate',{'prompt':row['input'],'max_new_tokens':32})
                match=result['status']==200 and result['body'].get('answer')==expected[row['id']]
                record={'id':row['id'],'expected':expected[row['id']],'actual':result,'match':match}
            except Exception as exc:
                match=False
                record={'id':row['id'],'expected':expected[row['id']],'error':str(exc),'match':False}
            stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
            if not match:mismatches.append(row['id'])
    print(f'Matched {len(rows)-len(mismatches)}/{len(rows)} saved baseline answers')
    if mismatches:raise SystemExit('Runtime mismatch: '+', '.join(mismatches))


if __name__=='__main__':main()
