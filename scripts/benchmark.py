"""A small client benchmark: preserve every success, HTTP error and timeout."""
import argparse
import concurrent.futures
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def request(url, row, mode, limit, timeout):
    started = time.perf_counter()
    record = dict(id=row['id'], concurrency=mode, max_new_tokens=limit,
                  request_id=None, status=None, error=None, model_id=None,
                  runtime_sha256=None, output_tokens=None, generation_seconds=None, queue_seconds=None)
    try:
        req = urllib.request.Request(url+'/generate', data=json.dumps({
            'prompt':row['input'], 'max_new_tokens':limit}).encode(), headers={'Content-Type':'application/json'})
        try:
            response = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            record['status'] = response.status
            payload = json.load(response)
        if record['status'] == 200:
            for field in ('request_id','model_id','runtime_sha256','output_tokens','generation_seconds','queue_seconds'):
                record[field] = payload[field]
        else:
            record['error'] = json.dumps(payload,ensure_ascii=False)
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['client_seconds'] = time.perf_counter()-started
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8091')
    parser.add_argument('--n',type=int,choices=[4,8],default=8)
    parser.add_argument('--output',type=Path,default=ROOT/'evidence/benchmark')
    parser.add_argument('--timeout',type=float,default=120)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists. Select a new directory to preserve old attempts.')
    with urllib.request.urlopen(args.url+'/ready',timeout=5) as response:
        ready = json.load(response)
    if ready.get('ready') is not True:
        raise SystemExit('Model not ready')
    rows = [json.loads(line) for line in (ROOT/'evidence/hw67/data/eval/golden.jsonl').read_text().splitlines()][:args.n]
    args.output.mkdir(parents=True)
    summaries, all_records = [], []
    with (args.output/'requests.jsonl').open('w') as raw:
        for concurrency in (1,2):
            started = time.perf_counter()
            records = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(request,args.url,row,concurrency,32,args.timeout) for row in rows]
                for future in concurrent.futures.as_completed(futures):
                    record = future.result()
                    records.append(record)
                    all_records.append(record)
                    raw.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                    raw.flush()
            wall = time.perf_counter()-started
            successful = [r for r in records if r['status']==200 and r['error'] is None]
            errors = [r for r in records if r not in successful]
            summaries.append(dict(concurrency=concurrency,n=len(records),successes=len(successful),
                errors=len(errors),batch_seconds=wall,successful_requests_per_second=len(successful)/wall,
                mean_success_seconds=statistics.mean(r['client_seconds'] for r in successful) if successful else None,
                max_success_seconds=max((r['client_seconds'] for r in successful),default=None),
                mean_error_seconds=statistics.mean(r['client_seconds'] for r in errors) if errors else None,
                max_error_seconds=max((r['client_seconds'] for r in errors),default=None),
                actual_output_tokens=sum(r['output_tokens'] for r in successful)))
    with (args.output/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (args.output/'summary.json').write_text(json.dumps({
        'request_ids':[r['id'] for r in rows], 'max_new_tokens':32,
        'ready':ready, 'modes':summaries,
        'sample_note':'CPU 4+4' if args.n==4 else '8+8',
        'measurement':'Full response latency; no TTFT, ITL or p95 estimate.'},indent=2)+'\n')
    if any(r['status']==200 and (r['model_id'] != ready['model_id'] or
           r['runtime_sha256'] != ready['runtime_sha256']) for r in all_records):
        raise SystemExit('Loaded model changed during benchmark; raw attempts preserved; comparison invalid')
    print(json.dumps(summaries,indent=2))


if __name__ == '__main__':
    main()
