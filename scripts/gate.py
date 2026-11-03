"""Recompute the saved full quality decision without loading model weights."""
import hashlib
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import journal
import lineage
import release


def main():
    pointer=ROOT/'.runtime/accepted.json'
    before=pointer.read_bytes()
    manifest=journal.read(ROOT/'.runtime/manifest-candidate.json')
    report=lineage.release_report(manifest)
    release.write_json(ROOT/'.runtime/release-report.json',report)
    if pointer.read_bytes()!=before:raise RuntimeError('Admission check changed accepted pointer')
    print('evaluation_run_id:',report['evaluation_run_id'])
    print('candidate accuracy:',report['metrics']['accuracy'])
    print('accepted.json SHA-256 before/after:',hashlib.sha256(before).hexdigest())
    print('PASS' if report['passed'] else 'REJECT:',report.get('reason') or '')
    raise SystemExit(0 if report['passed'] is True else 1)


if __name__=='__main__':main()
