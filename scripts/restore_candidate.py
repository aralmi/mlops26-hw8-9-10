"""Restore the separately supplied adapter archive after checksum verification."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from materialize import base_snapshot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'.runtime/artifacts/restored-candidate')
    args=p.parse_args()
    meta=json.loads((ROOT/'config/candidate-artifact.json').read_text())
    if hashlib.sha256(args.archive.read_bytes()).hexdigest()!=meta['sha256']:
        raise SystemExit('Archive checksum mismatch')
    if args.output.exists():raise SystemExit('Destination already exists')
    base=base_snapshot()
    with tarfile.open(args.archive,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(meta['files']) or {m.name for m in members}!=meta['files'].keys():
            raise SystemExit('Unexpected archive entries')
        for member in members:
            name=Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.size>50*1024*1024:
                raise SystemExit('Unsafe archive entry')
            payload=archive.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest()!=meta['files'][member.name]:
                raise SystemExit('File checksum mismatch: '+member.name)
            target=args.output/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    (args.output/'base').symlink_to(base,target_is_directory=True)
    print('Restored pinned adapter:',args.output)


if __name__=='__main__':main()
