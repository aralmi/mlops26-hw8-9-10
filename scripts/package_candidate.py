"""Make a separate adapter artifact; weights never enter the Git tree."""
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'.runtime/artifacts/candidate'
    files={str(p.relative_to(source)):p for p in sorted((source/'adapter').rglob('*')) if p.is_file()}
    files['source-run.json']=source/'source-run.json'
    output=ROOT/'.runtime/distribution/pandas-candidate.tar.gz'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w') as archive:
            for name,path in sorted(files.items()):
                payload=path.read_bytes()
                entry=tarfile.TarInfo(name);entry.size=len(payload);entry.mtime=0;entry.mode=0o644
                archive.addfile(entry,io.BytesIO(payload))
    meta=dict(filename=output.name,sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
              bytes=output.stat().st_size,training_run_id='1efd3e7eeda54bfc9ee161cf28065807',
              base_revision='c1899de289a04d12100db370d81485cdf75e47ca',
              files={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in files.items()})
    (ROOT/'config/candidate-artifact.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(output)
    print('SHA-256:',meta['sha256'],'bytes:',meta['bytes'])


if __name__=='__main__':main()
