"""Export small descriptors and hash the actual evidence, excluding model weights."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from source_snapshot import snapshot

ROOT=Path(__file__).resolve().parents[1]


def main():
    code=snapshot()
    check=json.loads((ROOT/'evidence/checks/full.txt.json').read_text())
    if check.get('exit_code')!=0 or check.get('source_sha256_before')!=code or check.get('source_sha256_after')!=code:
        raise SystemExit('Current source has no matching full green run. Record make check before exporting.')
    output=ROOT/'evidence/manifests'
    output.mkdir(parents=True,exist_ok=True)
    for name in ('accepted.json','manifest-baseline.json','manifest-candidate.json',
                 'baseline-import.json','release-report.json'):
        # Preserve pointer bytes so public SHA-256 agrees with before/after proofs.
        shutil.copy2(ROOT/'.runtime'/name,output/name)
    files={}
    for path in sorted((ROOT/'evidence').rglob('*')):
        if path.is_file() and path.name not in ('manifest.json','summary-verified.json'):
            files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'evidence/manifest.json').write_text(json.dumps(dict(
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files=files,code_sha256=code),indent=2)+'\n')
    print('Exported real descriptors and SHA-256 for',len(files),'evidence files.')


if __name__=='__main__':main()
