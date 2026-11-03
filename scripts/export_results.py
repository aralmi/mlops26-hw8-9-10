"""Export small descriptors and hash the actual evidence, excluding model weights."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'evidence/manifests'
    output.mkdir(parents=True,exist_ok=True)
    for name in ('accepted.json','manifest-baseline.json','manifest-candidate.json',
                 'baseline-import.json','release-report.json'):
        value=json.loads((ROOT/'.runtime'/name).read_text())
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    files={}
    for path in sorted((ROOT/'evidence').rglob('*')):
        if path.is_file() and path.name not in ('manifest.json','summary-verified.json'):
            files[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    code={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for folder in (ROOT,ROOT/'scripts',ROOT/'integration',ROOT/'verification',ROOT/'tests')
          for p in sorted(folder.glob('*.py'))}
    for name in ('pyproject.toml','uv.lock','Makefile','config/base-files.json'):
        code[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for path in sorted((ROOT/'config').glob('*.json')):
        code[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'evidence/manifest.json').write_text(json.dumps(dict(
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files=files,code_sha256=code),indent=2)+'\n')
    print('Exported real descriptors and SHA-256 for',len(files),'evidence files.')


if __name__=='__main__':main()
