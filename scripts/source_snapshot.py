"""Small source snapshot shared by command recording and result export."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def snapshot():
    paths=set()
    for folder in (ROOT,ROOT/'scripts',ROOT/'integration',ROOT/'verification',ROOT/'tests'):
        paths.update(folder.glob('*.py'))
    paths.update((ROOT/'config').glob('*.json'))
    paths.update(ROOT/name for name in ('pyproject.toml','uv.lock','Makefile','.python-version'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if p.is_file()}
