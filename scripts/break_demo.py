"""An intentional cache bug in an isolated copy; the working code stays intact."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT/'evidence/checks'
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hw8910-mutant-') as temporary:
        copy = Path(temporary)
        shutil.copy2(ROOT/'release.py', copy/'release.py')
        shutil.copytree(ROOT/'tests', copy/'tests')
        with (copy/'release.py').open('a') as stream:
            stream.write("\n# Deliberate defect in a disposable copy.\ndef cache_key(data_path, config):\n    return digest(str(data_path).encode())\n")
        result = subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                                cwd=copy,capture_output=True,text=True,timeout=30)
        text = result.stdout+result.stderr+f'\nExit code: {result.returncode}\n'
        (folder/'intentional-failure.txt').write_text(text)
        print(text)
        if result.returncode == 0 or 'FAIL: test_changed_bytes_at_same_path_invalidate_cache' not in text:
            raise SystemExit('Mutation was not detected by the expected assertion')
    print('The failing copy was removed. Working release.py was never changed.')


if __name__ == '__main__':
    main()
