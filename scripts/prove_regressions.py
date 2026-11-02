"""Run identical tests on the saved starter and on the fixed implementation."""
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = '7204896'


def record(command, path, env):
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1200)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '$ ' + ' '.join(map(str, command)) + '\n' + process.stdout + process.stderr + f'\nExit code: {process.returncode}\n'
    path.write_text(text)
    print(text, flush=True)
    return process.returncode


def main():
    logs = ROOT/'evidence/regressions'
    data = subprocess.check_output(['git','archive',ORIGINAL], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix='hw8910-original-') as temporary:
        original = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(original, filter='data')
        before_env = {**os.environ, 'HW8910_PROJECT':str(original),
                      'HW8910_RUNNER':'lab.py','HW8910_SERVICE':'service.py'}
        before = record([sys.executable,'scripts/run_integrations.py','--json',str(logs/'before.json')],
                        logs/'before.txt', before_env)
        result = json.loads((logs/'before.json').read_text())
        failed = {entry['test'] for entry in result['failures']}
        if before == 0 or result['tests'] != 7 or len(failed) != 7 or result['errors'] or result['skipped']:
            raise SystemExit('Evidence invalid: expected seven assertion failures, not import/CLI errors')
        after_env = {**os.environ, 'HW8910_PROJECT':str(ROOT),
                     'HW8910_RUNNER':'lab.py','HW8910_SERVICE':'service.py'}
        after = record([sys.executable,'scripts/run_integrations.py','--json',str(logs/'after.json')],
                       logs/'after.txt', after_env)
        if after != 0:
            raise SystemExit('Fixed lightweight adapter failed; inspect after.txt')
        target_env = {**after_env, 'HW8910_RUNNER':'flow_lab.py','HW8910_SERVICE':'api.py'}
        target = record([sys.executable,'scripts/run_integrations.py','--json',str(logs/'target.json')],
                        logs/'target.txt', target_env)
        if target != 0:
            raise SystemExit('Prefect/FastAPI target failed; inspect target.txt')
    difference = subprocess.check_output(['git','diff',ORIGINAL,'--','release.py','lab.py','service.py'],cwd=ROOT)
    (logs/'fix.diff').write_bytes(difference)
    print('Seven assertion failures before; seven passes on each corrected adapter.')


if __name__ == '__main__':
    main()
