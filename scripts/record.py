"""Preserve the real exit code and complete stdout/stderr of a command."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from source_snapshot import snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Command required after log path')
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    source_before = snapshot()
    with args.log.open('w') as stream:
        stream.write('$ ' + ' '.join(command) + '\n')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            stream.write(line)
            stream.flush()
        result = process.wait()
        stream.write(f'\nExit code: {result}\n')
    args.log.with_suffix(args.log.suffix+'.json').write_text(json.dumps({
        'command':command,'exit_code':result,'started_unix':started,'elapsed_seconds':time.time()-started,
        'source_sha256_before':source_before,'source_sha256_after':snapshot()},indent=2)+'\n')
    raise SystemExit(result)


if __name__ == '__main__':
    main()
