"""Сохранить фактический вывод команды, не меняя её код возврата."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error('Нужна команда')
    chunks = []
    with subprocess.Popen(args.command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        for line in proc.stdout:
            print(line,end='',flush=True)
            chunks.append(line)
        status = proc.wait()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(chunks)+f'\nExit code: {status}\n')
    sys.exit(status)


if __name__ == '__main__':
    main()
