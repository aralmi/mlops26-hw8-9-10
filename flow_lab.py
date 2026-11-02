"""Explicit STUB ONLY adapter preserving the original lab.py CLI for tests."""
import argparse
from pathlib import Path

import lab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path('.lab-prefect'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('init')
    run = sub.add_parser('run')
    run.add_argument('--report', default='report-B.json')
    run.add_argument('--fail-after-prepare', action='store_true')
    run.add_argument('--fail-after-pointer', action='store_true')
    args = parser.parse_args()
    if args.command == 'init':
        lab.initialize(args.workspace)
        return
    from flow import release_flow
    root = args.workspace
    release_flow(root, root/'input.txt', root/'manifest-B.json', root/'config.json',
                 args.fail_after_prepare, args.fail_after_pointer, str(root/args.report))


if __name__ == '__main__':
    main()
