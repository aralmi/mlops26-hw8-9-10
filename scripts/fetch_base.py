"""Explicit public HF download; no model is loaded or trained."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    from huggingface_hub import snapshot_download
    inventory=json.loads((ROOT/'config/base-files.json').read_text())
    path=snapshot_download(inventory['repository'],revision=inventory['revision'],
                           allow_patterns=list(inventory['files']))
    print('Pinned base snapshot:',path)


if __name__=='__main__':main()
