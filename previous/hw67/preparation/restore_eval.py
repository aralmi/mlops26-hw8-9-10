"""Восстановить только открытые данные оценки из приложения к сдаче."""
import hashlib
import zipfile
from pathlib import Path

from evaluation.contract import read_json


def main():
    source = Path('submission/evaluation-data.zip')
    manifest = read_json('docs/submission-manifest.json')
    if hashlib.sha256(source.read_bytes()).hexdigest() != manifest[str(source)]:
        raise ValueError('SHA256 приложения не совпадает')
    with zipfile.ZipFile(source) as archive:
        for name in archive.namelist():
            target = Path(name)
            if target.parent != Path('data/eval') or target.name not in {
                '.gitkeep','golden.jsonl','behavioral.jsonl','training-index.jsonl','train-attribution.jsonl'}:
                raise ValueError(f'Неожиданный путь: {name}')
            content = archive.read(name)
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'{target}: другие данные, отказ от перезаписи')
        for name in archive.namelist():
            target = Path(name)
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():
                target.write_bytes(archive.read(name))
    print('Данные оценки восстановлены; make status сверит DVC-версию')


if __name__ == '__main__':
    main()
