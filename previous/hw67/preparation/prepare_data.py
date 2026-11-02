"""Снимок ДЗ4, полный индекс train и новые исполняемые вопросы pandas.

Эталоны заданы до генерации моделью. Каждый проверяется исполнением кода
на закреплённом pandas; labels и проверочный код в prompt не передаются.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DOCS = 'https://pandas.pydata.org/pandas-docs/version/2.3.3/user_guide/'
SOURCES = dict(indexing='indexing.html', missing_data='missing_data.html',
               aggregation='groupby.html', reshape='merging.html', types_text='text.html')

# slice, setup, expression, checked value, two plausible wrong values.
BANK = [
    ('indexing', "s=pd.Series([13,27,41],index=['p','q','r'])", 'int(s.iloc[1])', '27', ['13','41']),
    ('indexing', "s=pd.Series([8,6,4],index=[20,10,30])", 'int(s.loc[10])', '6', ['8','4']),
    ('indexing', "s=pd.Series([2,5,9,12],index=['a','b','c','d'])", "s.loc['b':'c'].tolist()", '[5, 9]', ['[5]','[9, 12]']),
    ('indexing', "s=pd.Series([3,7,11,15])", 's.iloc[1:3].tolist()', '[7, 11]', ['[7, 11, 15]','[3, 7]']),
    ('indexing', "df=pd.DataFrame({'u':[2,8,5],'v':[9,3,6]})", "df.loc[df['u']>4,'v'].tolist()", '[3, 6]', ['[9]','[8, 5]']),
    ('indexing', "df=pd.DataFrame({'a':[1,2],'b':[3,4]},index=['x','y'])", "int(df.at['y','b'])", '4', ['2','3']),
    ('indexing', "df=pd.DataFrame({'a':[4,7],'b':[6,9]})", 'int(df.iat[0,1])', '6', ['7','4']),
    ('indexing', "s=pd.Series([10,20,30],index=['z','x','y'])", 's.sort_index().tolist()', '[20, 30, 10]', ['[10, 20, 30]','[30, 20, 10]']),
    ('indexing', "df=pd.DataFrame({'v':[8,2,6]},index=['a','b','c'])", "df.drop(index='b')['v'].tolist()", '[8, 6]', ['[2]','[8, 2]']),
    ('indexing', "s=pd.Series([1,2,3],index=['aa','bb','cc'])", "s.reindex(['cc','aa']).tolist()", '[3, 1]', ['[1, 3]','[2, 3]']),
    ('missing_data', 's=pd.Series([1,np.nan,3,np.nan])', 'int(s.isna().sum())', '2', ['0','4']),
    ('missing_data', 's=pd.Series([np.nan,4,np.nan,8])', 's.fillna(0).tolist()', '[0.0, 4.0, 0.0, 8.0]', ['[4.0, 8.0]','[0.0, 0.0, 0.0, 0.0]']),
    ('missing_data', "df=pd.DataFrame({'a':[1,np.nan,3],'b':[np.nan,2,4]})", 'df.dropna().index.tolist()', '[2]', ['[0, 1]','[0, 1, 2]']),
    ('missing_data', "df=pd.DataFrame({'a':[1,np.nan,3],'b':[np.nan,2,4]})", "df.dropna(subset=['a']).index.tolist()", '[0, 2]', ['[2]','[1]']),
    ('missing_data', 's=pd.Series([2,np.nan,8])', 'float(s.mean())', '5.0', ['10.0','3.3333333333333335']),
    ('missing_data', 's=pd.Series([np.nan,np.nan])', 'float(s.sum())', '0.0', ['2.0','1.0']),
    ('missing_data', "s=pd.Series([1,None,3],dtype='Int64')", 'int(s.count())', '2', ['3','1']),
    ('missing_data', 's=pd.Series([5,np.nan,np.nan,9])', 's.ffill().tolist()', '[5.0, 5.0, 5.0, 9.0]', ['[5.0, 9.0, 9.0, 9.0]','[5.0, 0.0, 0.0, 9.0]']),
    ('missing_data', 's=pd.Series([0,np.nan,4])', 's.notna().tolist()', '[True, False, True]', ['[False, True, False]','[False, False, True]']),
    ('missing_data', "df=pd.DataFrame({'a':[1,np.nan],'b':[np.nan,np.nan]})", "df.dropna(axis=1,how='all').columns.tolist()", "['a']", ["['a', 'b']","['b']"]),
    ('aggregation', "df=pd.DataFrame({'g':['a','b','a'],'v':[3,5,7]})", "df.groupby('g')['v'].sum().tolist()", '[10, 5]', ['[3, 12]','[5, 10]']),
    ('aggregation', "df=pd.DataFrame({'g':['x','x','y'],'v':[2,np.nan,6]})", "df.groupby('g')['v'].count().tolist()", '[1, 1]', ['[2, 1]','[1, 2]']),
    ('aggregation', "df=pd.DataFrame({'g':['x','x','y'],'v':[2,np.nan,6]})", "df.groupby('g').size().tolist()", '[2, 1]', ['[1, 1]','[1, 2]']),
    ('aggregation', "df=pd.DataFrame({'g':['a','a','b'],'v':[2,8,9]})", "df.groupby('g')['v'].mean().tolist()", '[5.0, 9.0]', ['[10.0, 9.0]','[2.0, 9.0]']),
    ('aggregation', "s=pd.Series(['b','a','b','c','b'])", "int(s.value_counts().loc['b'])", '3', ['2','5']),
    ('aggregation', 's=pd.Series([4,4,9,9,9])', 'int(s.nunique())', '2', ['5','3']),
    ('aggregation', "df=pd.DataFrame({'g':['a','a','b'],'v':[2,5,7]})", "df.groupby('g')['v'].transform('sum').tolist()", '[7, 7, 7]', ['[7, 7]','[2, 5, 7]']),
    ('aggregation', 's=pd.Series([2,3,5])', 's.cumsum().tolist()', '[2, 5, 10]', ['[2, 3, 5]','[10, 10, 10]']),
    ('aggregation', 's=pd.Series([1,4,7,10])', 'float(s.median())', '5.5', ['4.0','7.0']),
    ('aggregation', "df=pd.DataFrame({'g':['b','a','b'],'v':[1,2,3]})", "df.groupby('g',sort=False)['v'].sum().index.tolist()", "['b', 'a']", ["['a', 'b']","['b', 'a', 'b']"]),
    ('reshape', "a=pd.DataFrame({'k':[1,2],'x':[4,5]}); b=pd.DataFrame({'k':[2,3],'y':[8,9]})", "a.merge(b,on='k',how='inner')['k'].tolist()", '[2]', ['[1, 2, 3]','[1, 2]']),
    ('reshape', "a=pd.DataFrame({'k':[1,2]}); b=pd.DataFrame({'k':[2,3],'v':[8,9]})", "int(len(a.merge(b,on='k',how='left')))", '2', ['1','3']),
    ('reshape', "a=pd.DataFrame({'x':[1]},index=[7]); b=pd.DataFrame({'x':[2]},index=[8])", "pd.concat([a,b],ignore_index=True).index.tolist()", '[0, 1]', ['[7, 8]','[1, 2]']),
    ('reshape', "df=pd.DataFrame({'id':[1,2],'a':[3,4],'b':[5,6]})", "int(len(df.melt(id_vars='id')))", '4', ['2','6']),
    ('reshape', "df=pd.DataFrame({'r':['a','a','b'],'c':['x','y','x'],'v':[2,3,4]})", "float(df.pivot(index='r',columns='c',values='v').loc['a','y'])", '3.0', ['2.0','4.0']),
    ('reshape', "df=pd.DataFrame({'a':[1,1,2],'b':[3,3,4]})", 'int(len(df.drop_duplicates()))', '2', ['3','1']),
    ('reshape', "df=pd.DataFrame({'a':[1,1,2],'b':[3,4,5]})", "df.drop_duplicates(subset=['a'],keep='last')['b'].tolist()", '[4, 5]', ['[3, 5]','[3, 4, 5]']),
    ('reshape', "df=pd.DataFrame({'a':[[1,2],[3]],'b':['x','y']})", "df.explode('a')['a'].tolist()", '[1, 2, 3]', ['[[1, 2], [3]]','[1, 3]']),
    ('reshape', "a=pd.DataFrame({'x':[1,2]}); b=pd.DataFrame({'y':[3,4,5]})", "int(len(a.merge(b,how='cross')))", '6', ['5','2']),
    ('reshape', "df=pd.DataFrame({'a':[1,2],'b':[3,4]})", 'list(df.T.shape)', '[2, 2]', ['[4, 1]','[1, 4]']),
    ('types_text', "s=pd.Series(['aa','b','ccc'])", 's.str.len().tolist()', '[2, 1, 3]', ['[1, 1, 1]','[3, 1, 2]']),
    ('types_text', "s=pd.Series([' a ','b  '])", 's.str.strip().tolist()', "['a', 'b']", ["[' a ', 'b  ']","['a ', 'b ']"]),
    ('types_text', "s=pd.Series(['Ab','cD'])", 's.str.lower().tolist()', "['ab', 'cd']", ["['AB', 'CD']","['Ab', 'cD']"]),
    ('types_text', "s=pd.Series(['a.b','axb'])", "s.str.contains('.',regex=False).tolist()", '[True, False]', ['[True, True]','[False, False]']),
    ('types_text', "s=pd.Series(['red','blue','red'])", "s.map({'red':1,'blue':2}).tolist()", '[1, 2, 1]', ['[2, 1, 2]','[1, 1, 1]']),
    ('types_text', "s=pd.Series(['2','5'])", 's.astype(int).sum().item()', '7', ['25','3']),
    ('types_text', "s=pd.Series(['12','oops','7'])", "int(pd.to_numeric(s,errors='coerce').isna().sum())", '1', ['0','3']),
    ('types_text', "s=pd.to_datetime(pd.Series(['2024-02-03','2025-11-09']))", 's.dt.month.tolist()', '[2, 11]', ['[3, 9]','[2024, 2025]']),
    ('types_text', "s=pd.Series(['a','b','a'],dtype='category')", 's.cat.codes.tolist()', '[0, 1, 0]', ['[1, 2, 1]','[0, 0, 1]']),
    ('types_text', "s=pd.Series(['abc','def'])", 's.str.slice(0,2).tolist()', "['ab', 'de']", ["['bc', 'ef']","['abc', 'def']"]),
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def json_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hw4', type=Path, default=ROOT.parent.parent / 'mlops26-hw4/hw4-broken')
    args = parser.parse_args()
    source = args.hw4.resolve()
    if pd.__version__ != '2.3.3':
        raise ValueError('Эталоны закреплены на pandas 2.3.3')
    manifest = dict(source_repository='https://github.com/aralmi/mlops26-hw4',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(), files={})
    for name in ('train.pt','val.pt'):
        src = source / 'data/tokenized' / name
        dst = ROOT / 'data/tokenized' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha(src) != sha(dst):
            raise ValueError(f'{dst}: другой снимок; отказ от перезаписи')
        if not dst.exists():
            shutil.copyfile(src, dst)
        manifest['files'][name] = dict(sha256=sha(dst), size=dst.stat().st_size)
    raw = [json.loads(line) for line in (source/'data/train.jsonl').read_text().splitlines() if line.strip()]
    blob = torch.load(ROOT/'data/tokenized/train.pt', weights_only=True)
    raw_ids = [row['id'] for row in raw]
    token_ids = [row['id'] for row in blob['examples']]
    if raw_ids != token_ids:
        raise ValueError('Исходный train и тензоры расходятся по id/порядку')
    train_index = [dict(id=row['id'], group_id=row['id'],
                        input='\n\n'.join(m['content'] for m in row['messages'] if m['role']=='user')) for row in raw]
    manifest.update(raw_train_sha256=sha(source/'data/train.jsonl'), n_raw_train=len(raw),
                    n_tokenized_train=len(token_ids), ordered_ids_match=True,
                    tokenization='ДЗ4: src/tokenize_data.py, params.yaml и dvc.lock по source_commit')
    verifier = '''
import json, sys, torch
from pathlib import Path
from transformers import AutoTokenizer
from src.tokenize_data import encode_example
from src.config import load_params
p = load_params()
tok = AutoTokenizer.from_pretrained(p['model']['name'], local_files_only=True)
raw = [json.loads(line) for line in Path('data/train.jsonl').read_text().splitlines() if line.strip()]
blob = torch.load(Path(sys.argv[1])/'data/tokenized/train.pt', weights_only=True)
kept = []
for row in raw:
    item = encode_example(tok, row, p, p['tokenize']['max_seq_len'])
    item.pop('_meta')
    if any(label != -100 for label in item['labels']):
        kept.append(item)
if kept != blob['examples']:
    raise ValueError('Re-encoding does not reproduce HW4 tokenized train')
print(json.dumps(dict(reencoded_train=len(kept), all_ids_tokens_masks_labels_match=True)))
'''
    manifest['retokenization_proof'] = json.loads(subprocess.check_output(
        [sys.executable, '-c', verifier, str(ROOT)], cwd=source, text=True))
    json_lines(ROOT/'data/eval/training-index.jsonl', train_index)
    attribution = source/'data/attribution.jsonl'
    shutil.copyfile(attribution, ROOT/'data/eval/train-attribution.jsonl')
    manifest['attribution_sha256'] = sha(attribution)
    golden, review = [], []
    assert len(BANK) == 50
    for i, (group, setup, expression, expected, wrong) in enumerate(BANK, 1):
        env = dict(pd=pd, np=np)
        exec(setup, env)
        actual = repr(eval(expression, env))
        if actual != expected or expected in wrong or len(set(wrong)) != 2:
            raise ValueError(f'pandas-{i:03}: {actual} != {expected}')
        options = list(wrong)
        answer_pos = (i-1) % 3
        options.insert(answer_pos, expected)
        prompt = ('Using pandas 2.3.3, what is the result of the final expression?\n'
                  'Assume import pandas as pd and import numpy as np.\n'
                  f'{setup}\n{expression}\n' + '\n'.join(f'{label}: {value}' for label, value in zip('ABC',options)) +
                  '\nchoices: A, B, C, REFUSE. Return only a JSON object with the answer label.')
        row = dict(id=f'pandas-{i:03}', group_id=f'fixture-v1-{i:03}', input=prompt,
                   expected='ABC'[answer_pos], choices=['A','B','C','REFUSE'], slices=[group],
                   source=f'Original executable fixture v1; semantics: {DOCS}{SOURCES[group]}')
        golden.append(row)
        review.append(dict(id=row['id'], setup=setup, expression=expression, expected_value=expected,
                           actual_value=actual, expected=row['expected'], source=row['source']))
    behavior = [
        dict(id='behavior-format', group_id='behavior-selection',
             input='Pandas format check: s = pd.Series([6, 14, 22]). What is int(s.iloc[1])?\nA: 6\nB: 14\nC: 22\nchoices: A, B, C, REFUSE. Reply only with the required JSON.',
             expected='B', choices=['A','B','C','REFUSE'], slices=['format'], type='format', source=DOCS+'indexing.html'),
        dict(id='behavior-paraphrase', group_id='behavior-selection',
             input='A pandas Series has values 6, 14, 22 in that order. Choose the value at zero-based position one.\nA: 22\nB: 6\nC: 14\nchoices: A, B, C, REFUSE.',
             expected='C', choices=['A','B','C','REFUSE'], slices=['paraphrase'], type='paraphrase', source=DOCS+'indexing.html'),
        dict(id='behavior-refusal', group_id='behavior-outside-topic',
             input='What is the capital of France?\nA: Paris\nB: Rome\nC: Berlin\nchoices: A, B, C, REFUSE.',
             expected='REFUSE', choices=['A','B','C','REFUSE'], slices=['refusal'], type='refusal', source='Original out-of-domain probe; pandas-only boundary in evaluation/params.json'),
    ]
    from evaluation.contract import validate_golden
    proof = validate_golden(golden, train_index, behavior)
    manifest['leakage'] = proof
    manifest['golden_label_counts'] = dict(Counter(row['expected'] for row in golden))
    json_lines(ROOT/'data/eval/golden.jsonl', golden)
    json_lines(ROOT/'data/eval/behavioral.jsonl', behavior)
    json_file(ROOT/'docs/data-lineage.json', manifest)
    json_file(ROOT/'docs/golden-validation.json', dict(pandas=pd.__version__, checked=review))
    print(json.dumps(dict(**proof, executable_golden_passed=len(review)),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
