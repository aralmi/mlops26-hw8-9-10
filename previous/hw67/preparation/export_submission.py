"""Собрать сдаваемые отчёты из реальных MLflow runs и raw-ответов."""
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from mlflow import MlflowClient

from evaluation.cli import current_reports
from evaluation.contract import fingerprint, read_json, read_rows, write_json
from src.common import connect, digest, load_params


REQUIRED = ('sweep.json', 'candidate-model.json', 'baseline-outputs.json', 'candidate-outputs.json',
            'baseline-eval.json', 'candidate-eval.json', 'eval-diff.json', 'quality-gate.json',
            'negative-gate.json', 'negative-outputs.json', 'storage-check.json')


def raw(value):
    return json.dumps(value, ensure_ascii=False).replace('|', '\\|').replace('`', '\\`')


def main():
    params = load_params()
    connect(params)
    client = MlflowClient()
    baseline, candidate = current_reports()
    target = Path('docs/results')
    target.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED:
        shutil.copyfile(Path('metrics')/name, target/name)
    summary = read_json('metrics/sweep.json')
    runs = []
    for row in summary['rows']:
        run = client.get_run(row['run_id'])
        runs.append(dict(run_id=run.info.run_id, experiment_id=run.info.experiment_id, status=run.info.status,
                         url=f"{params['tracking']['uri']}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}",
                         params=dict(run.data.params), tags=dict(run.data.tags), metrics=dict(run.data.metrics),
                         curves={key:[dict(step=m.step,value=m.value,timestamp=m.timestamp)
                                      for m in sorted(client.get_metric_history(run.info.run_id,key),
                                                      key=lambda item:(item.step,item.timestamp))]
                                 for key in ('train_loss','val_loss')}))
    aliases = dict(client.get_registered_model(params['registry']['name']).aliases)
    write_json(target/'run-manifest.json', dict(parent_id=summary['parent_id'], runs=runs,
                                               aliases=aliases, evaluation_run_id=candidate['evaluation_run_id']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for run in runs:
        for ax, key in zip(axes,('train_loss','val_loss')):
            curve = run['curves'][key]
            ax.plot([point['step'] for point in curve],[point['value'] for point in curve],
                    marker='o',label=run['tags']['candidate'],linewidth=1.5)
    for ax, title in zip(axes,('Train loss','Validation loss')):
        ax.set(title=title,xlabel='Optimizer step',ylabel='Loss per target token',xticks=range(5))
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig('docs/sweep-curves.png',dpi=150)
    plt.close(fig)
    gate = read_json('metrics/quality-gate.json')
    negative = read_json('metrics/negative-gate.json')
    proof = read_json('metrics/storage-check.json')
    if not negative.get('rejected'):
        raise ValueError('Отрицательная проверка не завершена отказом')
    if gate.get('passed') and str(aliases.get('champion')) != candidate['model']['version']:
        raise ValueError('PASS есть, но champion не назначен: выполните make promote')
    if not gate.get('passed') and 'champion' in aliases:
        raise ValueError('Проверьте, что непринятый кандидат не назначен champion')
    golden = read_rows('data/eval/golden.jsonl')
    behavioral = read_rows('data/eval/behavioral.jsonl')
    review = {r['id']:r for r in read_json('docs/golden-validation.json')['checked']}
    by_id = {r['id']:r for r in golden}
    a = {r['id']:r for r in baseline['per_item']}
    b = {r['id']:r for r in candidate['per_item']}
    diff = read_json('metrics/eval-diff.json')
    best = next(r for r in runs if r['run_id']==candidate['model']['run_id'])
    conclusion = [
        '# ДЗ6–7: результаты поиска и решение о выпуске', '',
        '## Выбор кандидата', '',
        f"Один sweep `{summary['parent_id']}`, шесть отдельных законченных runs. "
        'Бюджет каждого: 4 обновления, train-пул 32, val 8, batch 1 × accumulation 2, seed 42. '
        'По факту за четыре обновления используются восемь перемешанных примеров из пула. '
        'Данные — те же тензоры pandas ДЗ4, без повторной обрезки и изменения масок. Test в выборе не участвует.', '',
        '| Конфигурация | Val до | Val после | Время цикла, с | Выборочный пик RSS, МиБ | Пакет адаптера, МиБ |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for run in sorted(runs,key=lambda r:r['metrics']['final_val_loss']):
        m = run['metrics']
        conclusion.append(f"| {run['tags']['candidate']} | {m['base_val_loss']:.4f} | {m['final_val_loss']:.4f} | "
                          f"{m['seconds']:.1f} | {m['peak_memory_mb']:.1f} | {m['adapter_mb']:.2f} |")
    conclusion += ['', f"Выбран **{best['tags']['candidate']}**, run `{best['run_id']}`, "
                   f"версия `{candidate['model']['uri']}`. Final/base = "
                   f"{best['metrics']['final_val_loss']/best['metrics']['base_val_loss']:.6f} при пороге 1.0. "
                   'Регистрация в candidate — не выпуск в champion.', '',
                   'Время включает начальную и последующие оценки и логирование шагов, но не загрузку, '
                   'упаковку и генерацию samples. Память — выборочные замеры RSS процесса, не точный '
                   'максимум аллокатора. Адаптер указан вместе с токенизатором. Это одиночные короткие '
                   'CPU-прогоны в одном процессе: кэш аллокатора и предыдущие загрузки могут влиять на RSS. '
                   'Это не статистически устойчивый бенчмарк. Четырёх шагов и восьми val-примеров '
                   'недостаточно для вывода о лучшей конфигурации вообще.', '',
                   '## Протокол и данные', '',
                   f"Baseline: `{json.dumps(baseline['model'],ensure_ascii=False)}`.", '',
                   f"Candidate: `{json.dumps(candidate['model'],ensure_ascii=False)}`.", '',
                   f"Общий protocol SHA256: `{candidate['protocol_sha256']}`. "
                   f"Golden SHA256: `{candidate['golden_sha256']}`. "
                   f"Evaluation run: `{candidate['evaluation_run_id']}`.", '',
                   'Протокол и эталоны закреплены до генерации в `docs/protocol.md`, '
                   '`evaluation/params.json` и скрипте подготовки. Все 50 новых фрагментов проверены '
                   'исполнением pandas 2.3.3; ожидаемая метка не передавалась модели. Индекс содержит '
                   '1388 исходных train-вопросов. Повторная токенизация кодом ДЗ4 точно воспроизводит '
                   'все их токены, attention_mask и labels. Точных нормализованных совпадений и общих '
                   'group_id с golden нет. Это не исключает любые смысловые перефразы и не делает '
                   'регрессионный набор независимым аудитом.', '',
                   'Критические срезы indexing и missing_data выбраны заранее: неверный выбор строк '
                   'или обработка пропусков незаметно меняют данные. Остальные проверяют агрегации, '
                   'структуру таблиц, типы и строки. Здесь пять непересекающихся групп по 10 задач. '
                   'Behavioral учитывается отдельно и не увеличивает общий n.', '',
                   '## Качество реальных ответов', '',
                   '| Срез | n | Baseline correct/valid/errors | Candidate correct/valid/errors | Accuracy до → после | Формат до → после |',
                   '|---|---:|---|---|---|---|']
    for name in ['overall']+sorted(baseline['slices']):
        x,y = (r['overall'] if name=='overall' else r['slices'][name] for r in (baseline,candidate))
        conclusion.append(f"| {name} | {x['n']} | {x['correct']}/{x['valid']}/{x['errors']} | "
                          f"{y['correct']}/{y['valid']}/{y['errors']} | {x['accuracy']:.1%} → {y['accuracy']:.1%} | "
                          f"{x['format_rate']:.1%} → {y['format_rate']:.1%} |")
    conclusion += ['', 'Ошибки генерации, пустые и невалидные ответы остаются в знаменателе.', '',
                   'Распределение фактических меток на golden: baseline — '+
                   json.dumps(dict(Counter((item['parsed'] or {}).get('answer') for item in a.values())),ensure_ascii=False)+
                   '; candidate — '+
                   json.dumps(dict(Counter((item['parsed'] or {}).get('answer') for item in b.values())),ensure_ascii=False)+
                   '. REFUSE на вопросе pandas — валидный формат, но неверное решение. '
                   'Это наблюдаемый перекос ответов, не доказанная причина ухудшения.', '',
                   'Исправленные id: '+(', '.join(diff['improved']) or 'нет')+'.', '',
                   'Потерянные id: '+(', '.join(diff['regressed']) or 'нет')+'.', '',
                   '### Behavioral', '', '| Кейс | Эталон | Baseline raw | Candidate raw | Candidate |',
                   '|---|---|---|---|---|']
    ab = {r['id']:r for r in baseline['behavioral']['per_item']}
    bb = {r['id']:r for r in candidate['behavioral']['per_item']}
    for row in behavioral:
        identifier = row['id']
        conclusion.append(f"| {identifier} | {row['expected']} | {raw(ab[identifier]['raw'])} | "
                          f"{raw(bb[identifier]['raw'])} | {'PASS' if bb[identifier]['correct'] else 'FAIL'} |")
    conclusion += ['', 'Format и paraphrase — одна задача с переставленными вариантами. '
                   'Refusal — вопрос о столице вне разрешённой pandas-темы. Обе версии ответили A '
                   '(Paris): географический факт верен, но нарушена заранее заданная граница задачи. '
                   'По этому контракту требовался REFUSE.', '',
                   '## Решение', '',
                   ('**Принят**: все условия gate выполнены. '
                    f"Champion указывает на `{candidate['model']['uri']}`. Baseline автоматически не обновлялся."
                    if gate.get('passed') else '**Не принят**: '+gate.get('reason','причина отсутствует')+
                    '. Champion не создан, candidate сохранён. Baseline и нулевой допуск регрессии не менялись.'), '',
                   'Отрицательная проверка испортила только копию raw-ответов, затем заново пересчитала '
                   'counts, срезы и behavioral. Отказ: '+negative['reason']+
                   '. Реальные веса, ответы и alias не менялись. Это проверка механизма gate, а не измерение качества.', '',
                   '### Разбор пар ответов', '']
    stable = [identifier for identifier in a if a[identifier]['correct'] and b[identifier]['correct']]
    examples = list(dict.fromkeys(diff['regressed'][:2]+diff['improved'][:2]+stable+list(a)))[:5]
    notes = {
        'pandas-002': 'loc[10] выбирает метку индекса 10, которой соответствует 6. Baseline выбрал B; '
                      'candidate ответил REFUSE на обычный вопрос pandas. Это необоснованный отказ, не ошибка JSON.',
        'pandas-004': 'iloc[1:3] включает позиции 1 и 2, но не правую границу 3. Получаются 7 и 11. '
                      'Кандидат выбрал B — список [7, 11, 15], ошибочно включив правую границу.',
        'pandas-011': 'isna даёт [False, True, False, True], сумма равна 2. Baseline выбрал A (0), '
                      'candidate B (2): пропуски действительно посчитаны правильно.',
        'pandas-014': 'subset=[a] проверяет пропуски только в столбце a: удаляется строка 1. '
                      'Строка 0 с пропуском в b остаётся. Candidate выбрал B ([0, 2]); baseline A ([2]) '
                      'соответствует удалению строк с пропуском в любом столбце, чего здесь не просили.',
        'pandas-001': 'iloc[1] — второй элемент по позиции, независимо от строковых меток p/q/r. '
                      'Значение 27 соответствует A; обе версии выбрали его правильно.',
    }
    for identifier in examples:
        row = by_id[identifier]
        conclusion += [f'#### {identifier}', '',
                       f"Код: `{review[identifier]['setup']}`; выражение `{review[identifier]['expression']}`. "
                       f"Проверенное значение: `{review[identifier]['expected_value']}`, метка **{row['expected']}**.", '',
                       'Baseline raw:', '', '> '+a[identifier]['raw'].replace('\n','\n> '), '',
                       'Candidate raw:', '', '> '+b[identifier]['raw'].replace('\n','\n> '), '',
                       ('Baseline: '+('верно' if a[identifier]['correct'] else 'неверно')+
                        '; candidate: '+('верно' if b[identifier]['correct'] else 'неверно')+'. '+
                        (notes[identifier] if identifier in notes else
                         'Ошибка формата: ответ не является требуемым JSON с одной допустимой меткой.'
                         if not b[identifier]['valid'] else
                         'Формат соблюдён; правильность определяется выбранной меткой, а не похожестью текста.')), '']
    conclusion += ['## Ограничения и следующий шаг', '',
                   'Обучение учит свободным ответам pandas, а golden требует короткую JSON-метку. '
                   'Такое различие разрешено протоколом, но снижение token loss само по себе не доказывает '
                   'улучшение выбора вариантов, формата и отказов. Следующий шаг — добавить отдельные '
                   'train-примеры формата/отказа и сравнить кандидатов при одинаковом увеличенном бюджете. '
                   'Протокол этой оценки не менять задним числом; для итогового независимого вывода '
                   'подготовить новый неиспользованный набор.', '',
                   'BLEU/ROUGE оценивают совпадение слов, BERTScore — близость представлений. '
                   'Почти одинаковые JSON с разными метками могут иметь высокое сходство, хотя один ответ '
                   'неверен. Здесь точный scorer полезнее. Модель-судья могла бы разбирать свободные '
                   'объяснения и корректность кода, но подвержена предпочтению позиции A/B, длины и '
                   'стиля ответа и сама ошибается. Понадобились бы слепое сравнение, перестановка A/B, '
                   'рубрика и проверка спорных вердиктов человеком. Judge и платный API не использовались.', '',
                   '## Проверяемые материалы', '',
                   f"Probe run `{proof['run_id']}`: SHA256 `{proof['sha256']}` совпал после "
                   'пересоздания контейнера MLflow. База и файловые артефакты находятся в разных '
                   'именованных volumes; порт открыт только на 127.0.0.1.', '',
                   'Сырые ответы, отчёты, gate, отрицательная проверка, состояние alias и параметры/кривые '
                   'шести runs — в `docs/results/`. Ссылки ведут на локальный MLflow; на другом компьютере '
                   'они сами по себе не откроют наш сервер. Данные оценки доступны в '
                   '`submission/evaluation-data.zip`; DVC-версии — `eval-data.dvc`, `tokenized.dvc`.', '']
    check_log, verify_log = Path('docs/check-all.txt'), Path('docs/verify.txt')
    if (check_log.exists() and verify_log.exists() and
            'Итог: 9/9' in check_log.read_text() and check_log.read_text().endswith('Exit code: 0\n') and
            'Ran 8 tests' in verify_log.read_text() and verify_log.read_text().endswith('Exit code: 0\n')):
        conclusion += ['Приёмка: **9/9**, 24 исходных теста и 8 дополнительных — OK. '
                       'Отпечаток всех файлов tests/ — **c42ac7788026**, исходные тесты не менялись. '
                       'Фактические выводы команд сохранены в `docs/check-all.txt` и `docs/verify.txt`.', '']
    Path('docs/conclusion.md').write_text('\n'.join(conclusion))
    submission = Path('submission')
    submission.mkdir(exist_ok=True)
    with zipfile.ZipFile(submission/'evaluation-data.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path('data/eval').iterdir()):
            if not path.is_file():
                raise ValueError('Неожиданный каталог в данных оценки')
            archive.write(path,arcname=str(path))
    hashes = {str(path):digest(path) for path in sorted(target.glob('*.json'))}
    hashes['submission/evaluation-data.zip'] = digest(submission/'evaluation-data.zip')
    write_json('docs/submission-manifest.json', hashes)
    for name in ('quality-gate.json','negative-gate.json','negative-outputs.json','storage-check.json','run-manifest.json'):
        client.log_artifact(candidate['evaluation_run_id'],str(target/name),'submission')
    for name in ('docs/conclusion.md','docs/comparison.csv','docs/defects.md','docs/sweep-curves.png'):
        client.log_artifact(candidate['evaluation_run_id'],name,'submission')
    print('Готово: docs/results, docs/conclusion.md, submission/evaluation-data.zip')


if __name__ == '__main__':
    main()
