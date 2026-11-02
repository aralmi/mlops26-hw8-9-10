"""Строгая оценка ответов и допуск без разрешённой регрессии."""

import json
import math


def _unique(items, name):
    if not isinstance(items, list) or not items:
        raise ValueError(f'{name}: нужен непустой список')
    result = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) or not item['id']:
            raise ValueError(f'{name}: неверный id')
        if item['id'] in result:
            raise ValueError(f'{name}: повтор id {item["id"]}')
        result[item['id']] = item
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Повтор JSON-ключа')
        result[key] = value
    return result


def _counts(items):
    n = len(items)
    correct = sum(item['correct'] for item in items)
    valid = sum(item['valid'] for item in items)
    errors = sum(item['error'] is not None for item in items)
    return dict(n=n, correct=correct, valid=valid, errors=errors,
                accuracy=correct / n if n else 0.0, format_rate=valid / n if n else 0.0)


def _aggregate(items):
    names = sorted({name for item in items for name in item['slices']})
    return dict(overall=_counts(items),
                slices={name: _counts([item for item in items if name in item['slices']]) for name in names},
                per_item=items)


def score(rows, outputs):
    """Верните overall, slices и per_item. Не удаляйте плохие ответы из знаменателя.

    Строгий ответ: JSON с единственным строковым полем answer из choices.
    outputs: [{id, raw, error, latency_seconds}, ...].
    В overall и каждом срезе: n, correct, valid, errors, accuracy, format_rate.
    В per_item: id, raw, parsed, valid, correct, error, slices.
    Несовпадающие/дублирующиеся id и пустой набор должны вызвать ValueError.
    """
    tasks = _unique(rows, 'rows')
    answers = _unique(outputs, 'outputs')
    if tasks.keys() != answers.keys():
        raise ValueError('Наборы id заданий и ответов не совпадают')
    items = []
    for identifier, row in tasks.items():
        choices = row.get('choices')
        slices = row.get('slices')
        if (not isinstance(choices, list) or not choices or
                any(not isinstance(c, str) or not c for c in choices) or
                len(set(choices)) != len(choices) or row.get('expected') not in choices):
            raise ValueError(f'{identifier}: неверные choices/expected')
        if (not isinstance(slices, list) or not slices or
                any(not isinstance(s, str) or not s for s in slices) or len(set(slices)) != len(slices)):
            raise ValueError(f'{identifier}: неверные slices')
        output = answers[identifier]
        raw, error = output.get('raw'), output.get('error')
        if not isinstance(raw, str) or (error is not None and (not isinstance(error, str) or not error)):
            raise ValueError(f'{identifier}: неверный контракт raw/error')
        parsed, valid = None, False
        if error is None:
            try:
                parsed = json.loads(raw, object_pairs_hook=_object,
                                    parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                valid = (isinstance(parsed, dict) and set(parsed) == {'answer'} and
                         isinstance(parsed['answer'], str) and parsed['answer'] in choices)
            except (ValueError, TypeError, RecursionError):
                parsed = None
        items.append(dict(id=identifier, raw=raw, parsed=parsed, valid=valid,
                          correct=bool(valid and parsed['answer'] == row['expected']),
                          error=error, slices=list(slices)))
    return _aggregate(items)


def _validate_metrics(report, name, minimum):
    if not isinstance(report, dict):
        raise ValueError(f'{name}: нет отчёта')
    by_id = _unique(report.get('per_item'), name)
    if len(by_id) < minimum:
        raise ValueError(f'{name}: недостаточно примеров')
    for item in by_id.values():
        if (type(item.get('valid')) is not bool or type(item.get('correct')) is not bool or
                'error' not in item or not isinstance(item.get('raw'), str) or 'parsed' not in item):
            raise ValueError(f'{name}: неполный per_item')
        error = item['error']
        if error is not None and (not isinstance(error, str) or not error):
            raise ValueError(f'{name}: неверная ошибка генерации')
        if item['correct'] and not item['valid'] or error is not None and (item['valid'] or item['correct']):
            raise ValueError(f'{name}: противоречивый per_item')
        slices = item.get('slices')
        if (not isinstance(slices, list) or not slices or
                any(not isinstance(s, str) or not s for s in slices) or len(set(slices)) != len(slices)):
            raise ValueError(f'{name}: неверные срезы')
    expected = _aggregate(list(by_id.values()))
    if not isinstance(report.get('slices'), dict) or report['slices'].keys() != expected['slices'].keys():
        raise ValueError(f'{name}: не совпадают имена срезов')
    for location, actual, calculated in [
        ('overall', report.get('overall'), expected['overall']),
        *[(key, report['slices'][key], value) for key, value in expected['slices'].items()],
    ]:
        if not isinstance(actual, dict):
            raise ValueError(f'{name}/{location}: нет метрик')
        for key in ('n', 'correct', 'valid', 'errors'):
            if type(actual.get(key)) is not int or actual[key] != calculated[key]:
                raise ValueError(f'{name}/{location}: {key} не соответствует per_item')
        for key in ('accuracy', 'format_rate'):
            value = actual.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or
                    not math.isclose(value, calculated[key], rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f'{name}/{location}: {key} не соответствует counts')
    return by_id


def _same_groups(first, second, name):
    if first.keys() != second.keys():
        raise ValueError(f'{name}: не совпадают id')
    if any(set(first[key]['slices']) != set(second[key]['slices']) for key in first):
        raise ValueError(f'{name}: не совпадает принадлежность к срезам')


def quality_gate(baseline, candidate, critical_slices):
    """Верните {passed: True, ...} или вызовите ValueError с причиной отказа.

    Проверьте совместимость протокола/набора/baseline, метрики и counts,
    все id и срезы, общую accuracy/format_rate, критические срезы и behavior.
    Missing, NaN, пустой набор и ошибки генерации означают отказ.
    Пороги задаются до измерения; учебный допуск регрессии равен нулю.
    """
    for name, report in [('baseline', baseline), ('candidate', candidate)]:
        if (not isinstance(report, dict) or report.get('status') != 'complete' or
                not isinstance(report.get('model'), dict) or not report['model'] or
                not isinstance(report['model'].get('kind'), str) or not report['model']['kind']):
            raise ValueError(f'{name}: неполный отчёт или не указана модель')
        model = report['model']
        if model['kind'] in ('base', 'adapter'):
            fields = ('name', 'revision') if model['kind'] == 'base' else ('name', 'version', 'run_id', 'uri')
            if any(not isinstance(model.get(key), str) or not model[key] for key in fields):
                raise ValueError(f'{name}: модель не закреплена конкретной версией')
            if model['kind'] == 'base' and (len(model['revision']) != 40 or
                    any(c not in '0123456789abcdef' for c in model['revision'])):
                raise ValueError(f'{name}: нужна commit-ревизия базы')
        elif model['kind'] != 'fixture':
            # fixture — только формат синтетических отчётов из исходных tests/.
            # Реальная обвязка дополнительно сверяет model с закреплёнными spec.
            raise ValueError(f'{name}: неизвестный вид модели')
        for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
            if not isinstance(report.get(key), str) or not report[key]:
                raise ValueError(f'{name}: отсутствует {key}')
    for key in ('protocol_sha256', 'golden_sha256', 'baseline_spec_sha256'):
        if baseline[key] != candidate[key]:
            raise ValueError(f'Несовместимые отчёты: {key}')
    a = _validate_metrics(baseline, 'baseline', 50)
    b = _validate_metrics(candidate, 'candidate', 50)
    _same_groups(a, b, 'golden')
    if len(baseline['slices']) < 3:
        raise ValueError('Нужны минимум три среза golden')
    ba = _validate_metrics(baseline.get('behavioral'), 'baseline behavioral', 3)
    bb = _validate_metrics(candidate.get('behavioral'), 'candidate behavioral', 3)
    _same_groups(ba, bb, 'behavioral')
    if (not isinstance(critical_slices, list) or not critical_slices or
            any(not isinstance(s, str) or s not in candidate['slices'] for s in critical_slices)):
        raise ValueError('Критические срезы не заданы или отсутствуют')
    reasons = []
    for name, report in [('baseline', baseline), ('candidate', candidate)]:
        if report['overall']['errors'] or report['behavioral']['overall']['errors']:
            reasons.append(f'{name}: есть ошибки генерации')
    for key in ('accuracy', 'format_rate'):
        count = 'correct' if key == 'accuracy' else 'valid'
        if candidate['overall'][count] < baseline['overall'][count]:
            reasons.append(f'Общая {key} ухудшилась')
    for name in critical_slices:
        if candidate['slices'][name]['correct'] < baseline['slices'][name]['correct']:
            reasons.append(f'Accuracy критического среза {name} ухудшилась')
    if candidate['behavioral']['overall']['correct'] != candidate['behavioral']['overall']['n']:
        reasons.append('Не все обязательные behavioral-кейсы пройдены')
    if reasons:
        raise ValueError('; '.join(reasons))
    return dict(passed=True, critical_slices=critical_slices, regression_tolerance=0,
                protocol_sha256=candidate['protocol_sha256'], golden_sha256=candidate['golden_sha256'],
                baseline_spec_sha256=candidate['baseline_spec_sha256'], candidate_model=candidate['model'])
