"""Format audit prose only; preserve quoted correspondence and pinned code."""
import collections
import json
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parents[1]
corpus = (OUT.parents[1] / 'docs/EXPLORER-LEARNING-REPORT-2026-10-19.ru.md').read_text()
conversation = OUT / 'evidence/conversation_messages.redacted.jsonl'
corpus += '\n'.join(json.loads(line)['text'] for line in conversation.read_text().splitlines())
counts = collections.Counter(re.findall(r'[А-Яа-яЁё]+', corpus.lower()))


def split_word(match):
    value = match.group(0)
    low = value.lower()
    best = {0: (0., [])}
    for end in range(2, len(value) + 1):
        choices = []
        for start in range(max(0, end - 24), end - 1):
            token = low[start:end]
            if start in best and counts[token] >= 3 and not (start == 0 and end == len(value)):
                score, parts = best[start]
                choices.append((score + math.log(counts[token]) - 5., parts + [value[start:end]]))
        if choices:
            best[end] = max(choices, key=lambda x: x[0])
    return ' '.join(best[len(value)][1]) if len(value) in best else value


SUBSTITUTIONS = {
    '291? нет: конкретныйаудит': 'конкретный аудит — ',
    'Следовательно, distribution shift структурно': 'Следовательно, сдвиг распределения структурно',
    'Начальнаякоманднаяпоза': 'Начальная командная поза',
    'Действующийfactoryинтерфейс/пользовательподтверждаютcommand-only;сентябрьскийконтурнеэквивалентентекущему.':
        'Действующий factory-интерфейс и уточнение пользователя подтверждают command-only; сентябрьский контур не эквивалентен текущему.',
    'Выводисправленв01/03/11/15/19;currentabsencefeedbackнеоспариваетсястарымиrawlogs.':
        'Вывод исправлен в01/03/11/15/19; отсутствие обратной связи в нынешней конфигурации не оспаривается старыми raw logs.',
    'Дажеаудитдолженпроверятьпозднейшиеисправления,а неразмножатьстарыеутверждения.':
        'Даже аудит должен проверять позднейшие исправления, а не повторять старые утверждения.',
}


def format_part(text):
    for a, b in SUBSTITUTIONS.items():
        text = text.replace(a, b)
    text = re.sub(r'(?<=[А-Яа-яЁё])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[А-Яа-яЁё])', ' ', text)
    text = re.sub(r'[А-Яа-яЁё]{20,}', split_word, text)
    text = re.sub(r'(?<=[,;])(?=[А-Яа-яЁёA-Za-z])', ' ', text)
    return text


def main():
    for p in sorted(OUT.glob('*.md')):
        fenced = False
        lines = []
        for line in p.read_text().splitlines(keepends=True):
            if line.startswith('```'):
                fenced = not fenced
            if fenced or line.startswith('```') or line.startswith('>') or re.match(r'^\[[^]]+\]:', line):
                lines.append(line)
            else:
                parts = re.split(r'(`[^`]*`|\]\([^)]*\)|https?://\S+)', line)
                lines.append(''.join(x if i % 2 else format_part(x) for i, x in enumerate(parts)))
        p.write_text(''.join(lines))
    print('Formatted report prose; quotations/code untouched')


if __name__ == '__main__':
    main()
