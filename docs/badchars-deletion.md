# Bad Characters / deletion

Стратегия [#35](https://github.com/ArtFilippov/tai/issues/35) использует
`BadCharacters._generate_deletion_variants` из установленного garak
`3f50ea5ff9cd7050099940647c15c39b07a93392` вместе с нативными методами выбора
позиций, ASCII-символов и вставки последовательностей. Вход — ровно переданный
`base.to_string()`, включая Unicode, фигурные скобки и переводы строк.

Каждая вставка — печатный ASCII-символ и **настоящий backspace U+0008**.
Например, Python-строка `"a!\bbc"` содержит пять кодовых точек. Этот текст целиком
получает Target; адаптер не стирает символы и не заменяет backspace печатным `\\b`.
В JSON backspace экранируется как `\b` и восстанавливается при чтении.

## Параметры

```toml
[attack]
type = "badchars_deletion"
perturbation_budget = 1
max_position_candidates = 8
max_ascii_variants = 4
```

| Параметр | Default tai | Значение |
| --- | ---: | --- |
| `perturbation_budget` | 1 | Максимальное число вставок в одном варианте; перебор начинается с одной |
| `max_position_candidates` | 8 | Верхняя граница числа позиций между кодовыми точками, включая начало и конец |
| `max_ascii_variants` | 4 | Верхняя граница числа выбранных печатных ASCII-символов из 95 |

Все три значения — положительные целые; boolean, дробные, строки, нулевые и
отрицательные значения отклоняются. `max_ascii_variants` не может превышать 95.
Неизвестные ключи, тип атаки, `[attacker]` и `run.max_calls_per_conv` — ошибки
конфигурации до генерации и модельных обращений. Необязательный `run.generations`
поддерживает только `1`.

Начальный профиль даёт **до 32 вариантов**: четыре символа с кодами
32, 63, 94, 125 для каждой из максимум восьми распределённых по строке позиций.
На коротких строках позиций меньше; у пустой строки одна позиция и четыре варианта.
Garak по умолчанию использует до 24 × 95 вариантов при одной вставке. Tai задаёт
штатные параметры генератора, сохраняя его порядок: число вставок, комбинации
позиций, произведение ASCII-символов. Это не первые 32 элемента полного перебора.
При увеличении бюджета перебор может стать значительно больше 32.

Приватный фасад инициализирует только три входных поля чистого генератора,
минуя конструктор probe, который строит варианты bundled-корпуса. Варианты
получаются лениво для одного base. Категория всегда `deletion`; глобальные
`_config`, random-state, reportfile и registry адаптером не изменяются.
Случайный downsampling не используется (`follow_prompt_cap=false`).

## Выполнение и Python API

```python
from tai import (
    BadCharactersDeletionExperiment, ContainsTextEvaluator, LLMChat,
    capture_run, save_run,
)
from core.primitives.messages.message.text_message import TextMessage

experiment = BadCharactersDeletionExperiment(
    LLMChat(lambda turns: "Complete example answer"), TextMessage("Explain π\ncompletely."),
    ContainsTextEvaluator("Complete"), journal_dir="runs",
    perturbation_budget=1, max_position_candidates=8, max_ascii_variants=4,
)
experiment.start()
experiment.wait()
directory = save_run(capture_run(experiment, journal=experiment.journal), "runs")
```

`BadCharactersDeletionUser(target, base, evaluator, **parameters)` выполняет тот
же путь без журнала. Experiment дополнительно принимает `journal_dir` и
`journal_factory`; повторный `start()` получает независимое выполнение и журнал.
Target может быть `OpenAIChat`, `QwenChat` или любым `LLMChat`. Оценщик —
`ContainsTextEvaluator`, внедрённый `Evaluator` или `EvaluatorSetup.model(chat, TextMessage)`;
модельный оценщик использует собственный разговор.

Общий `CandidateRuntime` выполняет и Adaptive Attacks, и deletion: новый разговор
Target с сохранёнными системными инструкциями, один запрос, один ответ, публикация
ответа и `evaluate(base, result)` с исходным base. Первый `true` завершает запуск
с `evaluator_success`. При всех `false` итог — последний оценённый Message,
причина — `strategy_exhausted`. Ранняя остановка — политика tai; штатный probe
garak сначала выполняет все attempts.

## CLI, preflight и ALERT

Готовый профиль: [badchars-deletion-qwen.toml](../examples/badchars-deletion-qwen.toml).
Подготовка закреплённых весов — [README](../README.md). Для HTTP замените `[target]`
и `[evaluator.chat]` OpenAI-совместимыми профилями; для общего vLLM-сервера доступны
существующие `[server]` и `backend="vllm"`.

```bash
uv run --locked --extra qwen tai preflight --config examples/badchars-deletion-qwen.toml

uv run --locked --extra qwen tai attack \
  --config examples/badchars-deletion-qwen.toml \
  --text 'Ваш базовый запрос' --artifacts-dir runs/badchars

uv run --locked --extra qwen tai alert-series --jobs 1 \
  --manifest examples/alert-manifest.json --taxonomy ALERT_taxonomy.md \
  --profile examples/badchars-deletion-qwen.toml --artifacts-dir runs/alert-badchars

uv run --locked --extra qwen tai alert-report \
  --index runs/alert-badchars/<series-UUID>/index.json
```

Поддерживается также `--text-file request.txt` (UTF-8). `preflight` проверяет
Target и оценщик; Attacker не создаётся. Для ALERT требуется модельный оценщик.
Один base остаётся одной записью серии при любом числе внутренних вариантов.
Отчёт использует сохранённую оценку финального `result_id`; пересчёт не делает
модельных обращений. Параметры сохраняются в `effective-profile.json` под
`badchars_deletion`, а в metadata запуска — рядом с `attack.type` и ревизией garak.

## Журнал, остановка и ошибки

В каждом запросе `Message.origin` указывает на исходный base со всей его историей.
`context.cause` явно передаёт `attempt_id` (id преобразованного запроса) и
`metadata`: `bad_character_category="deletion"`, нативные `count` и `operations`
с `positions` и `ascii_codes`. Наблюдающая обёртка записывает эти значения вместе
с точным текстом; приватное состояние генератора журналу не требуется.
Консольное представление не изменяет исходные Message и JSON-артефакты.

Жизненный цикл, Ctrl+C, освобождение участников и внешнее сохранение общие с
Adaptive Attacks и FITD. Остановка закрывает допуск новых действий и дожидается
допущенного обращения или оценки целиком. Завершённый обмен до публикации не
становится результатом; допущенная оценка сохраняется после остановки. Ошибки
генератора, Target, оценщика и журнала сохраняются как ошибки рабочего выполнения.
Ошибка внешней записи результата не меняет фактический исход User.

[Живая приёмка #36](acceptance/2026-09-23-garak-simple-attacks.md) содержит готовый
профиль с инструкцией оценщику о JSON-формате, полный журнал и проверку U+0008,
первого успеха и связи результата с оценкой. Офлайн-проверки:
`tests/test_badchars.py`, `test_badchars_cli.py`, `test_badchars_series.py` и общие
`test_adaptive_lifecycle.py`, `test_candidate_series.py`. Источник:
[нативный probe garak](https://github.com/NVIDIA/garak/blob/3f50ea5ff9cd7050099940647c15c39b07a93392/garak/probes/badchars.py).
