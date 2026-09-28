# Simple Adaptive Attacks

Реализация [#34](https://github.com/ArtFilippov/tai/issues/34) использует
`garak==3f50ea5ff9cd7050099940647c15c39b07a93392`: импортирует
`ADAPTIVE_TEMPLATE` и `SUFFIXES_PATH` из установленного пакета. Ресурс читается
через штатный `open(LocalDataPath, encoding="utf-8")` garak; пустые строки и
пробелы обрабатываются как в нативном probe. Копий шаблона и суффиксов в tai нет.

Это **статическая композиция шаблона и пяти cached-суффиксов** в исходном порядке,
без online search. Во все подстановки `goal` передаётся точный `base.to_string()`,
включая фигурные скобки и переводы строк. Встроенный seed-корпус не используется.
Каждый вариант — новый `Message` с происхождением от исходного base.

Перед каждым вариантом Target начинает новый разговор с сохранением своих
системных инструкций. Один вариант даёт один запрос и один полный ответ.
User предъявляет ответ, затем вызывает внедрённый `evaluate(base, result)`
с теми же объектами `Message`. При `true` причина завершения —
`evaluator_success`; после пяти `false` — `strategy_exhausted`, итогом остаётся
пятый оценённый ответ. **Ранняя остановка по оценщику — политика адаптера tai**:
штатные probes garak сначала выполняют все attempts.

## Python API

```python
from tai import (
    AdaptiveAttacksExperiment, ContainsTextEvaluator, LLMChat, Message,
    capture_run, save_run,
)

# Офлайн-пример с управляемой моделью; вместо неё можно передать OpenAIChat/QwenChat.
target = LLMChat(lambda turns: "Complete example answer")
experiment = AdaptiveAttacksExperiment(
    target, Message("Explain {this} fictional puzzle\ncompletely."),
    ContainsTextEvaluator("Complete"), journal_dir="runs",
)
experiment.start()
experiment.wait()
record = capture_run(experiment, journal=experiment.journal)
directory = save_run(record, "runs")
```

`AdaptiveAttacksUser(target, base, evaluator)` выполняет тот же рабочий путь без
журнала. `AdaptiveAttacksExperiment` добавляет независимый `RunJournal` каждому
запуску; параметры наблюдения — `journal_dir` и `journal_factory`. Интерфейсы
`start`, `stop`, `wait`, `status`, `result`, `snapshot`, `journal` и `journals`
совпадают с FITD. Повторный `start()` создаёт независимое выполнение.

Attacker не создаётся. Для модельного оценщика передайте
`EvaluatorSetup.model(evaluator_chat)`: его `LLMChat` должен отличаться от Target,
но может использовать тот же вычислительный ресурс. Сборка и чтение профиля не
выполняют инференс. Оценщик начинает новый разговор перед каждой парой.

## Одиночный запуск и проверка готовности

Готовый профиль — [examples/adaptive-attacks-qwen.toml](../examples/adaptive-attacks-qwen.toml).
Подготовка зависимостей и закреплённых локальных весов — [README](../README.md).
Из корня проекта:

```bash
uv run --locked --extra qwen tai preflight \
  --config examples/adaptive-attacks-qwen.toml

uv run --locked --extra qwen tai attack \
  --config examples/adaptive-attacks-qwen.toml \
  --text 'Ваш базовый запрос' --artifacts-dir runs/adaptive
```

Вместо `--text` можно передать `--text-file request.txt` (UTF-8).
`preflight` проверяет только Target и оценщик. Для OpenAI-совместимого сервера
замените настройки `[target]` и `[evaluator.chat]` соответствующими HTTP-профилями.

Выбор стратегии:

```toml
[attack]
type = "adaptive_attacks"
```

Без `[attack]` сохраняется прежний FITD-профиль; явный `type="fitd"` тоже выбирает
FITD. `tai attack` запускает выбранную стратегию, а `tai fitd` принимает только FITD.
Неизвестные типы и ключи отклоняются до допуска запуска. Adaptive Attacks не
принимает `[attacker]` или `run.max_calls_per_conv`: число вариантов фиксировано
ресурсом garak, оно не является лимитом раундов FITD. Необязательный
`run.generations` может быть только `1`.

## ALERT

Тот же профиль содержит модельный оценщик и подходит для полной серии:

```bash
uv run --locked --extra qwen tai alert-series --jobs 1 \
  --manifest examples/alert-manifest.json \
  --profile examples/adaptive-attacks-qwen.toml \
  --taxonomy ALERT_taxonomy.md --artifacts-dir runs/alert-adaptive

uv run --locked --extra qwen tai alert-report \
  --index runs/alert-adaptive/<series-UUID>/index.json
```

Один base остаётся одной записью ALERT, даже если выполнены все пять вариантов.
Каждый User получает отдельные разговоры, оценщик, состояние и журналы. `--jobs`
ограничивает активные запуски; координатор, консольные обновления и внешнее
сохранение общие с FITD. Отчёт использует только сохранённую оценку финального
`Message`; пересчёт не выполняет инференс. Подробности — [alert-series.md](alert-series.md).

## Остановка, ошибки и артефакты

Общий механизм находится в `attacks/shared/`: он управляет допуском действий,
участниками, фоновым выполнением и наблюдением операций. FITD сохраняет собственные
нативные решения; Adaptive Attacks использует те же операции сброса, обмена,
публикации и оценки. Новая стратегия не вводит отдельного жизненного цикла.

Ctrl+C закрывает допуск новых действий, ждёт уже допущенное обращение или оценку
целиком и сохраняет итог. Завершённый обмен сам по себе не становится результатом:
при остановке до публикации остаётся прежний результат либо `None`. Уже допущенная
оценка сохраняется и после остановки. Ошибки Target, оценщика, чтения ресурса и
журнала дают `FAILED`; освобождение участников выполняется общим механизмом.

Сохраняются существующие `events.jsonl`, `metadata.json` и, при наличии результата,
полный `result.txt`. Настройки содержат `attack.type`, `variants=5`, `generations=1`,
`online_search=false`, политику ранней остановки, исходный base, ревизию garak и
настроенных участников. `progress.max_calls_per_conv` равен `null`. В настройках
серии параметры находятся в `adaptive_attacks`, идентификатор — в `attack.type`.
Итоговые файлы записываются после ожидания User; ошибка внешнего сохранения
не меняет фактический исход рабочего выполнения.

Офлайн-проверки: `tests/test_adaptive_attacks.py`, `test_adaptive_lifecycle.py`,
`test_adaptive_cli.py`, `test_adaptive_series.py`, `test_candidate_series.py`; они используют настоящий
установленный garak, управляемые модели и локальный HTTP endpoint.

[Живая приёмка #36](acceptance/2026-09-23-garak-simple-attacks.md) содержит точный
базовый запрос, отдельный готовый профиль с инструкцией оценщику о JSON-формате,
команды CUDA/FP16 без внешней сети и полный журнал пяти вариантов. Источник:
[нативный probe garak](https://github.com/NVIDIA/garak/blob/3f50ea5ff9cd7050099940647c15c39b07a93392/garak/probes/adaptive_attacks.py).
