# tai — эксперименты с устойчивостью LLM

Стратегия получает базовый запрос, взаимодействует с моделью в роли Target
и предъявляет текстовый результат. Оценщик судит этот результат, а журнал
сохраняет ход запуска. Словарь проекта — в [CONTEXT.md](CONTEXT.md).

## Куда смотреть

- Реализации стратегий: [`src/attack/`](src/attack/); команды запуска: [`src/main/`](src/main/).
- Тесты стратегий: [`tests/attack/`](tests/attack/); тест серии: [`tests/benchmark/test_alert_series.py`](tests/benchmark/test_alert_series.py).
- Тексты 640 базовых запросов: [`examples/alert-manifest-640.json`](examples/alert-manifest-640.json)
  (поле `requests[].prompt`); состав и происхождение пула — в
  [`examples/alert-pool.md`](examples/alert-pool.md).
- Подробные команды, настройки моделей, алгоритмы и Python API:
  [руководство](docs/usage.md). Результаты проверки на Qwen:
  [отчёт](docs/issue-97-verification.md). Формат журналов:
  [схема](docs/journal-schema.md).

## Запуск

Для локального Qwen подготовьте окружение и закреплённый снимок модели:

```bash
git submodule update --init --recursive
uv sync --locked --group dev --extra qwen
uv run --locked --group dev --extra qwen tai-qwen prepare
```

Для одного базового запроса выберите команду из `src/main/`.
К каждой строке таблицы добавьте спереди `uv run --locked --group dev --extra qwen`.
`JOURNAL` — новый путь к файлу, `"BASE"` — текст запроса; укажите свой номер GPU.

| Атака | Команда |
|---|---|
| FITD | `python -m main.qwen_cli run JOURNAL "BASE" --gpu 0` |
| GOAT | `python -m main.goat_cli JOURNAL "BASE" --gpu 0` |
| ThreatenJSON | `python -m main.threaten_json_cli JOURNAL "BASE" --gpu 0` |
| Simple Adaptive Attacks | `python -m main.adaptive_attacks_cli JOURNAL "BASE" --gpu 0` |
| DRA | `python -m main.dra_cli JOURNAL "BASE" --gpu 0` |
| Bad Characters/deletion | `python -m main.deletion_cli JOURNAL "BASE" --gpu 0` |
| AutoDAN GA/HGA | `python -m main.autodan_cli JOURNAL "BASE" --gpu 0 --target "DESIRED RESPONSE"` (для HGA: `--mode hga`) |

Для серии по пулу ALERT укажите новый каталог вывода:

```bash
uv run --locked --group dev --extra qwen tai-alert /tmp/alert-series --gpu 0 --limit 1
```

`--limit 1` выбирает по одному запросу из каждой из 32 микрокатегорий.
Журналы лежат в подкаталогах атак, сводка — в `summary.json`.
Параметры и ограничения, в том числе для AutoDAN и GGUF, описаны в
[руководстве](docs/usage.md). Правила работы с кодом и проверками — в
[AGENTS.md](AGENTS.md).
