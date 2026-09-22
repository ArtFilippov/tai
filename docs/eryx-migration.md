# Выделение Eryx — 2026-09-22

Инструмент опубликован в приватном репозитории
[ArtFilippov/eryx](https://github.com/ArtFilippov/eryx) с новой историей и сохранённой
MIT-лицензией. [PR Eryx #1](https://github.com/ArtFilippov/eryx/pull/1) прошёл
автономный CI и слит. `tai` закрепляет merged SHA
`60556a59647dc0e2155b878cee6c0a6a023b9b97` через `.gitmodules` и gitlink `eryx/`.

## Интеграция

- Код runner, compatibility patches, workers, pins, defaults, схема и тесты — в Eryx.
- Проектные настройки и состояние перенесены в `.eryx/`; runtime-отчёты, логи и
  кэш игнорируются отдельными правилами.
- Численные лимиты сохранены. Marker принятой архитектуры (109 модулей), каталог
  правил и policy notes совпадают с исходными файлами. Native-конфиги изменились
  только заменой пути исключения `.quality` → `.eryx`.
- `ignored-parents` для garak, `stubPath`, uv source/index/dependency-metadata и
  optional dependencies Qwen сохранены. Политика garak больше не зашита в Eryx.
- В dev-группу `tai` добавлен `jsonschema==4.26.0` для автономной проверки схемы
  отчёта в тестах Eryx; остальные зависимости сохранены по lockfile.
- Старые скрипты и self-extracting-поставка удалены после переноса установщика,
  defaults и тестов. Происхождение зафиксировано в `eryx/ORIGIN.md`.
- Все CLI принимают `--project`; отчёты используют `eryx-report/v1`, события —
  `eryx-event/v1`, отчёт содержит commit SHA и dirty-признак запущенного Eryx.

## Проверки рабочего дерева

До и после миграции использовалось одно состояние исходников, включая существующие
локальные правки консоли ALERT. Эти правки не входят в коммит миграции.

| Проверка | До | После |
|---|---|---|
| Полный static check | exit 0, 0 диагностик | exit 0, 0 диагностик |
| Регрессии инструмента | exit 0, 42 теста | exit 0, 92 теста |
| Runtime pytest | exit 0, 659 тестов | exit 0, 659 тестов |

Единственное предупреждение pytest в обоих прогонах:
`PytestAssertRewriteWarning` для уже импортированного `qwen_fakes`.
Неразрешённых статических диагностик и ошибок исполнения инструментов нет.

```bash
uv run --locked --group dev --extra qwen python eryx/check.py
uv run --locked --group dev --extra qwen python -B -m unittest discover -s eryx/tests
uv run --locked --group dev --extra qwen pytest
```

Установка `bash eryx/install.sh --project . --extra qwen --json` завершилась с
exit 0 и шестью успешными native smoke-проверками. Повторная установка и
`eryx/apply.py --json` вернули `modified_files: []`; submodule остался чистым.
Финальный локальный отчёт: `.eryx/reports/latest.json`.

92 теста Eryx включают 42 перенесённые native-регрессии, 44 инфраструктурных теста
из исходной поставки и 6 новых тестов миграции/полного жизненного цикла. Проверены:
новый uv/src-проект, dry-run, повторная установка, rollback после реального сбоя
native probe, сохранение extras и проектных настроек, отдельный клон, настоящий
submodule по нестандартному пути с пробелами, SHA/dirty в отчёте, отсутствие записи
в инструмент при install/check/apply/accept, отказ в успехе без принятой архитектуры.
GitHub CI Eryx выполняет те же тесты без доступа к `tai`.

Дальнейшие исправления: сначала PR Eryx с автономной регрессией и проверкой
потребителя, после слияния — отдельный PR `tai`, закрепляющий merged SHA.
