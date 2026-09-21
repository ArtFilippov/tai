# Python Quality Bundle 1.0.0

Локальная интеграция tai и исправления исполнения описаны в
[`LOCAL-CHANGES.md`](LOCAL-CHANGES.md); учитывайте их при повторной установке поставки.

Строгая проверка одного Python-проекта с `src/` и `uv`. Linux, интерпретатор для
инструментов Python >= 3.12. Система не изменяет исходники и не создаёт baseline.
Установка скачивает зависимости через `uv`; они **не вложены** в `.run`.

## 1. Установка

Из корня репозитория:

```bash
bash /path/to/python-quality-1.0.0.run
```

Или с явным путём, включая пути с пробелами:

```bash
bash /path/to/python-quality-1.0.0.run --project /path/to/repository --json
```

`uv` должен быть установлен. Нужен интернет либо предварительно наполненный
uv cache/доступный индекс. Не запускайте установку через `sudo`.
Окружение должно позволять выполнять Bash, tar, gzip и sha256sum.
Поддерживаются обычные uv-проекты с `[project]`, `src/<import_package>/` и,
необязательно, `tests/`. Workspace/monorepo и симлинки в исходниках не поддерживаются.
Имена директорий пакетов должны быть допустимыми Python-идентификаторами.

Опции установщика:

```bash
bash /path/to/python-quality-1.0.0.run --help
bash /path/to/python-quality-1.0.0.run --project /repo --dry-run --json
bash /path/to/python-quality-1.0.0.run --extract-only /tmp/quality-installer-sources
```

Последняя команда не требует `uv` или интернета, распаковывает исходники
установщика в новую директорию, ничего не устанавливает. `--dry-run` не изменяет
репозиторий, но bootstrap может загрузить Python/tomlkit/packaging в кэш uv.

### Что меняет установщик

Добавляет точные dev-зависимости, обновляет `uv.lock`, синхронизирует виртуальное
окружение, устанавливает `quality/`, заменяет секции управляемых анализаторов в
`pyproject.toml` и `pyrightconfig.json`. Другие секции проекта сохраняются.
Для dev-группы устанавливается `requires-python >=3.12`; существующее ограничение
Python для runtime проекта, включая верхнюю границу, сохраняется. Если оно отсутствовало,
устанавливается `>=3.12`. Параметры индексов uv не заменяются.

После установки выполняются **реальные локальные smoke-тесты анализаторов**:
загрузка полного списка плагинов Pylint, разбор конфигураций, намеренная ошибка
типа Pyright, несовпадение docstring в pydoclint, запрещённый импорт Tach и запуск
`tach check-external`. Ошибка на этом этапе означает неуспешную установку, а не
автоматическое отключение сломавшегося инструмента.

При ошибке восстанавливаются изменённые управляемые файлы. Логи остаются в
`.quality/install.log` и `.quality/install-probes/`. Виртуальное окружение и кэш uv
могли измениться; после неудачи их можно пересинхронизировать командой `uv sync`.
Исходники приложения не редактируются. Установка не требует чистого Git tree;
перед ней всё же желательно сохранить текущие изменения в Git.

Повторная установка сохраняет изменённые `quality/readability.toml`, `scope.toml`
и ранее принятую архитектуру. Численные лимиты и содержимое файлов при одинаковом
вводе не меняются; временные логи smoke-тестов могут обновляться. Native-настройки
управляемых анализаторов вновь приводятся к профилю поставки.

**Установка НЕ инициализирует архитектуру автоматически.**

## 2. Инициализация архитектуры — обязательный отдельный шаг

```bash
uv run --locked --group dev python quality/init_architecture.py --json
```

Скрипт создаёт черновик `tach.toml`, выделяя компоненты по **существующим каталогам
и названиям слоёв**, и выполняет `tach sync`, чтобы записать реальные импорты.
Никакие файлы приложения не перемещаются.

Выбранная гранулярность: **предметная подсистема + слой**, например:

```text
src/shop/
  orders/
    domain/
      api.py
    application/
      api.py
    infrastructure/
      api.py
    interfaces/
      api.py
  bootstrap/
    main.py
```

Компонентами станут `shop.orders.domain`, `shop.orders.application` и т. д.
Контейнеры `shop` и `shop.orders` также получают явные границы для оставшихся файлов.
Распознаются `domain`, `application`/`use_cases`/`usecases`/`usecase`,
`infrastructure`/`adapters`, `interfaces`/`interface`, `bootstrap`/`composition`.
Нераспознанные директории получают **предварительный** слой Domain. Для плоского
проекта черновик будет грубым: границы нужно определить при ревью, а не выдавать
такое разбиение за автоматически найденную правильную архитектуру.

Матрица по умолчанию:

| Откуда | Куда разрешено, при явном `depends_on` |
|---|---|
| Bootstrap | В любые слои |
| Interface | Interface, UseCase, Domain |
| Infrastructure | Infrastructure, Domain |
| UseCase | UseCase, Domain |
| Domain | Domain |

Запрещённые направления дополнительно записаны в `cannot_depend_on`.
По умолчанию пакет экспортирует только `api.py` и его публичные символы,
а не всю реализацию. `Protocol` — контракт типов; интерфейс Tach — публичная
поверхность импорта. Это не одно и то же.

**Просмотрите и исправьте `tach.toml`:** состав компонентов, слой каждого, реальные
разрешённые зависимости и публичные экспорты. `tach sync` записывает текущее
состояние, в том числе нежелательные импорты. Сам по себе он не означает одобрения
архитектуры. Не запускайте sync автоматически перед каждой проверкой.

После ревью:

```bash
uv run --locked --group dev python quality/init_architecture.py --accept --json
```

Это синхронизирует классификацию слоёв с clean-architecture-плагином и сохраняет
`quality/architecture.ready.json`. Принятие конфигурации **не скрывает существующие
нарушения**: `check.py` сообщит их все. При ошибке исполнения или парсинга Tach
инициализация не подтверждается. Неинициализированный проект не может получить
зелёный общий результат.

Если `tach.toml` уже существовал, первый запуск оставляет его для ревью. Осознанно
создать новый черновик вместо него можно через `--force`.
После ручного изменения слоёв повторите `--accept`, чтобы синхронизировать плагин.

Граф файловых зависимостей можно сохранить отдельно, без сетевой визуализации:

```bash
uv run --locked --group dev tach map --output .quality/dependencies.json
```

## 3. Изменение параметров читаемости: один файл и одна команда

Редактируйте **`quality/readability.toml`**, затем:

```bash
uv run --locked --group dev python quality/apply.py --json
```

Применение обновляет значения Pylint для исходников и тестов. Также синхронизирует
общую область исключений из `quality/scope.toml`. Не переустанавливает пакеты,
не запускает анализаторы, не правит исходники и не сбрасывает архитектуру.
Это не глобальный слой конфигурации: границы живут в обычном `tach.toml`, типы —
в native Pyright config. Параметры читаемости не нужно дублировать вручную.

Невалидный TOML, неизвестный параметр, отрицательное/булево значение вместо числа
или противоречивые пределы завершают применение ошибкой. Изменённые, но ещё не
применённые лимиты дают `READABILITY_NOT_APPLIED` при запуске проверки.

| Параметр | production | tests |
|---|---:|---:|
| `max_line_length` | 100 | 100 |
| `max_module_lines` | 400 | 800 |
| `max_statements` | 30 | 60 |
| `max_complexity` | 8 | 10 |
| `max_branches` | 8 | 10 |
| `max_nested_blocks` | 4 | 4 |
| `max_locals` | 10 | 20 |
| `max_returns` | 4 | 6 |
| `max_bool_expr` | 4 | 4 |
| `max_args` | 5 | 10 |
| `max_positional_arguments` | 4 | 10 |
| `max_attributes` | 7 | 7 |
| `max_public_methods` | 12 | 30 |
| `max_parents` | 3 | 3 |
| `min_public_methods` | 2 | 0 |
| `min_similarity_lines` | 4 | 4; duplicate-code отключён для tests |
| `max_try_statements` | 1 | 3 |

`tests` наследует неуказанные параметры `production`. `max_statements` — не число
физических строк. `max_parents` ограничивает учитываемых Pylint предков, не только
непосредственные базы. В локальном профиле по согласованию исключены стандартные
базы исключений, `Protocol`, `Enum` и библиотечный каркас `garak.probes.base.Probe`
(точный список — `IGNORED_FRAMEWORK_PARENTS` в `_policy.py`). Pylint исключает также
предков указанной базы: для garak это `Configurable`; `FITD`, `IterativeProbe` и
собственные промежуточные классы продолжают учитываться при лимите 3.
Метрики вычисляются самим Pylint/McCabe; отдельного нового
измерителя здесь нет. Игнорирование комментариев/docstrings/imports/signatures
при поиске дублей оставлено включённым.

## 4. Проверка: одна команда

```bash
uv run --locked --group dev python quality/check.py
```

Машиночитаемый режим:

```bash
uv run --locked --group dev python quality/check.py --json
```

Все независимые задачи выполняются параллельно (до четырёх процессов-анализаторов,
с учётом числа CPU). Для Pylint исходники и тесты анализируются отдельными профилями;
внутри каждого `jobs=1`, чтобы не ослаблять проверку дублей и графа импортов из-за
разбиения проекта между воркерами. Используются процессы уже синхронизированного
окружения, без конкурентных `uv sync`. Fail-fast отсутствует.

Опции:

```bash
uv run --locked --group dev python quality/check.py --serial
uv run --locked --group dev python quality/check.py --timeout 900 --max-diagnostics 50
```

`--timeout` — лимит секунд **на одну задачу**, а не обещание времени всего анализа.
Обрезается только человекочитаемый stdout: полный JSON содержит все диагностики.
Для отключения синхронизации на повторных локальных запусках используйте
`uv run --no-sync --group dev ...` только с уже установленным актуальным окружением;
сам runner всё равно проверяет точные версии инструментов.

### Коды завершения

| Код | Значение |
|---:|---|
| 0 | Все запланированные проверки выполнены, сообщений нет |
| 1 | Найдены нарушения кода/архитектуры/документации |
| 2 | Неинициализированная архитектура, неверный конфиг/версии, crash, timeout, ошибка адаптера или выполнения |

Все уровни сообщений, в том числе Pylint `I`, делают проверку неуспешной.
Падение одного инструмента не отменяет остальные. Непонятный nonzero-вывод не
превращается в «нет ошибок»: сохраняются сырые логи и выдаётся код 2.

Результаты:

```text
.quality/reports/latest.json
.quality/reports/<run-id>/report.json
.quality/reports/<run-id>/raw/*.stdout.log
.quality/reports/<run-id>/raw/*.stderr.log
```

`--json` выводит один JSON-документ в stdout. Поля диагностики: `tool`, `rule`,
`severity`, `path`, `line`, `column`, `message`, `symbol`, `documentation_url`,
`fingerprint`, `job`, при необходимости `profile` и `is_infrastructure_error`.
Пути относительные, строки и колонки считаются с **1**; неизвестные значения — `null`.
Диагностики сортируются по пути, позиции, инструменту и правилу. Временные метки и
время выполнения естественно меняются. Правила Tach имеют имена из native diagnostics или явно помеченные категории
адаптера, а не вымышленные числовые ID upstream. Схема: `quality/report.schema.json`.

## 5. Покрытие, отключения и ограничения

Точные версии: `quality/tools.json`. Фактически загруженные Pylint-плагины и весь
их каталог сообщений записываются при установке в `quality/pylint-catalog.json`.
Все причины отключений находятся в `quality/policy-notes.json`.

* **Pylint 4.0.8:** core + все обнаруженные официальные extensions, кроме docparams.
  Читаемость, naming, структура, exception/logging practices, inference, дубли.
* **clean-architecture-linter-plugin 1.2.2:** архитектурные эвристики, DI, Protocol,
  ограничения I/O/мутабельности/структуры. Это сторонний **Alpha**-плагин, не доказатель
  полного Dependency Inversion для произвольного Python.
* **Pyright 1.1.414:** strict и дополнительные отключённые в strict diagnostics.
  `reportImportCycles` отдан Pylint. `pyright[nodejs]` обеспечивает Node runtime.
* **Tach 0.35.1:** границы, public API, слои, циклы, exact dependencies. Выполняются
  **оба** набора проверок `tach check` и `tach check-external`. Обёртка вызывает
  native API закреплённой версии и отдельно сохраняет exact/unused-dependencies:
  CLI JSON-режим этой версии завершает работу раньше этого дополнительного этапа.
* **pydoclint 0.9.1:** Google docstrings сопоставляются с сигнатурой/телом.

**Docstrings нигде не обязательны.** Отсутствующий docstring разрешён; существующий
проверяется, в том числе короткий и у private-функции. Пустой/некорректный docstring
не равен отсутствующему. Типы аргументов существующей документации ожидаются и в
сигнатуре, и в Google `Args:`. `check-arg-defaults` upstream применяется к NumPy,
поэтому не обещает дополнительную проверку defaults Google-docstrings.

`pylint.extensions.docparams` не загружается из-за коллизий W900x/W901x с
архитектурным плагином. Область документации покрывается Pylint docstyle и pydoclint;
это не утверждение побитового совпадения двух наборов правил.

В production Pylint отключены только обязательные docstrings и `reimported`,
`undefined-variable`, `used-before-assignment`, `possibly-used-before-assignment`:
соответствующие Pyright-проверки включены явно. В отличие от предварительного
обсуждения, **unused/private/unreachable и архитектурные W9001/W9003 сохранены**:
похожесть с Pyright/Tach не гарантирует полного совпадения покрытия.

В тестах сохранены строгие типы, ошибки исполнения, синтаксис, анализ исключений
и существующих docstrings. Разрешены pytest fixture arguments/reused names,
white-box protected access, тестовые классы с одной операцией, буквальные expected
values и повторяющиеся arrange/assert-фрагменты. Production-only clean architecture
правила выключены для тестов; тестовое правило W9101 (не более четырёх mock/patch-вызовов по эвристике плагина)
и anti-bypass W9501 остаются. W9102 отключено, чтобы не конфликтовать с разрешёнными
white-box тестами. Pyright в тестах не требует использовать результат каждого вызова
и не запрещает private access. Это отдельный согласованный тестовый профиль,
а не заявление, что все отключения заменяются другими инструментами.

`quality/scope.toml` исключает каталоги по **точному basename**: например `.venv`,
`build`, `dist`, `generated`, `vendor`, `migrations`. Обычный `models.py` не исключается.
Исключённые реально найденные директории перечислены в отчёте. Перепроверьте, что
в `generated/` или `migrations/` действительно лежит код, который нужно исключить.
Проверяются `src/` и `tests/`, не произвольные корневые `.py` скрипты, ноутбуки или
сами обслуживающие скрипты `quality/`. `.pyi` проверяет Pyright; остальные анализаторы
работают по своей нативной области. `__init__.py` не исключён.

Отложены: полный запрет внутренних комментариев, физическая длина функций,
community detection и проверка размещения обнаруженных сообществ, универсальное
«каждая зависимость только через Protocol/ABC». Не добавлены formatter, runtime
тесты, сканер уязвимостей, естественно-языковой spellcheck с системным Enchant.
Набор настроек не доказывает корректность программы и не запрещает все мыслимые
способы обхода статического анализа.

Подавления регулируются native-анализаторами, включая Pylint сообщения о
подавлении, anti-bypass плагина и правила Tach про причины. Отдельного нового
парсера запретов комментариев, hash-policy, автоисправления или скрытого baseline нет.

## 6. Git и агентский цикл

Закоммитьте `quality/`, `pyproject.toml`, `uv.lock`, `pyrightconfig.json`, `tach.toml`,
`quality/architecture.ready.json` и изменения `.gitignore`. Не коммитьте `.quality/`.
Никакие Git hooks или CI workflow автоматически не устанавливаются.

На свежем клоне сначала `uv sync --locked --group dev`, затем запускайте check.
Инструкции для агентов находятся в `quality/AGENTS.md`. Установщик не изменяет
корневой `AGENTS.md` проекта: существующие инструкции не теряются.

## 7. Источники спецификаций

- https://pylint.readthedocs.io/en/stable/user_guide/configuration/all-options.html
- https://pylint.readthedocs.io/en/stable/user_guide/checkers/extensions.html
- https://pypi.org/project/clean-architecture-linter-plugin/1.2.2/
- https://github.com/noah-goodrich/pylint-clean-architecture
- https://github.com/microsoft/pyright/blob/main/docs/configuration.md
- https://pypi.org/project/pyright/1.1.414/
- https://docs.gauge.sh/usage/configuration/
- https://docs.gauge.sh/usage/commands/
- https://jsh9.github.io/pydoclint/config_options.html
- https://docs.astral.sh/uv/concepts/projects/dependencies/

Состояние внешних описаний проверено 21 сентября 2026 г. Верхнеуровневые версии
зафиксированы; транзитивные версии фиксирует lockfile конкретного репозитория.
Сведения о проверках самого архива: `quality/BUILD-VALIDATION.md`.
