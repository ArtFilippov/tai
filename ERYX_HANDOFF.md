# Handoff: Eryx выделен, интеграция tai опубликована

Обновлено: 2026-09-22. Этот документ заменяет первоначальный план миграции.

## 1. Текущий статус

Выделение инструмента **выполнено**. Создан приватный репозиторий
[ArtFilippov/eryx](https://github.com/ArtFilippov/eryx) с новой историей и сохранённой
MIT-лицензией. Код, установщик, defaults, compatibility patches и оба исходных
набора тестов перенесены и адаптированы.

- **[PR Eryx #1](https://github.com/ArtFilippov/eryx/pull/1)** — `MERGED`.
  Автономный GitHub CI `native-tests` прошёл. В PR зафиксировано self-review;
  внешнего ревью на момент слияния не было.
- Merged SHA Eryx: **`60556a59647dc0e2155b878cee6c0a6a023b9b97`**.
- **[PR tai #28](https://github.com/ArtFilippov/tai/pull/28)** — `OPEN`, готов к ревью.
  Это отдельный PR принятия уже слитого инструмента.
- Ветка tai: **`chore/eryx-submodule`**, upstream — `origin/chore/eryx-submodule`.
- Коммит миграции tai: **`96913e9a95f9e22ace41fa77bf5714f1d60989d7`**,
  `replace Python Quality bundle with Eryx submodule`.
- `eryx/` — настоящий submodule с gitlink режима `160000` и записью в `.gitmodules`.
  HEAD submodule detached на merged SHA; локальных изменений инструмента нет.

После публикации миграции пользователь отдельно поручил **обновить этот handoff,
закоммитить все оставшиеся изменения и запушить**. Поэтому прежние локальные
изменения консоли ALERT сохраняются следующим коммитом в той же ветке вместе с
этим обновлением; они описаны в разделе 7. PR #28 получает этот дополнительный коммит.

## 2. Решения пользователя, которые сохраняют силу

1. Название — **Eryx**, в честь *Eryx jayakari*; репозиторий — private
   **`ArtFilippov/eryx`**.
2. История старого `quality/` не переносится. Актуальный код, исправления, тесты,
   MIT-лицензия и сведения о происхождении сохранены.
3. Код инструмента — **`eryx/`**, проектные настройки и артефакты — **`.eryx/`**.
   Вариант submodule в `.eryx/tool/` был явно отклонён.
4. Активный интерфейс переименован в Eryx. Старые названия допустимы в исторических
   документах и чтении legacy-конфигурации для миграции.
5. Исправления проходят через **PR Eryx**, затем после слияния — через
   **отдельный PR потребителя**, закрепляющий merged SHA.

## 3. Реализованная структура и владение файлами

```text
tai/
├── eryx/                         # submodule → ArtFilippov/eryx
│   ├── check.py, apply.py, init_architecture.py
│   ├── install.sh, install.py
│   ├── defaults/, tests/, .github/workflows/ci.yml
│   ├── tools.json, report.schema.json
│   ├── pyproject.toml, uv.lock
│   ├── AGENTS.md, README.md, PROFILE.md
│   └── COMPATIBILITY.md, ORIGIN.md, LICENSE, workers и compatibility patches
├── .eryx/
│   ├── readability.toml
│   ├── scope.toml
│   ├── architecture.ready.json
│   ├── pylint-tests.toml
│   ├── pylint-catalog.json
│   ├── policy-notes.json
│   ├── installation.json
│   ├── reports/                  # игнорируется Git
│   ├── logs/                     # игнорируется Git
│   └── cache/                    # игнорируется Git
├── .gitmodules
├── pyproject.toml, uv.lock, pyrightconfig.json, tach.toml
└── docs/eryx-migration.md
```

Настройки и состояние `.eryx/` версионируются в tai. **Не игнорировать `.eryx/`
целиком.** Отдельно игнорируются runtime-подкаталоги и native cache `.tach/`.
Старый `.quality/` оставлен в `.gitignore` только для исторических локальных отчётов.

Workers, manifest и schema берутся из фактически запущенного Eryx, настройки и
архитектура — из потребителя. Подпроцессы используют корень и окружение потребителя.
Обычные install/check/apply/accept не пишут в submodule. Кэш пакетов uv сохраняет
обычное расположение, кэши анализаторов принадлежат потребителю.

## 4. Что перенесено и сохранено

- 42 регрессионных теста и актуальные исправления установленного runner:
  `_architecture_compat.py`, Pylint/pathlib, создание `raw/` до запуска Pylint,
  сохранение Bootstrap в карте clean-architecture.
- Установщик, defaults и 44 инфраструктурных теста из payload исходной поставки.
  Происхождение и SHA-256 payload записаны в `eryx/ORIGIN.md`; MIT notice сохранён.
- Транзакционные install/reinstall, dry-run, rollback, реальные native probes,
  миграция legacy-настроек и выбор extras. Исходники инструмента установщик не копирует.
- Собственные `pyproject.toml`, `uv.lock` и CI Eryx, не требующие доступа к tai.
- Схемы **`eryx-report/v1`** и **`eryx-event/v1`**, прежние коды завершения `0/1/2`.
  Поля `eryx.commit` и `eryx.dirty` определяют версию запущенного checkout.

Политика tai сохранена:

- Фактические численные лимиты перенесены из старого файла, не заменены defaults.
- `architecture.ready.json` содержит `accepted: true`, **109 модулей**. Marker,
  каталог правил и policy notes совпали с исходными файлами. Черновик архитектуры
  tai не генерировался заново. Источник истины — `tach.toml`, не старое описание 97 модулей.
- `garak.probes.base.Probe` остаётся в native `ignored-parents` потребителя и его
  тестовом профиле; из общего кода Eryx эта проектная политика удалена.
- `stubPath = src/attacks/fitd/stubs`, uv sources/index/dependency-metadata и Qwen
  сохранены. В native analyzer config изменён только путь исключения состояния.
- В dev-группу tai добавлен `jsonschema==4.26.0` для тестов схемы отчёта Eryx.
  Остальные версии зависимостей сохранены.

Старые tracked `quality/`, `.run` и документы его поставки удалены после переноса.
Остаточные игнорируемые bytecode/отчёты не являются активной интеграцией.

## 5. Актуальные команды

Перед работой читать `AGENTS.md`, `docs/agents/domain.md`, `eryx/AGENTS.md`,
`eryx/PROFILE.md` и `eryx/COMPATIBILITY.md`. При работе с Issues дополнительно
читать `docs/agents/issue-tracker.md`.

Свежий клон текущей интеграции (до слияния PR #28 нужна указанная ветка):

```bash
git clone --recurse-submodules --branch chore/eryx-submodule git@github.com:ArtFilippov/tai.git
# Для существующего клона:
git submodule update --init --recursive
uv sync --locked --group dev
```

Текущее рабочее окружение содержит Qwen, поэтому сохранять `--extra qwen`:

```bash
uv sync --locked --group dev --extra qwen
uv run --locked --group dev --extra qwen python eryx/check.py
uv run --locked --group dev --extra qwen python -B -m unittest discover -s eryx/tests
uv run --locked --group dev --extra qwen pytest
```

Установка/обновление интеграции и применение настроек:

```bash
bash eryx/install.sh --project . --extra qwen --dry-run --json
bash eryx/install.sh --project . --extra qwen --json
# После изменения .eryx/readability.toml или .eryx/scope.toml:
uv run --locked --group dev --extra qwen python eryx/apply.py --json
```

Extras записываются в `.eryx/installation.json` и повторно используются установщиком.
Для нового потребителя без Qwen при первой установке опустить `--extra qwen`.
Все команды принимают `--project`, по умолчанию — текущий каталог. Отдельный клон:

```bash
uv run --locked --group dev --extra qwen python /path/to/eryx/check.py --project .
```

Архитектура tai уже принята. Только после изменения слоёв повторить:

```bash
uv run --locked --group dev --extra qwen python eryx/init_architecture.py --accept --json
```

Для нового проекта отдельно создать и отревьюить draft перед `--accept`.
Генерация draft поддерживает один пакет под `src/`; существующая принятая
multi-package-конфигурация поддерживается. Не запускать `tach sync` как обычную
проверку и не ослаблять правила ради зелёного результата.

## 6. Фактические результаты проверок

Все приведённые проверки действительно выполнялись 2026-09-22. Полные локальные
прогоны до/после миграции использовали одинаковое состояние Python-исходников,
включая изменения консоли из раздела 7.

| Проверка | Результат |
|---|---|
| Static check до миграции | **exit 0**, 0 диагностик |
| Исходные native-регрессии | **exit 0**, 42 теста |
| Runtime pytest до миграции | **exit 0**, 659 passed |
| Полный static check Eryx в tai | **exit 0**, 0 диагностик, все per-tool статусы passed |
| Eryx после выделения | **exit 0**, 92 теста: 42 native + 44 infrastructure + 6 новых |
| Runtime pytest после миграции | **exit 0**, 659 passed |
| Установка в tai с Qwen | **exit 0**, 6 успешных native smoke-проверок |
| Повторная установка и apply | **exit 0**, `modified_files: []` |
| GitHub CI Eryx #1 | **SUCCESS**, 92 теста без доступа к tai |

Проверены новый uv/src-проект, dry-run без записей, идемпотентность, rollback после
реального сбоя native probe, сохранение extras/настроек, отдельный клон, настоящий
submodule по нестандартному пути с пробелами, отсутствие записей в инструмент
после install/check/apply/accept и отказ в успехе без принятой архитектуры.

Свежий клон **коммита интеграции `96913e9`**, полученный с GitHub через
`--recurse-submodules`, проверен в каталоге с пробелами:

- `uv sync --locked --group dev` — **exit 0**;
- `uv run --locked --group dev python eryx/check.py` — **exit 0**, 0 диагностик;
- `uv run --locked --group dev pytest` — **exit 0**, **656 passed**;
- рабочие деревья потребителя и submodule остались чистыми.

656 вместо 659 относится к состоянию до добавления трёх тестовых случаев консоли
следующим коммитом. Это не пропуски тестов и не регрессия миграции.

Единственное предупреждение runtime pytest: прежнее `PytestAssertRewriteWarning`
для уже импортированного `qwen_fakes`. Неразрешённых статических диагностик и ошибок
исполнения инструментов нет. При обновлении этого handoff Python-код не менялся;
приведённые полные проверки покрывают сохраняемые изменения консоли.

Отчёт: `.eryx/reports/latest.json`; финальный локальный run:
`20260922T180431Z-4804ce10`, `eryx.commit` равен закреплённому SHA,
`eryx.dirty: false`. Сырые логи находятся рядом с архивным `report.json`.
Детали миграции: `docs/eryx-migration.md`.

## 7. Сохраняемые изменения консоли ALERT

Эта работа существовала до миграции и первоначально не вошла в `96913e9`.
По последующему прямому поручению пользователя она включается отдельным коммитом
в `chore/eryx-submodule`, вместе с обновлённым handoff:

- `src/core/runs/console_journal.py`: начало Exchange обновляет текущую операцию,
  а запрос и ответ передаются в консоль вместе после завершения обмена.
- `src/app/console_format.py`: показанная пара заменяется только завершённым
  Exchange; pending/failed-вызов сохраняет предыдущую пару.
- `tests/test_console_journal.py`, `tests/test_series_console.py`: регрессии
  завершённых, ожидающих и неуспешных обменов, включая пустой ответ и настоящий FITD.
- `README.md`, `docs/alert-series.md`: документация поведения завершённых пар.

До первого завершённого обмена с Target обе текстовые колонки пусты. Во время
следующего обращения остаётся предыдущая пара. Три дополнительных тестовых случая
объясняют разницу между 656 тестами исходного коммита интеграции и 659 тестами
полного рабочего дерева.

## 8. Что делать дальше

1. Ревью и слияние **PR tai #28**: миграция уже опубликована, повторно создавать
   репозиторий или переносить инструмент не требуется. PR остаётся открытым;
   поручение «закоммитить всё и запушить» не означает его автоматического слияния.
2. Следующие исправления инструмента делать в рабочей ветке внутри `eryx/`
   (сначала выйти из detached HEAD) либо в отдельном клоне.
3. Воспроизвести проблему на потребителе, добавить автономную регрессию Eryx,
   выполнить его тесты и проверки потребителя, открыть PR инструмента через `gh`.
4. После CI/review/merge закрепить точный merged SHA отдельным PR потребителя,
   включив необходимые интеграционные изменения и ссылку на PR Eryx.

Рабочий каталог: `/vg/isp/tai`; origin tai: `git@github.com:ArtFilippov/tai.git`.
Состояние публикации проверять через `git status`, `git log` и `gh pr view`:
этот handoff фиксирует выполненную миграцию и контекст следующего коммита,
но не заменяет актуальное состояние Git/GitHub.
