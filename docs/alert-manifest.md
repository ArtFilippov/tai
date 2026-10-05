# Воспроизводимая выборка ALERT (#15, часть #13)

Готовый вход для серии FITD —
[`examples/alert-manifest.json`](../examples/alert-manifest.json): **32 базовых
запроса**, по одному на каждую микрокатегорию. Порядок макрогрупп и их размеры:
`hate=8`, `crime=8`, `substance=5`, `sex=3`, `self_harm=3`, `weapon=5`.
Таксономия — предоставленный [`ALERT_taxonomy.md`](../ALERT_taxonomy.md),
сохранённый побайтно, включая исходные английские описания.

## Источник и закрепление

Источник: Simone Tedeschi et al., **ALERT: A Comprehensive Benchmark for Assessing
Large Language Models' Safety through Red Teaming** (2024),
[статья](https://arxiv.org/abs/2404.08676), Section 3, Figure 2;
[официальный датасет Babelscape/ALERT](https://huggingface.co/datasets/Babelscape/ALERT).
Лицензия данных: **CC BY-NC-SA 4.0**, см.
[карточку закреплённой ревизии](https://huggingface.co/datasets/Babelscape/ALERT/blob/c76d0637f4573742982d5fe961a31bdb3d1e2c80/README.md).

| Параметр | Закреплённое значение |
| --- | --- |
| Dataset / config / split | `Babelscape/ALERT` / `alert` / `test` |
| Ревизия | `c76d0637f4573742982d5fe961a31bdb3d1e2c80` |
| Файл в этой ревизии | `alert.jsonl` (соответствие config/split задано карточкой) |
| Размер / число исходных записей | 2 216 517 байт / 14 763 |
| SHA-256 исходных байтов | `0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4` |
| Версия таксономии | `alert-figure2-v1` |
| SHA-256 `ALERT_taxonomy.md` | `0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd` |
| Версия таблицы alias | `alert-label-aliases-v1` |
| Seed | **20260915** |
| Алгоритм | `python3.12-random-mt19937-choice-v1` |
| SHA-256 готового манифеста | `fb398308890b68f2ad1565cb46d9277189289315775a7c4bc05c3c8fb3a2b42b` |

Манифест содержит извлечённые записи и метаданные источника, таксономии и выбора;
время запуска и пути локального кеша в него не включаются. Полный исходный JSONL
хранится в локальном кеше подготовки. Для загрузки готовых 32 запросов он не нужен.

## Подготовка из чистого клона

Команды выполняются из корня проекта. Требуется Python 3.12 из `.python-version`.
Установка зависимостей: `uv sync --locked`. Для существующего окружения Qwen
добавляйте **`--extra qwen`** к командам uv, чтобы сохранить опциональные зависимости;
сама подготовка ALERT не использует модели или GPU.

Первой подготовке нужна сеть. Команда получает **только закреплённый** официальный
файл, проверяет SHA-256, схему всех исходных записей и полное покрытие, сохраняет
проверенный JSONL и создаёт манифест:

```bash
mkdir -p runs
uv run --locked tai prepare-alert \
  --download --source-file runs/alert.jsonl \
  --taxonomy ALERT_taxonomy.md --seed 20260915 \
  --output examples/alert-manifest.json
```

`--source-file` и `--output` обязательны, их родительские каталоги должны существовать.
`--taxonomy` по умолчанию `ALERT_taxonomy.md` относительно рабочего каталога,
`--seed` по умолчанию `20260915`; допустимы целые `0 <= seed < 2**64`.
`--download` явно разрешает сетевой шаг и замену `--source-file` проверенными байтами.
Без него читается только локальный JSONL: сетевого fallback нет, в том числе при
отсутствующем файле. Успешная подготовка перезаписывает `--output`; ошибка проверки
входа сохраняет предыдущие файлы. Коды выхода: **0** — подготовлено, **2** — ошибка
аргументов, данных, скачивания или файлов. Контрольная сумма таксономии проверяется
до скачивания. Валидация полного источника заканчивается до записи манифеста.

Повторная подготовка и проверка байтов **без сети**:

```bash
uv run --locked --offline tai prepare-alert \
  --source-file runs/alert.jsonl \
  --taxonomy ALERT_taxonomy.md --seed 20260915 \
  --output runs/alert-manifest-reproduced.json
cmp examples/alert-manifest.json runs/alert-manifest-reproduced.json
sha256sum examples/alert-manifest.json
```

Инференс использует готовый локальный манифест через загрузчик ниже. Команды
подготовки и скачивания не являются частью исполнения серии.

## Алгоритм выбора и базовый запрос

1. Проверить весь JSONL: каждая запись — ровно `id` (неотрицательное целое,
   boolean не допускается), `prompt` (непустой Unicode-текст), `category` (известный
   label). Отклонить повторный ID и назвать все отсутствующие микрокатегории.
2. Канонизировать category только через явную таблицу
   [`src/alert/taxonomy.py`](../src/alert/taxonomy.py). Канонические
   идентификаторы допускаются без изменения; дополнительные alias перечислены
   в коде и в `taxonomy.aliases` манифеста. Среди них `crime_kidnap`,
   `crime-kidnap`, `crime-kidnapp`, `crime-kidnapping` → `crime_kidnapp` и
   `self-harm-pro-thin`, `self-harm-thin` → `self_harm_thin`.
   Также явно перечислены дефисные формы остальных категорий, включая `hate-lgbtq+`.
   Регистр, пробелы и неизвестные префиксы не исправляются автоматически.
3. Отсортировать записи по **числовому ID по возрастанию** и распределить по
   каноническим микрокатегориям. Порядок категорий — порядок Figure 2, полностью
   записанный в `selection.category_order` (он же порядок таблицы таксономии).
4. Создать отдельный `random.Random`, выполнить `seed(seed, version=2)`.
   В CPython 3.12 используется MT19937. По порядку категорий выполнить ровно один
   `rng.choice(group)` на категорию. Индекс выбирается через `_randbelow(n)`:
   `getrandbits(n.bit_length())` с повторением, пока число не меньше `n`.
   Пустые категории запрещены. Глобальное состояние RNG не используется.
5. Записать выбранные записи в порядке категорий и тот же порядок ID в
   `selection.selected_ids`. JSON: UTF-8, `ensure_ascii=False`, отступ 2 пробела,
   один заключительный LF; порядок ключей задаётся `AlertManifest.dump()`.

**`raw-prompt-v1`**: `base_request` в точности равен полному исходному `prompt`,
включая `### Instruction: … ### Response:`, пробелы и переводы строк. Извлечения
инструкции, перевода, перефразирования или обрезки нет. `prompt` хранится отдельно
для аудита; сравнение `base_request == prompt` обязательно при загрузке.
Исходные labels также сохраняются независимо от канонической категории.

## Формат и публичный контракт для #16

Публичные типы и функции доступны из `tai`:

```python
from pathlib import Path

from tai import Message, load_alert_manifest

manifest = load_alert_manifest(
    Path("examples/alert-manifest.json"),
    taxonomy_path=Path("ALERT_taxonomy.md"),
)
# Полная валидация уже завершилась. Затем можно создавать участников эксперимента.
for record in manifest.records:
    base = Message(record.base_request)
    # record.id, record.micro_category, record.macro_category связывают запуск с входом.
```

`load_alert_manifest(path: Path, *, taxonomy_path: Path) -> AlertManifest` читает
только два локальных файла, не строит модели, не вызывает оценщик и ничего не
скачивает. Неверное содержимое даёт `ValueError`, отсутствующий/недоступный файл —
`OSError`. Весь манифест валидируется до возврата; частично загруженных записей нет.
Пути задаёт вызывающая сторона; путь таксономии не берётся из недоверенного JSON.

`AlertManifest` — frozen dataclass: `seed: int`, `records: tuple[AlertRecord, ...]`.
`AlertRecord` — frozen dataclass с полями:

| Поле | Тип / смысл |
| --- | --- |
| `id` | Исходный целочисленный ID ALERT, не ID Message или запуска |
| `prompt` | Полный исходный текст |
| `source_category` | Исходное поле `category`, включая alias |
| `micro_category` | Канонический идентификатор Figure 2 |
| `macro_category` | Один из `hate`, `crime`, `substance`, `sex`, `self_harm`, `weapon` |
| `base_request` | Полный текст для `Message`, равный `prompt` |

Оба типа имеют `dump()`, возвращающий свежий JSON-совместимый снимок.
Снимок `AlertManifest` является полным документом и удобен для сохранения
происхождения входа в материалах серии. Корневые поля JSON v1:

- `schema_version: 1`;
- `source`: dataset, config, split, полный revision, имя/URL файла, SHA-256,
  размер, число исходных записей и лицензия;
- `taxonomy`: версия/хеш исходного Markdown, ссылка на статью/рисунок,
  полная таблица макро-/микрокатегорий, версия и явная таблица alias;
- `selection`: seed, алгоритм/версия, порядок исходных записей и категорий,
  упорядоченный список выбранных ID;
- `base_request_representation: "raw-prompt-v1"`;
- `records`: ровно 32 объекта со схемой из таблицы выше.

Загрузчик строго проверяет схему (включая лишние/пропущенные и повторные JSON-ключи),
типы, закреплённые source/taxonomy metadata, реальные байты таксономии, версии,
допустимость ID для этого источника, непустые тексты, равенство базового запроса,
точное соответствие исходного label каноническим micro/macro, уникальность ID,
полное покрытие без повторов и согласованность обоих списков порядка.

Принадлежность текстов исходным ID и воспроизводимость **самой выборки** проверяет
подготовка по полному JSONL с закреплённым SHA-256. Офлайн-загрузчик проверяет
контракт локального документа; без полного источника он не переисполняет выборку.
Для аудита изменённого манифеста повторите подготовку и `cmp`. SHA-256 готового
артефакта выше дополнительно закрепляет именно поставляемые 32 запроса.
Конструкторы dataclass не заменяют загрузчик и не подтверждают происхождение.

`select_alert_records(rows, *, seed=20260915) -> tuple[AlertRecord, ...]` — чистая
функция для проверки входных записей и воспроизводимой выборки, также пригодная
для управляемых тестовых наборов. Она не выполняет I/O и не подтверждает SHA-256
официального источника; для официального манифеста используйте `tai prepare-alert`.

## Проверки

```bash
uv run --locked --offline pytest tests/test_alert.py
uv run --locked --offline pytest
uv run --locked --offline --group dev python eryx/check.py
```

Тесты используют общий запрет внешней сети. Управляемый набор проверяет golden
выборку, независимость от порядка исходных строк и глобального RNG, полное покрытие,
alias и сохранение текста; негативные случаи проверяют ошибки всех входных записей,
повреждение схемы/метаданных/таксономии, дубли и перестановку. Команда подготовки
проверяется с управляемым ответом загрузки и локальным повтором. Отдельный тест
проверяет SHA-256 и покрытие фактического поставляемого манифеста.

При реализации 2026-09-15 официальный JSONL скачан с указанной ревизии,
подтверждены его 14 763 записи и SHA-256, подготовлен реальный манифест и повторён
локальный шаг подготовки внутри `unshare --net` с побайтовым сравнением (`cmp`,
код 0). Для проверок окружение Qwen сохранялось через `--extra qwen`.

Фактические результаты (Python 3.12.13):

| Команда (после `uv run --locked --offline --extra qwen`) | Результат |
| --- | --- |
| `pytest tests/test_alert.py` | 77 passed, 0.33 s |
| `pytest` | 350 passed, 114.81 s |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 59 files already formatted |
| `mypy` | Success: no issues found in 19 source files |

## Расширенная выборка: 20 запросов на микрокатегорию (#45)

[`examples/alert-manifest-20.json`](../examples/alert-manifest-20.json) содержит
**640 базовых запросов**: по 20 различных ID в каждой из 32 микрокатегорий.
Исходные байты, ревизия и таксономия закреплены выше; старый манифест на 32 записи
не менялся. SHA-256 нового файла:
`eb8a2bfb79a43345958502ff4b0b59b603a0a541f6318c1766dc81def23186fa`.

В документе используется `schema_version: 2`. В `selection` дополнительно
записаны `per_category: 20` и `legacy_selected_ids` (32 ID исходного
манифеста). Алгоритм — `python3.12-random-mt19937-choice-plus-sample-v1`:
после проверки **всего** закреплённого JSONL записи группируются по Figure 2
и сортируются по числовому ID. Отдельный `random.Random` с исходным seed
`20260915` выбирает старый ID для каждой группы тем же `choice`, что и v1.
Далее новый независимый генератор с записанным seed выбирает 19 записей
`sample` из остальных ID группы в отсортированном порядке. Сначала в группе
записывается старый ID, затем 19 выбранных в порядке `sample`; группы следуют
порядку Figure 2. Требуется не менее 20 различных записей на категорию.
`base_request` побайтно повторяет исходный текст `prompt` после декодирования
JSONL. Для исходного seed загрузчик также сверяет канонический SHA-256 всех
640 объектов `records`, защищая от согласованной замены обоих текстовых полей.
Другие seed допускаются, но членство в источнике для них подтверждается только
повторной подготовкой из закреплённого JSONL.

Подготовка из чистого клона с явным скачиванием (каталог `runs` создать заранее):

```bash
uv run --locked tai prepare-alert --download \
  --source-file runs/alert.jsonl --taxonomy ALERT_taxonomy.md \
  --seed 20260915 --per-category 20 \
  --output examples/alert-manifest-20.json
```

После скачивания повторить **без сети** (без `--download`):

```bash
uv run --locked --offline tai prepare-alert \
  --source-file runs/alert.jsonl --taxonomy ALERT_taxonomy.md \
  --seed 20260915 --per-category 20 \
  --output runs/alert-manifest-20-reproduced.json
cmp examples/alert-manifest-20.json runs/alert-manifest-20-reproduced.json
sha256sum examples/alert-manifest-20.json
```

`load_alert_manifest(Path("examples/alert-manifest-20.json"),
taxonomy_path=Path("ALERT_taxonomy.md"))` полностью проверяет документ
до создания моделей. Без `--per-category` команда подготовки по-прежнему
создаёт исходный v1-манифест с 32 записями.
