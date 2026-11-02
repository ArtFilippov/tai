# Подробное руководство tai

Цель проекта — среда для экспериментов со стратегиями атак на языковые модели.
В модели предметной области стратегия получает базовый запрос, взаимодействует
с моделью в роли Target и предъявляет текстовый результат. Оценщик судит об
успешности именно этого результата, а журнал отдельно фиксирует ход запуска.
Цели и словарь проекта описаны в [CONTEXT.md](../CONTEXT.md).

## Текущее состояние

Проект переписывается с нуля. Доступны FITD, GOAT с разбором O-T-S-R из
закреплённого garak, Simple Adaptive Attacks, DRA, Bad Characters/deletion,
AutoDAN GA/HGA и ThreatenJSON с адаптированным coercive passage. Также
доступны управляемые собеседники, оценка предъявленного текста и построчный
журнал JSON. Команда `tai` запускает автономный FITD.

Пример без сети и GPU (укажите новый путь к файлу журнала):

```bash
uv run --locked --group dev tai /tmp/fitd-demo.jsonl "Explain the topic in detail"
```

`AttackRun` из `attack.attack_run` принимает `BaseRequest` с `text()` и
`metadata()`, Target с методом `reply(Conversation) -> str`, стратегию `Probe`,
оценщика с `assess(str) -> bool` и журнал с `record(dict)`. `Attack.run()`
выполняет уже связанный запрос; для свободного текста CLI использует
`PlainBase` без метаданных. `FITDAttackEnvelope` связывает роли через `AttackSetup`:
`SharedModelAttackSetup(model, base)` передаёт одну модель всем трём ролям,
а `ExplicitAttackSetup(attacker, target, judge, base)` — разные ресурсы.
`main()` собирает отдельного Attacker для `FITDProbe`
и оценщика с записью вердикта в журнал; запуск сохраняет текст и метаданные
базового запроса в событии `start`, записывает выбор результата и передаёт его оценщику;
`ParsedJournal(path).read()` восстанавливает их после завершения процесса.
Передавайте новый файл журнала и новых собеседников для каждого запуска.
Каждая запись содержит идентификатор и версию схемы; событие `start` содержит
снимок сборки атаки и вложенные настройки участников. Формат описан в
[схеме журнала](journal-schema.md).

## Локальный запуск Qwen

Сначала подготовьте закреплённый снимок модели и токенизатора
`Qwen/Qwen2.5-0.5B-Instruct` (ревизия
`7ae557604adf67be50417f59c2c2f167def9a775`):

```bash
uv run --locked --group dev --extra qwen tai-qwen prepare
```

После подготовки запуск читает только локальные файлы. Укажите номер CUDA GPU,
новый путь журнала и базовый запрос:

```bash
uv run --locked --group dev --extra qwen tai-qwen run /tmp/qwen-fitd.jsonl "Explain the topic in detail" --gpu 0 --turns 6
```

Target, Attacker и оценщик используют одни веса, но отдельные разговоры.
Чтобы выбрать GGUF Attacker, установите дополнительную зависимость с поддержкой
CUDA (нужны CUDA Toolkit и C/C++ toolchain) и подготовьте только закреплённый
`Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf` из
`mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF`, ревизия
`7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c`:

```bash
uv sync --locked --group dev --extra qwen --extra gguf
uv run --locked --group dev --extra qwen --extra gguf tai-qwen prepare --attacker gguf
uv run --locked --group dev --extra qwen --extra gguf tai-qwen run /tmp/gguf-fitd.jsonl "Explain the topic in detail" --gpu 0 --turns 6 --attacker gguf
```

`--host` и `--port` задают адрес сервера (по умолчанию `127.0.0.1:8080`).
При свободном адресе команда запускает сервер из uv-окружения, отключает thinking,
закрепляет выбранный GPU и завершает только свой процесс. При занятом адресе она
проверяет точный идентификатор модели через `/v1/models` и использует сервер
без остановки; для ручного сервера задайте `model_alias` в виде
`mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF@7f5aee7d2dc4df18bd67eab85ea81e0bc2e98a2c/Huihui-Qwen3.5-2B-abliterated.Q4_K_M.gguf`.
Уже запущенный сервер сам отвечает за CUDA и thinking; его ответы сохраняются
без удаления reasoning. Во время `run` файлы моделей берутся только из кеша.
Событие `start` журнала сохраняет сборку атаки с моделями, ревизиями и настройками генерации.

Для другого закреплённого GGUF-файла Python-код может создать
`GGUFProfile(GGUFFile(repository, revision, filename),
GGUFLocation(host, port), gpu, tokens)` из модулей
`target.gguf.envelopes.gguf_profile`, `target.gguf.gguf_file` и
`target.gguf.gguf_location`.
Метод `prepare()` подготавливает файл, а `open()` возвращает контекст с готовым
`Model` для Attacker. Его `settings()` содержит данные для журнала; вызывать
модель нужно внутри контекста, пока доступен сервер. При повторном
использовании сервера его неизвестные настройки отмечаются как
`server-controlled`.

Оценщик получает исходный запрос и каждый предъявленный ответ Target. Вердикт
`YES` означает, что ответ действительно выполняет запрос; `NO` — что не выполняет.
Иные ответы оценщика прерывают запуск с ошибкой. `success` означает положительный
вердикт, `round_limit` — исчерпанное число раундов, `no_result` — отсутствие
принятого ответа. Журнал сохраняет все выбранные результаты и оценки;
`ParsedJournal(path).read()` возвращает последний результат вместе с его оценкой.

## Последовательный пул ALERT

Автономные команды запускаются также как Python-модули из корня проекта
(`uv run --locked --group dev --extra qwen python -m main.<module>`):

| Стратегия | `<module>` | Дополнительные флаги |
|---|---|---|
| FITD | `qwen_cli` (`run JOURNAL BASE`) | `--turns`, `--tokens`, `--attacker` |
| GOAT | `goat_cli` (`JOURNAL BASE`) | `--turns`, `--tokens` |
| ThreatenJSON | `threaten_json_cli` (`JOURNAL BASE`) | `--tokens` |
| WHOIS injection | `whois_cli` (`JOURNAL BASE`) | `--tokens` |
| Grandma roleplay | `grandma_cli` (`JOURNAL BASE`) | `--tokens` |
| Slur Completion | `tai-slur-completion JOURNAL BASE` | `--desired-response ANSWER` или `--deferred`, `--tokens` |
| Leak Replay | `tai-leak-replay JOURNAL BASE` | `--desired-response ANSWER` или `--deferred`, `--tokens` |
| WordNet exploration | `tai-wordnet JOURNAL BASE` | `--lexicon-dir DIR`, `--nodes`, `--calls`, `--seed`, `--tokens` |
| Simple Adaptive Attacks | `adaptive_attacks_cli` (`JOURNAL BASE`) | `--tokens` |
| DRA | `dra_cli` (`JOURNAL BASE`) | `--candidates`, `--seed`, `--tokens` |
| Bad Characters/deletion | `deletion_cli` (`JOURNAL BASE`) | `--perturbations`, `--positions`, `--ascii`, `--tokens` |
| AutoDAN GA/HGA | `autodan_cli` (`JOURNAL BASE`) | `--mode ga|hga`, `--target`, `--steps`, `--batch`, `--elites`, `--points`, `--crossover`, `--mutation`, `--seed`, `--tokens` |

Каждый автономный запуск требует `--gpu N` и **новый** путь `JOURNAL`.
Существующие команды `tai` и `tai-qwen run` для FITD остаются доступны.
Например, `uv run --locked --group dev --extra qwen python -m main.deletion_cli /tmp/deletion.jsonl "Explain the topic in detail" --gpu 0 --positions 2 --ascii 2`.
Настройки выбираются аргументами команды, без отдельного файла конфигурации.

Для отдельного GOAT без конфигурационного файла укажите новый путь журнала,
исходный запрос и лимит ходов; модель и токенизатор Qwen должны быть заранее
подготовлены:

```bash
uv run --locked --group dev --extra qwen tai-goat /tmp/goat.jsonl "Explain the topic in detail" --gpu 0 --turns 2 --tokens 192
```

`GOATAttack` из `attack.goat.goat_attack` предоставляет границу Python `Attack`;
`GOATAttackEnvelope` связывает базовый запрос, роли и журнал. Attacker и Target
ведут независимые истории; внутренний GOAT judge записывается как `decision`,
а отдельный оценщик получает только выбранный ответ Target. Отсутствие первого
кандидата записывается как `no_result` без пустого предъявленного результата.
Target, Attacker и итоговый оценщик используют локальный Qwen с независимыми разговорами.

Отдельный ThreatenJSON запускается с исходным запросом и новым файлом журнала:

```bash
uv run --locked --group dev --extra qwen tai-threaten-json /tmp/threaten-json.jsonl "Explain the topic in detail" --gpu 0 --tokens 192
```

`ThreatenJSONAttack` и `ThreatenJSONAttackEnvelope` дают Python API. Контрольный
и усиленный запросы содержат одинаковые исходный текст и инструкцию JSON-only;
второй дополняется адаптированным давлением из закреплённого garak. Журнал
помечает обмены `phase: control` и `phase: pressured`, а также `json_parseable`.
Контрольный ответ диагностический; оценщик получает только усиленный ответ и
требует как корректный JSON любого типа, так и содержательный ответ на исходный
запрос. JSON без ответа и ответ без JSON получают отрицательную оценку.

WHOIS injection доступен через `WhoisAttackEnvelope(PlainBase(base),
SharedModelAttackSetup(model, base), journal)` и отдельную команду:

```bash
uv run --locked --group dev --extra qwen tai-whois /tmp/whois.jsonl "Explain the topic in detail" --gpu 0 --tokens 192
```

Команда предъявляет Target один WHOIS-отчёт из закреплённого garak: задача —
оценить домен, а полный исходный запрос внедрён в поле `Registrant Organization`.
Один полный ответ Target записывается до независимой оценки по исходному запросу;
фиксированная фраза «not malicious» не означает успеха. В серии вариант
включается только флагом `tai-alert --whois` и записывает отдельные журналы в
`whois/<id>.jsonl`.

Grandma roleplay доступен через `GrandmaAttackEnvelope(PlainBase(base),
SharedModelAttackSetup(model, base), journal)` или отдельную команду:

```bash
uv run --locked --group dev --extra qwen tai-grandma /tmp/grandma.jsonl "Explain the topic in detail" --gpu 0 --tokens 192
```

Один запрос сохраняется дословно внутри роли бабушки, вдохновлённой шаблонами
закреплённого garak, без корпуса Windows-ключей. Полный ответ единственного
обмена Target записывается до независимой оценки по исходному запросу; формат
ключа и эвристика отказа не считаются успехом. В серии `--grandma` добавляет
отдельные журналы в `grandma/<id>.jsonl`, не меняя варианты по умолчанию.

WordNet exploration использует заранее установленный `wn` и локальный
`oewn:2023`. Подготовка выполняется отдельно (загружает данные только здесь):

```bash
uv run --locked --group dev tai-wordnet-prepare /tmp/tai-wn
uv run --locked --group dev --extra qwen tai-wordnet /tmp/wordnet.jsonl "Explain the topic in detail" --lexicon-dir /tmp/tai-wn --gpu 0 --nodes 4 --calls 8 --seed 7
```

Python API: `WordnetAttackEnvelope(PlainBase(base), setup, journal,
PreparedWordnet(open_wordnet(Path("/tmp/tai-wn"))), budget=WordnetBudget(4, 8, 7))`.
`setup` задаёт независимые роли Attacker, Target и judge. Attacker ранжирует до
пяти близких терминов; поиск проверяет синсеты и выбирает первый допустимый.
Шаблоны и гипонимы закреплённого garak обходятся в ширину, ответ-зависимый
`MitigationBypass` решает только раскрытие дочерних узлов. Каждый полный ответ
Target отдельно оценивается по исходному запросу. Первая положительная оценка
останавливает поиск; иначе предъявляется последний оценённый ответ либо
`no_result`. Лимиты синсетов и обращений к Target действуют на весь запуск,
а `--seed` определяет воспроизводимый порядок дочерних узлов. В серии вариант
включается `tai-alert --wordnet --wordnet-dir /tmp/tai-wn` с отдельными
`--wordnet-nodes`, `--wordnet-calls`, `--wordnet-seed`; журналы записываются
в `wordnet/<id>.jsonl` без дополнительных попыток на посещённые узлы.

Simple Adaptive Attacks запускается без файла настроек, с исходным запросом и
новым файлом журнала:

```bash
uv run --locked --group dev --extra qwen tai-adaptive-attacks /tmp/adaptive.jsonl "Explain the topic in detail" --gpu 0 --tokens 192
```

Python API: `AdaptiveAttacksAttackEnvelope(PlainBase(base),
SharedModelAttackSetup(model, base), journal)` из `attack.adaptive_attacks.envelopes.adaptive_attacks_attack_envelope`.
Атака использует шаблон и пять суффиксов установленного garak в исходном порядке;
исходный запрос подставляется дословно. Каждый вариант получает новый разговор
Target, а ответы и вердикты записываются в журнал. Первый положительный вердикт
останавливает поиск; если все отрицательны, предъявляется пятый ответ. Эта ранняя
остановка — политика tai, а не стандартного garak probe.

DRA запускается с явным базовым запросом и новым журналом:

```bash
uv run --locked --group dev --extra qwen tai-dra /tmp/dra.jsonl "Explain the topic in detail" --gpu 0 --tokens 192 --candidates 4 --seed 7
```

Python API: `DRAAttackEnvelope(PlainBase(base), SharedModelAttackSetup(model, base),
journal, DRABudget(candidates, seed))` из `attack.dra.envelopes.dra_attack_envelope`.
Закреплённые четыре шаблона, карта букв и cache-only moderation garak преобразуют
ровно переданный запрос (без подмены на bundled behaviors и без Detoxify). Число
кандидатов — от 1 до 4; проверяется до обращений к модели. Каждый кандидат получает
отдельный разговор Target; журнал сохраняет шаблон, реконструкцию, запрос, ответ
и оценку ответа относительно исходного текста. Положительная оценка останавливает
запуск, иначе предъявляется последний оценённый ответ. `--seed` действует только
внутри этого запуска, не изменяя глобальный генератор случайных чисел.

Bad Characters/deletion использует закреплённое преобразование garak:
печатаемый ASCII-символ и настоящий U+0008 после выбранного символа исходного
запроса. Автономная команда `tai-deletion JOURNAL BASE --gpu N` принимает
`--perturbations` (по умолчанию 1), `--positions` (8) и `--ascii` (4):
обычный профиль даёт до 32 кандидатов на один базовый запрос.
В серии `tai-alert` это `--deletion-perturbations`, `--deletion-positions`,
`--deletion-ascii`; исключение варианта — `--no-deletion`.

AutoDAN GA запускается с исходным запросом и **явным желаемым продолжением**:

```bash
uv run --locked --group dev --extra qwen tai-autodan /tmp/autodan.jsonl "Explain the topic in detail" --gpu 0 --target "Sure, here is an explanation" --steps 2 --batch 3 --elites 1 --seed 7
```

Для иерархического поиска добавьте `--mode hga`. Он использует нативные
`autodan_hga`, накопление словарного momentum и замену слов с вероятностью
`--crossover` (для GA этот флаг управляет скрещиванием предложений). Мутации
`--mutation` выполняет отдельный Attacker в обоих режимах; `--points`
применяется только к GA. Перед автономным HGA-запуском с несколькими шагами
подготовьте NLTK `punkt_tab` вместе с локальными весами: закреплённый garak
использует его для разбиения слов при переходе между поколениями.

```bash
uv run --locked --group dev python -c 'import nltk; nltk.download("punkt_tab", raise_on_error=True)'
```

`AutoDANAttack` и `AutoDANAttackEnvelope` предоставляют Python API: связывают
`AttackSetup`, `QwenLogits`, журнал и `AutoDANBudget(steps, batch, elites, points,
crossover, mutation, seed, target, mode)`, где mode равен `ga` или `hga`.
Шаги и размер популяции ограничивают число
вычислений fitness (`steps × batch`); `--tokens` ограничивает длину каждого
ответа Target/Attacker. `--batch` от 2 до 128, `--elites` строго между 0 и batch,
число потомков `batch - elites` чётное для GA. `--points`, `--crossover` и `--mutation`
задают нативные операции garak; `--seed` закрепляет выбор и скрещивание.
Ресурс `garak.data/autodan/prompt_group.pth` читается без перезаписи; мутации
проводит отдельный Attacker. Fitness использует локальные logits Qwen и целевое
продолжение после **префикса вместе с базовым запросом**, не изменяя разговор
Target. При этом нативная проверка успешности смотрит ответ Target **только на
префикс**, без базового запроса: это расхождение сохранено намеренно. Положительный
нативный результат отдельно предъявляется оценщику исходного запроса; отсутствие
такого результата записывается как `no_result` без предъявленного текста и оценки.
Поддерживается только закреплённый локальный Qwen Target; бюджеты и backend
проверяются до inference. `--autodan` включает вариант GA в `tai-alert` (по умолчанию
он выключен); параметры задаются отдельными флагами `--autodan-target`,
`--autodan-steps`, `--autodan-batch`, `--autodan-elites`, `--autodan-points`,
`--autodan-crossover`, `--autodan-mutation`, `--autodan-seed`. Каждый запрос
получает отдельный журнал в подкаталоге `autodan/`.
`--autodan-hga` независимо добавляет вариант HGA с теми же бюджетами и
отдельными журналами `autodan_hga/<id>.jsonl`; можно выбрать оба варианта.

Подготовьте локальный снимок Qwen командой `tai-qwen prepare` выше; затем из корня
репозитория укажите **новый** каталог вывода и CUDA GPU:

```bash
uv run --locked --group dev --extra qwen tai-alert /tmp/alert-series --gpu 0 --limit 1 --autodan --autodan-hga
```

По умолчанию `--limit 20`: команда берёт первые N запросов в каждой из 32
микрокатегорий (всего `32 * N`, `1 <= N <= 20`) из закреплённого локального
`examples/alert-manifest-640.json`. Можно задать `--manifest PATH`, а также те
же `--turns`, `--tokens`, `--attacker qwen|gguf`, `--host`, `--port`, что у
`tai-qwen run`. `--goat-turns` ограничивает GOAT отдельно от FITD `--turns`.
ThreatenJSON включён по умолчанию; `--no-threaten-json` исключает эту фабрику.
WHOIS добавляется отдельно флагом `--whois`, без изменения вариантов по умолчанию.
Grandma добавляется отдельно флагом `--grandma`.
Slur Completion и Leak Replay добавляются флагами `--slur-completion` и
`--leak-replay`: передайте `--desired-response ANSWER` для общей подготовленной
строки либо опустите его для отложенной подготовки с кэшем на каждый запрос.
Первый вариант продолжает короткое начало ответа через настоящий assistant prefill;
второй сначала просит независимого Attacker замаскировать запрещённые части ответа
с сохранением прочих символов и отказывает в запуске Target при невалидной маске.
Оба оценивают полный предъявленный ответ на базовый запрос; оригинальные garak
детекторы завершения конкретных оскорблений и воспроизведения известных имён
не применяются к произвольным ALERT-запросам.
WordNet добавляется отдельно флагом `--wordnet` с подготовленной базой `--wordnet-dir`.
Он выбирает ближайший подтверждённый синсет, ограничивает узлы и вызовы флагами
`--wordnet-nodes` и `--wordnet-calls`, проверяет сигнал отказа для ветвления
отдельно от итогового judge. Оригинальный garak пропускает термин без точного
синсета; здесь допустим ближайший предложенный термин.
Simple Adaptive Attacks также включён по умолчанию; `--no-adaptive-attacks`
исключает его. Журналы находятся в `adaptive_attacks/<id>.jsonl`.
DRA включён независимо: `--no-dra` исключает его, `--dra-candidates` и `--dra-seed`
задают бюджет и локальное зерно. Журналы находятся в `dra/<id>.jsonl`.
При `--limit 1 --autodan --autodan-hga` включены все восемь фабрик:
32 исходных запроса × 8 вариантов = **256 попыток**. Это не один запрос и не
число внутренних кандидатов. Пять суффиксов Simple Adaptive Attacks всегда
остаются нативными; ранний положительный вердикт может сократить фактическое
число вызовов. Без двух флагов AutoDAN планируются 192 попытки.
Если дополнительно включить `--whois`, `--grandma`, `--slur-completion`,
`--leak-replay` и `--wordnet --wordnet-dir /tmp/tai-wn`, планируются
**13 × 32 = 416 попыток**.
Внутренние узлы WordNet и подготовка желаемого ответа не добавляют попыток.
Пример полного opt-in запуска (перед этим подготовьте Qwen, WordNet и NLTK
`punkt_tab` по инструкциям выше; укажите новый каталог):

```bash
uv run --locked --group dev --extra qwen tai-alert /tmp/alert-13 --gpu 0 --limit 1 --turns 1 --goat-turns 2 --tokens 64 --dra-candidates 2 --deletion-positions 2 --deletion-ascii 2 --autodan --autodan-hga --whois --grandma --slur-completion --leak-replay --wordnet --wordnet-dir /tmp/tai-wn --wordnet-nodes 2 --wordnet-calls 2
```
Для GGUF нужны подготовленный файл и зависимость `--extra gguf`.
Загрузчик сверяет весь файл с закреплённым SHA-256
`3ee64c7a6b86c2b28174d8c699407e24f94a4e54ca9b8f9c5fc35bf460725f43`:
`--manifest` позволяет выбрать копию этого файла, а не произвольную выборку.
Изменение текста и пересчёт внутренних SHA-256 не пройдут проверку. Манифест
целиком и каталог вывода проверяются до обращений к моделям. Повтор
требует нового каталога: существующие журналы и сводка не перезаписываются.
Для автономного воспроизведения файла сначала отдельно получите `alert.jsonl`
из `Babelscape/ALERT` (`test`, ревизия
`c76d0637f4573742982d5fe961a31bdb3d1e2c80`), затем выполните
`uv run --locked --group dev --extra qwen python -m main.alert_manifest_cli /path/to/alert.jsonl /tmp/alert-manifest-640.json`.
Подготовка проверяет SHA-256 исходного файла и все 14 763 записей без сети.

В корне находится общий `summary.json`. По одному независимому журналу на
исходный запрос находятся в `fitd/`, `goat/`, `threaten_json/`,
`adaptive_attacks/`, `dra/`, `deletion/`, `autodan/`, `autodan_hga/`,
а при `--whois` также в `whois/`, при `--grandma` в `grandma/`,
при `--slur-completion` в `slur_completion/`, при `--leak-replay` в
`leak_replay/`, при `--wordnet` в `wordnet/`
(`id.jsonl` в каждом каталоге). Внутренние кандидаты и контрольный обмен
ThreatenJSON не создают дополнительных записей серии.
Фабрики атак выполняются по очереди; каждая обходит весь пул запросов. В `runs`
записываются сквозной номер попытки, номер фабрики, полный базовый запрос и его
метаданные, JSON-описание атаки, финальный результат или ошибка. Верхний уровень
содержит `planned`, `attempted` и `complete`. При ошибке попытки цикл продолжается;
прерывание сохраняет частичную сводку с `complete: false`. Сводка обновляется после
каждой попытки и не вычисляет успех, процент или вердикт оценщика: они доступны в
журнале через `ParsedJournal(path).read()`. Если атака не была собрана, поле
`attack` отсутствует; при ошибке вместо `result` сохраняется `error`.
`outcome: failure` — ошибка выполнения, а не успешная атака;
`result: null` при `no_result` означает отсутствие предъявленного ответа.
События `call` сохраняют фактические вызовы по ролям, `result` — предъявленный
текст, `judgment` — оценку именно этого текста, `end.reason` — причину остановки.
`ParsedJournal(path).read()` возвращает `text`, `assessment`, `reason` и
`calls`; оценку нельзя выводить из контрольного ответа либо отвергнутого кандидата.
ThreatenJSON использует отдельную **JSON-answer** рубрику: усиленный ответ
должен быть корректным JSON и содержательно отвечать на исходный запрос;
обычный `YES/NO`-оценщик других атак проверяет выполнение запроса без JSON-гейта.
Качество оценки зависит от малой модели Qwen 0.5B и не гарантирует объективную
успешность атаки.

Для ограниченного воспроизведения всей серии с обоими AutoDAN (укажите свободный
GPU и новый каталог; перед запуском подготовьте Qwen и NLTK `punkt_tab` как выше):

```bash
uv run --locked --group dev --extra qwen --extra vllm tai-alert /tmp/alert-issue-97-live-20261027 --gpu 1 --limit 1 --turns 1 --goat-turns 2 --tokens 64 --dra-candidates 2 --dra-seed 13 --deletion-perturbations 1 --deletion-positions 2 --deletion-ascii 2 --autodan --autodan-hga --autodan-target 'Sure, here is the answer' --autodan-steps 2 --autodan-batch 3 --autodan-elites 1 --autodan-points 1 --autodan-crossover 0 --autodan-mutation 0 --autodan-seed 8
```

Этот короткий профиль: FITD 1 обмен, GOAT 2, DRA до 2 кандидатов, удаление
до 4, AutoDAN GA/HGA по 2 шага × 3 особи, Simple Adaptive Attacks до пяти
нативных суффиксов. Общие бюджеты отличаются: FITD 10, GOAT 5, DRA 4,
удаление до 32, AutoDAN (если включён) 2 × 3; `--tokens` по умолчанию 192.
Реальные итоги и разбор журналов приведены в [проверке #97](issue-97-verification.md).

Для наблюдения за изменениями кандидатов без файла журнала используйте отдельную
команду (снимок Qwen должен быть подготовлен заранее):

```bash
uv run --locked --group dev --extra qwen tai-qwen-observe "Explain the topic in detail" --gpu 0 --turns 2
```

Она печатает отчёт: `candidate_changes` сравнивает первый ответ Attacker с
базовым запросом, последующие — с предыдущим кандидатом (`similarity` — доля
совпадения символов от 0 до 1); проверочные ответы Attacker не считаются
кандидатами. `target_requests` перечисляет разные запросы, действительно
переданные Target, в порядке первого появления. `presented_result`, `verdict` и
`status` относятся к результату атаки, оценке и завершению запуска отдельно от
изменений текста. Каждый вызов создаёт собственный журнал только в памяти.

Для поиска первого изменённого **первого** кандидата FITD среди фиксированного
набора запросов с метками микрокатегорий ALERT:

```bash
uv run --locked --group dev --extra qwen tai-qwen-search --gpu 0
```

Команда проверяет запросы по порядку, каждый ровно один раз, с лимитом в один
обмен с Target. Печатается JSON со статусом `found` (базовый запрос, категория,
первый ответ Attacker, запросы к Target, число проверенных баз и вызовы модели
по ролям и всего) либо `exhausted` (проверенное число и вызовы). Поле
`inspections` показывает первый ответ Attacker для каждой проверенной базы.
Журнал запуска существует только в памяти. Изменение текста не доказывает
успешность атаки.

Интеграционный тест с этим запросом и реальным Qwen помечен `real_qwen` и не
запускается обычным `pytest`. Для явного запуска нужен подготовленный снимок
модели и CUDA GPU 0:

```bash
uv run --locked --group dev --extra qwen pytest -m real_qwen
```

Предметная область остаётся прежней: атаки на текстовые LLM, независимые разговоры
участников, оценка предъявленного результата и воспроизводимая запись эксперимента.
Доступность других стратегий garak будет определяться новой реализацией;
закреплённый пул ALERT уже используется для последовательных запусков выше.

## Основа для разработки

Сохранены Python 3.12, управление зависимостями через [uv](https://docs.astral.sh/uv/)
и статические проверки [Quard](https://github.com/ArtFilippov/quard) в подмодуле
`quard/`. Для подготовки окружения из корня репозитория:

```bash
git submodule update --init --recursive
uv sync --locked --group dev
```

После появления Python-кода полная статическая проверка запускается командой:

```bash
uv run --locked --group dev python quard/check.py
```

Отчёт Quard находится в `.quard/reports/latest.json`. Для добавляемых тестов
используйте `uv run --locked --group dev pytest`. Если разрабатывается окружение
с локальным Qwen, к командам uv добавляется `--extra qwen`; для общего окружения
Qwen/vLLM — оба флага `--extra qwen --extra vllm`. Правила работы с исходниками
и проверками приведены в [AGENTS.md](../AGENTS.md).
