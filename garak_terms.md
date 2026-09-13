# Термины garak

Этот документ фиксирует рабочий словарь garak для red teaming и оценки LLM-систем. Как и в [словаре PyRIT](pyrit_terms.md), термины разделены на два уровня: ключевая доменная модель и сокращенный справочник по основным garak-терминам. Отдельная колонка показывает соответствия PyRIT.

Основа — документация и исходники локальной копии [NVIDIA/garak](garak/) на commit `3f50ea5ff9cd7050099940647c15c39b07a93392`. Маппинг сравнивает смысл и ответственность компонентов, а не совместимость API.

## Как читать маппинг

- Термин без префикса — близкое смысловое соответствие; `≈` — частичное или составное соответствие с ограничениями, указанными в описании.
- `+` в колонке PyRIT означает, что один термин garak объединяет несколько понятий PyRIT. `+` в колонке garak означает сборку из нескольких сущностей, а не имя отдельного класса.
- Если понятие PyRIT собирается из нескольких терминов garak, у этих строк **одинаковый маппинг**. Например, `Detector` и `Evaluator` вместе реализуют оценивание; `Message` и `Turn` вместе дают содержимое сообщения и роль отправителя.
- **Каждая таблица отсортирована по порядку соответствующей таблицы в `pyrit_terms.md`.** Для составного маппинга используется самое раннее понятие PyRIT; слагаемые также перечислены в порядке PyRIT. Строки без подходящего соответствия, в том числе составного, стоят в конце.

## Ключевые доменные термины

Это основные понятия, через которые удобно понимать garak; названия процессов здесь не обязательно обозначают Python-классы.

| Термин garak | Краткое значение | Примеры | Маппинг на PyRIT |
|---|---|---|---|
| `Target` | Тестируемая система: модель, chatbot, API или приложение, доступное через generator. [^core] | чат-модель, корпоративный REST endpoint | `Target` |
| `Generator` | Адаптер, через который garak получает ответы target. Обычно представляет тестируемую систему; тот же интерфейс может использоваться для вспомогательных моделей. [^generator] | `openai.OpenAIGenerator`, `rest.RestGenerator` | ≈ `Target` |
| `Goal` | Формулировка поведения, которого пытается добиться probe или конкретная попытка. Встречается в `Probe.goal`, `Attempt.goal` и параметрах отдельных атак. [^probe] | извлечь system prompt, вызвать токсичный ответ | `Objective` |
| `Intent` | Тип поведения или failure mode, который хотят вызвать. Код типологии может обозначать целый класс поведения; конкретный goal или stub уточняет отдельную цель. Intents включают не только вредные действия. [^cas] | `S005hate`, `T009indirect` | ≈ `Objective` |
| `Probe` | Исполняемая проверка: задает или строит prompts, описывает искомое поведение и организует взаимодействие с target. Обычный probe может объединять цель, исходные данные, прием и алгоритм; `IntentProbe` разделяет прием и intents. [^probe] | `dan.Dan_11_0`, `tap.TAP` | ≈ `Objective` + `Seed` + `Technique` + `Strategy` |
| `Stub` | Исходный контент, связанный с intent, из которого probe строит тестовые prompts. [^cas] | `TextStub`, `ConversationStub` | ≈ `Seed` |
| `Payload` | Подставляемый материал: фрагмент prompt, название сущности, строка или контент, появление которого проверяют в ответе. Не обязательно готовая атака. [^payload] | название продукта из `keyedprod_win10`, фрагмент для кодирования | ≈ `Seed` |
| `Technique` | Прием, которым probe пытается вызвать выбранное поведение. В CAS отделен от intent; описывается, в частности, тегами `demon:*`. [^cas] [^taxonomy] | roleplay, encoding, persuasion | `Technique` |
| `Adaptive probing` | Поиск следующих атакующих запросов с учетом ответов или оценок target; алгоритм находится внутри probe и его вспомогательного кода. [^adaptive] | TAP, PAIR, GOAT | ≈ `Strategy` |
| `Scan` / `Run` | Запуск набора проверок против target. При заданных рисках, intents, данных и критериях приближается к кампании PyRIT; сам по себе run может быть произвольным набором probes. [^core] [^config] | выбранные probes и intents в `run.spec` | ≈ `Scenario` |
| `Context-Aware Scanning` / `CAS` | Подход, разделяющий «как тестировать» и «какое поведение вызывать»: выбранные techniques применяются к intents, для которых подбираются данные и detectors. В исследованной версии развивается как экспериментальный механизм. [^cas] | одна roleplay technique на нескольких safety intents | ≈ `Scenario` + `Workflow` |
| `Attempt` | Запись конкретного тестового взаимодействия: исходный prompt или история, goal, outputs, scores и metadata. Может содержать несколько генераций; в iterative probe отдельный attempt создается на каждом шаге. [^attempt] [^probe] | один prompt с несколькими ответами target | ≈ `Attack` + `Evidence` |
| `Vulnerability scanning` / `Red teaming` | Систематический поиск нежелательного поведения AI-систем статическими, динамическими и адаптивными проверками. [^core] | поиск утечек, jailbreaks, hallucinations | ≈ `AI red teaming` |
| `Red-team model` / `Attack model` | Вспомогательная модель, генерирующая атакующие prompts или следующие ходы. Это роль модели, которая также подключается через generator. [^adaptive] | `atkgen.Tox.redteamer`, attack model в TAP | `Adversarial target` |
| `Prompt` | Контент или история, передаваемые generator для получения ответа. На границе `Generator.generate()` это `Conversation`. [^attempt] [^generator] | пользовательский запрос, история с system prompt | `Prompt` |
| `Jailbreak` | Попытка или результат обхода ограничений target; также класс проверяемого поведения. Конкретный probe реализует прием или алгоритм такого обхода. [^core] [^adaptive] | DAN, TAP, GCG | `Jailbreak` |
| `Prompt injection` | Внедрение инструкций, которые перенаправляют поведение target относительно исходной задачи. [^injection] | семейства `promptinject`, `latentinjection` | `Prompt injection` |
| `Latent injection` | Инъекция, спрятанная в обрабатываемом контенте: резюме, отчете и т.п. Ближайший аналог XPIA по вектору, но probes могут лишь собирать такой контекст внутри prompt, без реальной записи во внешний канал. [^injection] | `latentinjection.LatentInjectionResume` | ≈ `XPIA` |
| `Single-turn probing` | Проверка с одним запросом к target на тестовый случай. Повторные генерации для того же prompt не превращают ее в многоходовую атаку. [^probe] [^generator] | отправка заранее подготовленного jailbreak prompt | `Single-turn attack` |
| `Multi-turn probing` | Проверка с последовательными запросами; следующие сообщения могут зависеть от ответов, а история — дополняться или изменяться. [^probe] [^adaptive] | `IterativeProbe`, `goat.GOATAttack` | `Multi-turn attack` |
| `Conversation` | Последовательность сообщений с ролями. Один `Attempt` может хранить несколько conversation histories — по одной на генерацию. [^attempt] | system → user → assistant → user | `Conversation` |
| `Initial conversation` | Готовая история, переданная как входной prompt до получения новых ответов. Это использование `Conversation`, а не отдельный тип. [^attempt] [^probe] | system prompt и few-shot turns | ≈ `Prepended conversation` |
| `Buff` | Преобразует исходные attempts и создает их варианты: меняет кодирование, формулировку или язык prompt. Может иметь обратное преобразование ответа. [^buff] | Base64, paraphrase, low-resource-language translation | ≈ `Prompt transformation` |
| `Detector` | Первая часть оценивания: анализирует attempt и выдает оценки обнаружения заданного поведения. Для итогового решения используется вместе с evaluator. [^scoring] | поиск строк, классификатор токсичности, LLM judge | ≈ `Scoring` |
| `Evaluator` | Вторая часть оценивания: интерпретирует detector scores, применяет критерий PASS/FAIL и агрегирует результаты. Работает с оценками, а не вместо detector с сырым ответом. [^scoring] | `ThresholdEvaluator`, `ZeroToleranceEvaluator` | ≈ `Scoring` |
| `Detector score` | Число от `0.0` до `1.0`: степень или факт обнаружения поведения по семантике detector. `None` означает отсутствие оценки; rationale и категория не упакованы в единый объект score. [^scoring] | `0.0`, `1.0`, `0.8`, `None` | ≈ `Score` |
| `Refusal` / `Mitigation` | Отказ или защитная реакция target. Refusal-related detectors garak часто оценивают противоположное событие — обход защиты или отсутствие отказа. [^judge] | `mitigation.MitigationBypass`, `judge.Refusal` | `Refusal` |
| `Risk tags` / `Safety intents` | Классификации риска или поведения. Risk tags и safety-часть intent typology вместе помогают сопоставлять категории вреда; все теги и все intents к вреду не сводятся. [^taxonomy] [^cas] | `S005hate`, теги content safety | ≈ `Harm category` |
| `Dataset` | Набор исходных данных для проверки. В garak нет единого обязательного контейнера: используются payload groups, intent stubs и загрузчики данных отдельных probes. [^payload] [^cas] | локальные prompts, Hugging Face datasets | ≈ `Dataset` |
| `Fixed scan configuration` | Фиксация probes, intents, данных, параметров генерации и критериев позволяет использовать scan как воспроизводимую оценку. Это способ организации запуска, а не отдельный класс `Benchmark`. [^core] [^config] | одинаковый набор тестов для нескольких targets | ≈ `Benchmark` |
| `ASR` | Доля FAIL среди оцененных outputs для пары probe/detector: `fails / (passes + fails)`. `None` исключается. Это не автоматически доля уникальных многоходовых атак. [^scoring] | 12 FAIL из 100 оцененных outputs → `0.12` | ≈ `ASR` |
| `Harness` | Координатор процесса: запускает probes на generator, detectors на полученных attempts и evaluator на оценках. [^harness] | `ProbewiseHarness`, `PxD` | ≈ `Workflow` |
| `Modality` | Типы входа и выхода, декларируемые компонентами. Реальная поддержка зависит от конкретного generator и probe. [^generator] [^probe] | `text`, `image`, `audio`, `video`, `3d` | `Modality` |
| `Generator capabilities` | Отдельные признаки возможностей адаптера: modalities, несколько генераций за запрос, параллельные вызовы. Единой модели всех возможностей PyRIT здесь нет. [^generator] | `modality`, `supports_multiple_generations`, `parallel_capable` | ≈ `Capability` |
| `PASS` / `FAIL` / `SKIP` | Итог проверки с точки зрения target: PASS — критерий пройден, FAIL — обнаружено нежелательное поведение, SKIP — нет оцененных результатов. Не единый outcome всей атаки. [^scoring] | высокий hit score обычно дает FAIL | ≈ `Attack outcome` |
| `Report` | Журнал наблюдений и результатов scan, а также построенные из него сводки. Содержит prompts, outputs, detector scores и metadata. [^report] | `.report.jsonl`, `.report.html` | `Evidence` |
| `Hitlog` | Выборка срабатываний detector: prompt, output, score, goal и идентификаторы. Файл создается при первом hit. [^report] | `.hitlog.jsonl` | `Evidence` |

<details>
<summary>Все термины</summary>

Здесь перечислены классы, поля, механизмы и составные конструкции. Повторение понятия из первой таблицы уточняет его технический смысл.

| Термин garak | Что означает | Маппинг на PyRIT |
|---|---|---|
| `Harness` | Координатор полного цикла probing → detection → evaluation. Ближе к внешнему уровню исполнения; логика конкретной атаки остается в probe. [^harness] | ≈ `Executor` |
| `ProbewiseHarness` | Harness, который выбирает detectors по рекомендациям каждого probe; для `IntentProbe` базовый harness уточняет их по intent каждого attempt. [^harness] | ≈ `Executor` |
| `PxD` | Harness для явно выбранных probes и detectors: обычные probes проверяются всеми заданными detectors. Для `IntentProbe` действует общая intent-based маршрутизация. [^harness] | ≈ `Executor` |
| `Probe` | Базовый исполняемый компонент проверки; обычно объединяет алгоритм и набор prompts. Возвращает attempts, а внешнее оценивание организует harness. [^probe] | ≈ `AttackStrategy` / `Attack` + `SeedDataset` |
| `IntentProbe` | Probe, применяющий одну technique к stubs выбранных intents. Строит prompts и сохраняет их связь с intent; один instance может породить множество тестов. [^probe] [^cas] | ≈ `AttackStrategy` / `Attack` + `SeedDataset` |
| `IterativeProbe` | Основа адаптивных многоходовых probes: формирует следующие attempts по ответам предыдущих, поддерживает ограничения числа вызовов и попыток. [^probe] | ≈ `AttackStrategy` / `Attack` |
| `TreeSearchProbe` | Основа поиска по дереву: обход узлов, генерация prompts и выбор дальнейших ветвей по detector feedback. Не синоним конкретного алгоритма TAP. [^probe] | ≈ `AttackStrategy` / `Attack` |
| `tap.TAP` / `tap.PAIR` | Конкретные реализации адаптивного поиска jailbreak prompts с атакующей и оценивающей моделями. Итерации поиска сами по себе не означают единый диалог с неизменяемой историей target. [^adaptive] | ≈ `AttackStrategy` / `Attack` |
| `goat.GOATAttack` | Многоходовый probe с отдельными историями attacker и target, генерацией следующих ходов и проверкой jailbreak. [^adaptive] | ≈ `AttackStrategy` / `Attack` |
| `Attempt` | Изменяемая запись теста: `prompt`, `goal`, `notes`, `conversations`, `outputs`, `detector_results`, `status`, `uuid`. Одновременно служит контекстом, носителем результатов и входом detector; не гарантирует границу целой атаки. [^attempt] | ≈ `AttackContext` + `AttackResult` + `Scorable` |
| Probe configuration / `DEFAULT_PARAMS` | Параметры конкретного probe, загружаемые через `Configurable`: лимиты, данные, модели, параметры алгоритма. Отдельного общего объекта `AttackParameters` нет. [^probe] [^config] | ≈ `AttackParameters` |
| `Evaluator.test()` | Проверка одного detector score: `True` означает PASS target. Для attack outcome нужны семантика detector и правила агрегации по outputs/ходам; исключения и `None` требуют отдельной интерпретации. [^scoring] | ≈ `AttackOutcome` |
| `IntentProbe` + конкретный `Stub` | Составная конструкция «прием + исходный материал с intent», приближающая один атомарный тест. Один stub может развернуться в несколько prompts; конкретную цель и границу теста нужно определить. [^probe] [^cas] | ≈ `AtomicAttack` |
| `eval` / `probe_summary` | Записи отчета с агрегатами по probe/detector, counts и, когда доступны, intents. Это части сводного результата проверки. [^scoring] [^report] | ≈ `ScenarioResult` |
| `digest` / `technique_intent_matrix` | Сводка run, включая матрицу techniques × intents по тегам `demon:*`. В матрице `score` — pass rate; counts могут объединять несколько detectors для одних outputs. [^report] | ≈ `ScenarioResult` |
| `Generator` | Общий интерфейс получения ответов системы: `generate(Conversation)` возвращает список `Message` или пустых результатов. По охвату уже универсального `PromptTarget`, который может представлять и storage target. [^generator] | ≈ `PromptTarget` |
| Target generator / `--target_type` / `--target_name` | Generator и имя системы, выбранные для основного scan. Target type определяет адаптер, target name — конкретную модель или endpoint в терминах этого адаптера. [^core] [^config] | `objective_target` |
| `redteamer` / `red_team_model_*` / `attack_model_*` | Вспомогательный generator и его настройки внутри атакующих probes. Названия зависят от реализации; не путать с основным generator scan. [^adaptive] | `adversarial_target` / `adversarial_chat` |
| `Generator.modality` / `supports_multiple_generations` / `parallel_capable` | Декларации отдельных возможностей generator. Не содержат общего набора требований к editable history, streaming audio и другим возможностям PyRIT. [^generator] | ≈ `TargetCapabilities` |
| Generator configuration | Настройки адаптера в `plugins.generators` плюс его атрибуты и преобразование conversation в формат API. Единой обертки capabilities + policy + normalization pipeline нет. [^generator] [^config] | ≈ `TargetConfiguration` |
| `Probe.modality` / `_modality_match()` | Требуемые входные modalities и проверка их совместимости с generator. При несовпадении harness пропускает probe; это ограниченный аналог проверки требований к target. [^harness] | ≈ `TargetRequirements` |
| `Message.text` / `data_path` / `data_type` | Текст и поля вложения внутри сообщения. Приближают отдельные части мультимодального сообщения, но отдельной атомарной сущности с memory identity нет. [^attempt] | ≈ `MessagePiece` |
| `Message` | Содержимое одного сообщения: текст, язык, вложение и notes. Для аналога сообщения PyRIT с ролью используется вместе с `Turn`. [^attempt] | ≈ `Message` |
| `Turn` | Роль `system`/`user`/`assistant` плюс `content: Message`. Вместе с `Message` дает сообщение с ролью; это одна реплика, а не пара «запрос + ответ». [^attempt] | ≈ `Message` |
| `Conversation` | Объект с `turns` и `notes`, представляющий одну историю. `Attempt.conversations` хранит истории разных генераций; отдельного PyRIT-подобного `conversation_id` у этого объекта нет. [^attempt] | ≈ `Conversation` |
| Подготовка и исполнение в `Probe` | Совместная работа создания attempts, перевода, buff hooks, вызовов generator и записи в report. Функции PyRIT normalizer распределены между методами probe и службами garak. [^probe] | ≈ `PromptNormalizer` |
| `Generator._conversation_to_list()` / API-specific formatting | Преобразование conversation в список ролей и содержимого либо иной формат конкретного API. Реализация зависит от generator. [^generator] | ≈ `MessageNormalizer` |
| `Buff` | Расширяемый преобразователь attempts; `transform()` может выдать несколько производных attempts. Шире converter одного значения и не возвращает PyRIT-подобный `ConverterResult`. [^buff] | ≈ `Converter` |
| `LangProvider` / `langservice` | Механизм локализации prompts и обратного перевода ответов для detectors. Перевод применяется отдельно от системы buffs. [^language] | ≈ `Converter` |
| `Probe._buff_hook()` | Применение выбранных buffs до вызова target, с возможностью сохранять исходные prompts и ограничивать число вариантов. Не следует считать несколько buffs обязательным последовательным pipeline. [^probe] | ≈ `request_converters` |
| `Buff.untransform()` / `post_buff_hook` | Обратное преобразование после ответа target, если buff его требует. Например, обратный перевод ответа после перевода запроса. [^buff] | ≈ `response_converters` |
| `reverse_langprovider` / `reverse_translation_outputs` | Обратный перевод ответов для оценивания на языке detector; переводные outputs хранятся отдельно. [^probe] [^language] | ≈ `response_converters` |
| `Detector` | Компонент, выдающий scores по attempt. Для аналога boolean scorer с настраиваемым порогом используется вместе с evaluator; числовые оценки может давать самостоятельно. [^scoring] | ≈ `Scorer` |
| `Evaluator` | Пороговая и агрегирующая часть оценивания: интерпретирует scores detector и формирует отчет. В составном аналоге scorer отвечает за решение по числовой оценке; также выполняет работу уровня отчетности. [^scoring] | ≈ `Scorer` |
| `Attempt.detector_results` | Словарь «имя detector → список оценок». Для структурированного аналога `Score` дополнительно нужны соответствующий output и metadata detector; rationale не гарантирован. [^attempt] [^scoring] | ≈ `Score` |
| `ThresholdEvaluator` | Дает PASS при `score < threshold`, FAIL при `score >= threshold`. Вместе с detector превращает числовую оценку в boolean-решение; полярность решения — со стороны target. [^scoring] | ≈ `true_false` |
| `ZeroToleranceEvaluator` | Дает PASS только при нулевом detector score; любое ненулевое обнаружение считается FAIL. [^scoring] | ≈ `true_false` |
| Числовой результат `Detector.detect()` | Оценка в диапазоне `0.0`–`1.0` либо `None`. Совпадение диапазона с PyRIT не означает совпадения смысла: score может быть бинарным hit, уверенностью классификатора или другой мерой. [^scoring] | ≈ `float_scale` |
| `primary_detector` | Основной detector, рекомендуемый probe для его failure mode. Приближает objective scorer, только если действительно проверяет достижение нужной цели. [^probe] [^harness] | ≈ `objective_scorer` |
| `intentservice.get_detectors()` | Выбор detectors по intent с поиском ближайшего определенного соответствия в иерархии. Это механизм привязки критериев, а не scorer сам по себе. [^cas] | ≈ `objective_scorer` |
| `judge.Jailbreak` / `judge.ModelAsJudge` | LLM-based detectors для проверки jailbreak или заданного goal. Конкретная цель берется из настроек и полей, предусмотренных данным detector. [^judge] | ≈ `objective_scorer` |
| `mitigation.MitigationBypass` / `mitigation.ModernBERTRefusal` / `judge.Refusal` | Детектирование non-refusal или обхода mitigation. Если в PyRIT `true` означает отказ, нужна инверсия смысла; отсутствие отказа не доказывает достижение objective. [^judge] | ≈ `refusal_scorer` |
| `extended_detectors` | Дополнительные detectors к primary detector. Их результаты — дополнительные измерения, которые не обязательно образуют одну итоговую метрику успеха. [^probe] [^harness] | ≈ `auxiliary_scorers` |
| `goal` / `goal_str` / `custom_goals` | Формулировки целей в metadata и настройках probes. Для PyRIT seed потребовалась бы упаковка: здесь это строки или списки, а не типизированные `SeedObjective`. [^probe] [^adaptive] | ≈ `SeedObjective` |
| `Attempt.targets` / `notes["triggers"]` | Искомые строки или маркеры в ответе target, используемые соответствующими detectors. Могут задавать проверяемую часть цели, но не заменяют ее полное описание; `targets` здесь не означает тестируемые системы. [^attempt] [^scoring] | ≈ `SeedObjective` |
| `Stub` | Исходный контент с привязкой к intent. Может выражать цель как запрос или давать материал для prompt; не разделяет эти роли на обязательные типы PyRIT. [^cas] | ≈ `SeedObjective` + `SeedPrompt` |
| `TextStub` | Текстовый stub, который probe расширяет или вставляет в атакующий шаблон. [^cas] | ≈ `SeedPrompt` |
| `ConversationStub` | Stub, содержащий conversation для дальнейшего построения проверки. [^cas] | ≈ `SeedPrompt` |
| `Probe.prompts` | Готовые исходные prompts: строки, messages или conversations, из которых создаются attempts. [^probe] | ≈ `SeedPrompt` |
| Конкретизированный `intent` + связанные `stubs` | Возможная сборка спецификации «что атаковать». Для аналога `AttackSeedGroup` нужно выделить ровно одну цель и связанные материалы; код intent сам по себе этого инварианта не обеспечивает. [^cas] | ≈ `AttackSeedGroup` |
| Шаблоны в `_prompts_from_stub()` | Переиспользуемая часть приема до подстановки контента stub. Например, roleplay-шаблон и варианты персонажей в `grandma.GrandmaIntent`; исполняемый метод целиком не является seed group. [^cas] | ≈ `AttackTechniqueSeedGroup` |
| `PayloadGroup` | Коллекция строк payloads с metadata: имя, типы, optional intent и detector settings. Может служить источником dataset; не гарантирует структуру `SeedGroup` или «ровно один objective». [^payload] | ≈ `SeedDataset` |
| `garak.intents.base.Intent` | Базовый Python-класс поставщика stubs через `stubs()`. Участвует в подготовке набора исходных данных; отличается от доменного intent и не является объектом с полной спецификацией цели. [^cas] | ≈ `SeedDataset` |
| Набор intent stubs / `get_intent_stubs()` | Контент для выбранного intent, собираемый из типологии, файлов и Python-поставщиков. Вместе с выбором intents образует источник данных для кампании. [^cas] | ≈ `SeedDataset` |
| `tags` | MISP-format metadata компонентов: риск, качество, техника и другие свойства. Только соответствующее подмножество risk tags сопоставляется категориям вреда. [^taxonomy] | ≈ `HarmCategory` |
| `Trait typology` / CTMS | Иерархия поведения: Chat, Tasks, Meta, Safety. Для harm mapping используется релевантная часть типологии, а не все intents; `demon:*` относится к отдельной таксономии techniques. [^cas] [^taxonomy] | ≈ `HarmCategory` |
| `.report.jsonl` | Персистентный журнал одного run: setup, attempts, scores, aggregates и прочие записи. Покрывает часть задач хранения результатов, но не является общей БД conversations, datasets и components. [^report] | ≈ `Memory` / `CentralMemory` |
| Plugin classpath + configuration | Имя вида `probes.module.Class` и настройки идентифицируют реализацию и ее параметры. Это только материал для аналога `ComponentIdentifier`: content-addressed hash и дерево child identifiers не обеспечиваются. [^plugins] [^config] | ≈ `ComponentIdentifier` |
| `garak._plugins` / `load_plugin()` | Обнаружение, описание и загрузка компонентов. Общий plugin discovery охватывает probes, detectors, generators, harnesses и buffs. [^plugins] | ≈ `Registry` |
| `payloads.Director` | Отдельный каталог payload groups: индексирование, поиск по типам и загрузка по имени. Не универсальный registry всех datasets. [^payload] | ≈ `Registry` |
| `PluginCache` / `enumerate_plugins()` / `plugin_info()` | Каталог metadata классов plugins, перечисление доступных компонентов и получение их описаний. [^plugins] | ≈ `Class Registry` |
| `PluginProvider` | Registry/cache уже созданных instances. Ключ — класс и строковое представление `config_root`, а не content hash конфигурации. [^plugins] | ≈ `Instance Registry` |
| Startup service initialization | Инициализация и проверка `langservice` и `intentservice` при создании harness. Частично соответствует startup-роли initializer, но не общему механизму регистрации компонентов PyRIT. [^harness] | ≈ `Initializer` |
| `garak.core.yaml` / `garak.site.yaml` / `--config` | Встроенная, пользовательская и run-конфигурации. Основные секции: `system`, `run`, `plugins`, `reporting`; CLI может переопределять значения. [^config] | ≈ `.pyrit_conf` |
| `garak` / `python -m garak` | CLI запуска оценки target. `--spec` выбирает probes, buffs и intent scope; detectors задаются отдельно или выбираются по рекомендациям и intents. [^config] | ≈ `pyrit_scan` |
| `--interactive` / `GarakTerminal` | Интерактивный режим probing на базе `cmd2`; ближайший функциональный аналог shell PyRIT. [^config] | ≈ `pyrit_shell` |
| `ASR` в evaluator | Метрика по оцененным outputs для пары probe/detector. Для PyRIT-подобной доли успешных атак отдельно определяются граница атаки, учет генераций, detectors и пропусков. [^scoring] | ≈ `ASR` |
| Attack generation в `atkgen.Tox` / `tap.TAP` | Генерация новых атакующих prompts внутри probe с помощью вспомогательной модели. Это встроенная ответственность, а не отдельное семейство executors; обычный garak `Generator` этой роли не подразумевает. [^adaptive] | ≈ `Prompt Generator` |
| Buff-based augmentation / `paraphrase.PegasusT5` | Порождение вариантов исходных attempts, например нескольких перефразировок prompt. Составной механизм из buff и его модели/правил преобразования. [^buff] | ≈ `Fuzzer` |
| `suffix.GCG` | Генерация adversarial suffix через Greedy Coordinate Gradient; требует совместимого generator и доступа, нужного алгоритму. [^gcg] | `GCG` |
| `suffix.GCGCached` | Применение заранее вычисленных GCG suffixes. Использует результат техники, но не запускает gradient-based поиск во время scan. [^gcg] | ≈ `GCG` |
| `ATTEMPT_NEW` / `ATTEMPT_STARTED` / `ATTEMPT_COMPLETE` | Состояния жизненного цикла записи attempt. Завершенность ничего не говорит об успехе атаки; это не `AttackOutcome`. [^attempt] | Нет соответствия в словаре |
| `generations` / `generations_this_call` | Число ответов, запрашиваемых для одного входа. Создает несколько историй/выборок, а не несколько последовательных ходов. [^generator] [^attempt] | Нет соответствия в словаре |
| `run.seed` / `--seed` | Seed генератора псевдослучайных чисел для воспроизводимости. Не исходный материал атаки и не PyRIT `Seed`. [^config] | Нет соответствия в словаре |
| `run_id` / `Attempt.uuid` | UUID конкретного запуска или записи attempt. Не `EvaluationIdentifier`, `AtomicAttackIdentifier` или content-addressed идентификатор конфигурации. [^attempt] [^report] | Нет соответствия в словаре |
| `Tier` | Приоритет/значимость probe: `OF_CONCERN`, `COMPETE_WITH_SOTA`, `INFORMATIONAL`, `UNLISTED`. Не категория вреда и не severity отдельного ответа. [^taxonomy] | Нет соответствия в словаре |
| `Calibration` / `Z-score` / `DEFCON` | Сравнение pass rate с распределением результатов набора моделей и преобразование в относительные/абсолютные grades. Не detector score и не ASR. [^calibration] | Нет соответствия в словаре |
| `DetectorMetrics` | Метрики качества самого detector на размеченных данных, в том числе sensitivity/specificity. Используются, например, при расчете доверительного интервала ASR; не оценка target. [^calibration] | Нет соответствия в словаре |
| `cas.Policy` | Экспериментальное представление разрешенности поведения по intent typology. Это policy поведения target, а не PyRIT `CapabilityHandlingPolicy` с выбором `ADAPT`/`RAISE`; API в этой версии незавершен. [^cas] | Нет соответствия в словаре |

</details>

## Важные границы соответствий

1. **`Generator` ≈ `PromptTarget`, а не `Prompt Generator`.** Только generator, используемый как attacker model внутри соответствующего probe, участвует в генерации атакующих prompts.
2. **Один garak `Turn` — одна реплика.** PyRIT `Attack turn` из исходного словаря приблизительно соответствует паре garak turns: user + assistant. Один `Attempt` может включать несколько генераций или представлять лишь шаг многоходовой атаки.
3. **`Detector` + `Evaluator` дают составной аналог оценивания.** Высокий hit score обычно означает FAIL target. `None` не равно отказу, неуспеху атаки или ошибке; оно означает отсутствие пригодной оценки. Non-refusal также не равен достижению objective.
4. **`Scenario` собирается из конфигурации и компонентов.** Практический аналог: выбранные probes/techniques + intents/goals + stubs/payloads + target + harness + detectors/evaluator. Ни один только probe, intent или selector не равен всей кампании.
5. **Метрики требуют одинаковой единицы учета.** ASR по outputs, по завершенным многоходовым атакам и по объединенным detector judgments — разные величины. В report digest `score` может означать pass rate, то есть иметь противоположное направление detector hit score.
6. **Внешний канал XPIA нужно моделировать отдельно.** `LatentInjectionMixin` строит контекст с инъекцией, но сам по себе не выполняет полный workflow «записать payload во внешнюю систему → дождаться чтения target → проверить эффект».

## Источники

Ссылки ведут в исследованную локальную копию garak; трактовки PyRIT взяты из [исходного словаря](pyrit_terms.md).

[^core]: [README](garak/README.md), [How garak runs](garak/docs/source/how.rst).
[^generator]: [`Generator`, генерации и формат conversation](garak/garak/generators/base.py), [генераторы и target API](garak/docs/source/extending.generator.rst).
[^probe]: [`Probe`, `TreeSearchProbe`, `IterativeProbe`, `IntentProbe`](garak/garak/probes/base.py).
[^attempt]: [`Message`, `Turn`, `Conversation`, `Attempt`](garak/garak/attempt.py).
[^cas]: [Context-Aware Scanning](garak/docs/source/cas.rst), [`Intent` и stubs](garak/garak/intents/base.py), [`intentservice`](garak/garak/services/intentservice.py), [trait typology](garak/garak/data/cas/trait_typology.json), [intent → detectors](garak/garak/data/cas/intent_detectors.json), [`GrandmaIntent`](garak/garak/probes/grandma.py), [`Policy`](garak/garak/cas.py).
[^payload]: [Payloads](garak/docs/source/payloads.rst), [`PayloadGroup` и `Director`](garak/garak/payloads.py), [пример `keyedprod_win10`](garak/garak/data/payloads/keyedprod_win10.json).
[^taxonomy]: [Tags and taxonomies](garak/docs/source/tags_taxonomies.rst), [MISP tags](garak/garak/data/tags.misp.tsv), [payload typology](garak/garak/data/typology_payloads.tsv), [`Tier`](garak/garak/probes/_tier.py).
[^adaptive]: [`atkgen.Tox`](garak/garak/probes/atkgen.py), [`tap.TAP` и `tap.PAIR`](garak/garak/probes/tap.py), [`goat.GOATAttack`](garak/garak/probes/goat.py).
[^injection]: [`promptinject`](garak/garak/probes/promptinject.py), [`LatentInjectionMixin` и конкретные probes](garak/garak/probes/latentinjection.py).
[^buff]: [`Buff`](garak/garak/buffs/base.py), [encoding buffs](garak/garak/buffs/encoding.py), [translation buff](garak/garak/buffs/low_resource_languages.py), [paraphrase buffs](garak/garak/buffs/paraphrase.py).
[^language]: [Translation](garak/docs/source/translation.rst), [`LangProvider`](garak/garak/langproviders/base.py), [`langservice`](garak/garak/services/langservice.py).
[^scoring]: [`Detector`](garak/garak/detectors/base.py), [`Evaluator`, thresholds, PASS/FAIL, ASR](garak/garak/evaluators/base.py).
[^judge]: [Mitigation/refusal detectors](garak/garak/detectors/mitigation.py), [LLM judges и интерпретация refusal](garak/garak/detectors/judge.py).
[^harness]: [`Harness` и modality checking](garak/garak/harnesses/base.py), [`ProbewiseHarness`](garak/garak/harnesses/probewise.py), [`PxD`](garak/garak/harnesses/pxd.py).
[^report]: [Открытие run report](garak/garak/command.py), [запись eval и hitlog](garak/garak/evaluators/base.py), [report digest и technique–intent matrix](garak/garak/analyze/report_digest.py), [`Report`](garak/garak/report.py).
[^plugins]: [Plugin discovery, `PluginCache`, `PluginProvider`](garak/garak/_plugins.py).
[^config]: [Конфигурация](garak/docs/source/configurable.rst), [`_config`](garak/garak/_config.py), [core defaults](garak/garak/resources/garak.core.yaml), [`Configurable`](garak/garak/configurable.py), [selection spec](garak/garak/_spec.py), [CLI](garak/garak/cli.py), [interactive shell](garak/garak/interactive.py).
[^calibration]: [`Calibration`](garak/garak/analyze/calibration.py), [DEFCON grades](garak/garak/analyze/__init__.py), [`DetectorMetrics`](garak/garak/analyze/detector_metrics.py), [документация detector metrics](garak/docs/source/detector_metrics.rst).
[^gcg]: [`suffix.GCG` и `suffix.GCGCached`](garak/garak/probes/suffix.py).
