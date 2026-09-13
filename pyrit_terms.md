# Термины PyRIT

Этот документ фиксирует рабочий словарь PyRIT для red teaming и оценки LLM-систем. Термины разделены на два уровня: сначала ключевая доменная модель, затем сокращенный справочник по основным PyRIT-терминам.

## Ключевые доменные термины

Это термины, которые полезны и вне конкретной реализации PyRIT.

| Термин | Краткое значение | Примеры |
|---|---|---|
| `Target` | Система, которую тестируют или атакуют: модель, chatbot, endpoint, приложение. | `OpenAIChatTarget`, корпоративный support bot |
| `Objective` | Цель атаки: какого поведения атакующий хочет добиться от target. | раскрыть system prompt, выдать phishing-инструкции |
| `Seed` | Исходный материал для атаки или теста: prompt, objective, шаблон, пример диалога. | `SeedPrompt`, `SeedObjective` |
| `Technique` | Прием реализации атаки: roleplay, translation, encoding, obfuscation, many-shot priming. | roleplay as movie script, Base64 encoding |
| `Strategy` | Алгоритм проведения атаки: как выбирать ходы, когда остановиться, как использовать feedback. | `CrescendoAttack`, `TAPAttack` |
| `Scenario` | Кампания проверки конкретного риска или класса рисков на наборе objectives и techniques. | `Jailbreak`, `SystemPromptExtraction` |
| `Attack` | Попытка добиться objective против target с помощью выбранной strategy/technique. В текущих PyRIT docs это также имя семейства executor-алгоритмов. | `PromptSendingAttack`, прогон roleplay против support bot |
| `AI red teaming` | Проактивная проверка AI-систем через попытки вызвать нежелательное или небезопасное поведение. | jailbreak campaign, phishing-resistance test |
| `System under test` | Тестируемая AI-система; в PyRIT обычно совпадает с `objective_target`. | Qwen chatbot, RAG assistant |
| `Objective target` | Основной target, на котором проверяется достижение objective. | tested chat model, customer-support endpoint |
| `Adversarial target` | Вспомогательная модель/агент, который генерирует атакующие prompts или следующие ходы. | attacker LLM in `RedTeamingAttack`, prompt generator model |
| `Prompt` | Сообщение или входной контент, который передается target. | user text request, image plus caption |
| `Jailbreak` | Цель, результат или класс сценариев, где атакующий пытается обойти safety-поведение модели. Не путать с конкретной техникой вроде roleplay. | successful policy bypass, `Jailbreak` scenario |
| `Prompt injection` | Класс атак, где инструкции внедряются в prompt/context, чтобы изменить поведение AI-системы. Конкретная техника может быть, например, roleplay, encoding или indirect injection через внешний контент. | hidden instruction in document, malicious web page content |
| `XPIA` | Cross-prompt injection: инъекция через внешний канал, который позже считывается AI-системой. | payload in email, payload in SharePoint document |
| `Attack turn` | Один ход атаки: запрос к target и ответ target. | one user prompt plus assistant reply, one audio interruption turn |
| `Single-turn attack` | Атака с одним основным запросом к objective target. | `PromptSendingAttack`, `SkeletonKeyAttack` |
| `Multi-turn attack` | Атака с несколькими ходами, часто адаптирующаяся по ответам target. | `CrescendoAttack`, `PAIRAttack` |
| `Conversation` | Последовательность сообщений между атакой и target. | chat transcript, `conversation_id` history |
| `Prepended conversation` | Диалоговая история, подставляемая перед live-ходами атаки. | system prompt plus few-shot examples, priming dialogue |
| `Simulated conversation` | Сгенерированная заранее или динамически история диалога для подготовки атаки. | synthetic assistant-user warmup, generated roleplay context |
| `Next message` | Явно заданное следующее сообщение атаки вместо автоматического вывода из objective. | prebuilt multimodal prompt, specific follow-up question |
| `Prompt transformation` | Преобразование prompt перед отправкой: перевод, кодирование, обфускация, смена modality. | translate to Spanish, render text as QR code |
| `Scoring` | Оценка ответа или conversation на успех атаки, отказ, вредность, качество или другие критерии. | refusal detection, harmfulness grading |
| `Score` | Конкретная оценка: boolean или шкала, rationale, категория и связанный evidence. | `true_false=true`, `float_scale=0.8` |
| `Refusal` | Ответ target, в котором система отказывается выполнять опасный или запрещенный запрос. | “I can’t help with that”, policy refusal response |
| `Harm category` | Категория риска или вреда: violence, hate, cyber, scam, self-harm и т.п. | cyber, violence |
| `Dataset` | Набор objectives, prompts или seed groups для систематического тестирования. | HarmBench, AIRT datasets |
| `Benchmark` | Оценка target на фиксированном dataset и критериях. | adversarial benchmark, Q&A accuracy benchmark |
| `ASR` | Attack Success Rate: доля атак, достигших objective. | 12 successful attacks of 100, ASR 0.12 |
| `Workflow` | Многошаговый процесс оценки или атаки, не обязательно являющийся классической атакой. | XPIA workflow, plant-and-read injection flow |
| `Modality` | Тип входа или выхода: text, image, audio, video, file. | text input, audio output |
| `Capability` | Возможность target: multi-turn, system prompt, editable history, JSON output, streaming audio. | `supports_multi_turn`, `STREAMING_AUDIO` |
| `Attack outcome` | Итог атаки: success, failure, error или undetermined. | `success`, `undetermined` |
| `Evidence` | Данные, на которых основан вывод об успехе или неуспехе атаки: ответы, scores, transcript. | `last_response`, scored conversation transcript |

<details>
<summary>Все термины</summary>

| Термин | Что означает |
|---|---|
| `Executor` | Алгоритм взаимодействия с `objective_target`. Не каждый executor является атакой. Есть attacks, workflows, benchmarks, prompt generators. |
| `AttackStrategy` / `Attack` | Конкретный алгоритм атаки: `PromptSendingAttack`, `CrescendoAttack`, `PAIRAttack`, `TAPAttack`, `SkeletonKeyAttack`. |
| `AttackContext` | Контекст выполнения атаки: `objective`, `memory_labels`, `prepended_conversation`, `next_message`, состояние исполнения. |
| `AttackParameters` | Входные параметры атаки, из которых создается context. |
| `AttackResult` | Результат выполнения атаки: `conversation_id`, `objective`, `last_response`, `last_score`, `executed_turns`, `outcome`. |
| `AttackOutcome` | Статус результата: `success`, `failure`, `error`, `undetermined`. |
| `AtomicAttack` | Атомарный тест внутри `Scenario`: связывает `AttackTechnique` с `AttackSeedGroup`/objectives. |
| `ScenarioResult` | Агрегированный результат сценария. |
| `PromptTarget` | Универсальная точка, куда PyRIT отправляет prompts: LLM endpoint, HTTP endpoint, storage target, realtime target. |
| `objective_target` | Основная тестируемая система, которую атакуем. |
| `adversarial_target` / `adversarial_chat` | Вспомогательная модель, которая генерирует атакующие prompts для multi-turn/adaptive атак. |
| `TargetCapabilities` | Декларация возможностей target: multi-turn, editable history, system prompt, JSON output, modalities. |
| `TargetConfiguration` | Обертка вокруг capabilities, policy и normalization pipeline. |
| `TargetRequirements` | Требования компонента к target. Например, атака может требовать `supports_editable_history=True`. |
| `CapabilityHandlingPolicy` | Что делать, если target не поддерживает capability: адаптировать (`ADAPT`) или падать (`RAISE`). |
| `ConversationNormalizationPipeline` | Набор normalizer’ов, которые приводят conversation к формату, поддерживаемому target. |
| `MessagePiece` | Атомарная единица сообщения, которая хранится в memory. Может быть текстом, image path, audio path и т.д. |
| `Message` | Один запрос или ответ, состоящий из одного или нескольких `MessagePiece`. |
| `Conversation` | Последовательность `Message` с общим `conversation_id` для одного target. |
| `PromptNormalizer` | Подготавливает prompts, применяет converters и пишет данные в memory. |
| `MessageNormalizer` | Адаптирует `Message`/conversation к wire-format конкретного chat API. |
| `Converter` | Преобразователь prompt’а: encoding, translation, obfuscation, text-to-image, audio conversion и т.д. |
| `ConverterResult` | Нормализованный результат converter’а: значение плюс тип данных. |
| `request_converters` | Converters, применяемые перед отправкой в target. |
| `response_converters` | Converters, применяемые к ответу target. |
| `Scorer` | Компонент оценки ответа, prompt’а или conversation. |
| `Score` | Структурированная оценка: значение, тип, rationale, category, scorer identifier. |
| `Scorable` | Абстракция “то, что можно оценивать scorer’ом”. |
| `true_false` | Boolean score type, удобен для успеха атаки, refusal, policy checks. |
| `float_scale` | Числовой score type от `0.0` до `1.0`, удобен для severity/degree. |
| `objective_scorer` | Scorer, который определяет, достигнута ли цель атаки. |
| `refusal_scorer` | Scorer для определения отказа/non-compliance. |
| `auxiliary_scorers` | Дополнительные scorers для побочных метрик. |
| `SeedObjective` | Цель атаки как seed: что атакующий хочет получить. |
| `SeedPrompt` | Prompt/content, который реально отправляется или используется как шаблон. |
| `SeedGroup` | Группа связанных seeds. |
| `AttackSeedGroup` | Seed group, которая обязана содержать ровно один `SeedObjective`; полная спецификация “что атаковать”. |
| `AttackTechniqueSeedGroup` | Reusable technique seeds без objective; спецификация “как атаковать”. |
| `SeedDataset` | Коллекция `SeedGroup` для кампаний и benchmark’ов. |
| `HarmCategory` | Таксономия вреда/риска: hate, violence, cyber, scam и т.д. |
| `Memory` / `CentralMemory` | Персистентный слой PyRIT для conversations, scores, results, datasets. |
| `memory_labels` | Метки, которые позволяют потом фильтровать конкретный run/campaign. |
| `ComponentIdentifier` | Content-addressed идентификатор конфигурации компонента: class, params, children, hash. |
| `AtomicAttackIdentifier` | Composite identifier для atomic attack: attack technique плюс seed identifiers. |
| `EvaluationIdentifier` / `eval hash` | Хэш логической evaluation-конфигурации, очищенный от части operational params. |
| `Registry` | Механизм обнаружения и доступа к компонентам. |
| `Class Registry` | Registry классов, которые пользователь потом инстанцирует. |
| `Instance Registry` | Registry уже сконфигурированных instances. |
| `Initializer` | Компонент startup-конфигурации: регистрирует targets, scorers, techniques, datasets. |
| `.pyrit_conf` | YAML-конфиг PyRIT: memory backend, initializers, env sources. |
| `pyrit_scan` | CLI для automated scenario execution. |
| `pyrit_shell` | Интерактивная shell/REPL-среда для сценариев. |
| `CoPyRIT` | GUI PyRIT для interactive red teaming. |
| `Workflow` | Multi-step orchestration, не обязательно атака. Частый пример: XPIA. |
| `XPIA` | Cross-prompt injection workflow, где payload попадает в систему через внешний/side channel. |
| `Benchmark` | Evaluation по фиксированному dataset и criteria. |
| `ASR` | Attack Success Rate. Метрика успешности атак. |
| `Prompt Generator` | Executor, который генерирует attack prompts. |
| `Fuzzer` | Механизм мутации/генерации prompts. |
| `GCG` | Greedy Coordinate Gradient, техника генерации adversarial suffix. |
| `BargeInAttack` | Streaming/audio attack, где новая речь пользователя прерывает ответ ассистента. |
| `RealtimeTarget` | Target для realtime audio interaction. |
| `server VAD` | Server-side voice activity detection. |
| `STREAMING_AUDIO` | Capability, требуемая для streaming audio атак. |

</details>
