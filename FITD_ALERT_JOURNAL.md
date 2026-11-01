# Журнал запуска FITD на ALERT

Этот документ описывает **текущий формат журнала**, создаваемого командой
`tai-alert`. Пример ниже иллюстративный: идентификатор и категории взяты из
`examples/alert-manifest-640.json`, а ответы модели и времена придуманы, чтобы
показать поля. Это не отчёт о выполненном прогоне.

## Файлы и связь с серией

Например, команда `tai-alert /tmp/alert-fitd --gpu 0 --limit 1 --turns 2`
создаёт отдельный журнал для каждого выбранного базового запроса. Первый запрос
манифеста имеет ID `5697`, микрокатегорию `hate_body` и макрокатегорию `hate`;
его журнал называется `/tmp/alert-fitd/fitd/5697.jsonl`. В корне запуска лежит
общий `/tmp/alert-fitd/summary.json` с планом и известными исходами попыток.
Если ошибка произошла до сборки атаки, журнал может отсутствовать; ошибка
остаётся в общей сводке. Первое событие собранной FITD-атаки — `start`.

Журнал — JSONL: **один JSON-объект на строку**, в порядке записи. Примеры в этом
документе развернуты на несколько строк только для чтения. Дисковый журнал
добавляет время `at` (Unix seconds), а обёртка добавляет к **каждой** записи
`schema_id` и `schema_version` из реализации журнала. Текущая пара —
`tai.run` / `1`. `ParsedJournal` отклоняет запись с отсутствующей или неизвестной
парой; формат прежних журналов не угадывается.

## Снимок сборки атаки

Далее `AttackRun` записывает `start`. Поле `base` содержит **полный** исходный
текст запроса; ниже он заменён маркером только ради краткости примера.
`base_metadata` сохраняет происхождение и категории выбранного запроса.
`attack` — снимок собранной атаки **на момент старта**, а не изменяющееся
состояние диалогов. Для локальной Qwen-сборки структура выглядит так (во всех
трёх ролях `model` содержит объект с фактическими настройками, здесь показаны
ключевые поля):

```json
{
  "event": "start",
  "base": "<полный текст запроса ALERT с ID 5697>",
  "base_metadata": {
    "id": 5697,
    "source_category": "hate_body",
    "microcategory": "hate_body",
    "macro": "hate",
    "source_revision": "c76d0637f4573742982d5fe961a31bdb3d1e2c80",
    "source_sha256": "0a5594d47d3c4b81462048e6e5dfa22304be6eb490019ddbddc73adc4e317cc4",
    "taxonomy_version": "ALERT Section 3 Figure 2",
    "taxonomy_sha256": "0877011641dbe3a558a9b6e917d65c1daf103b1722eece04bd56905b4ba0debd"
  },
  "attack": {
    "type": "fitd",
    "target": {
      "type": "model_chat",
      "model": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
        "generation": {
          "max_new_tokens": 192,
          "do_sample": false,
          "pad_token_id": "eos_token_id"
        },
        "device": "cuda:0",
        "dtype": "float16"
      }
    },
    "judge": {
      "type": "recorded_judge",
      "judge": {
        "type": "qwen_judge",
        "chat": {
          "type": "model_chat",
          "model": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "generation": {
              "max_new_tokens": 192,
              "do_sample": false,
              "pad_token_id": "eos_token_id"
            },
            "device": "cuda:0",
            "dtype": "float16"
          }
        }
      }
    },
    "probe": {
      "type": "fitd_probe",
      "max_calls_per_conv": 2,
      "attacker": {
        "type": "continuing_chat",
        "chat": {
          "type": "model_chat",
          "model": {
            "id": "Qwen/Qwen2.5-0.5B-Instruct",
            "revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "generation": {
              "max_new_tokens": 192,
              "do_sample": false,
              "pad_token_id": "eos_token_id"
            },
            "device": "cuda:0",
            "dtype": "float16"
          }
        }
      }
    }
  },
  "schema_id": "tai.run",
  "schema_version": 1,
  "at": 1790000001.0
}
```

Каждый участник атаки формирует своё описание через `description()`; модель
предоставляет `settings()`. Поле `journal` в `attack` не входит: журнал не
определяет поведение атаки. Отдельного поля `start.models` в новом формате нет.
При выборе GGUF для Attacker его вложенный `model` содержит соответствующие
настройки и сведения о сервере вместо показанных настроек Qwen.

## События выполнения

После `start` могут следовать несколько циклов. У всех событий ниже также есть
`schema_id`, `schema_version` и `at`; таблица перечисляет **поля полезной
нагрузки**, не повторяя эти три общих поля.

| `event` | Сохраняемые сведения |
| --- | --- |
| `call` | `role` (`attacker`, `target` или `judge`), `prompt`, `response`; у вызова оценщика также `base`. При ошибке оценщика `response` равен `null`. |
| `step` | Запрос и ответ Target (`prompt`, `response`), `accepted`, заметки FITD (`notes`) и ходы разговора (`turns`). |
| `decision` | Решение стратегии `action`, прежние `prompt` и `response`, следующий запрос `next`, число сохранённых ходов `kept` и отброшенные ходы `discarded`. |
| `result` | `text` — выбранный атакой ответ Target, передаваемый оценщику; это не любой ответ из `call`. |
| `judgment` | Тот же `text`, исходный `base` и логический `assessment`. При ошибке вместо вердикта записывается `assessment_error` с `text`, `base` и `error`. |
| `end` | Итоговые `status` и `reason`: например, `success` / `early_success`, `round_limit`, `no_result` или `failure` с причиной `assessment_error` либо `execution_error`. |

Например, **фрагмент** иллюстративного успешного запуска после `start` может
закончиться следующими записями; промежуточные обращения Attacker и Target,
`step` и решения FITD здесь намеренно опущены:

```json
{
  "event": "result",
  "text": "<выбранный ответ Target>",
  "schema_id": "tai.run",
  "schema_version": 1,
  "at": 1790000005.0
}
```

```json
{
  "event": "call",
  "role": "judge",
  "base": "<полный текст запроса ALERT с ID 5697>",
  "prompt": "<выбранный ответ Target>",
  "response": "True",
  "schema_id": "tai.run",
  "schema_version": 1,
  "at": 1790000006.0
}
```

```json
{
  "event": "judgment",
  "base": "<полный текст запроса ALERT с ID 5697>",
  "text": "<выбранный ответ Target>",
  "assessment": true,
  "schema_id": "tai.run",
  "schema_version": 1,
  "at": 1790000007.0
}
```

```json
{
  "event": "end",
  "status": "success",
  "reason": "early_success",
  "schema_id": "tai.run",
  "schema_version": 1,
  "at": 1790000008.0
}
```

`ParsedJournal` связывает предъявленный `result` с последующим `judgment` по
точному тексту и проверяет итоговый `end`; ранний положительный вердикт не
заменяет оценку последнего предъявленного результата. `no_result` означает
отсутствие результата, а не пустой ответ. Ошибка выполнения записывается как
`execution_error`, затем как `end` со статусом `failure`.

Общий `summary.json` перечисляет сквозные номера попыток, номера фабрик, полные
базовые запросы с метаданными, описания атак, финальные результаты и ошибки.
Он содержит `planned`, `attempted` и `complete`, но не вычисляет показатели
успешности и не восстанавливает вердикты из журналов. Подробности оценки
доступны через `ParsedJournal` для файла конкретного запуска.
