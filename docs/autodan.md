# AutoDAN: локальный GA/HGA-поиск для явного базового запроса

`attack.type = "autodan"` запускает ограниченный генетический поиск префикса.
`attack.mode = "ga"` (по умолчанию) выбирает sentence-level GA, `"hga"` —
иерархический поиск по словам с momentum-словарём из закреплённого garak.
`attack.expected` — ожидаемое продолжение ответа для logits-fitness, а не
базовый запрос: последний передаётся отдельно в `tai attack --text` / `--text-file`
или берётся из записи ALERT. Target должен использовать `backend = "qwen"` с
локальными весами; `preflight` проверяет также logits-scoring. Attacker владеет
отдельным разговором и выполняет модельные мутации. Оценщик tai может работать
на другом backend.

Порядок закреплённого garak сохраняется: fitness считает вероятность
`expected` после строки `prefix + " " + base` под системными инструкциями Target;
нативная проверка отправляет Target **только prefix**, без base. Проверка
успеха отвергает ответы, содержащие штатные строки отказа garak. Только
проверенный prefix предъявляется как результат атаки, после чего отдельный
оценщик tai получает исходный base и именно этот prefix. Если нативный поиск
ничего не нашёл, итоговый `result.txt` отсутствует, статус `COMPLETED` с
причиной `strategy_exhausted`, а история fitness, выборов, мутаций и проверок
остаётся в `events.jsonl`. Fitness не считается оценкой успеха.

Пример профиля: [examples/autodan-qwen.toml](../examples/autodan-qwen.toml).
Для HGA в примере укажите `mode = "hga"` и удалите `num_points`.
В `[attack]` задаются `mode`, `num_steps` (1–100, по умолчанию 25), `batch_size`
(2–64, по умолчанию 8), `num_elites` (1–63 и меньше `batch_size`, по умолчанию 2),
`crossover_rate` (0–1, по умолчанию 0.5), `mutation_rate` (0–1, по умолчанию 0.1)
и `seed` (0–4294967295, по умолчанию 0). Только GA использует `num_points`
(1–32, по умолчанию 5); HGA отвергает этот параметр. Бюджет fitness — не более
`num_steps × batch_size` вычислений logits, нативных проверок — до `num_steps`,
модельных мутаций — до `(num_steps - 1) × (batch_size - num_elites)`; успешная
проверка останавливает поиск раньше. В `[run]` допускается только `generations = 1`.
Режим, seed и эффективные бюджеты/параметры сохраняются в профиле и журнале.
Black-box fallback не предусмотрен.

Для HGA перед запуском один раз подготовьте лексические данные NLTK:
`uv run --locked --group dev python -m nltk.downloader stopwords wordnet`.
`preflight` отдельно проверяет их наличие (`HGAResources`), не скачивая их при
запуске. HGA сохраняет лучшие префиксы и обновляет top-30 momentum-словарь по
отрицательным fitness-потерям всей популяции; затем заменяет слова с вероятностью
`crossover_rate` с помощью WordNet и применяет мутации через отдельный Attacker.
Словарь и новая популяция пишутся в `autodan_momentum` после каждого перехода.
Одно вычисление fitness не меняет историю Target и не считается вызовом Target;
генерации и оценивания завершаются до остановки User.

```bash
uv run --locked --extra qwen tai preflight --config examples/autodan-qwen.toml
uv run --locked --extra qwen tai attack --config examples/autodan-qwen.toml \
  --text 'Ваш базовый запрос' --artifacts-dir runs/autodan
uv run --locked --extra qwen tai alert-series --manifest examples/alert-manifest.json \
  --profile examples/autodan-qwen.toml --taxonomy ALERT_taxonomy.md \
  --artifacts-dir runs/series --jobs 1
```

Python API: `AutoDANUser(target, attacker, base, evaluator, expected="...", mode="hga")` и
`AutoDANExperiment(target, attacker, base, evaluator, expected="...", mode="hga", journal_dir=...)`.
`target` должен предоставлять независимую операцию `score_continuation(prompt,
expected) -> float`; `QwenChat` реализует её на тех же весах, что и разговор.
