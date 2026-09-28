# AutoDAN: локальный GA-поиск для явного базового запроса

`attack.type = "autodan"` запускает ограниченный генетический поиск префикса.
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
Параметры `num_steps`, `batch_size`, `num_elites`, `crossover_rate`, `num_points`,
`mutation_rate`, `seed` задаются в `[attack]`; вложенный `[run]` допускает только
`generations = 1`. Иерархический GA и black-box fallback не поддерживаются.
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

Python API: `AutoDANUser(target, attacker, base, evaluator, expected="...")` и
`AutoDANExperiment(target, attacker, base, evaluator, expected="...", journal_dir=...)`.
`target` должен предоставлять независимую операцию `score_continuation(prompt,
expected) -> float`; `QwenChat` реализует её на тех же весах, что и разговор.
