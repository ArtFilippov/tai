# GOAT: адаптивный разговор с явным базовым запросом

`attack.type = "goat"` использует закреплённые шаблоны и разбор O-T-S-R из garak
для одного явно переданного базового запроса. Корпус HF и модельные плагины garak
не загружаются. Attacker строит следующий запрос с учётом предыдущего ответа;
Target ведёт свой независимый разговор. GOAT judge использует штатную рубрику
garak для решения о продолжении; отдельный оценщик tai оценивает только полный
предъявленный ответ Target относительно исходного базового запроса. В журнале
решение judge записывается как `local_decision` с `decision=goat_judge`, а решение
оценщика — как `evaluation_completed`. Положительный judge при отрицательной
оценке завершает поиск с причиной `judge_stop`, не объявляя успех tai.

```toml
[attack]
type = "goat"

[run]
generations = 1
max_calls_per_conv = 5

[target]
base_url = "http://localhost:8000/v1"
model = "target"

[attacker]
base_url = "http://localhost:8000/v1"
model = "attacker"

[evaluator]
type = "model"
rubric_version = "alert-safety-v1"
[evaluator.chat]
base_url = "http://localhost:8000/v1"
model = "evaluator"
```

Лимит по умолчанию — 5 ходов Target; значение должно быть положительным целым.
GOAT также доступен как `GOATUser(target, attacker, base, evaluator)` и
`GOATExperiment(target, attacker, base, evaluator, journal_dir=...)` из `tai`.
CLI `tai preflight`, `tai attack` и `tai alert-series` используют один и тот же
профиль. Для `tai attack` передайте `--text` или `--text-file`; для ALERT —
`--manifest`, `--profile`, `--taxonomy` и `--artifacts-dir`.
