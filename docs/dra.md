# DRA: маскировка и восстановление базового запроса

`attack.type = "dra"` принимает один явный базовый запрос. Закреплённый garak
собирает варианты из четырёх штатных шаблонов, карты букв и cache-only moderation:
отсутствующий в кеше токен считается безвредным. Bundled-корпус запросов и
Detoxify не загружаются. Генерация использует отдельный RNG на запуск, не меняя
общесистемный `random`; каждому варианту соответствует новый разговор Target.

```toml
[attack]
type = "dra"
max_variants = 4
seed = 0
toxic_trunc = 0.5
benign_trunc = 0.5
benign_full_token_prob = 0.33
```

`max_variants` — от 1 до 4; если он меньше четырёх, garak сначала генерирует
все варианты и выбирает заданное число через локальный RNG. `seed` —
неотрицательное целое. `toxic_trunc` и `benign_trunc` — вероятности в `[0, 1)`,
`benign_full_token_prob` — в `[0, 1]`. Иные поля атаки отклоняются до запуска.
Профиль по умолчанию соответствует примеру выше.

Каждый полный ответ Target предъявляется как результат атаки и оценивается с
исходным базовым запросом; при первом успехе запуск завершается. Журнал содержит
вариант, ответ, идентификатор предъявленного результата и оценку. Остановка,
сохранение и повторный запуск используют общие операции User. Attacker не нужен.

```python
from tai import DRAExperiment, Message

user = DRAExperiment(target, Message("Ваш базовый запрос"), evaluator, max_variants=2, seed=42)
user.start()
user.wait()
result = user.result()
```

Для CLI: `tai preflight --config examples/dra-qwen.toml`, `tai attack --config
examples/dra-qwen.toml --text 'Ваш базовый запрос' --artifacts-dir runs/dra`.
Для ALERT: `tai alert-series --manifest examples/alert-manifest.json --taxonomy
ALERT_taxonomy.md --profile examples/dra-qwen.toml --artifacts-dir runs/series
--jobs 1`. Эффективные параметры находятся в `effective-profile.json` под `dra`.
Модельный оценщик требуется для ALERT; локальному Qwen нужны `--extra qwen` и
подготовленные веса.
