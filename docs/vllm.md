# Ленивый QwenServer на vLLM

`backend = "vllm"` доступен в `fitd`, `preflight` и `alert-series`.
Профиль: [`examples/fitd-alert-vllm.toml`](../examples/fitd-alert-vllm.toml).
Один `QwenServer` обслуживает независимые `LLMChat` всех ролей и запусков.

## Подготовка отдельно от эксперимента

Linux x86_64, Python 3.12. Закреплены **vLLM 0.6.4.post1**, **PyTorch
2.5.1+cu121**, torchvision 0.20.1+cu121, xformers 0.0.28.post3 и Transformers
4.45.2. Полное отдельное окружение фиксирует
[`environments/vllm/uv.lock`](../environments/vllm/uv.lock). Оно не устанавливается
в основной tai venv и не меняет Transformers из `--extra qwen`.

```bash
uv sync --locked --group dev
uv sync --locked --project environments/vllm
environments/vllm/.venv/bin/huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 \
  --cache-dir "$HOME/.cache/tai/huggingface" \
  --include config.json generation_config.json tokenizer.json tokenizer_config.json \
    vocab.json merges.txt model.safetensors
```

В примере задайте **абсолютный `server.python`** вашего окружения и нужный
`server.device = "cuda:N"`. При ограниченной квоте можно установить окружение
через `UV_PROJECT_ENVIRONMENT=/absolute/path/to/vllm-env uv sync --locked
--project environments/vllm`, затем указать его `bin/python` в профиле.
Все пути кеша расширяют `~`; snapshot должен быть подготовлен заранее.
Веса: SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.

Совместимость проверена на TITAN RTX (compute capability 7.5), драйвере
545.23.08, CUDA runtime 12.1. Используются FP16, XFORMERS и eager execution.
vLLM 0.6.3.post1 был отклонён: его CLI привязывает сокет ко всем интерфейсам
даже при `--host 127.0.0.1`; закреплённая 0.6.4.post1 учитывает host.
`lm-format-enforcer` задаётся явно: старый outlines импортирует отсутствующий
модуль pyairports даже для обычного запроса. Генерация не ограничивается JSON
grammar; строгий boolean проверяет обычный модельный оценщик tai.
Lockfile использует исторический cutoff зависимостей; исключения для CUDA wheels
учитывают даты их загрузки в индекс. pyairports остаётся транзитивной зависимостью
outlines, но этот backend не используется.

## Настройки

`[server]` обязателен для vLLM-ролей: `python`, `model`, закреплённый 40-символьный
`revision`, `cache_dir`, `device`. `dtype` только `float16`.

| Поле | Default | Значение в ALERT-примере |
| --- | ---: | ---: |
| `max_model_len` | 8192 | 8192 |
| `max_num_seqs` | 32 | 32 |
| `gpu_memory_utilization` | 0.3 | 0.5 |
| `startup_timeout` | 300 s | 300 s |
| `shutdown_timeout` | 10 s | 10 s |

У этой версии vLLM профилирование памяти учитывает другие процессы на GPU.
На выбранной занятой GPU значение 0.3 не оставило KV-cache; 0.5 прошло реальный
smoke-test и серию. Лимит памяти задаётся явно и сохраняется в настройках.
`max_num_seqs` ограничивает серверный scheduler, `--jobs` — активных User.
Контекст учитывает полную историю и запрошенный ответ; превышение возвращает
ошибку конкретному запуску, а не обрезает полный Message.

Каждая роль содержит только `backend`, `system`, `generation`, `timeout` (default
120 s, отдельно от старта сервера). Генерация использует ту же схему, что Qwen:
`max_new_tokens`, `do_sample`, `repetition_penalty`; `temperature/top_p/top_k`
разрешены при sampling. В HTTP: `max_tokens`, `temperature=0` при greedy,
`top_p=1`, `top_k=-1` без ограничения, `seed=0`, `n=1`, `stream=false`.
Настройки сохраняют эффективные значения и наблюдённую версию сервера.

## Запуск

```bash
uv run --locked tai preflight --config examples/fitd-alert-vllm.toml \
  --report runs/vllm-preflight.json
uv run --locked tai alert-series --jobs 4 \
  --manifest examples/alert-manifest.json --profile examples/fitd-alert-vllm.toml \
  --taxonomy ALERT_taxonomy.md --artifacts-dir runs/alert-vllm
uv run --locked tai alert-report --index runs/alert-vllm/<series-UUID>/index.json
```

Дочерний процесс получает `CUDA_VISIBLE_DEVICES=N`, `HF_HUB_OFFLINE=1` и
`TRANSFORMERS_OFFLINE=1`; runtime не скачивает веса. Сервер обслуживает loopback;
случайные отдельные API key и served model identity исключают принятие чужого
процесса за свой при гонке за свободный порт. Проверяются процесс, `/v1/models`
и `/version`. Ошибка старта общая для всех ожидающих запросов; повторного старта
сломавшегося ресурса нет. Лог vLLM — отдельный `vllm.log` внутри серии, рядом
с preflight-отчётом либо в каталоге одиночного fitd.

Python API: `QwenServer(ServerConfig(...), log_path)` — контекстный менеджер с
идемпотентным `close()`. `LLMChat.cancellable(partial(server.generate,
generation=...), ...)` создаёт самостоятельный разговор; штатная сборка использует
`QwenServerChat`, добавляющий в диагностические настройки сервер и параметры роли.
`settings()` показывает `not_started/starting/ready/failed/closed` без загрузки.
При закрытии ресурс ждёт допущенные запросы и очищает группу процессов.

Фактические результаты: [приёмка 2026-09-22](acceptance/2026-09-22-alert-vllm.md).
