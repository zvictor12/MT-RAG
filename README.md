# mt-rag

Локальная инфраструктура:

```text
Python pipeline -> Elasticsearch 9.4.3 (Docker)
                -> ELSER sparse retrieval (CPU, trial)
                -> BGE-M3 dense + sparse indices (без trial)
Python pipeline -> Ollama + Qwen (host, NVIDIA GPU)
```

## 1. Python

```bash
cp .env.example .env
uv sync
```

Проект использует Python 3.12. Базовое окружение не устанавливает Torch и
тяжёлые метрики. Для поиска/реранкинга или полного эксперимента используем:

```bash
make sync-ml
make sync-experiment  # ML + локальные Task B/C метрики
```

ML-extra использует официальный PyTorch CUDA 13.2 wheel. Драйвер может быть
новее bundled CUDA runtime; системный CUDA Toolkit и conda не нужны.

## 2. BGE-M3 и reranker

Модели скачиваются с закреплённых ревизий Hugging Face в
`~/.cache/mtrag/models`:

```bash
make models-bge       # только query encoder для dense + sparse
make models-reranker  # только cross-encoder
make models           # обе модели
```

[BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) и
[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
содержат примерно по 2.27 ГБ
весов, то есть выбранные файлы занимают около 4.6 ГБ на диске без учёта
окружения и download cache. В эксперименте они загружаются в FP16 и строго
по очереди: совместная резидентность BGE/reranker с Ollama не входит в бюджет
RTX 3060 6 ГБ.

Параметры моделей задаются в секции `[models]` файла
`configs/experiment.toml`.

Если длинные запросы дают CUDA OOM, сначала уменьшите BGE batch до `16`, а
reranker batch до `4` (при необходимости до `8`/`2`); `max_length=512`
оставьте неизменным для сопоставимости прогонов. BGE выдаёт за один проход нормализованный
dense-вектор размерности 1024 и sparse-веса по ID токенов. Это в точности тот
формат и tokenizer, которыми построены восстановленные Elasticsearch-индексы.

В основном эксперименте используется официальный PyTorch checkpoint, а не
сторонние `philipchung/bge-m3-onnx` или `aapot/bge-m3-onnx`: ONNX-экспорты не
являются drop-in заменой `FlagEmbedding`, не закреплены здесь на той же ревизии
и требуют отдельной проверки численного и ranking parity. Их имеет смысл
исследовать позже только как CPU fallback.

Если `MODEL_ROOT` переопределён при скачивании, синхронно задайте
`BGE_MODEL_PATH`/`RERANKER_MODEL_PATH` в `.env` и `bge_path`/`reranker_path` в
`configs/experiment.toml`.

## 3. Ollama

На Arch Ollama работает нативно, чтобы GPU не зависел от NVIDIA Container
Toolkit:

```bash
sudo systemctl enable --now ollama
make ollama-pull
```

Модель `qwen3.5:4b-q4_K_M` занимает 3.4 ГБ. Контекст ограничен до 8192,
thinking выключен, генерация детерминирована. Перед загрузкой BGE или reranker
освобождаем VRAM:

```bash
make ollama-unload
```

Текущий Ollama digest закреплён в `configs/experiment.toml`. Если повторный
`ollama pull` обновил тег, preflight покажет оба digest; обновляйте значение
осознанно. Новые ответы получат отдельную fingerprint-ревизию внутри того же
`RUN_DIR`, поэтому старые результаты не перезаписываются.

## 4. Elasticsearch

```bash
make es-up
```

Обычный `es-up` не включает trial. Elasticsearch доступен только локально на
`127.0.0.1:9200`; данные индексов находятся в Docker volume.

## 5. Snapshot-артефакты

Скачанные ZIP храним отдельно от распакованных Elasticsearch repositories:

```text
artifacts/elasticsearch/archives/bge-m3/{dense,sparse}/
```

Распаковываем их так, чтобы `index-*`, `snap-*.dat`, `meta-*.dat` и `indices/`
лежали непосредственно в соответствующих каталогах:

```text
artifacts/elasticsearch/snapshots/
  elser/
  bge-m3/
    dense/{clapnq,cloud,govt,fiqa}/
    sparse/{clapnq,cloud,govt,fiqa}/
```

Имена BGE-архивов соответствуют каталогам, например
`mtrag_bge_m3_dense_clapnq.zip` распаковывается в
`bge-m3/dense/clapnq/`.

Восстановление восьми BGE-индексов не требует платной лицензии:

```bash
make bge-restore
```

ELSER настраиваем только после загрузки `mtrag_elser_snapshot.zip`:

```bash
make elser-setup
```

Команда сначала проверяет snapshot и только затем активирует trial, создаёт
локальный ELSER endpoint и восстанавливает четыре индекса. Поэтому отсутствие
архива не расходует trial.

## 6. Проверка

```bash
make test
```

## 7. Эксперимент

Единственный источник параметров — `configs/experiment.toml`. В нём явно
описаны запросы (`queries`), retrieval-рецепты (`pipelines`), generation jobs
и именованные очереди (`schedules`). До запуска просмотрите построенный DAG:

```bash
make sync-experiment
make experiment-plan
make experiment-preflight
make experiment-run
make experiment-status
make experiment-results
```

При запуске DAG строится один раз и сохраняется в
`runs/<campaign>/plans/<hash>.json`. Scheduler передаёт дочернему процессу
готовую стадию из этого плана: каждый subprocess не перечитывает весь schedule
и не строит граф заново. Пары имени и ревизии артефакта передаются единым
`ArtifactRef`, поэтому data flow между rewrite, retrieval, rerank и generation
виден непосредственно в плане.

По умолчанию запускается `schedules.bge`, состояние и результаты лежат в
`runs/main`. Другую очередь выбираем без изменения Python-кода:

```bash
make experiment-plan EXPERIMENT_SCHEDULE=elser
make experiment-run EXPERIMENT_SCHEDULE=elser
```

Для полностью независимой кампании используйте новый каталог:

```bash
make experiment-plan RUN_DIR=runs/pilot-01
make experiment-run RUN_DIR=runs/pilot-01
make experiment-status RUN_DIR=runs/pilot-01
```

`schedules.bge` сейчас считает явно перечисленные `last`, `qwen_t0`,
`qwen_t02` и `gold` outputs. Dense, sparse, RRF и reranked — независимые
эксперименты: код ничего не объявляет победителем и не включает/выключает
reranker по скрытому gate. После Task A нужный контекст для Task C выбирается
прямой ссылкой вроде `contexts = "bge_t02.rrf_reranked"` в TOML.
Запросы первого turn копируются без изменений; Qwen вызывается только при
наличии истории.

Task B/C не стартуют автоматически вслед за retrieval. После просмотра Task A
запускайте нужную очередь явно:

```bash
make experiment-run EXPERIMENT_SCHEDULE=task_b
make experiment-run EXPERIMENT_SCHEDULE=task_c_bge_last_rrf_reranked
# после восстановления ELSER:
make experiment-run EXPERIMENT_SCHEDULE=task_c_elser_last_reranked
```

Трёхролевой history agent отдельно формулирует уточнения, отвечает на них по
пронумерованным репликам и собирает standalone query. Первый turn остаётся
identity. Его ELSER-вариант сразу включает reranker; Task A и полный Task C
запускаются отдельными очередями:

```bash
make experiment-run EXPERIMENT_SCHEDULE=elser_agentic_reranked
make experiment-run EXPERIMENT_SCHEDULE=task_c_elser_agentic_reranked
```

После восстановления ELSER продолжите тот же campaign directory:

```bash
make experiment-run RUN_DIR=runs/main EXPERIMENT_SCHEDULE=elser
```

Чтобы добавить третью температуру или другой prompt, добавьте новую
`[queries.<name>]`, pipeline и ссылки в schedule. Никакой Python менять не
нужно.

Dense-поиск здесь использует восстановленный Elasticsearch `int8_hnsw` с
`num_candidates = max(100, top_k * dense_candidate_multiplier)`, по умолчанию
500 кандидатов для top-50. Лучшие 100 (`oversample=2`) пересчитываются по
исходным float-векторам после int8-поиска. Это быстрый approximate search, а
не точное воспроизведение paper baseline на FAISS `IndexFlatIP`;
нормализованные BGE-M3 векторы и `dot_product` при этом семантически совместимы.

Прогон возобновляемый. Глобальной блокировки `run-definition.json` больше нет.
Fingerprint применяется локально как адрес конкретного артефакта: он включает
только его prompt/model/temperature и upstream inputs. Поэтому изменение
`qwen_t02` создаёт новую ревизию этого эксперимента, но `bge_last` остаётся
переиспользуемым. Результаты лежат в читаемой структуре
`experiments/<pipeline>/<output>/<fingerprint>/` и
`generation/<job>/<fingerprint>/`.
Поля `retrieval.bge_index_revision` и `retrieval.elser_index_revision` — явные
версии содержимого Elasticsearch; меняйте соответствующее значение после
замены восстановленного индекса.

Scheduler допускает только одну GPU-стадию, но параллелит независимую CPU/ES
работу в пределах `run.cpu_slots`. На границах стадий и model batches действует
thermal guard. Измерение GPU `>=86°C` или CPU `>=90°C` ставит работу на паузу;
продолжение происходит после `<=72°C`/`<=80°C` в течение 30 секунд.
Перегрев не завершает прогон: работа остаётся на паузе до охлаждения.
Если датчик недоступен, в лог пишется предупреждение и защита только для этого
датчика отключается.

Для одного `RUN_DIR` запускайте один scheduler: lock защищает manifest и cache
от двух конкурирующих процессов. Параллельные варианты перечисляйте внутри
одного schedule — scheduler сам распределит CPU/ES и GPU stages.

Task B/C evaluation включается отдельно для каждого job полем
`evaluate = true`. Его можно временно поставить в `false`, не меняя retrieval
артефакты. Внутри одного schedule сначала выполняются все Qwen generation jobs,
затем Ollama выгружается и все jobs оцениваются одной загрузкой DeBERTa.

## 8. Промпты и воспроизводимость

Тексты промптов находятся в `src/mtrag/llm/templates/`. Каждый query и
generator ссылается на свой файл и задаёт температуру непосредственно в
`configs/experiment.toml`.
`think=false` задаётся в `src/mtrag/llm/ollama_client.py`, а воспроизводимый
`seed` — в `configs/experiment.toml`.

Содержимое prompt входит и в SQLite cache key, и в fingerprint артефакта;
ручной номер версии увеличивать не нужно.

## 9. Оценка

Для Task A тонкий адаптер вызывает `prepare_results_dict`, `load_qrels` и
`evaluate` непосредственно из соседнего `mt-rag-benchmark`. Локальной копии
формул nDCG/Recall нет:

```bash
uv run --extra evaluation python scripts/evaluate_retrieval.py \
  --input path/to/retrieval.jsonl \
  --benchmark-root ../mt-rag-benchmark \
  --output path/to/retrieval-report.json
```

Для Task B/C также вызывается официальный `run_algorithmic_judges`: Recall,
RougeL, BERTScore, BertKPrec, extractiveness и RB-agg. BERTScore пары считаются
батчами одной загруженной
[microsoft/deberta-xlarge-mnli](https://huggingface.co/microsoft/deberta-xlarge-mnli),
а не 5894
одиночными `compute`-вызовами на полном Task C:

```bash
make sync-evaluation
uv run --extra evaluation python scripts/evaluate_generation.py \
  --input path/to/generation.jsonl \
  --output path/to/generation-evaluated.jsonl \
  --batch-size 2
```

Рядом автоматически создаётся `generation-evaluated_summary.json` со средними
по задачам; passages сначала усредняются внутри задачи, чтобы диалоги с большим
числом контекстов не получали больший вес.

DeBERTa-XLarge — отдельная evaluation-only модель на 750M параметров; она
скачается при первом таком запуске и не участвует в runtime RAG pipeline.

На 6 ГБ VRAM запускайте эту фазу отдельно от Ollama/BGE/reranker; при OOM
используйте `--batch-size 1`. Локальный Qwen не используется как судья своих же
ответов: LLM judges из официального IBM контура остаются отдельной внешней
проверкой, а основной локальный выбор конфигурации опирается на детерминированные
retrieval и algorithmic generation metrics.
