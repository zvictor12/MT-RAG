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
make diagnose
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

Начальные настройки smoke-скриптов рассчитаны на короткие benchmark-запросы:

```dotenv
BGE_BATCH_SIZE=32
BGE_MAX_LENGTH=512
RERANKER_BATCH_SIZE=8
RERANKER_MAX_LENGTH=512
```

Полный DAG берёт те же значения из секции `[models]` файла
`configs/experiment.toml`, а не из `.env`.

Если длинные запросы дают CUDA OOM, сначала уменьшите BGE batch до `16`, а
reranker batch до `4` (при необходимости до `8`/`2`); `max_length=512`
оставьте неизменным для сопоставимости прогонов. Меняйте `.env` для smoke и
`[models]` для эксперимента. BGE выдаёт за один проход нормализованный
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
make ollama-smoke
```

Модель `qwen3.5:4b-q4_K_M` занимает 3.4 ГБ. Контекст ограничен до 8192,
thinking выключен, генерация детерминирована. Перед загрузкой BGE или reranker
освобождаем VRAM:

```bash
make ollama-unload
```

Текущий Ollama digest закреплён в `configs/experiment.toml`. Если повторный
`ollama pull` обновил тег, preflight покажет оба digest; обновляйте значение
осознанно и используйте новый `RUN_DIR`, чтобы не смешать ответы разных весов.

## 4. Elasticsearch

```bash
make es-up
make infra-check
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
make infra-check
make smoke-prompts
make ollama-unload
make smoke-models
make test
# только после make elser-setup:
make smoke-search
```

`infra-check` показывает GPU, версии сервисов, тип лицензии, наличие Ollama
модели и все восстановленные `mtrag-*` индексы. `smoke-models` последовательно
проверяет BGE dense/sparse retrieval и reranker, освобождая модель после каждого
процесса. Ollama перед этой проверкой выгружается явно.

## 7. Эксперимент

Единственный источник параметров — `configs/experiment.toml`. До полного
прогона просмотрите DAG, затем запустите его:

```bash
make sync-experiment
make experiment-plan
make experiment-preflight
make experiment-run
make experiment-status
make experiment-results
```

По умолчанию состояние и результаты лежат в `runs/main`. Для независимого
прогона используйте новый каталог во всех трёх командах:

```bash
make experiment-plan RUN_DIR=runs/pilot-01
make experiment-run RUN_DIR=runs/pilot-01
make experiment-status RUN_DIR=runs/pilot-01
```

Фаза `bge` сравнивает одинаковым контуром варианты `last`, `qwen_t0`
(`temperature=0.0`) и `qwen_t02` (`temperature=0.2`): BGE dense, sparse, RRF
и RRF с reranker. Отдельный `gold` остаётся oracle baseline. Reranker считается
полезным для варианта только при приросте
`nDCG@5 >= 0.01` и paired-bootstrap probability of improvement `>= 0.95`.
Запросы первого turn копируются без изменений; Qwen вызывается только когда
у вопроса уже есть история разговора.
Победитель rewrite выбирается по BGE-dense, а общий BGE-победитель — между
dense, sparse, RRF и прошедшими gate reranked-вариантами. `Task C (last)`
сохраняется отдельно как baseline; ещё один Task C строится только для
выбранного BGE-контура. Task B использует эталонные контексты.

После восстановления ELSER продолжите тот же run расширенной фазой:

```bash
make experiment-run RUN_DIR=runs/main EXPERIMENT_PHASE=full
```

ELSER получает выбранный rewrite-вариант. Если BGE подтвердил пользу reranker,
финальный выбор всё равно сравнивает и базовый, и reranked ELSER с BGE, поэтому
ухудшивший ELSER reranker не становится обязательным.

Dense-поиск здесь использует восстановленный Elasticsearch `int8_hnsw` с
`num_candidates = max(100, top_k * dense_candidate_multiplier)`, по умолчанию
500 кандидатов для top-50. Лучшие 100 (`oversample=2`) пересчитываются по
исходным float-векторам после int8-поиска. Это быстрый approximate search, а
не точное воспроизведение paper baseline на FAISS `IndexFlatIP`;
нормализованные BGE-M3 векторы и `dot_product` при этом семантически совместимы.

Прогон возобновляемый. `manifest.json`, `events.jsonl`, stage-логи, SQLite
cache, дописываемые JSONL checkpoints, predictions, метрики и решения лежат
внутри `RUN_DIR`. Повторный `experiment-run` пропускает успешные стадии и
продолжает прерванные. Первый `Ctrl-C` корректно останавливает дочерние
процессы и сохраняет статус, второй завершает их принудительно.
Параметры эксперимента, включая температуры rewrite-вариантов, и версии обоих промптов фиксируются в
`run-definition.json`; если они изменились, resume просит новый `RUN_DIR`.
Секция `[thermal]` исключена из checksum: её можно менять во время run.

`candidates/*.jsonl` сохраняют текст, title, исходные scores и компоненты RRF
для reranker/Task C. `predictions/task_a/*.jsonl` намеренно компактны: только
обязательные `document_id` и rank-preserving `score`, чтобы уложиться в
официальный лимит 20 МБ.

Scheduler допускает только одну GPU-стадию, но параллелит независимую CPU/ES
работу в пределах `run.cpu_slots`. На границах стадий и model batches действует
thermal guard. Два подряд измерения GPU `>=80°C` или CPU `>=90°C` ставят работу
на паузу; продолжение происходит после `<=72°C`/`<=80°C` в течение 30 секунд.
Верхние значения не завершают прогон: работа остаётся на паузе до охлаждения.
Если датчик недоступен, в лог пишется предупреждение и защита только для этого
датчика отключается.

Task B/C algorithmic evaluation по умолчанию включена
(`generation.run_algorithmic_metrics=true`) и запускается после выгрузки
Qwen. Чтобы сначала только сохранить ответы и отложить тяжёлый BERTScore,
выключите её в TOML до запуска нового `RUN_DIR`.

## 8. Промпты и воспроизводимость

Оба исследовательских промпта находятся в
`src/mtrag/llm/prompts.py`: `_REWRITE_SYSTEM_PROMPT` строит самостоятельный
поисковый запрос, `_GENERATOR_SYSTEM_PROMPT` формирует grounded-ответ по
пассажам. Температуры rewriter заданы в `rewriting.variants`, а generator —
в `generation.temperature` из `configs/experiment.toml` (сейчас `0.1`).
`think=false` задаётся в `src/mtrag/llm/ollama_client.py`, а воспроизводимый
`seed` — в `configs/experiment.toml`.

При изменении промпта увеличьте рядом соответствующий
`REWRITE_PROMPT_VERSION` или `GENERATOR_PROMPT_VERSION`. Версия входит в ключ
SQLite cache, поэтому старые ответы не будут незаметно переиспользованы в новом
прогоне.

## 9. Оценка

Для Task A используется встроенный evaluator: он считает nDCG и Recall на
`1/3/5/10`, включает отсутствующие predictions как нули и умеет выпустить
IBM-совместимый enriched JSONL и aggregate CSV:

```bash
uv run python scripts/evaluate_retrieval.py \
  --input path/to/retrieval.jsonl \
  --benchmark-root ../mt-rag-benchmark \
  --output path/to/retrieval-report.json \
  --official-output path/to/retrieval-evaluated.jsonl
```

Для Task B/C выбран алгоритмический набор метрик из IBM evaluator: Recall,
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
