# ML pipeline — HW4

Полный CPU-пайплайн классификации табличных данных: логарифмический attention-блок,
модель, Dataset/DataLoader, обучение, оценка и сохранение модели.

## Установка

Требуется Python 3.12 или 3.13 и `uv`. Все команды выполняются из каталога репозитория.

```bash
uv sync --locked
```

Используется CPU-сборка PyTorch; CUDA и GPU не требуются. После установки зависимостей
генерация данных, обучение и тесты работают без сети.

## Запуск пайплайна

```bash
uv run --locked ml-pipeline train --epochs 20 --output artifacts/demo
uv run --locked ml-pipeline evaluate --checkpoint artifacts/demo/model.pt --samples 128 --seed 987
```

Первая команда генерирует бинарную задачу с четырьмя признаками и сохраняет:

- `model.pt`: веса, конфигурацию, параметры нормализации и историю обучения;
- `history.json`: train loss, validation loss и validation accuracy по эпохам;
- `metrics.json`: начальный и конечный validation loss, accuracy и путь к чекпойнту.

`evaluate` загружает модель в новом процессе и оценивает другую выборку. Обе команды
выводят JSON. Ошибки входных данных возвращают код завершения `2`.
Также доступен запуск через `uv run --locked python -m ml_pipeline`.

Для своих данных нужен CSV с числовыми признаками и столбцом `target`, содержащим
целочисленные метки `0` и `1`. Порядок признаков при обучении и оценке должен совпадать.

```bash
uv run --locked ml-pipeline train --csv data/train.csv --target target --output artifacts/csv
uv run --locked ml-pipeline evaluate --checkpoint artifacts/csv/model.pt --csv data/test.csv
```

Python API поддерживает больше двух классов через `TrainConfig(num_classes=...)`.

## Устройство

```text
CSV / синтетические данные
  → TableDataset
  → детерминированное разделение train / validation
  → Standardizer, обученный только на train
  → DataLoader
  → токены признаков → AttentionBlock → logits
  → CrossEntropyLoss + Adam → оценка → checkpoint
```

`LogAttention` использует `log(softplus(Q))` и `log(softplus(K))`, чтобы отрицательные
линейные проекции не приводили к NaN. Поддерживаются padding, causal и явная boolean-маска;
`True` запрещает внимание. У полностью замаскированной строки веса внимания нулевые.
Возвращаемые веса усредняются по головам. AttentionBlock содержит residual-связи,
LayerNorm и feed-forward сеть. Выход модели — logits для cross-entropy.

Seed управляет инициализацией, разделением и перемешиванием. Последний неполный батч
сохраняется, метрики усредняются по числу объектов. Чекпойнт загружается на CPU
с `weights_only=True`; он предназначен для инференса, а не продолжения состояния Adam.

```text
src/ml_pipeline/
  blocks.py      # LogAttention, AttentionBlock
  models.py      # TabularClassifier
  data.py        # Dataset, CSV, нормализация, DataLoader
  training.py    # обучение, оценка, predict, checkpoint
  cli.py         # train / evaluate
tests/
  unit/          # контракты отдельных компонентов
  e2e/           # запуск CLI в отдельных процессах
  test_*.py      # integration, property, regression, smoke
```

## Тесты

```bash
uv run --locked pytest
uv run --locked pytest --cov=ml_pipeline --cov-branch --cov-report=term-missing
uv run --locked ruff check .
uv run --locked ruff format --check .
```

Минимальное общее покрытие — **90%**, учитываются ветвления и CLI-процессы.
Маркеры зарегистрированы строго; каждый можно запустить отдельно:

| Маркер | Что проверяется | Команда |
| --- | --- | --- |
| `unit` | Численный результат attention, градиенты, формы, ошибки, CSV, разбиение, метрики | `uv run --locked pytest -m unit` |
| `integration` | Снижение loss, качество на независимых данных, воспроизводимость, checkpoint round-trip | `uv run --locked pytest -m integration` |
| `property` | Инвариантность к перестановке батча, вероятности, полнота DataLoader; по 20 примеров Hypothesis | `uv run --locked pytest -m property` |
| `regression` | Отрицательные проекции, полная маска, causal-mask при возврате весов | `uv run --locked pytest -m regression` |
| `smoke` | Одна эпоха через публичный API | `uv run --locked pytest -m smoke` |
| `e2e` | CSV → обучение → файлы → отдельный процесс оценки; синтетический запуск и ошибки CLI | `uv run --locked pytest -m e2e` |

Тесты используют CPU, небольшие локальные данные, один поток и временные каталоги.
Проверки качества требуют validation accuracy и accuracy на независимой синтетической
выборке не ниже 0.85. Сравнение повторных запусков проверяет историю и веса.
Это проверка на детерминированной учебной задаче, а не оценка качества на произвольных CSV.

Для удобства доступны `make install`, `make lint`, `make test`, `make coverage`,
`make train`, `make evaluate`. CI запускает lint, форматирование и полный набор тестов
с покрытием на Python 3.12 и 3.13.

## Работа с Git

Используется Gitflow с отдельными коммитами для этапов:

- `main` — стабильные релизы, помеченные тегами;
- `develop` — интеграция изменений;
- `feature/pipeline-tests` — тесты контрактов и полного пайплайна;
- `feature/pipeline-implementation` — компоненты и обучение;
- `test/pipeline-validation` — проверка, исправления и автоматизация тестирования;
- `release/1.0.0` — подготовка релиза с последующим merge в `main` и `develop`.

Слияния выполняются с `--no-ff`, чтобы этапы оставались видны в истории:

```bash
git log --graph --oneline --all --decorate
```

`.gitignore` исключает локальные папки `пример`, `.codex`, `.agents`, `.aws`, окружения,
кэши, секреты, данные, веса и отчёты тестирования. `uv.lock` хранится в репозитории.
