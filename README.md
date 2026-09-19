# VAIS Voice — ML/R&D

Reproducible KZ/RU synthetic speech detection and generation research infrastructure.

Исследование подлинности и генерации казахской, русской и смешанной речи.
Сейчас это **инфраструктура, не SaaS и не готовая модель**. Нет реального benchmark,
обучения, TTS, API или UI. Detection — Phase 1, VAIS VoiceGen — Phase 2.

## Быстрый старт

Python 3.11 рекомендуется (поддерживается 3.11–3.12).

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m vais_voice.doctor
python scripts/smoke_pipeline.py
pytest --cov=vais_voice
ruff check .
ruff format --check .
```

Linux: python3.11 -m venv .venv, затем source .venv/bin/activate.
Smoke создаёт **20 математических сигналов, не речь** (10 условных real / 10 synthetic).
Dummy всегда возвращает 0.5. Метрики проверяют только код, не качество модели.
Результаты: runs/smoke/runs/dummy/{metrics.json,predictions.csv,config.yaml,
dataset_snapshot.json,experiment.json,report.md}. Повтор: --output runs/smoke-v2.
Существующие данные не перезаписываются.

## Pipeline на собственных данных

Проверьте лицензии/согласия, положите аудио в data/raw/, manifest в
data/manifests/local/kzru_v1.jsonl и настройте configs/data/kzru_v1.yaml.

```bash
vais-voice prepare --config configs/data/kzru_v1.yaml
vais-voice split --manifest data/processed/kzru_v1/manifest.parquet --preparation data/processed/kzru_v1/preparation.json
vais-voice predict --detector dummy --manifest data/processed/kzru_v1/manifest.split.parquet --preparation data/processed/kzru_v1/split_preparation.json --output runs/dummy/predictions.csv
vais-voice evaluate --manifest data/processed/kzru_v1/manifest.split.parquet --preparation data/processed/kzru_v1/split_preparation.json --predictions runs/dummy/predictions.csv --output runs/dummy
```

Prepare: безопасные пути, лимиты, хеши, mono/resample 16 kHz, JSONL + Parquet.
Split: train.parquet, val.parquet, test.parquet и manifest.split.parquet рядом с audio/.
Декодирование через libsndfile; неподдерживаемые форматы требуют контролируемой
конвертации. Автоматического FFmpeg fallback пока нет.

Training пока только preflight:
`python -m vais_voice.train --config configs/detection/baseline.yaml --dry-run`.
Без dry-run команда отказывается. Evaluate не загружает checkpoints.
`python -m vais_voice.generation` честно сообщает об отсутствии VoiceGen.

## Schema и leakage

[Контракт manifest](data/manifests/README.md). Языки: kk, ru, kk_ru, other.
Synthetic требует generator_id; у real он null. Родитель parent_sample_id должен
присутствовать в manifest; циклы запрещены.

> No derived samples from the same source recording may cross dataset split boundaries.
> No speaker overlap between protected train/test splits.
> For unseen-generator evaluation: no generator overlap between train and test.

Транзитивные группы: speaker_id/related_speaker_ids, source_id, parent_sample_id,
одинаковые исходные SHA-256. ID должны быть глобально согласованы.
Готовые splits проверяются; автоматические детерминированы seed.
Все три выборки требуют оба класса; маленький корпус может быть отклонён.

По умолчанию known-generator. Для unseen задайте unseen_generators: эти генераторы
и связанные группы попадают только в test. Отчитывайтесь по отдельному holdout-срезу
с real-reference: общий test может содержать и известные генераторы.
Нельзя называть весь mixed test unseen. Test не предназначен для выбора порогов.

Метрики: ROC-AUC, EER, FPR@TPR95; срезы language/codec/condition/generator.
Линейная интерполяция ROC; одноклассовые срезы получают null.
FPR@TPR95 — описательная test-метрика, не production-порог.
Калибровки и доверительных интервалов пока нет. CSV обязан точно покрывать test IDs.
Provenance внешней модели не подтверждается.

## Структура

```text
configs/{data,detection,generation}/
data/{raw,processed,manifests}/
src/vais_voice/
  data/           # physical preparation, validation, CLI, split export
  datasets/       # schema.py, manifest.py, split.py: logical dataset
  preprocessing/  # bounded decoding and resampling
  detection/      # Detector Protocol + explicit dummy
  generation/     # GenerationNotImplementedError
  evaluation/     # metrics and reports
  utils/          # paths, hashes, provenance
scripts/
notebooks/
tests/
runs/
```

Аудио, raw/processed, локальные manifests, checkpoints и runs gitignored.
Snapshots могут содержать чувствительные метаданные: не публикуйте без проверки.
MIT относится к коду, не к внешним данным/моделям. Права и согласия проверяются до ingestion.

## Окружение и Docker

PyTorch не входит в base. Extra .[ml] содержит unpinned torch/torchaudio.
Версии и CUDA wheels выбираем после проверки QAIRU. Для научного эксперимента
зафиксируйте фактически установленное окружение.
Отчёт сохраняет package versions, seed, Git commit/dirty, config, dataset/split versions,
хеши; checkpoint пока null.

```bash
docker build -t vais-voice:core .
docker run --rm vais-voice:core
docker run --rm vais-voice:core pytest -q
docker run --rm vais-voice:core python scripts/smoke_pipeline.py
```

Python 3.11 + FFmpeg + libsndfile, core/dev dependencies, непривилегированный пользователь.
Optional build args TORCH_VERSION и TORCH_INDEX_URL задают совместимую пару torch/torchaudio.
GPU runtime требует совместимый host driver, NVIDIA Container Toolkit, --gpus all.
QAIRU пока не проверен. CUDA в репозиторий не кладём; Docker Compose не нужен.
GitHub Actions: Windows/Linux lint/tests и CPU Docker build.

## Этапы

1. Phase 0: schema → prepare → protected splits → dummy inference → evaluation/report.
2. Phase 1: согласованные/versioned реальные и synthetic данные → detector → benchmark.
3. Phase 2: KZ/RU VoiceGen с правами на голоса и независимыми holdouts.
4. После проверенных моделей: API/deployment.

Fixture-метрики никогда не являются научными результатами или гарантией подлинности.
