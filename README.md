# VAIS Voice — ML/R&D

Reproducible KZ/RU synthetic speech detection and generation research infrastructure.

Synthetic dataset v0.1 provides provenance-aware text inventory, deterministic planning,
resumable adapter execution, canonical manifests, and machine-readable reports. Detector Baseline
v0.1 adds real PyTorch training, validation-only checkpoint and threshold selection, protected-test
benchmark export, and checkpoint inference. It does not implement VAIS VoiceGen. See
[`docs/synthetic-generation.md`](docs/synthetic-generation.md) for the protocol and commands.

Исследование подлинности и генерации казахской, русской и смешанной речи.
Сейчас это **исследовательская инфраструктура, не production antifraud**. Training-код и два
baseline реализованы, но опубликованного научного checkpoint ещё нет: локальный пилотный корпус
слишком мал. Synthetic v0.1 создаёт данные через настраиваемые внешние TTS-адаптеры; Detection —
Phase 1, VAIS VoiceGen — Phase 2.

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
Dummy всегда возвращает 0.5. Evaluator помечает одинаковые scores как
`degenerate_constant_scores`, а ROC-AUC/EER/FPR@TPR95 записывает как `null`.
Метрики проверяют только код, не качество модели.
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

Training preflight:
`python -m vais_voice.train --config configs/detection/baseline.yaml --dry-run`.
Настоящее обучение Baseline A:
`python -m vais_voice.train --config configs/detection/baseline.yaml`.
Baseline B использует frozen официальный `torchaudio` Wav2Vec2 encoder и MLP head:
`python -m vais_voice.train --config configs/detection/baseline_ssl.yaml`.
Обе конфигурации требуют полноценный split corpus с двумя классами в train/val/test; synthetic
должны пройти quality gate и иметь `training_eligible=true`. Checkpoint и threshold выбираются
только по validation. Test запускается после фиксации лучшего checkpoint.
`python -m vais_voice.generation --help` показывает команды inventory/plan/run/report. Это pipeline
датасета, а не VoiceGen.

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
  generation/     # text inventory, generator adapters, plan/run/report
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

PyTorch не входит в base. Для проверенного Windows/NVIDIA окружения RTX 3060 Ti используется
зафиксированная пара PyTorch/torchaudio 2.11.0 с CUDA 13.0:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-gpu-cu130.txt
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

На другом сервере CUDA wheel выбирается заново и фиксируется в отчёте эксперимента.
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
GitHub Actions: Windows/Linux lint/tests, CPU Docker build и полный smoke pipeline
внутри только что построенного image.

## Dataset ingestion

The canonical manifest is the strict JSONL/Parquet contract in
`src/vais_voice/datasets/schema.py`. Local source adapters translate source-specific metadata
into that contract; they never edit raw files and never download datasets. The supported internal
language codes are `kk`, `ru`, `kk_ru`, and `other` (never `kz`). `kk_ru` requires an explicit
source annotation or user mapping; it is not inferred from occasional borrowed words.

Source configs under `configs/data/sources/` pin the official release/config or track, license,
URLs, citation, DOI, revision, and adapter layout version. Review those upstream terms again when
changing a revision. A downloadable dataset is not assumed to permit unrestricted use. Raw
datasets and local manifests remain ignored because their licenses, size, and access conditions
differ from this repository's MIT-licensed code. The verification matrix and exact layouts are in
[`docs/dataset-layouts.md`](docs/dataset-layouts.md).

To build a bounded local subset, review `configs/data/kzru_real_v01.yaml` and its source configs,
then run:

```bash
python -m vais_voice.data.prepare --config configs/data/kzru_real_v01.yaml
```

The immutable output contains processed audio, canonical JSONL and Parquet manifests,
`dataset_report.json`, and a provenance-rich `preparation.json`. Limits and filters live in YAML;
exact-content duplicates are reported and retained unless `duplicate_policy: error` is selected.
ASVspoof is tagged `external_sanity` and is excluded by the default `include_roles: [primary]`.
Build it only as an explicitly separate reference dataset; it is not KZ/RU corpus data.

KSC2 accepts a reviewed CSV/TSV/Parquet table or discovers the official FLAC/TXT sidecar tree.
FLEURS accepts the official headerless seven-column TSV layout and explicit `kk_kz`/`ru_ru`
source configs. ASVspoof maps
protocol `bonafide` to `real`, `spoof` to `synthetic`, and stores attack/system identifiers in
`attack_id` (also used as the required synthetic `generator_id`). KSC2 and both FLEURS adapters
and ASVspoof DF adapters have been exercised on tiny real local subsets. ASVspoof remains an
external-sanity source and is not included in the primary KZ/RU pilot manifest.

To add a source, implement `DatasetAdapter` under `vais_voice.datasets.adapters`, register it in
the adapter registry, add explicit source provenance YAML, and test it with tiny generated audio
and synthetic metadata only. Preserve the research invariants: no derivatives of one recording
may cross splits, protected speaker-independent tests have no speaker overlap, and unseen-generator
tests have no generator overlap with training.

## Этапы

1. Phase 0: schema → prepare → protected splits → dummy inference → evaluation/report.
2. Phase 1A: два detector baseline → validation selection → checkpoint → inference API.
3. Phase 1B: versioned real/synthetic corpus → настоящее обучение → protected benchmark.
4. Phase 2: KZ/RU VoiceGen с правами на голоса и независимыми holdouts.

Fixture-метрики никогда не являются научными результатами или гарантией подлинности.
