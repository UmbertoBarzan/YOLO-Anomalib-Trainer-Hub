# Visiofy Anomalib Orchestrator

This repository hosts a small web service that queues and executes computer vision training jobs on a single GPU. Incoming HTTP requests describe a dataset and the model families to run (Anomalib or YOLO). The service prepares the dataset folders, launches the appropriate training loops, and keeps basic queue statistics.

## Features
- Flask API with `/train` for job submission and `/status` for queue inspection
- Background job queue per GPU (`engine/gpu_manager.py`)
- Dataset preparation helpers for Anomalib and YOLO (`engine/data_preparer.py`)
- Training orchestration with Anomalib models (Patchcore, Padim, EfficientAd) and Ultralytics YOLO variants (`engine/trainer.py`)
- Configurable paths and default job templates under `configs/`

## Prerequisites
- Linux or macOS environment with NVIDIA GPU drivers (CUDA optional but recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- Raw RGB images named by UUID under `raw_images/` (see below)

## Quick Start

```bash
# 1. Create the conda environment (CPU-only Torch)
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu conda env create -f environment.yml
# (or `PIP_EXTRA_INDEX_URL=... conda env update --file environment.yml --prune` to refresh an existing env)

# 2. Activate the environment
conda activate yoloanomalibhub

# 3. Launch the API server
python main.py
# Server listens on http://0.0.0.0:5000
```

> Note: `environment.yml` stores `PIP_EXTRA_INDEX_URL` so pip keeps using the CPU Torch wheels after activation, but you must also set the variable when running `conda env create` as shown above.

## API

### `POST /train`
Submit a training job. The payload must include:
- `training_id`: unique string identifier
- `worker_type`: `AnomalibDetection` or `YoloDetection`
- `dataset`: metadata used by the data preparer (UUIDs, labels/compliance, optional `metadata` map)
- Optional `config_yaml`: list of config overrides. If omitted, defaults in `configs/anomalib_jobs.yaml` or `configs/yolo_jobs.yaml` are used.

Example request:

```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "training_id": "demo-001",
    "worker_type": "YoloDetection",
    "dataset": {
      "training_id": "demo-001",
      "metadata": {"0": "normal", "1": "anomaly"},
      "dataset": [
        {"uuid": "img-001", "labels": ["0"]},
        {"uuid": "img-002", "labels": ["1"]}
      ]
    }
  }'
```

### `GET /status`
Returns queued and running job counts:

```json
{
  "queued": 0,
  "running": 1
}
```

## Dataset Preparation
- Place source images in `raw_images/` using the same UUID and extension reported in the job payload (e.g. `raw_images/<uuid>.jpg`).
- Each dataset item must declare at least a `uuid`, optional `split` (`train` | `val` | `test`) and, for Anomalib, a `compliance` flag (`0` good, `1` anomaly, anything else uncertain). Non-normal samples are automatically moved out of the training split.
- For YOLO jobs provide `annotations` with `class_id` and `bbox` (centre-based `cxcywh`). Bboxes can be normalised or absolute (image size is read at runtime). If only a class label is given, a full-image bounding box is used as a fallback.
- Missing raw images now raise a descriptive error instead of generating empty placeholders, so ingest failures surface immediately.

## Outputs
- Training checkpoints saved under `models/<run_name>/`
- Logs written to `logs/` (`webserver.log`, `trainer.log`, etc.)
- YOLO data yaml files produced in `generated_yolo/`

## Known Gaps
- `engine/data_preparer.py` does not yet generate YOLO label files; adjust if your workflow requires bounding boxes.
- Default YAML templates assume two classes (`normal`, `anomaly`). Update metadata or configs for other setups.
- Ultralytics writes settings under `~/.config/Ultralytics`; ensure that path is writable in your environment.

## Development Tips
- Use `conda run -n yoloanomalibhub python ...` for scripted commands without activating the environment globally.
- Keep `configs/global.yaml` in sync with your storage layout.
- Export an updated `environment.yml` after adding dependencies (`conda env export --from-history > environment.yml`).
