import os
from pathlib import Path
from typing import List
from threading import Lock

import torch
import yaml
from anomalib.data import Folder
from anomalib.models import EfficientAd, Padim, Patchcore
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import ModelCheckpoint

from core.config_loader import ConfigLoader
from core.logger import create_logger
from core.utils import ensure_dir, load_yaml
from engine.data_preparer import DataPreparer
from ultralytics import YOLO

ANOMALIB_MODELS = {
    "patchcore": Patchcore,
    "padim": Padim,
    "efficientad": EfficientAd,
}

YOLO_WEIGHTS = {
    "yolov8": "yolov8n.pt",
}

lock = Lock()


class Trainer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Trainer, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = create_logger("trainer")
        self.cfg = ConfigLoader()
        self._ensure_ultralytics_settings()

    # ------------------------------------------------------------------
    # internal utilities
    # ------------------------------------------------------------------
    def _ensure_ultralytics_settings(self) -> None:
        """Persist Ultralytics settings within the project tree to avoid $HOME writes."""
        settings_path = Path(self.cfg.get_path("generated_yolo")) / "ultralytics_settings.json"
        ensure_dir(settings_path.parent)
        os.environ.setdefault("ULTRALYTICS_SETTINGS", str(settings_path.resolve()))
        if not settings_path.exists():
            settings_path.write_text("{}")

    def _resolve_config_list(self, config_yaml, default_path: Path) -> List[dict]:
        if config_yaml:
            if isinstance(config_yaml, list):
                return config_yaml
            if isinstance(config_yaml, dict):
                return [config_yaml]
            raise TypeError("config_yaml must be a dict or list of dicts")

        configs = load_yaml(default_path)
        if isinstance(configs, dict):
            configs = [configs]
        if not isinstance(configs, list):
            raise ValueError(f"Invalid configs loaded from {default_path}")
        return configs

    # ------------------------------------------------------------------
    # training primitives
    # ------------------------------------------------------------------
    def _train_anomalib_model(self, config: dict, dataset_path: Path) -> None:
        model_name = config.get("name", "anomalib_model")
        batch_size = config.get("train_batch_size", 32)
        epochs = config.get("epochs", 10)
        size = config.get("size", 256)
        model_type = config.get("model", "").lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info("[Anomalib] Training %s with dataset %s", model_name, dataset_path)

        model_cls = ANOMALIB_MODELS.get(model_type)
        if not model_cls:
            raise ValueError(f"Unsupported Anomalib model '{model_type}'")

        model_params = dict(config.get("model_params", {}))
        model_params.setdefault("backbone", "resnet18")
        model_params.setdefault("layers", ["layer1", "layer2", "layer3"])

        datamodule = Folder(
            root=dataset_path,
            image_size=size,
            train_batch_size=batch_size,
        )

        model = model_cls(input_size=(size, size), **model_params)

        output_dir = Path("models") / model_name
        ensure_dir(output_dir)

        checkpoint = ModelCheckpoint(
            dirpath=str(output_dir),
            filename="{epoch}-{val_loss:.2f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )

        trainer = LightningTrainer(
            max_epochs=epochs,
            accelerator="gpu" if device == "cuda" else "cpu",
            callbacks=[checkpoint],
            log_every_n_steps=1,
            enable_progress_bar=False,
        )

        with lock:
            trainer.fit(model=model, datamodule=datamodule)

        self.logger.info("[Anomalib] Completed training for %s", model_name)

    def _train_yolo_model(
        self,
        config: dict,
        dataset_path: Path,
        class_names: List[str],
        training_id: str,
    ) -> None:
        model_name = config.get("name", "yolo_model")
        epochs = config.get("epochs", 50)
        batch_size = config.get("train_batch_size", 32)
        size = config.get("size", 640)
        model_type = config.get("model", "yolov8").lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        weights_candidate = config.get("weights") or YOLO_WEIGHTS.get(model_type)
        if not weights_candidate:
            raise ValueError(
                f"No weights defined for YOLO model '{model_type}'. Provide 'weights' in config."
            )

        weights_path = Path(weights_candidate)
        if not weights_path.is_absolute():
            weights_path = Path.cwd() / weights_path
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found at {weights_path}")

        self.logger.info("[YOLO] Training %s (%s) using dataset %s", model_name, model_type, dataset_path)

        model = YOLO(str(weights_path))

        generated_dir = Path(self.cfg.get_path("generated_yolo"))
        ensure_dir(generated_dir)
        yaml_name = f"{training_id}_{model_name}.yaml".replace("/", "_")
        yolo_yaml_path = generated_dir / yaml_name

        yolo_yaml = {
            "train": str(dataset_path / "train/images"),
            "val": str(dataset_path / "val/images"),
            "nc": len(class_names),
            "names": class_names,
        }
        test_dir = dataset_path / "test/images"
        if test_dir.exists():
            yolo_yaml["test"] = str(test_dir)

        with open(yolo_yaml_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(yolo_yaml, fh, sort_keys=False)

        ensure_dir(Path("models"))

        with lock:
            model.train(
                data=str(yolo_yaml_path),
                epochs=epochs,
                imgsz=size,
                batch=batch_size,
                project="models",
                name=model_name,
                device=device,
                **config.get("model_params", {}),
            )

        self.logger.info("[YOLO] Completed training for %s", model_name)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run_anomalib_job(self, dataset_data: dict, config_yaml: list | dict | None = None) -> None:
        training_id = dataset_data.get("training_id", "unknown")
        job_dir = Path(self.cfg.get_path("datasets_dir")) / f"job_{training_id}"
        preparer = DataPreparer(job_dir=job_dir)
        dataset_path = preparer.prepare_anomalib(dataset_data)

        configs = self._resolve_config_list(config_yaml, Path(self.cfg.get_path("anomalib_jobs")))

        for config in configs:
            if config.get("disabled", False):
                continue
            self._train_anomalib_model(config, dataset_path)

    def run_yolo_job(self, dataset_data: dict, config_yaml: list | dict | None, metadata: dict | None = None) -> None:
        training_id = dataset_data.get("training_id", "unknown")
        job_dir = Path(self.cfg.get_path("datasets_dir")) / f"job_{training_id}"
        preparer = DataPreparer(job_dir=job_dir)
        dataset_path, class_names = preparer.prepare_yolo(dataset_data)

        configs = self._resolve_config_list(config_yaml, Path(self.cfg.get_path("yolo_jobs")))

        for config in configs:
            if config.get("disabled", False):
                continue
            self._train_yolo_model(config, dataset_path, class_names, training_id)
