from core.logger import create_logger
from core.config_loader import ConfigLoader
from core.utils import load_yaml, ensure_dir
from pathlib import Path
import torch
import yaml
from threading import Lock
from pytorch_lightning import Trainer as LightningTrainer
from pytorch_lightning.callbacks import ModelCheckpoint

from anomalib.data import Folder
from anomalib.models import Patchcore, Padim, EfficientAd
from ultralytics import YOLO

from engine.data_preparer import DataPreparer

# Mappatura nome modello -> classe o file
MODEL_MAP = {
    # anomalib
    'patchcore': Patchcore,
    'padim': Padim,
    'efficientad': EfficientAd,

    # yolo
    'yolov8': 'yolov8n.pt',
    'yolov10': 'yolov810.pt',
    'yolov11': 'yolov8n.pt',
    'yolov12': 'yolov12n.pt',
    # TODO: aggiungere altri modelli custom se necessari
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
    def _train_anomalib_model(self, config: dict, dataset: str):
        """
        Esegue il training Anomalib su un dataset già preparato (in cartella)
        """
        model_name = config.get("name", "anomalib_model")
        batch_size = config.get("train_batch_size", 32)
        epochs = config.get("epochs", 10)
        size = config.get("size", 256)
        model_type = config.get("model", "").lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Anomalib Model: {model_name} ({model_type})")

        model_cls = MODEL_MAP.get(model_type)
        if not model_cls:
            self.logger.warning(f"Modello Anomalib non supportato: {model_type}")
            return

        model_params = config.get("model_params", {})
        if "backbone" not in model_params:
            model_params["backbone"] = "resnet18"
        if "layers" not in model_params:
            model_params["layers"] = ["layer1", "layer2", "layer3"]

        datamodule = Folder(
            root=Path(self.cfg.get_path("datasets_dir")) / dataset,
            image_size=size,
            train_batch_size=batch_size,
        )

        model = model_cls(input_size=(size, size), **model_params)

        ensure_dir(Path("models") / model_name)

        checkpoint = ModelCheckpoint(
            dirpath=f"models/{model_name}",
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
            enable_progress_bar=False
        )

        with lock:
            trainer.fit(model=model, datamodule=datamodule)

        self.logger.info(f"[Anomalib] Training completato per {model_name}")

    def _train_yolo_model(self, config: dict, dataset: str):
        """
        Esegue il training YOLO su un dataset già preparato (in cartella)
        """
        model_name = config.get("name", "yolo_model")
        epochs = config.get("epochs", 50)
        batch_size = config.get("train_batch_size", 32)
        size = config.get("size", 640)
        model_type = config.get("model", "yolov8").lower()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"[YOLO] Model: {model_name} ({model_type})")

        model_path = MODEL_MAP.get(model_type, "yolov8n.pt")
        model = YOLO(model_path)

        dataset_path = Path(self.cfg.get_path("datasets_dir")) / dataset
        yolo_yaml = {
            "train": str(dataset_path / "train/images"),
            "val": str(dataset_path / "val/images"),
            "nc": 2,
            "names": ["normal", "anomaly"]
            # TODO: ricavare dinamicamente da metadata (non ancora fatto)
        }

        yolo_yaml_path = Path(self.cfg.get_path("generated_yolo")) / f"{dataset}.yaml"
        ensure_dir(yolo_yaml_path.parent)
        with open(yolo_yaml_path, "w") as f:
            yaml.safe_dump(yolo_yaml, f)

        with lock:
            model.train(
                data=str(yolo_yaml_path),
                epochs=epochs,
                imgsz=size,
                batch=batch_size,
                project="models",
                name=model_name,
                device=device,
                **config.get("model_params", {})
            )

        self.logger.info(f"[YOLO] Training completato per {model_name}")

    def run_anomalib_job(self, dataset_data: dict, config_yaml: list = None):
        """
        Prepara dataset + training per Anomalib
        """
        training_id = dataset_data.get("training_id", "unknown")

        # Crea la job_dir dinamicamente
        job_dir = Path(self.cfg.get_path("datasets_dir")) / f"job_{training_id}"

        # Istanzia DataPreparer passandogli la job_dir
        self.preparer = DataPreparer(job_dir=job_dir)

        # Prepara il dataset e ottieni il path relativo al dataset creato
        dataset_dir = self.preparer._prepare_anomalib(job_dir, dataset_data)

        # Se non specificato nel JSON, usa quello di default
        configs = config_yaml or load_yaml(Path(self.cfg.get_path("anomalib_jobs")))

        for config in configs:
            if not config.get("disabled", False):
                self._train_anomalib_model(config, Path(dataset_dir).name)


    def run_yolo_job(self, dataset_data: dict, config_yaml: list, metadata: dict = None):
        """
        Prepara dataset + training per YOLO
        """
        training_id = dataset_data.get("training_id", "unknown")
        job_dir = Path(self.cfg.get_path("datasets_dir")) / f"job_{training_id}"
        preparer = DataPreparer(job_dir=job_dir)

        dataset_dir = preparer._prepare_yolo(job_dir, dataset_data)

        configs = config_yaml or load_yaml(Path(self.cfg.get_path("yolo_jobs")))

        for config in configs:
            if not config.get("disabled", False):
                self._train_yolo_model(config, Path(dataset_dir).name)

