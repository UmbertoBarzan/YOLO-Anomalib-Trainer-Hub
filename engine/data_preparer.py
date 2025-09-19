import shutil
import yaml
from pathlib import Path
from core.utils import ensure_dir, save_yaml
from core.logger import create_logger

class DataPreparer:
    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        ensure_dir(self.job_dir)
        self.logger = create_logger("datapreparer")

    def prepare(self, job_data: dict):
        dataset_data = job_data.get("dataset", {})
        dataset_name = job_data.get("training_id", "unknown_dataset")
        
        # Dataset dinamico (usato nei train)
        dataset_dir = self.job_dir / "datasets"
        ensure_dir(dataset_dir)

        # Scrivi immagini a partire da UUID + compliance/labels
        if job_data.get("worker_type") == "AnomalibDetection":
            dataset_path = self._prepare_anomalib(dataset_dir, dataset_data)
        elif job_data.get("worker_type") == "YoloDetection":
            dataset_path = self._prepare_yolo(dataset_dir, dataset_data)
        else:
            raise ValueError("worker_type non valido o mancante")

        # Salva config YAML custom, se fornito, o imposta default
        anomalib_yaml = self._save_configs(
            job_data, dataset_name, dataset_dir, mode="anomalib"
        )
        yolo_yaml = self._save_configs(
            job_data, dataset_name, dataset_dir, mode="yolo"
        )

        return {
            "dataset": dataset_name,
            "dataset_path": str(dataset_path),
            "anomalib_yaml": anomalib_yaml,
            "yolo_yaml": yolo_yaml,
        }

    def _prepare_anomalib(self, target_dir: Path, dataset_data: dict):
        '''
        Prepara il dataset per l'addestramento di anomalib in base all'UUID
        e compliance. Sostituisce le immagini mancanti con un placeholder
        per evitare errori durante l'addestramento.
        '''
        dataset = dataset_data["dataset"]
        anomalib_dir = target_dir / "anomalib"
        for cls in ["0_good", "1_anomaly", "2_uncertain"]:
            ensure_dir(anomalib_dir / cls)

        for item in dataset:
            uuid = item["uuid"]
            compliance = int(item.get("compliance", 0))

            if compliance == 0:
                category = "0_good"
            elif compliance == 1:
                category = "1_anomaly"
            else:
                category = "2_uncertain"

            dest = anomalib_dir / category / f"{uuid}.jpg"
            img_path = Path("raw_images") / f"{uuid}.jpg"
            if img_path.exists():
                shutil.copy(img_path, dest)
            else:
                with open(dest, "wb") as f:
                    f.write(b"")  # placeholder

        return anomalib_dir

    def _prepare_yolo(self, target_dir: Path, dataset_data: dict):
        '''
        Prepara il dataset per l'addestramento di YOLO in base all'UUID
        e labels. Sostituisce le immagini mancanti con un placeholder
        per evitare errori durante l'addestramento.
        '''
        dataset = dataset_data["dataset"]
        metadata = dataset_data.get("metadata", {"0": "normal", "1": "anomaly"})
        
        yolo_dir = target_dir / "yolo"
        for split in ["train", "val"]:
            for cls_name in metadata.values():
                ensure_dir(yolo_dir / split / "images" / cls_name)

        for i, item in enumerate(dataset):
            uuid = item["uuid"]
            labels = item.get("labels", [])
            cls_index = labels[0] if labels else "0"
            class_name = metadata.get(cls_index, "unknown")
            split = "train" if i % 2 == 0 else "val"

            dest = yolo_dir / split / "images" / class_name / f"{uuid}.jpg"
            img_path = Path("raw_images") / f"{uuid}.jpg"
            if img_path.exists():
                shutil.copy(img_path, dest)
            else:
                with open(dest, "wb") as f:
                    f.write(b"")

        return yolo_dir

    def _save_configs(self, job_data: dict, dataset_name: str, target_dir: Path, mode: str):
        """
        Scrive il config YAML custom se è incluso nel JSON.
        Altrimenti ritorna il path al file YAML di default.
        """
        config_data = job_data.get("config_yaml")
        if config_data:
            config_path = target_dir / f"{mode}_job.yaml"
            save_yaml(config_data, config_path)
            self.logger.info(f"[Custom] Config YAML salvato: {config_path}")
            return str(config_path)

        # default fallback
        default_path = Path("configs") / f"{mode}_jobs.yaml"
        self.logger.info(f"[Default] Uso config YAML: {default_path}")
        return str(default_path)
