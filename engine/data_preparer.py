from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

from core.config_loader import ConfigLoader
from core.logger import create_logger
from core.utils import ensure_dir


class DataPreparer:
    """Utility class that materialises datasets on disk for training jobs."""

    ALLOWED_SPLITS = {"train", "val", "test"}

    def __init__(self, job_dir: Path):
        self.job_dir = Path(job_dir)
        ensure_dir(self.job_dir)
        self.logger = create_logger("datapreparer")

        self.cfg = ConfigLoader()
        self.raw_images_dir = Path(self.cfg.get_path("raw_images"))
        if not self.raw_images_dir.exists():
            raise FileNotFoundError(
                f"Raw images directory not found: {self.raw_images_dir.resolve()}"
            )

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    def prepare_anomalib(self, dataset_data: dict) -> Path:
        """Create an Anomalib-compatible folder structure.

        Expected entry fields:
            - uuid: str
            - compliance: 0 (good) | 1 (anomaly) | other (uncertain)
            - split (optional): train | val | test
            - extension / file_name (optional): used to resolve the image on disk
        """
        entries = self._extract_entries(dataset_data)

        target_root = self.job_dir / "anomalib"
        self._reset_dir(target_root)

        for item in entries:
            uuid = self._require_uuid(item)
            compliance = self._resolve_compliance(item)
            split = self._resolve_split(item, default="train" if compliance == 0 else "test")
            if compliance != 0 and split == "train":
                self.logger.warning(
                    "Forcing split for anomalous sample %s to 'test' (compliance=%s)",
                    uuid,
                    compliance,
                )
                split = "test"

            category = self._compliance_to_category(compliance)
            src = self._resolve_source_path(uuid, item)
            dest = target_root / split / category / self._build_filename(uuid, item, src.suffix)
            self._copy_image(uuid, item, dest, src_override=src)

        return target_root

    def prepare_yolo(self, dataset_data: dict) -> Tuple[Path, List[str]]:
        """Create a YOLO dataset with images/labels per split.

        Each dataset entry must provide:
            - uuid: str
            - split (optional): train | val | test
            - annotations (preferred): list of objects with `class_id` and `bbox`
              where bbox is expressed in cxcywh format (normalised or with width/height info)
            - labels (fallback): list of class ids; if no bbox is provided, a full-image
              bounding box will be generated as a last resort.
        """
        entries = self._extract_entries(dataset_data)
        class_map, remap = self._build_class_map(dataset_data.get("metadata"), entries)

        target_root = self.job_dir / "yolo"
        self._reset_dir(target_root)

        for split in self.ALLOWED_SPLITS:
            ensure_dir(target_root / split / "images")
            ensure_dir(target_root / split / "labels")

        for idx, item in enumerate(entries):
            uuid = self._require_uuid(item)
            default_split = "train" if idx % 2 == 0 else "val"
            split = self._resolve_split(item, default_split)

            src = self._resolve_source_path(uuid, item)
            width, height = self._get_image_size(src)
            dest_img = target_root / split / "images" / self._build_filename(uuid, item, src.suffix)
            self._copy_image(uuid, item, dest_img, src_override=src)

            labels_path = target_root / split / "labels" / f"{uuid}.txt"
            yolo_lines = self._build_yolo_labels(uuid, item, remap, width, height)
            labels_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

        ordered_names = [class_map[str(i)] for i in range(len(class_map))]
        return target_root, ordered_names

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _extract_entries(self, dataset_data: dict) -> List[dict]:
        dataset = dataset_data.get("dataset")
        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Dataset payload must include a non-empty 'dataset' list")
        return dataset

    def _require_uuid(self, item: dict) -> str:
        uuid = item.get("uuid")
        if not uuid:
            raise ValueError("Dataset item is missing required field 'uuid'")
        return str(uuid)

    def _resolve_compliance(self, item: dict) -> int:
        value = item.get("compliance", 0)
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid compliance value for item {item.get('uuid')}: {value}") from exc

    def _compliance_to_category(self, compliance: int) -> str:
        if compliance == 0:
            return "good"
        if compliance == 1:
            return "bad"
        return "uncertain"

    def _resolve_split(self, item: dict, default: str) -> str:
        split = item.get("split", default)
        if isinstance(split, str):
            split = split.lower()
        else:
            split = default

        if split not in self.ALLOWED_SPLITS:
            self.logger.warning("Invalid split '%s', falling back to '%s'", split, default)
            split = default
        return split

    def _build_filename(self, uuid: str, item: dict, suffix: str | None = None) -> str:
        extension = suffix or item.get("extension") or Path(item.get("file_name", "")).suffix
        if not extension:
            extension = ".jpg"
        if not str(extension).startswith("."):
            extension = f".{extension}"
        return f"{uuid}{extension.lower()}"

    def _resolve_source_path(self, uuid: str, item: dict) -> Path:
        explicit = item.get("file_name") or item.get("filename") or item.get("path")
        candidates: Iterable[Path]

        if explicit:
            candidates = [self.raw_images_dir / explicit]
        else:
            requested_ext = item.get("extension")
            requested_ext = requested_ext.lstrip(".") if isinstance(requested_ext, str) else None
            base_names = [f"{uuid}.{requested_ext}"] if requested_ext else []
            base_names.extend([f"{uuid}.jpg", f"{uuid}.png", f"{uuid}.jpeg"])
            candidates = [self.raw_images_dir / name for name in base_names]

        for path in candidates:
            if path and path.exists():
                return path

        raise FileNotFoundError(
            f"Unable to locate raw image for uuid '{uuid}' in {self.raw_images_dir}"
        )

    def _copy_image(self, uuid: str, item: dict, destination: Path, src_override: Path | None = None) -> Path:
        source = src_override or self._resolve_source_path(uuid, item)
        ensure_dir(destination.parent)
        shutil.copy(source, destination)
        return source

    def _reset_dir(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)
        ensure_dir(path)

    def _get_image_size(self, image_path: Path) -> Tuple[int, int]:
        with Image.open(image_path) as img:
            return img.width, img.height

    def _build_class_map(
        self, metadata: Dict[str, str] | None, entries: List[dict]
    ) -> Tuple[Dict[str, str], Dict[str, int]]:
        id_to_name: Dict[str, str] = {}
        if metadata:
            for key, value in metadata.items():
                id_to_name[str(key)] = str(value)

        if not id_to_name:
            collected_ids = set()
            for item in entries:
                for ann in item.get("annotations", []) or []:
                    class_id = ann.get("class_id") or ann.get("label") or ann.get("class")
                    if class_id is not None:
                        collected_ids.add(str(class_id))
                for class_id in item.get("labels", []) or []:
                    collected_ids.add(str(class_id))

            if not collected_ids:
                raise ValueError("Unable to infer class labels for YOLO dataset generation")

            for cid in collected_ids:
                id_to_name[cid] = f"class_{cid}"

        sorted_original_keys = sorted(id_to_name.keys(), key=lambda x: int(x))
        normalised_map: Dict[str, str] = {}
        remap: Dict[str, int] = {}
        for new_idx, original_key in enumerate(sorted_original_keys):
            normalised_map[str(new_idx)] = id_to_name[original_key]
            remap[original_key] = new_idx

        return normalised_map, remap

    def _build_yolo_labels(
        self,
        uuid: str,
        item: dict,
        remap: Dict[str, int],
        width: int,
        height: int,
    ) -> List[str]:
        annotations = item.get("annotations") or []
        lines: List[str] = []

        if annotations:
            for ann in annotations:
                original_class = ann.get("class_id") or ann.get("label") or ann.get("class")
                if original_class is None:
                    raise ValueError(f"Missing class id in annotation for sample {uuid}")
                class_key = str(original_class)
                if class_key not in remap:
                    raise ValueError(
                        f"Annotation for {uuid} references unknown class id '{class_key}'"
                    )
                bbox = self._extract_bbox(uuid, ann, item, width, height)
                lines.append(self._format_yolo_line(remap[class_key], bbox))
        else:
            labels = item.get("labels") or []
            if labels:
                class_key = str(labels[0])
                if class_key not in remap:
                    raise ValueError(
                        f"Sample {uuid} references unknown class id '{class_key}' in 'labels'"
                    )
                self.logger.warning(
                    "No bounding boxes supplied for %s; using a full-image box for class %s",
                    uuid,
                    class_key,
                )
                bbox = (0.5, 0.5, 1.0, 1.0)
                lines.append(self._format_yolo_line(remap[class_key], bbox))
            else:
                # negative sample (no objects); YOLO expects an empty label file
                lines = []

        return lines

    def _extract_bbox(
        self,
        uuid: str,
        annotation: dict,
        item: dict,
        width: int,
        height: int,
    ) -> Tuple[float, float, float, float]:
        raw_bbox = annotation.get("bbox") or annotation.get("box") or annotation.get("coordinates")
        if raw_bbox is None:
            raise ValueError(f"Annotation for {uuid} is missing bbox information")

        fmt = (annotation.get("format") or "cxcywh").lower()
        bbox_values, normalised = self._unpack_bbox(raw_bbox, annotation)

        if len(bbox_values) != 4:
            raise ValueError(f"BBox for {uuid} must contain four values, got {bbox_values}")

        values = [float(v) for v in bbox_values]

        if not normalised or any(v > 1.0 for v in values):
            values = self._normalise_bbox(values, fmt, width, height, uuid)
        else:
            values = self._coerce_bbox_format(values, fmt, uuid)

        values = [min(max(v, 0.0), 1.0) for v in values]
        return values[0], values[1], values[2], values[3]

    def _unpack_bbox(self, raw_bbox, annotation: dict) -> Tuple[List[float], bool]:
        normalised = annotation.get("normalized") or annotation.get("normalised")

        if isinstance(raw_bbox, dict):
            values = raw_bbox.get("value") or raw_bbox.get("values") or raw_bbox.get("bbox")
            normalised = raw_bbox.get("normalized", normalised)
            annotation.setdefault("format", raw_bbox.get("format", annotation.get("format")))
        else:
            values = raw_bbox

        if values is None:
            raise ValueError("BBox structure does not contain usable coordinates")

        if normalised is None:
            # heuristics: consider values normalised if they are already within [0, 1]
            normalised = all(isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0 for v in values)

        return list(map(float, values)), bool(normalised)

    def _normalise_bbox(
        self,
        values: List[float],
        fmt: str,
        width: int,
        height: int,
        uuid: str,
    ) -> List[float]:
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for {uuid}: {width}x{height}")

        if fmt == "cxcywh":
            cx, cy, w, h = values
            cx /= width
            cy /= height
            w /= width
            h /= height
        elif fmt == "xywh":
            x, y, w, h = values
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            w /= width
            h /= height
        elif fmt == "xyxy":
            x_min, y_min, x_max, y_max = values
            cx = ((x_min + x_max) / 2) / width
            cy = ((y_min + y_max) / 2) / height
            w = abs(x_max - x_min) / width
            h = abs(y_max - y_min) / height
        else:
            raise ValueError(f"Unsupported bbox format '{fmt}' for {uuid}")

        return [cx, cy, w, h]

    def _coerce_bbox_format(self, values: List[float], fmt: str, uuid: str) -> List[float]:
        if fmt == "cxcywh":
            return values
        if fmt == "xywh":
            x, y, w, h = values
            return [x + w / 2, y + h / 2, w, h]
        if fmt == "xyxy":
            x_min, y_min, x_max, y_max = values
            return [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                abs(x_max - x_min),
                abs(y_max - y_min),
            ]
        raise ValueError(f"Unsupported bbox format '{fmt}' for {uuid}")

    def _format_yolo_line(self, class_index: int, bbox: Tuple[float, float, float, float]) -> str:
        cx, cy, w, h = bbox
        return f"{class_index} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
