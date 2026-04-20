"""Pole detector — Stanford GridMapping CNN inference (Source 17).

Paper: Wang, Majumdar & Rajagopal (2023), Nature Communications
Repo:  https://github.com/wangzhecheng/GridMapping
Data:  https://figshare.com/articles/dataset/22723171
License: MIT

Detects utility poles in street-level imagery using a ResNet-based CNN.
Falls back to heuristic (no-op) if PyTorch is unavailable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ..utils.config_loader import load_settings
from ..utils.logger import get_logger

log = get_logger(__name__)

try:
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    log.warning(
        "PyTorch not available — pole detection will return empty results. "
        "Install with: pip install torch torchvision pillow"
    )


class PoleDetector:
    """Detect utility poles from street-view images using GridMapping CNN.

    When PyTorch is unavailable, returns an empty GeoDataFrame and logs
    a warning — the pipeline continues with OSM-only pole data.
    """

    def __init__(self) -> None:
        cfg = load_settings()
        ml_cfg = cfg["ml_inference"]
        self._weights_path = Path(ml_cfg["pole_detector"]["model_weights"])
        self._conf_threshold = ml_cfg["pole_detector"]["confidence_threshold"]
        self._model = None
        self._transform = None

        if _TORCH_AVAILABLE and self._weights_path.exists():
            self._load_model()

    def detect(self, image_records: list[dict]) -> gpd.GeoDataFrame:
        """Detect poles in a list of imagery records.

        Parameters
        ----------
        image_records:
            List of dicts: {lat, lon, image_path, provider}

        Returns
        -------
        GeoDataFrame of detected pole locations with confidence scores.
        """
        if not image_records:
            return self._empty_pole_gdf()

        if not _TORCH_AVAILABLE:
            log.warning("PoleDetector: PyTorch unavailable — returning empty results")
            return self._empty_pole_gdf()

        if self._model is None:
            log.warning("PoleDetector: model not loaded — returning empty results")
            return self._empty_pole_gdf()

        records = []
        for rec in image_records:
            img_path = Path(rec.get("image_path", ""))
            if not img_path.exists():
                continue

            confidence = self._infer_single(img_path)
            if confidence >= self._conf_threshold:
                records.append({
                    "lat":        rec["lat"],
                    "lon":        rec["lon"],
                    "confidence": confidence,
                    "geometry":   Point(rec["lon"], rec["lat"]),
                    "source":     "GridMapping_ML",
                    "provider":   rec.get("provider", ""),
                    "node_type":  "pole",
                    "inferred":   True,
                })

        if not records:
            return self._empty_pole_gdf()

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
        log.info("PoleDetector: detected %d poles from %d images", len(gdf), len(image_records))
        return gdf

    # ------------------------------------------------------------------
    # Model loading and inference
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load GridMapping ResNet pole detection model."""
        try:
            import torchvision.models as models
            model = models.resnet50(pretrained=False)
            # GridMapping uses binary classification head
            import torch.nn as nn
            model.fc = nn.Linear(model.fc.in_features, 2)
            state_dict = torch.load(
                self._weights_path, map_location=torch.device("cpu")
            )
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            self._transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            log.info("PoleDetector: GridMapping model loaded from %s", self._weights_path)
        except Exception as exc:
            log.error("PoleDetector: failed to load model: %s", exc)
            self._model = None

    def _infer_single(self, image_path: Path) -> float:
        """Return pole detection probability [0,1] for a single image."""
        try:
            img = Image.open(image_path).convert("RGB")
            tensor = self._transform(img).unsqueeze(0)
            with torch.no_grad():
                output = self._model(tensor)
                proba = torch.softmax(output, dim=1)[0, 1].item()
            return proba
        except Exception as exc:
            log.debug("Inference failed for %s: %s", image_path, exc)
            return 0.0

    def _empty_pole_gdf(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            columns=["lat", "lon", "confidence", "source", "node_type", "inferred", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
