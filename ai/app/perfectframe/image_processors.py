"""Image processing helpers used by the frame extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from uuid import uuid4

import numpy as np

from app.perfectframe.schemas import (
    Image,
    ImageExtension,
    ImageResolution,
    Images,
    ImagesBatch,
)
from app.perfectframe.video_processors import _import_cv2

NIMA_RESIZE_INTERPOLATION = "INTER_LINEAR"


class ImageProcessor(ABC):
    """Abstract image processor."""

    @staticmethod
    @abstractmethod
    def normalize_images(images: Images, target_size: ImageResolution) -> ImagesBatch:
        """Normalize images for scoring."""

    @staticmethod
    @abstractmethod
    def save_image(
        image: Image,
        output_directory: Path,
        output_format: ImageExtension,
        filename: str | None = None,
    ) -> Path:
        """Save a single image to disk."""


class OpenCVImage(ImageProcessor):
    """OpenCV-backed image utilities."""

    @staticmethod
    def normalize_images(images: Images, target_size: ImageResolution) -> ImagesBatch:
        cv2 = _import_cv2()
        if not images:
            return np.empty(
                (0, target_size.height, target_size.width, 3),
                dtype=np.float32,
            )
        normalized = []
        for image in images:
            resized = cv2.resize(
                image,
                (target_size.width, target_size.height),
                interpolation=getattr(cv2, NIMA_RESIZE_INTERPOLATION),
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized.append(rgb)
        return np.array(normalized, dtype=np.float32) / 255.0

    @staticmethod
    def save_image(
        image: Image,
        output_directory: Path,
        output_format: ImageExtension,
        filename: str | None = None,
    ) -> Path:
        cv2 = _import_cv2()
        output_directory.mkdir(parents=True, exist_ok=True)
        resolved_filename = filename or f"{uuid4().hex}{output_format.value}"
        output_path = output_directory / resolved_filename
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to save image to {output_path}.")
        return output_path
