"""Scorer abstractions for aesthetic benchmark models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


class FrameScorer(ABC):
    """Abstract frame scorer."""

    @abstractmethod
    def score(self, frame_path: Path) -> float:
        """Return a single aesthetic score for the provided frame."""


class PyiqaScorer(FrameScorer):
    """Adapter that exposes a pyiqa metric as a frame scorer."""

    def __init__(
        self,
        metric_name: str,
        device: str | None = None,
        call_kwargs: dict[str, object] | None = None,
    ) -> None:
        try:
            import pyiqa
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pyiqa is not installed. Install the benchmark project with the "
                "`scorers` extra after installing a CUDA-enabled PyTorch wheel."
            ) from exc

        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "torch is not installed. Install a CUDA-enabled PyTorch wheel "
                "before running the real scorer benchmark."
            ) from exc

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._metric = pyiqa.create_metric(metric_name, device=resolved_device)
        self._metric.eval()
        self._device = resolved_device
        self._call_kwargs = dict(call_kwargs or {})

    def score(self, frame_path: Path) -> float:
        with Image.open(frame_path) as image:
            rgb = image.convert("RGB")
            tensor = self._torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self._device)

        with self._torch.inference_mode():
            result = self._metric(tensor, **self._call_kwargs)
        if hasattr(result, "detach"):
            result = result.detach()
        if hasattr(result, "cpu"):
            result = result.cpu()
        if hasattr(result, "reshape"):
            result = result.reshape(-1)
        if hasattr(result, "numel") and result.numel() == 0:
            raise RuntimeError(f"{frame_path} produced an empty score output.")
        if hasattr(result, "numel") and result.numel() >= 1:
            return float(result[0].item())
        if hasattr(result, "item"):
            return float(result.item())
        return float(result)


ScorerFactory = Callable[[str | None], FrameScorer]


def build_default_scorer_registry() -> dict[str, ScorerFactory]:
    """Return the supported scorer registry."""

    return {
        "nima": lambda device: PyiqaScorer("nima", device=device),
        "clipiqa": lambda device: PyiqaScorer("clipiqa", device=device),
        "qalign": lambda device: PyiqaScorer(
            "qalign",
            device=device,
            call_kwargs={"task_": "aesthetic"},
        ),
    }
