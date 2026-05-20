from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.perfectframe.image_processors import (
    NIMA_RESIZE_INTERPOLATION,
    OpenCVImage,
)
from app.perfectframe.schemas import ImageResolution


def test_normalize_images_returns_empty_float32_batch_for_empty_input() -> None:
    result = OpenCVImage.normalize_images([], ImageResolution(224, 224))

    assert result.shape == (0, 224, 224, 3)
    assert result.dtype == np.float32


def test_normalize_images_uses_inter_linear_and_normalizes_rgb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resize_mock = MagicMock(
        return_value=np.array(
            [
                [[0, 64, 255], [255, 128, 0]],
                [[10, 20, 30], [40, 50, 60]],
            ],
            dtype=np.uint8,
        )
    )
    cvt_color_mock = MagicMock(
        side_effect=lambda image, code: image[..., ::-1]
    )
    cv2_mock = MagicMock(
        INTER_LINEAR=1,
        COLOR_BGR2RGB=99,
        resize=resize_mock,
        cvtColor=cvt_color_mock,
    )

    monkeypatch.setattr(
        "app.perfectframe.image_processors._import_cv2",
        lambda: cv2_mock,
    )

    images = [np.zeros((4, 4, 3), dtype=np.uint8)]
    result = OpenCVImage.normalize_images(images, ImageResolution(2, 2))

    resize_mock.assert_called_once_with(
        images[0],
        (2, 2),
        interpolation=cv2_mock.INTER_LINEAR,
    )
    cvt_color_mock.assert_called_once_with(resize_mock.return_value, cv2_mock.COLOR_BGR2RGB)
    assert NIMA_RESIZE_INTERPOLATION == "INTER_LINEAR"
    assert result.dtype == np.float32
    assert result.shape == (1, 2, 2, 3)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert result[0, 0, 0].tolist() == pytest.approx([1.0, 64.0 / 255.0, 0.0])
