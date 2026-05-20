from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from app.core.config import settings
from app.perfectframe.image_evaluators import (
    NIMAEvaluator,
    SharpnessVarianceEvaluator,
    build_image_evaluator,
)
from app.perfectframe.weights import (
    NimaWeightsUnavailableError,
    _download_nima_weights,
)


def _build_evaluator(weights_path: Path, session_mock: MagicMock) -> NIMAEvaluator:
    with patch("onnxruntime.InferenceSession", return_value=session_mock):
        return NIMAEvaluator(weights_path)


@pytest.fixture
def restore_nima_ort_settings():
    original = (
        settings.NIMA_ORT_INTRA_OP_THREADS,
        settings.NIMA_ORT_INTER_OP_THREADS,
        settings.NIMA_ORT_EXECUTION_MODE,
        settings.NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS,
    )
    try:
        yield
    finally:
        (
            settings.NIMA_ORT_INTRA_OP_THREADS,
            settings.NIMA_ORT_INTER_OP_THREADS,
            settings.NIMA_ORT_EXECUTION_MODE,
            settings.NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS,
        ) = original


def test_nima_evaluator_weighted_mean_returns_known_score(tmp_path: Path) -> None:
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="input_1", configure_mock=None)]
    session.get_inputs.return_value[0].name = "input_1"

    weights_path = tmp_path / "weights.onnx"
    weights_path.write_bytes(b"")

    evaluator = _build_evaluator(weights_path, session)

    uniform = np.full((1, 10), 0.1, dtype=np.float32)
    session.run.return_value = [uniform]

    images = np.zeros((1, 224, 224, 3), dtype=np.float32)
    scores = evaluator.evaluate_images(images)

    expected = float(np.sum(np.arange(1, 11) * 0.1) / np.sum(np.arange(1, 11)))
    assert len(scores) == 1
    assert scores[0] == pytest.approx(expected)


def test_nima_evaluator_handles_non_array_predictions(tmp_path: Path) -> None:
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input_1"
    session.run.return_value = ["not-an-array"]

    weights_path = tmp_path / "weights.onnx"
    weights_path.write_bytes(b"")

    evaluator = _build_evaluator(weights_path, session)

    images = np.zeros((1, 224, 224, 3), dtype=np.float32)
    assert evaluator.evaluate_images(images) == []


def test_nima_evaluator_initializes_cpu_session_options(tmp_path: Path) -> None:
    import onnxruntime as ort

    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input_1"
    weights_path = tmp_path / "weights.onnx"
    weights_path.write_bytes(b"")

    with patch("onnxruntime.InferenceSession", return_value=session) as session_ctor:
        evaluator = NIMAEvaluator(weights_path)

    _, kwargs = session_ctor.call_args
    sess_options = kwargs["sess_options"]
    assert kwargs["providers"] == ["CPUExecutionProvider"]
    assert sess_options.intra_op_num_threads == 4
    assert sess_options.inter_op_num_threads == 1
    assert sess_options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert (
        sess_options.graph_optimization_level
        == ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    assert evaluator.provider == "CPUExecutionProvider"
    assert evaluator.intra_op_num_threads == 4
    assert evaluator.inter_op_num_threads == 1
    assert evaluator.execution_mode == "sequential"
    assert evaluator.graph_optimization_level == "ORT_ENABLE_ALL"


def test_nima_evaluator_applies_setting_overrides(
    tmp_path: Path,
    restore_nima_ort_settings,
) -> None:
    import onnxruntime as ort

    settings.NIMA_ORT_INTRA_OP_THREADS = 3
    settings.NIMA_ORT_INTER_OP_THREADS = 2
    settings.NIMA_ORT_EXECUTION_MODE = "parallel"
    settings.NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS = False

    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input_1"
    weights_path = tmp_path / "weights.onnx"
    weights_path.write_bytes(b"")

    with patch("onnxruntime.InferenceSession", return_value=session) as session_ctor:
        evaluator = NIMAEvaluator(weights_path)

    _, kwargs = session_ctor.call_args
    sess_options = kwargs["sess_options"]
    assert sess_options.intra_op_num_threads == 3
    assert sess_options.inter_op_num_threads == 2
    assert sess_options.execution_mode == ort.ExecutionMode.ORT_PARALLEL
    assert (
        sess_options.graph_optimization_level
        == ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    assert evaluator.intra_op_num_threads == 3
    assert evaluator.inter_op_num_threads == 2
    assert evaluator.execution_mode == "parallel"
    assert evaluator.graph_optimization_level == "ORT_DISABLE_ALL"


def test_nima_ort_setting_normalization_falls_back_to_safe_defaults() -> None:
    from app.core.config import Settings

    normalized = Settings(
        POSTGRES_DB="db",
        POSTGRES_USER="user",
        POSTGRES_PASSWORD="pw",
        NIMA_ORT_INTRA_OP_THREADS=0,
        NIMA_ORT_INTER_OP_THREADS=-1,
        NIMA_ORT_EXECUTION_MODE="broken",
        NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS="true",
    )

    assert normalized.NIMA_ORT_INTRA_OP_THREADS == 4
    assert normalized.NIMA_ORT_INTER_OP_THREADS == 1
    assert normalized.NIMA_ORT_EXECUTION_MODE == "sequential"
    assert normalized.NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS is True


def test_build_image_evaluator_falls_back_when_weights_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.onnx"
    evaluator = build_image_evaluator(missing)
    assert isinstance(evaluator, SharpnessVarianceEvaluator)


def test_build_image_evaluator_falls_back_when_weights_path_is_none() -> None:
    evaluator = build_image_evaluator(None)
    assert isinstance(evaluator, SharpnessVarianceEvaluator)


def test_build_image_evaluator_falls_back_when_session_init_raises(
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "weights.onnx"
    weights_path.write_bytes(b"corrupt")

    with patch(
        "onnxruntime.InferenceSession",
        side_effect=RuntimeError("corrupt onnx"),
    ):
        evaluator = build_image_evaluator(weights_path)

    assert isinstance(evaluator, SharpnessVarianceEvaluator)


def test_build_image_evaluator_returns_nima_when_session_init_succeeds(
    tmp_path: Path,
) -> None:
    weights_path = tmp_path / "weights.onnx"
    weights_path.write_bytes(b"")

    session = MagicMock()
    session.get_inputs.return_value = [MagicMock()]
    session.get_inputs.return_value[0].name = "input_1"

    with patch("onnxruntime.InferenceSession", return_value=session):
        evaluator = build_image_evaluator(weights_path)

    assert isinstance(evaluator, NIMAEvaluator)


def test_sharpness_evaluator_returns_positive_scores_for_random_input() -> None:
    rng = np.random.default_rng(seed=42)
    images = rng.random((3, 8, 8, 3), dtype=np.float32)
    scores = SharpnessVarianceEvaluator().evaluate_images(images)
    assert len(scores) == 3
    assert all(score >= 0 for score in scores)


def test_download_nima_weights_writes_file_on_success(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.onnx"

    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.raise_for_status.return_value = None
    response.iter_bytes.return_value = [b"abc", b"def"]

    with patch("httpx.stream", return_value=response):
        _download_nima_weights(weights_path)

    assert weights_path.read_bytes() == b"abcdef"


def test_download_nima_weights_raises_on_http_error(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.onnx"

    with patch(
        "httpx.stream",
        side_effect=httpx.ConnectError("boom"),
    ):
        with pytest.raises(NimaWeightsUnavailableError):
            _download_nima_weights(weights_path)

    assert not weights_path.exists()


def test_download_nima_weights_raises_on_empty_response(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.onnx"

    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    response.raise_for_status.return_value = None
    response.iter_bytes.return_value = []

    with patch("httpx.stream", return_value=response):
        with pytest.raises(NimaWeightsUnavailableError):
            _download_nima_weights(weights_path)

    assert not weights_path.exists()
