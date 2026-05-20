from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.final_edit_agent import FinalEditAgent, FinalEditImageState
from app.services.final_edit_planner import ImageEditPlan
from app.services.final_edit_tools import (
    AESTHETIC_TARGET_TONE_PROFILE,
    AESTHETIC_TONE_METRIC_KEYS,
    FinalEditToolRegistry,
    analyze_image_tone,
    build_aesthetic_color_grading_params,
    normalize_tool_params,
)
from tests.utils.session_support import FakeUpscaler


def _write_test_image(path: Path, shape: tuple[int, int, int] = (16, 16, 3)) -> None:
    image = np.full(shape, 120, dtype=np.uint8)
    image[:, : shape[1] // 2, 2] = 220
    cv2.imwrite(str(path), image)


def _compute_luminance(image: np.ndarray) -> np.ndarray:
    return (
        0.114 * image[:, :, 0].astype(np.float32)
        + 0.587 * image[:, :, 1].astype(np.float32)
        + 0.299 * image[:, :, 2].astype(np.float32)
    )


def _mean_for_percentile(values: np.ndarray, percentile: float) -> float:
    flattened = values.reshape(-1)
    threshold = float(np.percentile(flattened, percentile))
    return float(np.mean(flattened[flattened >= threshold]))


@pytest.fixture(autouse=True)
def fake_realesrgan_upscaler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.final_edit_tools.get_realesrgan_upscaler",
        lambda: FakeUpscaler(),
    )


def test_normalize_tool_params_clamps_values() -> None:
    assert normalize_tool_params("upscale", {"scale": 8}) == {"scale": 2}
    assert normalize_tool_params("denoise", {"strength": 4}) == {"strength": 1.0}
    assert normalize_tool_params("filter", {"filter_name": "Valencia"}) == {
        "filter_name": "Valencia"
    }
    assert normalize_tool_params(
        "color_grading",
        {"temperature": "invalid", "saturation": -3, "brightness": 3},
    ) == {
        "contrast": 0.0,
        "highlights": 0.0,
        "shadows": 0.0,
        "vibrance": 0.0,
        "temperature": "warm",
        "temperature_strength": 1.0,
        "tint": 0.0,
        "tint_strength": 0.0,
        "saturation": -100.0,
        "tone_curve_shadow_lift": 0.0,
        "brightness": 1.0,
    }
    assert normalize_tool_params("background_blur", {"blur_radius": 99}) == {
        "blur_radius": 20
    }


def test_normalize_tool_params_supports_new_color_grading_fields() -> None:
    assert normalize_tool_params(
        "color_grading",
        {
            "contrast": 180,
            "highlights": -140,
            "shadows": 25,
            "vibrance": -12,
            "saturation": -18,
            "temperature": "cool",
            "tone_curve_shadow_lift": 130,
        },
    ) == {
        "contrast": 100.0,
        "highlights": -100.0,
        "shadows": 25.0,
        "vibrance": -12.0,
        "temperature": "cool",
        "temperature_strength": 1.0,
        "tint": 0.0,
        "tint_strength": 0.0,
        "saturation": -18.0,
        "tone_curve_shadow_lift": 100.0,
        "brightness": 0.0,
    }


def test_analyze_image_tone_returns_expected_metrics() -> None:
    image = np.full((32, 32, 3), 80, dtype=np.uint8)
    image[:, :16, 2] = 180
    image[:, 16:, 0] = 20

    metrics = analyze_image_tone(image)

    assert tuple(metrics.keys())[: len(AESTHETIC_TONE_METRIC_KEYS)] == (
        AESTHETIC_TONE_METRIC_KEYS
    )
    assert "highlight_area_ratio" in metrics
    assert all(isinstance(value, float) for value in metrics.values())
    assert metrics["white_point"] >= metrics["black_point"]


def test_build_aesthetic_color_grading_params_skips_small_gaps() -> None:
    image = np.dstack(
        [
            np.full((48, 48), 70, dtype=np.uint8),
            np.full((48, 48), 90, dtype=np.uint8),
            np.full((48, 48), 110, dtype=np.uint8),
        ]
    )

    params, debug = build_aesthetic_color_grading_params(image)

    assert debug["current_tone"] == analyze_image_tone(image)
    assert set(debug["gap_by_metric"]) == set(AESTHETIC_TONE_METRIC_KEYS)
    assert set(debug["skip_by_metric"]) == set(AESTHETIC_TONE_METRIC_KEYS)
    assert debug["skip_by_metric"]["temperature"] is True
    assert debug["skip_by_metric"]["tint"] is True
    assert params["temperature_strength"] == 0.0
    assert params["tint_strength"] == 0.0


def test_analyze_image_tone_uses_trimmed_brightness_for_small_white_patch() -> None:
    image = np.full((96, 96, 3), 90, dtype=np.uint8)
    image[4:10, 4:10] = 255

    metrics = analyze_image_tone(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    raw_mean_brightness = round(float(np.mean(l_channel)), 2)
    baseline_metrics = analyze_image_tone(np.full((96, 96, 3), 90, dtype=np.uint8))

    assert metrics["brightness"] < raw_mean_brightness
    assert metrics["brightness"] == baseline_metrics["brightness"]
    assert metrics["highlight_area_ratio"] < 0.03


def test_build_aesthetic_color_grading_params_attenuates_small_bright_highlights() -> None:
    small_highlights = np.full((128, 128, 3), 95, dtype=np.uint8)
    small_highlights[8:31, 8:31] = 255
    small_highlights[8:30, 36:56] = 220
    broad_overexposure = np.full((128, 128, 3), 235, dtype=np.uint8)

    small_params, small_debug = build_aesthetic_color_grading_params(small_highlights)
    broad_params, broad_debug = build_aesthetic_color_grading_params(broad_overexposure)
    highlight_target = AESTHETIC_TARGET_TONE_PROFILE["white_point"]
    raw_highlight_adjustment = max(
        -100.0,
        min(
            100.0,
            (
                (highlight_target["target"] - small_debug["current_tone"]["white_point"])
                / max(highlight_target["tolerance"], 1.0)
            )
            * 45.0,
        ),
    )

    assert (
        small_debug["current_tone"]["highlight_area_ratio"]
        < broad_debug["current_tone"]["highlight_area_ratio"]
    )
    assert small_params["brightness"] == 0.0
    assert small_params["highlights"] == round(raw_highlight_adjustment * 0.35, 2)
    assert small_params["spotlight_protection"] is True
    assert abs(small_params["brightness"]) < abs(broad_params["brightness"])


def test_build_aesthetic_color_grading_params_zeroes_highlights_for_sub_threshold_spotlight() -> None:
    image = np.full((128, 128, 3), 95, dtype=np.uint8)
    image[8:28, 8:28] = 255
    image[8:33, 36:56] = 220

    params, debug = build_aesthetic_color_grading_params(image)

    assert debug["current_tone"]["highlight_area_ratio"] < 0.03
    assert debug["current_tone"]["white_point"] >= AESTHETIC_TARGET_TONE_PROFILE["white_point"]["target"]
    assert params["brightness"] == 0.0
    assert params["highlights"] == 0.0
    assert params["spotlight_protection"] is True


def test_tool_registry_upscale_changes_shape() -> None:
    registry = FinalEditToolRegistry(upscale_enabled=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    result, metadata = registry.execute("upscale", image, {"scale": 4})

    assert result.shape == (16, 16, 3)
    assert metadata == {}


def test_tool_registry_excludes_upscale_when_disabled() -> None:
    registry = FinalEditToolRegistry(upscale_enabled=False)

    assert "upscale" not in registry.tool_names


def test_normalize_tool_params_upscale_forces_two_x() -> None:
    assert normalize_tool_params("upscale", {"scale": 4}) == {"scale": 2}


def test_tool_registry_background_blur_preserves_shape() -> None:
    registry = FinalEditToolRegistry()
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result, metadata = registry.execute("background_blur", image, {"blur_radius": 4})

    assert result.shape == image.shape
    assert metadata == {}


def test_tool_registry_filter_applies_pixels_and_records_metadata() -> None:
    registry = FinalEditToolRegistry()
    image = np.full((12, 12, 3), 110, dtype=np.uint8)
    image[:, :6, 2] = 220

    result, metadata = registry.execute("filter", image, {"filter_name": "Reyes"})

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not np.array_equal(result, image)
    assert metadata == {
        "filter_applied": True,
        "selected_filter": "Reyes",
        "selected_filter_backend": "pilgram2",
        "selected_filter_callable": "reyes",
    }


def test_tool_registry_color_grading_changes_pixels_and_preserves_dtype() -> None:
    registry = FinalEditToolRegistry()
    image = np.full((16, 16, 3), 90, dtype=np.uint8)
    image[:, :, 2] = 140
    image[:, 8:, :] = 200

    result, metadata = registry.execute(
        "color_grading",
        image,
        {
            "contrast": -18,
            "highlights": -12,
            "shadows": 20,
            "vibrance": -25,
            "saturation": -18,
            "temperature": "cool",
            "tone_curve_shadow_lift": 12,
        },
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.any(result != image)
    assert metadata == {}


def test_tool_registry_color_grading_preserves_specular_core_with_spotlight_protection() -> None:
    registry = FinalEditToolRegistry()
    image = np.full((128, 128, 3), 110, dtype=np.uint8)
    image[8:28, 8:28] = 255
    image[8:30, 36:56] = 220

    result, metadata = registry.execute(
        "color_grading",
        image,
        {
            "contrast": 0,
            "highlights": -12,
            "shadows": 0,
            "vibrance": 0,
            "saturation": 0,
            "temperature": "warm",
            "temperature_strength": 0.0,
            "tint": 0,
            "tint_strength": 0.0,
            "tone_curve_shadow_lift": 0,
            "brightness": 0.0,
            "spotlight_protection": True,
        },
    )

    input_luminance = _compute_luminance(image)
    result_luminance = _compute_luminance(result)
    input_top_one_percent = _mean_for_percentile(input_luminance, 99.0)
    result_top_one_percent = _mean_for_percentile(result_luminance, 99.0)
    input_midtone_mean = float(np.mean(input_luminance[(input_luminance >= 100.0) & (input_luminance <= 140.0)]))
    result_midtone_mean = float(np.mean(result_luminance[(input_luminance >= 100.0) & (input_luminance <= 140.0)]))

    assert result_top_one_percent >= input_top_one_percent
    assert result_midtone_mean <= input_midtone_mean + 1.0
    assert metadata == {}


@pytest.mark.asyncio
async def test_final_edit_agent_overrides_aesthetic_color_grading_params(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    agent = FinalEditAgent(FinalEditToolRegistry())
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        owner_persona="aesthetic",
        plan=ImageEditPlan(
            image_index=0,
            content="케이크",
            strategy="tone gap 보정",
            tools=["color_grading"],
            params={
                "color_grading": {
                    "contrast": 99,
                    "highlights": 99,
                    "shadows": 99,
                    "vibrance": 99,
                    "saturation": 99,
                    "temperature": "warm",
                    "tone_curve_shadow_lift": 99,
                }
            },
        ),
    )

    result = await agent.run(state)

    assert result.fallback_to_original is False
    assert result.metadata["aesthetic_tone_target_applied"] is True
    assert "aesthetic_spotlight_protection_applied" in result.metadata
    assert "aesthetic_tone_debug" in result.metadata
    assert result.metadata["current_tool_params"] != normalize_tool_params(
        "color_grading",
        state.plan.params["color_grading"],
    )


@pytest.mark.asyncio
async def test_final_edit_agent_keeps_planner_params_for_non_aesthetic(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    agent = FinalEditAgent(FinalEditToolRegistry())
    planned_params = {
        "contrast": -8,
        "highlights": -10,
        "shadows": 6,
        "vibrance": -4,
        "saturation": -6,
        "temperature": "warm",
        "tone_curve_shadow_lift": 4,
    }
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        owner_persona="friendly",
        plan=ImageEditPlan(
            image_index=0,
            content="케이크",
            strategy="friendly 보정",
            tools=["color_grading"],
            params={"color_grading": planned_params},
        ),
    )

    result = await agent.run(state)

    assert result.fallback_to_original is False
    assert result.metadata["aesthetic_tone_target_applied"] is False
    assert result.metadata["aesthetic_spotlight_protection_applied"] is False
    assert result.normalized_params["color_grading"] == normalize_tool_params(
        "color_grading",
        planned_params,
    )
    assert result.executed_tools[0] == "color_grading"
    assert "aesthetic_tone_debug" not in result.metadata


@pytest.mark.asyncio
async def test_final_edit_agent_writes_edited_image(tmp_path: Path) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    agent = FinalEditAgent(
        FinalEditToolRegistry(upscale_enabled=True),
        upscale_enabled=True,
    )
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        plan=ImageEditPlan(
            image_index=0,
            content="딸기 케이크",
            strategy="업스케일 후 샤프닝",
            tools=["upscale", "sharpen"],
            params={
                "upscale": {"scale": 2},
                "sharpen": {"strength": 0.4},
            },
        ),
    )

    result = await agent.run(state)

    assert result.output_path is not None
    assert result.output_path.exists()
    output_image = cv2.imread(str(result.output_path))
    assert output_image is not None
    assert output_image.shape[:2] == (32, 32)
    assert result.fallback_to_original is False


@pytest.mark.asyncio
async def test_final_edit_agent_strips_disabled_upscale_before_execution(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    agent = FinalEditAgent(
        FinalEditToolRegistry(upscale_enabled=False),
        upscale_enabled=False,
    )
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        plan=ImageEditPlan(
            image_index=0,
            content="cake",
            strategy="upscale only",
            tools=["upscale"],
            params={"upscale": {"scale": 2}},
        ),
        owner_persona="friendly",
    )

    result = await agent.run(state)

    assert result.fallback_to_original is False
    assert result.executed_tools == []
    assert result.output_path == image_path


@pytest.mark.asyncio
async def test_final_edit_agent_records_selected_filter_metadata(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    agent = FinalEditAgent(
        FinalEditToolRegistry(upscale_enabled=True),
        upscale_enabled=True,
    )
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        plan=ImageEditPlan(
            image_index=0,
            content="cake",
            strategy="warm cafe mood",
            tools=["filter"],
            params={"filter": {"filter_name": "Valencia"}},
        ),
    )

    result = await agent.run(state)

    assert result.fallback_to_original is False
    assert result.executed_tools == ["filter", "upscale"]
    assert result.metadata["selected_filter"] == "Valencia"
    assert result.metadata["filter_applied"] is True
    assert result.metadata["selected_filter_backend"] == "pilgram2"
    assert result.metadata["selected_filter_callable"] == "valencia"
    assert result.output_path is not None
    output_image = cv2.imread(str(result.output_path))
    assert output_image is not None
    assert output_image.shape[:2] == (32, 32)


@pytest.mark.asyncio
async def test_final_edit_agent_falls_back_when_filter_callable_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    monkeypatch.setattr(
        "app.services.final_edit_tools._import_pilgram2",
        lambda: object(),
    )
    agent = FinalEditAgent(FinalEditToolRegistry())
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        plan=ImageEditPlan(
            image_index=0,
            content="cake",
            strategy="warm cafe mood",
            tools=["filter"],
            params={"filter": {"filter_name": "Valencia"}},
        ),
    )

    result = await agent.run(state)

    assert result.fallback_to_original is True
    assert result.output_path == image_path
    assert "tool_failed:filter" in (result.failure_reason or "")


@pytest.mark.asyncio
async def test_final_edit_agent_falls_back_when_tool_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "draft.png"
    _write_test_image(image_path)
    registry = FinalEditToolRegistry()
    original_execute = registry.execute

    def failing_execute(tool_name, image, params):
        if tool_name == "denoise":
            raise RuntimeError("boom")
        return original_execute(tool_name, image, params)

    registry.execute = failing_execute  # type: ignore[method-assign]
    agent = FinalEditAgent(registry)
    state = FinalEditImageState(
        image_path=image_path,
        working_dir=tmp_path / "edited",
        plan=ImageEditPlan(
            image_index=0,
            content="매장 전경",
            strategy="디노이즈",
            tools=["denoise"],
            params={"denoise": {"strength": 0.4}},
        ),
    )

    result = await agent.run(state)

    assert result.fallback_to_original is True
    assert result.output_path == image_path
    assert "tool_failed:denoise" in (result.failure_reason or "")
