from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from langgraph.graph import END, START, StateGraph
import numpy as np

from app.core.config import settings
from app.services.final_edit_planner import (
    DEFAULT_OWNER_PERSONA,
    ImageEditPlan,
    normalize_image_edit_plan,
)
from app.services.final_edit_filter_catalog import resolve_final_edit_filter_definition
from app.services.final_edit_runtime import import_cv2
from app.services.final_edit_tools import (
    build_aesthetic_color_grading_params,
    normalize_tool_params,
)

if TYPE_CHECKING:
    from app.services.final_edit_tools import FinalEditToolRegistry


@dataclass
class FinalEditImageState:
    image_path: Path
    working_dir: Path
    plan: ImageEditPlan
    owner_persona: str = DEFAULT_OWNER_PERSONA
    session_id: str = ""
    final_index: int = 0
    current_tool_index: int = 0
    current_image: np.ndarray | None = None
    output_path: Path | None = None
    fallback_to_original: bool = False
    failure_reason: str | None = None
    normalized_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    executed_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class FinalEditAgent:
    """LangGraph agent that applies an image edit plan to a single image."""

    def __init__(
        self,
        tool_registry: FinalEditToolRegistry,
        tool_event_sink: Callable[[str, FinalEditImageState, str], None] | None = None,
        upscale_enabled: bool | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.tool_event_sink = tool_event_sink
        self.upscale_enabled = (
            settings.FINAL_EDIT_ENABLE_UPSCALE
            if upscale_enabled is None
            else upscale_enabled
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(FinalEditImageState)
        graph.add_node("initialize", self._initialize)
        graph.add_node("validate_plan", self._validate_plan)
        graph.add_node("run_tool", self._run_tool)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "validate_plan")
        graph.add_conditional_edges(
            "validate_plan",
            self._should_continue,
            {
                "run_tool": "run_tool",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "run_tool",
            self._should_continue,
            {
                "run_tool": "run_tool",
                "finalize": "finalize",
            },
        )
        graph.add_edge("finalize", END)
        return graph.compile()

    def _initialize(self, state: FinalEditImageState) -> FinalEditImageState:
        cv2 = import_cv2()
        image = cv2.imread(str(state.image_path))
        if image is None:
            state.fallback_to_original = True
            state.failure_reason = "image_load_failed"
            state.output_path = state.image_path
            state.metadata["output_source"] = "original_fallback"
            return state
        state.current_image = image
        state.metadata["input_shape"] = self._shape_to_string(image)
        return state

    def _validate_plan(self, state: FinalEditImageState) -> FinalEditImageState:
        try:
            state.plan = normalize_image_edit_plan(
                state.plan,
                upscale_enabled=self.upscale_enabled,
                owner_persona=state.owner_persona,
            )
            state.normalized_params = {
                tool_name: normalize_tool_params(
                    tool_name,
                    state.plan.params.get(tool_name, {}),
                )
                for tool_name in state.plan.tools
            }
        except Exception as exc:
            state.fallback_to_original = True
            state.failure_reason = f"plan_validation_failed:{exc.__class__.__name__}"
            state.output_path = state.image_path
            state.metadata["output_source"] = "original_fallback"
        return state

    def _run_tool(self, state: FinalEditImageState) -> FinalEditImageState:
        if state.current_image is None:
            state.fallback_to_original = True
            state.failure_reason = state.failure_reason or "image_not_initialized"
            state.output_path = state.image_path
            state.metadata["output_source"] = "original_fallback"
            return state

        tool_name = state.plan.tools[state.current_tool_index]
        params = state.normalized_params.get(tool_name, {})
        if tool_name == "color_grading":
            state.metadata["aesthetic_tone_target_applied"] = False
            state.metadata["aesthetic_spotlight_protection_applied"] = False
        if tool_name == "color_grading" and state.owner_persona == DEFAULT_OWNER_PERSONA:
            params, tone_debug = build_aesthetic_color_grading_params(state.current_image)
            state.metadata["aesthetic_tone_target_applied"] = True
            state.metadata["aesthetic_spotlight_protection_applied"] = bool(
                params.get("spotlight_protection", False)
            )
            state.metadata["aesthetic_tone_debug"] = tone_debug
        if tool_name == "filter":
            try:
                filter_definition = resolve_final_edit_filter_definition(
                    params["filter_name"]
                )
            except Exception:
                state.metadata["current_tool_params"] = params
            else:
                state.metadata["current_tool_params"] = {
                    **params,
                    "backend": "pilgram2",
                    "callable": filter_definition.pilgram2_name,
                }
        else:
            state.metadata["current_tool_params"] = params
        state.metadata["current_tool_order"] = state.current_tool_index + 1
        state.metadata["current_tool_input_shape"] = self._shape_to_string(
            state.current_image
        )
        if self.tool_event_sink is not None:
            self.tool_event_sink("started", state, tool_name)
        try:
            state.current_image, tool_metadata = self.tool_registry.execute(
                tool_name,
                state.current_image,
                params,
            )
        except Exception as exc:
            state.fallback_to_original = True
            state.failure_reason = f"tool_failed:{tool_name}:{exc.__class__.__name__}"
            state.output_path = state.image_path
            state.metadata["output_source"] = "original_fallback"
            if self.tool_event_sink is not None:
                self.tool_event_sink("failed", state, tool_name)
            return state

        state.executed_tools.append(tool_name)
        state.metadata[f"tool:{state.current_tool_index}"] = tool_name
        if tool_metadata:
            for key, value in tool_metadata.items():
                state.metadata[key] = value
        state.metadata["current_tool_output_shape"] = self._shape_to_string(
            state.current_image
        )
        state.current_tool_index += 1
        if self.tool_event_sink is not None:
            self.tool_event_sink("completed", state, tool_name)
        return state

    def _should_continue(
        self,
        state: FinalEditImageState,
    ) -> Literal["run_tool", "finalize"]:
        if state.fallback_to_original:
            return "finalize"
        if state.current_tool_index < len(state.plan.tools):
            return "run_tool"
        return "finalize"

    def _finalize(self, state: FinalEditImageState) -> FinalEditImageState:
        if state.fallback_to_original or state.current_image is None:
            state.output_path = state.image_path
            state.metadata["output_source"] = "original_fallback"
            return state
        if not state.plan.tools:
            state.output_path = state.image_path
            state.metadata["output_source"] = "original_fallback"
            return state

        cv2 = import_cv2()
        state.working_dir.mkdir(parents=True, exist_ok=True)
        output_path = state.working_dir / f"edited-{state.plan.image_index:03d}.png"
        cv2.imwrite(str(output_path), state.current_image)
        state.output_path = output_path
        state.metadata["output_source"] = "edited"
        state.metadata["final_output_shape"] = self._shape_to_string(state.current_image)
        return state

    @staticmethod
    def _shape_to_string(image: np.ndarray) -> str:
        height, width = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        return f"{width}x{height}x{channels}"

    async def run(self, state: FinalEditImageState) -> FinalEditImageState:
        result = await self.graph.ainvoke(state)
        if isinstance(result, FinalEditImageState):
            return result
        return FinalEditImageState(**result)
