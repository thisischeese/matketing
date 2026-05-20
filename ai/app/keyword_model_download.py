"""Helpers for provisioning the local GGUF keyword model."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

DEFAULT_KEYWORD_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "exaone-3.5-7.8b-instruct"
    / "q4_k_m"
    / "model.gguf"
)
DEFAULT_KEYWORD_MODEL_HF_REPO_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF"
DEFAULT_KEYWORD_MODEL_HF_FILENAME = "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"

logger = logging.getLogger(__name__)


def _hf_hub_download(*, repo_id: str, filename: str, repo_type: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
    )


def ensure_keyword_model_available(model_path: Path | None = None) -> Path:
    """Ensure the default local GGUF model file exists."""

    model_path = model_path or DEFAULT_KEYWORD_MODEL_PATH

    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = model_path.with_suffix(f"{model_path.suffix}.download")

    logger.info(
        "Keyword model is missing; downloading the configured text-only GGUF.",
        extra={
            "keyword_model_repo_id": DEFAULT_KEYWORD_MODEL_HF_REPO_ID,
            "keyword_model_filename": DEFAULT_KEYWORD_MODEL_HF_FILENAME,
            "keyword_model_path": str(model_path),
        },
    )

    try:
        downloaded = Path(
            _hf_hub_download(
                repo_id=DEFAULT_KEYWORD_MODEL_HF_REPO_ID,
                filename=DEFAULT_KEYWORD_MODEL_HF_FILENAME,
                repo_type="model",
            )
        )
        shutil.copyfile(downloaded, temp_path)
        temp_path.replace(model_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Failed to download the keyword model GGUF from "
            f"{DEFAULT_KEYWORD_MODEL_HF_REPO_ID}/"
            f"{DEFAULT_KEYWORD_MODEL_HF_FILENAME} to {model_path}."
        ) from exc

    if not model_path.exists():
        raise RuntimeError(
            f"Keyword model download completed, but file is missing: {model_path}"
        )

    return model_path
