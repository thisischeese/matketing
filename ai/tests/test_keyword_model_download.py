from pathlib import Path

import pytest

from app import keyword_model_download
from app.keyword_model_download import (
    DEFAULT_KEYWORD_MODEL_HF_FILENAME,
    DEFAULT_KEYWORD_MODEL_HF_REPO_ID,
    ensure_keyword_model_available,
)


def test_ensure_keyword_model_available_returns_existing_file(tmp_path) -> None:
    model_path = tmp_path / "models" / "model.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"existing")

    resolved = ensure_keyword_model_available(model_path)

    assert resolved == model_path
    assert model_path.read_bytes() == b"existing"


def test_ensure_keyword_model_available_downloads_missing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "models" / "model.gguf"
    downloaded = tmp_path / "cache" / "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"downloaded")

    def fake_hf_hub_download(*, repo_id: str, filename: str, repo_type: str) -> str:
        assert repo_type == "model"
        assert repo_id == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF"
        assert filename == "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"
        return str(downloaded)

    monkeypatch.setattr(
        "app.keyword_model_download._hf_hub_download",
        fake_hf_hub_download,
    )

    resolved = ensure_keyword_model_available(model_path)

    assert resolved == model_path
    assert model_path.read_bytes() == b"downloaded"


def test_ensure_keyword_model_available_uses_default_path_when_not_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded = tmp_path / "cache" / "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"
    default_model_path = (
        tmp_path / "models" / "exaone-3.5-7.8b-instruct" / "q4_k_m" / "model.gguf"
    )
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"default")

    def fake_hf_hub_download(*, repo_id: str, filename: str, repo_type: str) -> str:
        assert repo_id == DEFAULT_KEYWORD_MODEL_HF_REPO_ID
        assert filename == DEFAULT_KEYWORD_MODEL_HF_FILENAME
        assert repo_type == "model"
        return str(downloaded)

    monkeypatch.setattr(keyword_model_download, "_hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(
        keyword_model_download,
        "DEFAULT_KEYWORD_MODEL_PATH",
        default_model_path,
    )

    resolved = ensure_keyword_model_available()

    assert resolved == default_model_path
    assert default_model_path.read_bytes() == b"default"
