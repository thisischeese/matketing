from pathlib import Path

import pytest

from app.core.config import (
    DEFAULT_CAPTION_MODEL_HEALTH_ENDPOINT,
    DEFAULT_CAPTION_MODEL_SERVER_PORT,
    DEFAULT_CAPTION_MODEL_TIMEOUT_SECONDS,
    DEFAULT_CANONICAL_KEYWORD_EMBEDDING_DIM,
    DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_CACHE_DIR,
    DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME,
    DEFAULT_KEYWORD_MODEL_HEALTH_ENDPOINT,
    DEFAULT_KEYWORD_MODEL_SERVER_PORT,
    DEFAULT_KEYWORD_MODEL_TIMEOUT_SECONDS,
    Settings,
)


@pytest.fixture
def env_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DB", "testdb")
    monkeypatch.setenv("POSTGRES_USER", "tester")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("REDIS_PASSWORD", "rsecret")
    monkeypatch.setenv("SESSION_TTL_SECONDS", "1800")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("S3_ACCESS_KEY", "access")
    monkeypatch.setenv("S3_SECRET_KEY", "secret")
    monkeypatch.setenv("S3_BUCKET_NAME", "bucket")
    monkeypatch.setenv("S3_REGION", "ap-northeast-2")
    monkeypatch.setenv("CLOUDFRONT_DOMAIN", "cdn.example.com")
    monkeypatch.setenv("KEYWORD_MODEL_BASE_URL", "http://keyword-server:8001")
    monkeypatch.setenv("KEYWORD_MODEL_NAME", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ")
    monkeypatch.setenv("KEYWORD_MODEL_CHAT_ENDPOINT", "/v1/chat/completions")
    monkeypatch.setenv("KEYWORD_MODEL_HEALTH_ENDPOINT", "/ready")
    monkeypatch.setenv("KEYWORD_MODEL_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("KEYWORD_MODEL_MAX_TOKENS", "32")
    monkeypatch.setenv("KEYWORD_MODEL_TEMPERATURE", "0.2")
    monkeypatch.setenv("KEYWORD_MODEL_TOP_P", "0.85")
    monkeypatch.setenv("KEYWORD_MODEL_ENABLED", "true")
    monkeypatch.setenv("CAPTION_MODEL_BASE_URL", "http://caption-server:8002")
    monkeypatch.setenv("CAPTION_MODEL_NAME", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ")
    monkeypatch.setenv("CAPTION_MODEL_CHAT_ENDPOINT", "/v1/chat/completions")
    monkeypatch.setenv("CAPTION_MODEL_HEALTH_ENDPOINT", "/readyz")
    monkeypatch.setenv("CAPTION_MODEL_TIMEOUT_SECONDS", "75")
    monkeypatch.setenv("CAPTION_MODEL_MAX_TOKENS", "96")
    monkeypatch.setenv("CAPTION_MODEL_TEMPERATURE", "0.6")
    monkeypatch.setenv("CAPTION_MODEL_TOP_P", "0.88")
    monkeypatch.setenv("CAPTION_MODEL_ENABLED", "true")
    monkeypatch.setenv("FINAL_EDIT_ENABLE_UPSCALE", "true")
    monkeypatch.setenv("CANONICAL_KEYWORD_RESOLVER_ENABLED", "true")
    monkeypatch.setenv(
        "CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME",
        "intfloat/multilingual-e5-small",
    )
    monkeypatch.setenv("CANONICAL_KEYWORD_EMBEDDING_DIM", "384")


def _set_required_postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_DB", "db")
    monkeypatch.setenv("POSTGRES_USER", "u")
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")


def test_settings_parses_infra_fields(env_setup: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.POSTGRES_HOST == "project-postgres"
    assert settings.POSTGRES_PORT == 5432
    assert settings.POSTGRES_SCHEMA == "public"
    assert settings.REDIS_HOST == "project-redis"
    assert settings.REDIS_PORT == 6379
    assert settings.REDIS_DB == 0
    assert settings.SESSION_TTL_SECONDS == 1800
    assert settings.DEBUG is True
    assert settings.LOG_INCLUDE_RAW_IDENTIFIERS is True
    assert settings.LOG_EVENT_PREVIEW_MAX_LEN == 200

    assert settings.keyword_model_client.base_url == "http://keyword-server:8001"
    assert (
        settings.keyword_model_client.model_name
        == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"
    )
    assert settings.keyword_model_client.chat_endpoint == "/v1/chat/completions"
    assert settings.keyword_model_client.health_endpoint == "/ready"
    assert settings.keyword_model_client.timeout_seconds == 45
    assert settings.keyword_model_client.max_tokens == 32
    assert settings.keyword_model_client.temperature == 0.2
    assert settings.keyword_model_client.top_p == 0.85
    assert settings.keyword_model_client.enabled is True

    assert settings.caption_model_client.base_url == "http://caption-server:8002"
    assert (
        settings.caption_model_client.model_name
        == "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"
    )
    assert settings.caption_model_client.chat_endpoint == "/v1/chat/completions"
    assert settings.caption_model_client.health_endpoint == "/readyz"
    assert settings.caption_model_client.timeout_seconds == 75
    assert settings.caption_model_client.max_tokens == 96
    assert settings.caption_model_client.temperature == 0.6
    assert settings.caption_model_client.top_p == 0.88
    assert settings.caption_model_client.enabled is True
    assert settings.FINAL_EDIT_ENABLE_UPSCALE is True

    assert settings.CANONICAL_KEYWORD_RESOLVER_ENABLED is True
    assert (
        settings.CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME
        == "intfloat/multilingual-e5-small"
    )
    assert settings.CANONICAL_KEYWORD_EMBEDDING_DIM == 384
    assert settings.postgres_search_path == "public"


def test_postgres_dsn_format(env_setup: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.postgres_dsn == (
        "postgresql+asyncpg://tester:secret@project-postgres:5432/testdb"
    )


def test_postgres_search_path_uses_custom_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("POSTGRES_SCHEMA", "custom_schema")

    settings = Settings(_env_file=None)

    assert settings.postgres_search_path == "custom_schema,public"


def test_redis_url_with_password(env_setup: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.redis_url == "redis://:rsecret@project-redis:6379/0"


def test_redis_url_without_password(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)
    monkeypatch.delenv("REDIS_DB", raising=False)
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)

    settings = Settings(_env_file=None)

    assert settings.redis_url == "redis://project-redis:6379/0"


def test_default_container_hosts_and_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("POSTGRES_PORT", raising=False)
    monkeypatch.delenv("REDIS_HOST", raising=False)
    monkeypatch.delenv("REDIS_PORT", raising=False)

    settings = Settings(_env_file=None)

    assert settings.POSTGRES_HOST == "project-postgres"
    assert settings.POSTGRES_PORT == 5432
    assert settings.REDIS_HOST == "project-redis"
    assert settings.REDIS_PORT == 6379


def test_hosts_and_ports_can_still_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("POSTGRES_HOST", "test_pg")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("REDIS_HOST", "test_redis")
    monkeypatch.setenv("REDIS_PORT", "6380")

    settings = Settings(_env_file=None)

    assert settings.POSTGRES_HOST == "test_pg"
    assert settings.POSTGRES_PORT == 6543
    assert settings.REDIS_HOST == "test_redis"
    assert settings.REDIS_PORT == 6380


def test_redis_db_defaults_to_zero_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.delenv("REDIS_DB", raising=False)

    settings = Settings(_env_file=None)

    assert settings.REDIS_DB == 0
    assert settings.redis_url.endswith("/0")


def test_settings_load_from_env_file_with_only_shared_db_vars(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "POSTGRES_DB=testdb",
                "POSTGRES_USER=tester",
                "POSTGRES_PASSWORD=secret",
                "REDIS_PASSWORD=rsecret",
            ]
        ),
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.POSTGRES_HOST == "project-postgres"
    assert settings.POSTGRES_PORT == 5432
    assert settings.REDIS_HOST == "project-redis"
    assert settings.REDIS_PORT == 6379
    assert settings.REDIS_DB == 0
    assert settings.postgres_dsn == (
        "postgresql+asyncpg://tester:secret@project-postgres:5432/testdb"
    )
    assert settings.redis_url == "redis://:rsecret@project-redis:6379/0"


def test_s3_configured_when_required_fields_exist(env_setup: None) -> None:
    settings = Settings(_env_file=None)

    assert settings.s3_configured is True
    assert settings.CLOUDFRONT_DOMAIN == "cdn.example.com"


def test_model_defaults_resolve_without_env(env_setup: None) -> None:
    assert DEFAULT_KEYWORD_MODEL_SERVER_PORT == 8001
    assert DEFAULT_KEYWORD_MODEL_TIMEOUT_SECONDS == 60.0
    assert DEFAULT_KEYWORD_MODEL_HEALTH_ENDPOINT == "/health"

    assert DEFAULT_CAPTION_MODEL_SERVER_PORT == 8002
    assert DEFAULT_CAPTION_MODEL_TIMEOUT_SECONDS == 600.0
    assert DEFAULT_CAPTION_MODEL_HEALTH_ENDPOINT == "/health"

    assert (
        DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME
        == "intfloat/multilingual-e5-small"
    )
    assert DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_CACHE_DIR.name == "canonical-keywords"
    assert DEFAULT_CANONICAL_KEYWORD_EMBEDDING_DIM == 384


def test_model_client_defaults_apply_when_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.delenv("KEYWORD_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("KEYWORD_MODEL_CHAT_ENDPOINT", raising=False)
    monkeypatch.delenv("KEYWORD_MODEL_NAME", raising=False)
    monkeypatch.delenv("KEYWORD_MODEL_HEALTH_ENDPOINT", raising=False)
    monkeypatch.delenv("KEYWORD_MODEL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("CAPTION_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("CAPTION_MODEL_CHAT_ENDPOINT", raising=False)
    monkeypatch.delenv("CAPTION_MODEL_NAME", raising=False)
    monkeypatch.delenv("CAPTION_MODEL_HEALTH_ENDPOINT", raising=False)
    monkeypatch.delenv("CAPTION_MODEL_TIMEOUT_SECONDS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.keyword_model_client.base_url == "http://keyword-server:8001"
    assert settings.keyword_model_client.model_name is None
    assert settings.keyword_model_client.chat_endpoint == "/v1/chat/completions"
    assert (
        settings.keyword_model_client.health_endpoint
        == DEFAULT_KEYWORD_MODEL_HEALTH_ENDPOINT
    )
    assert (
        settings.keyword_model_client.timeout_seconds
        == DEFAULT_KEYWORD_MODEL_TIMEOUT_SECONDS
    )

    assert settings.caption_model_client.base_url == "http://caption-server:8002"
    assert settings.caption_model_client.model_name is None
    assert settings.caption_model_client.chat_endpoint == "/v1/chat/completions"
    assert (
        settings.caption_model_client.health_endpoint
        == DEFAULT_CAPTION_MODEL_HEALTH_ENDPOINT
    )
    assert (
        settings.caption_model_client.timeout_seconds
        == DEFAULT_CAPTION_MODEL_TIMEOUT_SECONDS
    )
    assert settings.caption_model_client.max_tokens == 512


def test_reference_caption_rag_defaults_apply_when_env_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.delenv("REFERENCE_CAPTION_RAG_ENABLED", raising=False)
    monkeypatch.delenv("REFERENCE_CAPTION_MAX_REFERENCES", raising=False)

    settings = Settings(_env_file=None)

    assert settings.REFERENCE_CAPTION_RAG_ENABLED is True
    assert settings.REFERENCE_CAPTION_MAX_REFERENCES == 2


def test_reference_caption_rag_env_vars_are_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("REFERENCE_CAPTION_RAG_ENABLED", "false")
    monkeypatch.setenv("REFERENCE_CAPTION_MAX_REFERENCES", "4")

    settings = Settings(_env_file=None)

    assert settings.REFERENCE_CAPTION_RAG_ENABLED is False
    assert settings.REFERENCE_CAPTION_MAX_REFERENCES == 4


def test_final_edit_upscale_flag_env_var_is_applied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("FINAL_EDIT_ENABLE_UPSCALE", "false")

    settings = Settings(_env_file=None)

    assert settings.FINAL_EDIT_ENABLE_UPSCALE is False


def test_model_tunable_env_vars_are_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("KEYWORD_MODEL_BASE_URL", "http://host.docker.internal:8080")
    monkeypatch.setenv("KEYWORD_MODEL_NAME", "exaone-text")
    monkeypatch.setenv("KEYWORD_MODEL_CHAT_ENDPOINT", "/v1/chat/completions")
    monkeypatch.setenv("KEYWORD_MODEL_HEALTH_ENDPOINT", "/healthz")
    monkeypatch.setenv("KEYWORD_MODEL_TIMEOUT_SECONDS", "15")
    monkeypatch.setenv("KEYWORD_MODEL_MAX_TOKENS", "16")
    monkeypatch.setenv("KEYWORD_MODEL_TEMPERATURE", "0.3")
    monkeypatch.setenv("KEYWORD_MODEL_TOP_P", "0.75")
    monkeypatch.setenv("KEYWORD_MODEL_ENABLED", "false")
    monkeypatch.setenv("CAPTION_MODEL_BASE_URL", "http://host.docker.internal:9090")
    monkeypatch.setenv("CAPTION_MODEL_NAME", "exaone-text")
    monkeypatch.setenv("CAPTION_MODEL_HEALTH_ENDPOINT", "/caption-health")
    monkeypatch.setenv("CAPTION_MODEL_TIMEOUT_SECONDS", "25")
    monkeypatch.setenv("CAPTION_MODEL_MAX_TOKENS", "120")
    monkeypatch.setenv("CAPTION_MODEL_TEMPERATURE", "0.55")
    monkeypatch.setenv("CAPTION_MODEL_TOP_P", "0.82")
    monkeypatch.setenv("CAPTION_MODEL_ENABLED", "false")

    settings = Settings(_env_file=None)

    assert settings.keyword_model_client.base_url == "http://host.docker.internal:8080"
    assert settings.keyword_model_client.model_name == "exaone-text"
    assert settings.keyword_model_client.health_endpoint == "/healthz"
    assert settings.keyword_model_client.timeout_seconds == 15
    assert settings.keyword_model_client.max_tokens == 16
    assert settings.keyword_model_client.temperature == 0.3
    assert settings.keyword_model_client.top_p == 0.75
    assert settings.keyword_model_client.enabled is False

    assert settings.caption_model_client.base_url == "http://host.docker.internal:9090"
    assert settings.caption_model_client.model_name == "exaone-text"
    assert settings.caption_model_client.health_endpoint == "/caption-health"
    assert settings.caption_model_client.timeout_seconds == 25
    assert settings.caption_model_client.max_tokens == 120
    assert settings.caption_model_client.temperature == 0.55
    assert settings.caption_model_client.top_p == 0.82
    assert settings.caption_model_client.enabled is False


def test_final_edit_model_name_matches_env_example(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.delenv("FINAL_EDIT_MODEL_NAME", raising=False)

    settings = Settings(_env_file=None)
    env_example = Path(__file__).resolve().parents[1] / ".env.example"
    assert (
        f"FINAL_EDIT_MODEL_NAME={settings.FINAL_EDIT_MODEL_NAME}"
        in env_example.read_text(encoding="utf-8")
    )


def test_s3_not_configured_when_bucket_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("S3_ACCESS_KEY", "access")
    monkeypatch.setenv("S3_SECRET_KEY", "secret")
    monkeypatch.setenv("S3_REGION", "ap-northeast-2")
    monkeypatch.delenv("S3_BUCKET_NAME", raising=False)

    settings = Settings(_env_file=None)

    assert settings.s3_configured is False


def test_log_identifier_policy_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required_postgres(monkeypatch)
    monkeypatch.setenv("LOG_INCLUDE_RAW_IDENTIFIERS", "false")
    monkeypatch.setenv("LOG_EVENT_PREVIEW_MAX_LEN", "80")

    settings = Settings(_env_file=None)

    assert settings.LOG_INCLUDE_RAW_IDENTIFIERS is False
    assert settings.LOG_EVENT_PREVIEW_MAX_LEN == 80
