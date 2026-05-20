from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR.parent / ".env"

DEFAULT_KEYWORD_MODEL_SERVER_PORT = 8001
DEFAULT_KEYWORD_MODEL_HOST_DIR = "./.models/keyword/exaone-3.5-7.8b-instruct/q4_k_m"
DEFAULT_KEYWORD_MODEL_PATH = (
    ROOT_DIR / "models" / "exaone-3.5-7.8b-instruct" / "q4_k_m" / "model.gguf"
)
DEFAULT_KEYWORD_MODEL_HF_REPO_ID = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF"
DEFAULT_KEYWORD_MODEL_HF_FILENAME = "EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"
DEFAULT_KEYWORD_MODEL_TIMEOUT_SECONDS = 60.0
DEFAULT_KEYWORD_MODEL_HEALTH_ENDPOINT = "/health"

DEFAULT_CAPTION_MODEL_SERVER_PORT = 8002
DEFAULT_CAPTION_MODEL_HOST_DIR = "./.models/caption/exaone-4.0-32b/q4_k_m"
DEFAULT_CAPTION_MODEL_PATH = (
    ROOT_DIR / "models" / "exaone-4.0-32b" / "q4_k_m" / "model.gguf"
)
DEFAULT_CAPTION_MODEL_HF_REPO_ID = "LGAI-EXAONE/EXAONE-4.0-32B-GGUF"
DEFAULT_CAPTION_MODEL_HF_FILENAME = "EXAONE-4.0-32B-Q4_K_M.gguf"
DEFAULT_CAPTION_MODEL_TIMEOUT_SECONDS = 600.0
DEFAULT_CAPTION_MODEL_HEALTH_ENDPOINT = "/health"
DEFAULT_FINAL_EDIT_MODEL_SERVER_PORT = 49155
DEFAULT_FINAL_EDIT_MODEL_TIMEOUT_SECONDS = 180.0
DEFAULT_FINAL_EDIT_MODEL_HEALTH_ENDPOINT = "/health"

DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_CACHE_DIR = (
    ROOT_DIR / "models" / "canonical-keywords"
)
DEFAULT_CANONICAL_KEYWORD_EMBEDDING_DIM = 384

DEFAULT_NIMA_WEIGHTS_PATH = ROOT_DIR / "models" / "nima" / "weights.onnx"
DEFAULT_NIMA_WEIGHTS_URL = (
    "https://huggingface.co/BKDDFS/nima_weights/resolve/main/weights.onnx"
)
DEFAULT_NIMA_ORT_INTRA_OP_THREADS = 4
DEFAULT_NIMA_ORT_INTER_OP_THREADS = 1
DEFAULT_NIMA_ORT_EXECUTION_MODE = "sequential"
DEFAULT_NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS = True
DEFAULT_REALESRGAN_WEIGHTS_PATH = (
    ROOT_DIR / "models" / "realesrgan" / "RealESRGAN_x2plus.pth"
)
DEFAULT_REALESRGAN_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/"
    "RealESRGAN_x2plus.pth"
)


@dataclass(frozen=True)
class LlamaModelClientSettings:
    base_url: str
    model_name: str | None
    chat_endpoint: str
    health_endpoint: str
    api_key: str | None
    timeout_seconds: float
    max_tokens: int
    temperature: float
    top_p: float
    enabled: bool


@dataclass(frozen=True)
class LlamaModelServerSettings:
    host_dir: str
    hf_repo_id: str
    hf_filename: str
    server_port: int
    ctx_size: int
    gpu_layers: int


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "AI Service"
    ENVIRONMENT: Literal["local", "staging", "production"] = "local"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "DEBUG"
    LOG_INCLUDE_RAW_IDENTIFIERS: bool = True
    LOG_EVENT_PREVIEW_MAX_LEN: int = 200

    DEBUG: bool = True

    POSTGRES_HOST: str = "project-postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_SCHEMA: str = "public"

    REDIS_HOST: str = "project-redis"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0

    SESSION_TTL_SECONDS: int = 3600

    S3_SECRET_KEY: str | None = None
    S3_ACCESS_KEY: str | None = None
    S3_BUCKET_NAME: str | None = None
    S3_REGION: str | None = None
    CLOUDFRONT_DOMAIN: str | None = None

    NIMA_WEIGHTS_URL: str = DEFAULT_NIMA_WEIGHTS_URL
    NIMA_ORT_INTRA_OP_THREADS: int = DEFAULT_NIMA_ORT_INTRA_OP_THREADS
    NIMA_ORT_INTER_OP_THREADS: int = DEFAULT_NIMA_ORT_INTER_OP_THREADS
    NIMA_ORT_EXECUTION_MODE: str = DEFAULT_NIMA_ORT_EXECUTION_MODE
    NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS: bool = (
        DEFAULT_NIMA_ORT_ENABLE_ALL_OPTIMIZATIONS
    )
    REALESRGAN_WEIGHTS_URL: str = DEFAULT_REALESRGAN_WEIGHTS_URL

    KEYWORD_MODEL_HOST_DIR: str = DEFAULT_KEYWORD_MODEL_HOST_DIR
    KEYWORD_MODEL_HF_REPO_ID: str = DEFAULT_KEYWORD_MODEL_HF_REPO_ID
    KEYWORD_MODEL_HF_FILENAME: str = DEFAULT_KEYWORD_MODEL_HF_FILENAME
    KEYWORD_MODEL_SERVER_PORT: int = DEFAULT_KEYWORD_MODEL_SERVER_PORT
    KEYWORD_MODEL_BASE_URL: str = (
        f"http://keyword-server:{DEFAULT_KEYWORD_MODEL_SERVER_PORT}"
    )
    KEYWORD_MODEL_NAME: str | None = None
    KEYWORD_MODEL_CHAT_ENDPOINT: str = "/v1/chat/completions"
    KEYWORD_MODEL_HEALTH_ENDPOINT: str = DEFAULT_KEYWORD_MODEL_HEALTH_ENDPOINT
    KEYWORD_MODEL_API_KEY: str | None = None
    KEYWORD_MODEL_CTX_SIZE: int = 2048
    KEYWORD_MODEL_GPU_LAYERS: int = 20
    KEYWORD_MODEL_MAX_TOKENS: int = 64
    KEYWORD_MODEL_TEMPERATURE: float = 0.1
    KEYWORD_MODEL_TOP_P: float = 0.9
    KEYWORD_MODEL_ENABLED: bool = True
    KEYWORD_MODEL_TIMEOUT_SECONDS: float = DEFAULT_KEYWORD_MODEL_TIMEOUT_SECONDS

    CAPTION_MODEL_HOST_DIR: str = DEFAULT_CAPTION_MODEL_HOST_DIR
    CAPTION_MODEL_HF_REPO_ID: str = DEFAULT_CAPTION_MODEL_HF_REPO_ID
    CAPTION_MODEL_HF_FILENAME: str = DEFAULT_CAPTION_MODEL_HF_FILENAME
    CAPTION_MODEL_SERVER_PORT: int = DEFAULT_CAPTION_MODEL_SERVER_PORT
    CAPTION_MODEL_BASE_URL: str = (
        f"http://caption-server:{DEFAULT_CAPTION_MODEL_SERVER_PORT}"
    )
    CAPTION_MODEL_NAME: str | None = None
    CAPTION_MODEL_CHAT_ENDPOINT: str = "/v1/chat/completions"
    CAPTION_MODEL_HEALTH_ENDPOINT: str = DEFAULT_CAPTION_MODEL_HEALTH_ENDPOINT
    CAPTION_MODEL_API_KEY: str | None = None
    CAPTION_MODEL_CTX_SIZE: int = 2048
    CAPTION_MODEL_GPU_LAYERS: int = 20
    CAPTION_MODEL_MAX_TOKENS: int = 512
    CAPTION_MODEL_TEMPERATURE: float = 1
    CAPTION_MODEL_TOP_P: float = 0.9
    CAPTION_MODEL_ENABLED: bool = True
    CAPTION_MODEL_TIMEOUT_SECONDS: float = DEFAULT_CAPTION_MODEL_TIMEOUT_SECONDS

    FINAL_EDIT_MODEL_SERVER_PORT: int = DEFAULT_FINAL_EDIT_MODEL_SERVER_PORT
    FINAL_EDIT_MODEL_BASE_URL: str = (
        f"http://vlm-server:{DEFAULT_FINAL_EDIT_MODEL_SERVER_PORT}"
    )
    FINAL_EDIT_MODEL_CHAT_ENDPOINT: str = "/v1/chat/completions"
    FINAL_EDIT_MODEL_HEALTH_ENDPOINT: str = DEFAULT_FINAL_EDIT_MODEL_HEALTH_ENDPOINT
    FINAL_EDIT_MODEL_API_KEY: str | None = None
    FINAL_EDIT_MODEL_MAX_TOKENS: int = 2048
    FINAL_EDIT_MODEL_TEMPERATURE: float = 0.2
    FINAL_EDIT_MODEL_TOP_P: float = 0.9
    FINAL_EDIT_MODEL_TIMEOUT_SECONDS: float = DEFAULT_FINAL_EDIT_MODEL_TIMEOUT_SECONDS
    FINAL_EDIT_MODEL_NAME: str = "Qwen/Qwen2.5-VL-3B-Instruct-AWQ"
    FINAL_EDIT_MODEL_ENABLED: bool = True
    FINAL_EDIT_ENABLE_UPSCALE: bool = False

    CANONICAL_KEYWORD_RESOLVER_ENABLED: bool = True
    CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME: str = (
        DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME
    )
    CANONICAL_KEYWORD_EMBEDDING_MODEL_CACHE_DIR: Path = (
        DEFAULT_CANONICAL_KEYWORD_EMBEDDING_MODEL_CACHE_DIR
    )
    CANONICAL_KEYWORD_EMBEDDING_DIM: int = (
        DEFAULT_CANONICAL_KEYWORD_EMBEDDING_DIM
    )
    REFERENCE_CAPTION_RAG_ENABLED: bool = True
    REFERENCE_CAPTION_MAX_REFERENCES: int = 2

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "local", "dev", "development"}:
                return True
        return value

    @field_validator(
        "NIMA_ORT_INTRA_OP_THREADS",
        "NIMA_ORT_INTER_OP_THREADS",
        mode="before",
    )
    @classmethod
    def normalize_nima_ort_thread_count(cls, value: object, info) -> int:
        default = (
            DEFAULT_NIMA_ORT_INTRA_OP_THREADS
            if info.field_name == "NIMA_ORT_INTRA_OP_THREADS"
            else DEFAULT_NIMA_ORT_INTER_OP_THREADS
        )
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    @field_validator("NIMA_ORT_EXECUTION_MODE", mode="before")
    @classmethod
    def normalize_nima_ort_execution_mode(cls, value: object) -> str:
        if not isinstance(value, str):
            return DEFAULT_NIMA_ORT_EXECUTION_MODE
        normalized = value.strip().lower()
        if normalized in {"sequential", "parallel"}:
            return normalized
        return DEFAULT_NIMA_ORT_EXECUTION_MODE

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @computed_field
    @property
    def postgres_search_path(self) -> str:
        normalized_schema = self.POSTGRES_SCHEMA.strip() or "public"
        if normalized_schema == "public":
            return "public"
        return f"{normalized_schema},public"

    @computed_field
    @property
    def redis_url(self) -> str:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @computed_field
    @property
    def s3_configured(self) -> bool:
        return all(
            [
                self.S3_SECRET_KEY,
                self.S3_ACCESS_KEY,
                self.S3_BUCKET_NAME,
                self.S3_REGION,
            ]
        )

    @computed_field
    @property
    def keyword_model_client(self) -> LlamaModelClientSettings:
        return LlamaModelClientSettings(
            base_url=self.KEYWORD_MODEL_BASE_URL.rstrip("/"),
            model_name=self.KEYWORD_MODEL_NAME,
            chat_endpoint=self.KEYWORD_MODEL_CHAT_ENDPOINT,
            health_endpoint=self.KEYWORD_MODEL_HEALTH_ENDPOINT,
            api_key=self.KEYWORD_MODEL_API_KEY,
            timeout_seconds=self.KEYWORD_MODEL_TIMEOUT_SECONDS,
            max_tokens=self.KEYWORD_MODEL_MAX_TOKENS,
            temperature=self.KEYWORD_MODEL_TEMPERATURE,
            top_p=self.KEYWORD_MODEL_TOP_P,
            enabled=self.KEYWORD_MODEL_ENABLED,
        )

    @computed_field
    @property
    def caption_model_client(self) -> LlamaModelClientSettings:
        return LlamaModelClientSettings(
            base_url=self.CAPTION_MODEL_BASE_URL.rstrip("/"),
            model_name=self.CAPTION_MODEL_NAME,
            chat_endpoint=self.CAPTION_MODEL_CHAT_ENDPOINT,
            health_endpoint=self.CAPTION_MODEL_HEALTH_ENDPOINT,
            api_key=self.CAPTION_MODEL_API_KEY,
            timeout_seconds=self.CAPTION_MODEL_TIMEOUT_SECONDS,
            max_tokens=self.CAPTION_MODEL_MAX_TOKENS,
            temperature=self.CAPTION_MODEL_TEMPERATURE,
            top_p=self.CAPTION_MODEL_TOP_P,
            enabled=self.CAPTION_MODEL_ENABLED,
        )

    @computed_field
    @property
    def final_edit_model_client(self) -> LlamaModelClientSettings:
        return LlamaModelClientSettings(
            base_url=self.FINAL_EDIT_MODEL_BASE_URL.rstrip("/"),
            model_name=self.FINAL_EDIT_MODEL_NAME,
            chat_endpoint=self.FINAL_EDIT_MODEL_CHAT_ENDPOINT,
            health_endpoint=self.FINAL_EDIT_MODEL_HEALTH_ENDPOINT,
            api_key=self.FINAL_EDIT_MODEL_API_KEY,
            timeout_seconds=self.FINAL_EDIT_MODEL_TIMEOUT_SECONDS,
            max_tokens=self.FINAL_EDIT_MODEL_MAX_TOKENS,
            temperature=self.FINAL_EDIT_MODEL_TEMPERATURE,
            top_p=self.FINAL_EDIT_MODEL_TOP_P,
            enabled=self.FINAL_EDIT_MODEL_ENABLED,
        )

    @computed_field
    @property
    def keyword_model_server(self) -> LlamaModelServerSettings:
        return LlamaModelServerSettings(
            host_dir=self.KEYWORD_MODEL_HOST_DIR,
            hf_repo_id=self.KEYWORD_MODEL_HF_REPO_ID,
            hf_filename=self.KEYWORD_MODEL_HF_FILENAME,
            server_port=self.KEYWORD_MODEL_SERVER_PORT,
            ctx_size=self.KEYWORD_MODEL_CTX_SIZE,
            gpu_layers=self.KEYWORD_MODEL_GPU_LAYERS,
        )

    @computed_field
    @property
    def caption_model_server(self) -> LlamaModelServerSettings:
        return LlamaModelServerSettings(
            host_dir=self.CAPTION_MODEL_HOST_DIR,
            hf_repo_id=self.CAPTION_MODEL_HF_REPO_ID,
            hf_filename=self.CAPTION_MODEL_HF_FILENAME,
            server_port=self.CAPTION_MODEL_SERVER_PORT,
            ctx_size=self.CAPTION_MODEL_CTX_SIZE,
            gpu_layers=self.CAPTION_MODEL_GPU_LAYERS,
        )


settings = Settings()
