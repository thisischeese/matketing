# AI Service

This directory contains the standalone FastAPI service for AI-related APIs.

## Structure

- `app/`: FastAPI application package
- `scripts/`: local helper scripts
- `tests/`: pytest-based tests

## Package management

This project uses `uv` as the single source of truth for dependency management.

- Dependency definitions live in `pyproject.toml`
- Locked versions live in `uv.lock`
- Do not use `requirements.txt`

## Quick start

```powershell
cd ai
uv sync
uv run fastapi dev app/main.py
```

Run the production server locally with:

```powershell
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Runtime model configuration

The AI service calls separately served model APIs over HTTP. The active runtime
configuration lives in `app/core/config.py` and is centered on remote client
settings, not local model server provisioning.

Keyword extraction settings:

- `KEYWORD_MODEL_BASE_URL`
- `KEYWORD_MODEL_CHAT_ENDPOINT`
- `KEYWORD_MODEL_HEALTH_ENDPOINT`
- `KEYWORD_MODEL_API_KEY`
- `KEYWORD_MODEL_TIMEOUT_SECONDS`
- `KEYWORD_MODEL_MAX_TOKENS`
- `KEYWORD_MODEL_TEMPERATURE`
- `KEYWORD_MODEL_TOP_P`
- `KEYWORD_MODEL_ENABLED`

Caption generation settings:

- `CAPTION_MODEL_BASE_URL`
- `CAPTION_MODEL_CHAT_ENDPOINT`
- `CAPTION_MODEL_HEALTH_ENDPOINT`
- `CAPTION_MODEL_API_KEY`
- `CAPTION_MODEL_TIMEOUT_SECONDS`
- `CAPTION_MODEL_MAX_TOKENS`
- `CAPTION_MODEL_TEMPERATURE`
- `CAPTION_MODEL_TOP_P`
- `CAPTION_MODEL_ENABLED`

Final edit planner settings:

- `FINAL_EDIT_MODEL_BASE_URL`
- `FINAL_EDIT_MODEL_CHAT_ENDPOINT`
- `FINAL_EDIT_MODEL_HEALTH_ENDPOINT`
- `FINAL_EDIT_MODEL_API_KEY`
- `FINAL_EDIT_MODEL_TIMEOUT_SECONDS`
- `FINAL_EDIT_MODEL_MAX_TOKENS`
- `FINAL_EDIT_MODEL_TEMPERATURE`
- `FINAL_EDIT_MODEL_TOP_P`
- `FINAL_EDIT_MODEL_NAME`
- `FINAL_EDIT_MODEL_ENABLED`
- `FINAL_EDIT_ENABLE_UPSCALE`

## Keyword extraction runtime

FastAPI no longer calls a keyword GGUF model directly during inference.
Keyword extraction sends HTTP requests to a separately served model endpoint.

- Base URL: `KEYWORD_MODEL_BASE_URL`
- Chat endpoint: `KEYWORD_MODEL_CHAT_ENDPOINT`
- Default target: `http://keyword-server:8001/v1/chat/completions`

At startup, FastAPI only checks connectivity to the remote keyword server. If
the server is unavailable, the app still starts and `process-utterance` can
return `503` until the server becomes reachable.

## Keyword extraction flow

When `process-utterance` receives the first shop-owner utterance, the keyword
extraction flow works in two steps:

1. The remote keyword model extracts 1 to 3 raw keyword candidates as `keywords`.
2. FastAPI normalizes each keyword and stores the normalized values as
   `draft_keywords`.

The remote model contract is:

```json
{
  "keywords": ["막걸리", "파전"]
}
```

FastAPI persists the normalized keyword state in Redis. The public API response
still returns only the generated caption and guide text.

## Canonical keyword flow

After `draft_keywords` are produced, FastAPI resolves each draft keyword to the
nearest canonical keyword code.

1. The AI service embeds each `draft_keyword` with `intfloat/multilingual-e5-small`.
2. It queries `maketing.canonical_keywords` with pgvector cosine similarity.
3. The matched row's `code` is stored as `final_keyword:*` in Redis.
4. If lookup fails, the original `draft_keyword` is used as the fallback.

Current implementation notes:

- `draft_keyword:*` remains the human-readable normalized keyword.
- `final_keyword:*` is the canonical `code` when resolution succeeds.
- User-facing caption and guide generation still use readable draft keywords.

## Weather tag flow

During the first `process-utterance` call, FastAPI evaluates reusable weather
tags from the request body's `weather` payload and stores them in the session
for later text-generation steps.

- Stored Redis fields: `weather_tag:1`, `weather_tag:2`, ...
- Current tag groups:
  - precipitation: `PRECIP_CLEAR`, `PRECIP_CLOUDY`, `PRECIP_RAIN`, `PRECIP_HEAVY_RAIN`
  - temperature: `TEMP_FREEZING`, `TEMP_COLD`, `TEMP_MILD`, `TEMP_HOT`, `TEMP_SCORCHING`
  - humidity: `HUMID_LOW`, `HUMID_HIGH`
  - special conditions: `SPECIAL_FINE_DUST`, `SPECIAL_TYPHOON`, `SPECIAL_SEASONAL_CHANGE`

## Local development

Local development can stay flexible as long as FastAPI can reach running model
servers.

If FastAPI runs in Docker and a model server runs separately on the host, use:

```env
KEYWORD_MODEL_BASE_URL=http://host.docker.internal:8080
KEYWORD_MODEL_CHAT_ENDPOINT=/v1/chat/completions
```

One working local example for a separately served keyword model is:

```powershell
docker run --rm -p 8080:8080 `
  -v C:\path\to\models:/models `
  ghcr.io/ggml-org/llama.cpp:server `
  -m /models/model.gguf --port 8080 -c 2048 --host 0.0.0.0
```

If both services run under Compose, keep the default
`KEYWORD_MODEL_BASE_URL=http://keyword-server:8001`.

Legacy local GGUF download helpers still exist for keyword-model maintenance,
but the active FastAPI inference path depends on remote model servers.

## Linux deployment notes

Development was done on Windows, but deployment is expected to run on Linux.
Keep the following in mind when preparing the server:

- Do not copy `ai/.venv` from Windows to Linux. Create a new virtual environment on Linux and run `uv sync` there.
- FastAPI no longer needs `llama-cpp-python` or a local GGUF mount to serve keyword extraction requests.
- Any GGUF model file mounting happens in the separately served model container, not in FastAPI.
- First startup may still take time if orientation weights need to be downloaded. Plan for this in container startup and health checks.
