from __future__ import annotations

import boto3
from botocore.config import Config

from app.core.config import settings


def build_s3_client():
    client_options: dict[str, object] = {
        "region_name": settings.S3_REGION,
        "aws_access_key_id": settings.S3_ACCESS_KEY,
        "aws_secret_access_key": settings.S3_SECRET_KEY,
    }
    if settings.S3_ENDPOINT_URL:
        client_options["endpoint_url"] = settings.S3_ENDPOINT_URL
    if settings.S3_FORCE_PATH_STYLE:
        client_options["config"] = Config(s3={"addressing_style": "path"})
    return boto3.client(
        "s3",
        **client_options,
    )


def normalize_s3_key(value: str) -> str:
    return value.lstrip("/")
