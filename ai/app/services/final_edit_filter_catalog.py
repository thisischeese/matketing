from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.core.config import ROOT_DIR

FINAL_EDIT_FILTER_CATALOG_PATH = ROOT_DIR / "app" / "assets" / "final_edit_filters.json"


class FinalEditFilterCatalogError(RuntimeError):
    """Raised when the final-edit filter catalog is invalid or unusable."""


@dataclass(frozen=True)
class FinalEditFilterDefinition:
    name: str
    description: str
    pilgram2_name: str


@lru_cache(maxsize=1)
def load_final_edit_filter_catalog() -> tuple[FinalEditFilterDefinition, ...]:
    try:
        payload = json.loads(FINAL_EDIT_FILTER_CATALOG_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FinalEditFilterCatalogError(
            "Final-edit filter catalog file is missing."
        ) from exc
    except json.JSONDecodeError as exc:
        raise FinalEditFilterCatalogError(
            "Final-edit filter catalog is invalid JSON."
        ) from exc

    if not isinstance(payload, list) or not payload:
        raise FinalEditFilterCatalogError(
            "Final-edit filter catalog must be a non-empty list."
        )

    definitions: list[FinalEditFilterDefinition] = []
    seen_names: set[str] = set()
    seen_pilgram2_names: set[str] = set()

    for item in payload:
        if not isinstance(item, dict):
            raise FinalEditFilterCatalogError(
                "Each final-edit filter catalog item must be an object."
            )
        name = str(item.get("name", "")).strip()
        description = str(item.get("description", "")).strip()
        pilgram2_name = str(item.get("pilgram2_name", "")).strip()
        if not name or not description or not pilgram2_name:
            raise FinalEditFilterCatalogError(
                "Each final-edit filter catalog item must include name, description, and pilgram2_name."
            )

        normalized_name = name.casefold()
        normalized_pilgram2_name = pilgram2_name.casefold()
        if normalized_name in seen_names:
            raise FinalEditFilterCatalogError(
                f"Duplicate final-edit filter name found: {name}"
            )
        if normalized_pilgram2_name in seen_pilgram2_names:
            raise FinalEditFilterCatalogError(
                f"Duplicate final-edit pilgram2_name found: {pilgram2_name}"
            )
        seen_names.add(normalized_name)
        seen_pilgram2_names.add(normalized_pilgram2_name)
        definitions.append(
            FinalEditFilterDefinition(
                name=name,
                description=description,
                pilgram2_name=pilgram2_name,
            )
        )

    return tuple(definitions)


def get_final_edit_filter_name_lookup() -> dict[str, str]:
    return {
        definition.name.casefold(): definition.name
        for definition in load_final_edit_filter_catalog()
    }


def resolve_final_edit_filter_name(value: Any) -> str:
    if value is None:
        raise FinalEditFilterCatalogError(
            "filter params must include a non-empty filter_name."
        )
    normalized = str(value).strip()
    if not normalized:
        raise FinalEditFilterCatalogError(
            "filter params must include a non-empty filter_name."
        )
    canonical_name = get_final_edit_filter_name_lookup().get(normalized.casefold())
    if canonical_name is None:
        raise FinalEditFilterCatalogError(
            f"Unsupported filter requested: {normalized}"
        )
    return canonical_name


def resolve_final_edit_filter_definition(value: Any) -> FinalEditFilterDefinition:
    canonical_name = resolve_final_edit_filter_name(value)
    for definition in load_final_edit_filter_catalog():
        if definition.name == canonical_name:
            return definition
    raise FinalEditFilterCatalogError(
        f"Unsupported filter requested: {canonical_name}"
    )
