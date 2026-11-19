"""A small JSON Schema (draft 2020-12) validator for the subset the files in schemas/ use.

No third-party package. Supported keywords: type (one name or a list), required, properties,
additionalProperties (true, false or a schema), enum, items (one schema), minimum, maximum,
pattern (re.search, as in the standard), minItems, maxItems. Annotations (`$schema`, `$id`,
`title`, `description`) are ignored. Any other keyword is an error, so a schema cannot drift
outside the subset unnoticed (`check_schema`).

    errors = validate(instance, schema)      # [] when valid, else "path: message" lines
"""
from __future__ import annotations

import json
import pathlib
import re

ANNOTATIONS = frozenset({"$schema", "$id", "title", "description"})
KEYWORDS = frozenset({"type", "required", "properties", "additionalProperties", "enum", "items",
                      "minimum", "maximum", "pattern", "minItems", "maxItems"})
TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})
DRAFT = "https://json-schema.org/draft/2020-12/schema"


def _is_type(value: object, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise ValueError(f"unknown type {name!r}")


def check_schema(schema: object, path: str = "$") -> list[str]:
    """Problems that keep a schema out of the supported subset (empty list: it is inside)."""
    if isinstance(schema, bool):
        return []
    if not isinstance(schema, dict):
        return [f"{path}: a schema must be an object"]
    problems = []
    for key in schema:
        if key not in KEYWORDS and key not in ANNOTATIONS:
            problems.append(f"{path}: unsupported keyword {key!r}")
    kind = schema.get("type")
    for name in ([kind] if isinstance(kind, str) else kind if isinstance(kind, list) else []):
        if name not in TYPES:
            problems.append(f"{path}: unknown type {name!r}")
    if "type" in schema and not isinstance(kind, (str, list)):
        problems.append(f"{path}: type must be a string or a list")
    for name, sub in (schema.get("properties") or {}).items():
        problems += check_schema(sub, f"{path}.{name}")
    if "items" in schema:
        problems += check_schema(schema["items"], f"{path}[]")
    if isinstance(schema.get("additionalProperties"), dict):
        problems += check_schema(schema["additionalProperties"], f"{path}.*")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (re.error, TypeError):
            problems.append(f"{path}: bad pattern")
    return problems


def validate(instance: object, schema: object, path: str = "$") -> list[str]:
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: nothing is allowed here"]
    errors: list[str] = []
    if "type" in schema:
        names = [schema["type"]] if isinstance(schema["type"], str) else schema["type"]
        if not any(_is_type(instance, name) for name in names):
            return [f"{path}: expected {'|'.join(names)}, got {type(instance).__name__}"]
    if "enum" in schema and not any(type(instance) is type(option) and instance == option
                                    for option in schema["enum"]):
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
    if isinstance(instance, str) and "pattern" in schema and not re.search(schema["pattern"], instance):
        errors.append(f"{path}: {instance[:60]!r} does not match {schema['pattern']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: {instance} is below {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: {instance} is above {schema['maximum']}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            errors.append(f"{path}: {len(instance)} items, at least {schema['minItems']} needed")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: {len(instance)} items, at most {schema['maxItems']} allowed")
        if "items" in schema:
            for index, item in enumerate(instance):
                errors += validate(item, schema["items"], f"{path}[{index}]")
    if isinstance(instance, dict):
        for name in schema.get("required", ()):
            if name not in instance:
                errors.append(f"{path}: missing required key {name!r}")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for name, value in instance.items():
            if name in properties:
                errors += validate(value, properties[name], f"{path}.{name}")
            elif extra is False:
                errors.append(f"{path}: unexpected key {name!r}")
            elif isinstance(extra, dict):
                errors += validate(value, extra, f"{path}.{name}")
    return errors


SCHEMA_DIR = pathlib.Path(__file__).resolve().parent.parent / "schemas"


def load_schema(name: str, directory: pathlib.Path = SCHEMA_DIR) -> dict:
    """schemas/<name>.schema.json"""
    return json.loads((directory / f"{name}.schema.json").read_text(encoding="utf-8"))
