from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .models import AnnotationField, FieldType, GraphExtractionSchema


def _field_schema(field: AnnotationField) -> dict[str, Any]:
    if field.type == FieldType.OBJECT:
        properties = {child.name: _field_schema(child) for child in field.children}
        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        required = [child.name for child in field.children if child.required]
        if required:
            schema["required"] = required
    elif field.type == FieldType.LIST:
        if field.item_type == FieldType.OBJECT:
            properties = {child.name: _field_schema(child) for child in field.children}
            item: dict[str, Any] = {
                "type": "object",
                "properties": properties,
                "additionalProperties": False,
            }
            required = [child.name for child in field.children if child.required]
            if required:
                item["required"] = required
        else:
            item = {"type": _json_type(field.item_type)}
        schema = {"type": "array", "items": item}
    elif field.type == FieldType.CATEGORY:
        schema = {"type": "string", "enum": field.allowed_values}
    else:
        schema = {"type": _json_type(field.type)}

    if field.multiple and field.type not in {FieldType.LIST, FieldType.OBJECT}:
        schema = {"type": "array", "items": schema}
    if field.description:
        schema["description"] = field.description
    if field.examples:
        schema["examples"] = field.examples
    return schema


def _json_type(field_type: FieldType) -> str:
    return {
        FieldType.STRING: "string",
        FieldType.INTEGER: "integer",
        FieldType.NUMBER: "number",
        FieldType.BOOLEAN: "boolean",
        FieldType.CATEGORY: "string",
        FieldType.LIST: "array",
        FieldType.OBJECT: "object",
    }[field_type]


def compile_schema(fields: list[AnnotationField]) -> dict[str, Any]:
    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise ValueError("Field names must be unique")
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Annotation",
        "type": "object",
        "properties": {field.name: _field_schema(field) for field in fields},
        "additionalProperties": False,
    }
    required = [field.name for field in fields if field.required]
    if required:
        schema["required"] = required
    Draft202012Validator.check_schema(schema)
    return schema


def _typed_graph_object(
    type_name: str,
    base_properties: dict[str, Any],
    required: list[str],
    custom_fields: list[AnnotationField],
    description: str = "",
) -> dict[str, Any]:
    properties = dict(base_properties)
    properties["type"] = {"const": type_name}
    properties.update({field.name: _field_schema(field) for field in custom_fields})
    result = {
        "type": "object",
        "properties": properties,
        "required": [*required, "type", *[field.name for field in custom_fields if field.required]],
        "additionalProperties": False,
    }
    if description:
        result["description"] = description
    return result


def _type_list_description(values: list[str]) -> str:
    return ", ".join(values) if values else "any defined node type"


def compile_graph_schema(definition: GraphExtractionSchema) -> dict[str, Any]:
    if not definition.node_types:
        raise ValueError("Define at least one node type")
    node_base: dict[str, Any] = {
        "id": {"type": "string", "description": "Stable identifier reused by edge endpoints"},
        "label": {"type": "string", "description": "Human-readable entity label from the text"},
    }
    if definition.include_evidence:
        node_base["evidence"] = {"type": "string", "description": "Supporting source text"}
    node_variants = [
        _typed_graph_object(
            item.name,
            node_base,
            ["id", "label"],
            item.properties,
            description=(
                f"Node type '{item.name}'"
                + (f" ({item.label})" if item.label else "")
                + (f": {item.description}" if item.description else "")
            ),
        )
        for item in definition.node_types
    ]

    edge_variants = []
    for item in definition.edge_types:
        allowed_sources = _type_list_description(item.source_types)
        allowed_targets = _type_list_description(item.target_types)
        direction = "directed" if item.directed else "undirected"
        rule = (
            f"Edge type '{item.name}' is {direction}. "
            f"Allowed source node types: {allowed_sources}. "
            f"Allowed target node types: {allowed_targets}."
        )
        if item.label:
            rule += f" Label: {item.label}."
        if item.description:
            rule += f" Meaning: {item.description}"
        edge_base: dict[str, Any] = {
            "source": {
                "type": "string",
                "description": (
                    "ID of an existing node in the nodes array. "
                    f"For edge type '{item.name}', that node must have one of these types: {allowed_sources}."
                ),
            },
            "target": {
                "type": "string",
                "description": (
                    "ID of an existing node in the nodes array. "
                    f"For edge type '{item.name}', that node must have one of these types: {allowed_targets}."
                ),
            },
        }
        if definition.include_evidence:
            edge_base["evidence"] = {"type": "string", "description": "Text supporting this relation"}
        if definition.include_confidence:
            edge_base["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
        edge_variants.append(
            _typed_graph_object(
                item.name,
                edge_base,
                ["source", "target"],
                item.properties,
                description=rule,
            )
        )

    edge_rules = [
        (
            f"{item.name}: source={_type_list_description(item.source_types)}; "
            f"target={_type_list_description(item.target_types)}; "
            f"{'directed' if item.directed else 'undirected'}"
        )
        for item in definition.edge_types
    ]

    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ExtractedGraph",
        "type": "object",
        "properties": {
            "nodes": {
                "type": "array",
                "description": "Entities found in the text. Every edge endpoint must reference an ID from this array.",
                "items": {"anyOf": node_variants},
            },
            "edges": {
                "type": "array",
                "description": (
                    "Relations found in the text. Obey these endpoint type rules exactly: "
                    + (" | ".join(edge_rules) if edge_rules else "No edges are allowed.")
                ),
                "items": {"anyOf": edge_variants} if edge_variants else {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                **({"maxItems": 0} if not edge_variants else {}),
            },
        },
        "required": ["nodes", "edges"],
        "additionalProperties": False,
    }
    Draft202012Validator.check_schema(schema)
    return schema


def openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert project JSON Schema to OpenAI's strict Structured Outputs subset."""

    def nullable(value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") == "null":
            return value
        variants = value.get("anyOf")
        if isinstance(variants, list) and any(item.get("type") == "null" for item in variants):
            return value
        return {"anyOf": [value, {"type": "null"}]}

    def convert(value: Any) -> Any:
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"$schema", "examples", "default"}:
                continue
            if key == "const":
                converted["enum"] = [convert(item)]
                continue
            converted["anyOf" if key == "oneOf" else key] = convert(item)
        if converted.get("type") == "object":
            properties = converted.get("properties", {})
            originally_required = set(converted.get("required", []))
            converted["properties"] = {
                name: child if name in originally_required else nullable(child)
                for name, child in properties.items()
            }
            converted["required"] = list(properties)
            converted["additionalProperties"] = False
        return converted

    result = convert(deepcopy(schema))
    Draft202012Validator.check_schema(result)
    return result


def validation_errors(value: Any, schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    return [f"{'/'.join(map(str, error.absolute_path)) or '$'}: {error.message}" for error in errors]


def build_prompt(
    schema: dict[str, Any],
    instructions: str,
    examples: list[dict[str, Any]],
    include_schema: bool = True,
) -> str:
    if schema.get("title") == "ExtractedGraph":
        parts = [
            "Extract a graph from the supplied text.",
            "Identify only nodes and edges supported by the text. Use stable, concise node IDs and reuse those exact IDs in edge source and target fields.",
            "For every edge, obey the allowed source and target node types stated in that edge type's schema description.",
            "Do not invent missing entities or relations. Return empty arrays when none are present.",
        ]
    else:
        parts = ["Extract structured annotations from the supplied text."]
    if include_schema:
        parts.extend([
            "Return only one JSON object that conforms exactly to this JSON Schema:",
            json.dumps(schema, ensure_ascii=False, indent=2),
        ])
    else:
        parts.append("Return only the structured result defined by the response format.")
    if instructions.strip():
        parts.extend(["Additional instructions:", instructions.strip()])
    if examples:
        parts.append("Follow these labeled examples:")
        for example in examples:
            if "text" in example and "annotation" in example:
                parts.append(
                    "Example input:\n"
                    + str(example["text"])
                    + "\nExample structured output:\n"
                    + json.dumps(example["annotation"], ensure_ascii=False)
                )
            else:
                parts.append(json.dumps(example, ensure_ascii=False))
    return "\n\n".join(parts)


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response did not contain a JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value
