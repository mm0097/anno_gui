import json

import pytest

from automatic_annotations.models import (
    AnnotationField,
    EdgeTypeDefinition,
    FieldType,
    GraphExtractionSchema,
    NodeTypeDefinition,
)
from automatic_annotations.schema import (
    compile_graph_schema,
    compile_schema,
    openai_strict_schema,
    parse_json_object,
    validation_errors,
)


def test_compiles_nested_schema_and_validates():
    fields = [
        AnnotationField(
            name="actors",
            type=FieldType.LIST,
            item_type=FieldType.OBJECT,
            children=[
                AnnotationField(name="id", required=True),
                AnnotationField(name="kind", type=FieldType.CATEGORY, allowed_values=["person", "org"]),
            ],
        ),
        AnnotationField(name="stance", type=FieldType.CATEGORY, allowed_values=["support", "oppose"], required=True),
    ]
    schema = compile_schema(fields)
    valid = {"actors": [{"id": "a", "kind": "person"}], "stance": "support"}
    assert validation_errors(valid, schema) == []
    assert validation_errors({"actors": [], "stance": "other"}, schema)


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        compile_schema([AnnotationField(name="x"), AnnotationField(name="x")])


def test_parse_json_from_fence_and_surrounding_text():
    assert parse_json_object("```json\n{\"ok\": true}\n```") == {"ok": True}
    assert parse_json_object("Result: {\"ok\": true} done") == {"ok": True}


def test_compiles_graph_ontology_to_strict_extraction_schema():
    definition = GraphExtractionSchema(
        node_types=[NodeTypeDefinition(
            name="person",
            properties=[AnnotationField(name="role", required=True)],
        )],
        edge_types=[EdgeTypeDefinition(
            name="supports", source_types=["person"], target_types=["person"]
        )],
    )
    schema = compile_graph_schema(definition)
    valid = {
        "nodes": [{"id": "a", "label": "Alice", "type": "person", "role": "author"}],
        "edges": [{"source": "a", "target": "a", "type": "supports", "confidence": 0.9}],
    }
    assert validation_errors(valid, schema) == []
    assert validation_errors({"nodes": [], "edges": [{"source": "a", "target": "b", "type": "unknown"}]}, schema)


def test_openai_schema_requires_all_fields_and_makes_optional_fields_nullable():
    schema = compile_schema([
        AnnotationField(name="required_value", required=True),
        AnnotationField(name="optional_value"),
    ])
    strict = openai_strict_schema(schema)
    assert set(strict["required"]) == {"required_value", "optional_value"}
    assert strict["properties"]["optional_value"] == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    assert "$schema" not in strict


def test_openai_graph_schema_uses_supported_unions_and_enum_discriminators():
    schema = compile_graph_schema(GraphExtractionSchema(
        node_types=[NodeTypeDefinition(name="person"), NodeTypeDefinition(name="place")]
    ))
    strict = openai_strict_schema(schema)
    variants = strict["properties"]["nodes"]["items"]["anyOf"]
    assert variants[0]["properties"]["type"] == {"enum": ["person"]}
    assert "oneOf" not in json.dumps(strict)


def test_graph_schema_passes_edge_endpoint_rules_to_model():
    schema = compile_graph_schema(GraphExtractionSchema(
        node_types=[NodeTypeDefinition(name="person"), NodeTypeDefinition(name="organization")],
        edge_types=[EdgeTypeDefinition(
            name="works_for",
            label="Works for",
            description="Employment relationship",
            source_types=["person"],
            target_types=["organization"],
        )],
    ))
    edge_variant = schema["properties"]["edges"]["items"]["anyOf"][0]
    assert "Allowed source node types: person" in edge_variant["description"]
    assert "Allowed target node types: organization" in edge_variant["description"]
    assert "must have one of these types: person" in edge_variant["properties"]["source"]["description"]
    assert "must have one of these types: organization" in edge_variant["properties"]["target"]["description"]
    assert "works_for: source=person; target=organization" in schema["properties"]["edges"]["description"]

    strict = openai_strict_schema(schema)
    strict_edge = strict["properties"]["edges"]["items"]["anyOf"][0]
    assert "Allowed source node types: person" in strict_edge["description"]
