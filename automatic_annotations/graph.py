from __future__ import annotations

import io
import json
import pickle
import re
from typing import Any

import networkx as nx
import pandas as pd

from .models import AnnotationField, EdgeTypeDefinition, FieldType, GraphExtractionSchema, NodeTypeDefinition


def _schema_identifier(value: Any, fallback: str) -> str:
    identifier = re.sub(r"\W+", "_", str(value or "").strip().lower()).strip("_")
    if not identifier or identifier[0].isdigit():
        identifier = fallback
    return identifier


def _field_type(values: list[Any]) -> FieldType:
    populated = [value for value in values if value is not None]
    if not populated:
        return FieldType.STRING
    if all(isinstance(value, bool) for value in populated):
        return FieldType.BOOLEAN
    if all(isinstance(value, int) and not isinstance(value, bool) for value in populated):
        return FieldType.INTEGER
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in populated):
        return FieldType.NUMBER
    if all(isinstance(value, list) for value in populated):
        return FieldType.LIST
    return FieldType.STRING


def _graph_annotation(record: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("annotation", "graph", "parsed", "extraction"):
        value = record.get(key)
        if isinstance(value, dict) and ("nodes" in value or "edges" in value):
            return value
    if "nodes" in record or "edges" in record:
        return record
    return None


def normalize_graph_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    node_type_map: dict[str, str] = {}
    edge_type_map: dict[str, str] = {}
    for index, record in enumerate(records):
        annotation = _graph_annotation(record)
        if not annotation:
            continue
        nodes = []
        for node_index, node in enumerate(annotation.get("nodes", [])):
            if not isinstance(node, dict):
                continue
            copied = dict(node)
            copied["id"] = str(copied.get("id") or copied.get("label") or f"row_{index}_node_{node_index}")
            raw_type = str(copied.get("type") or "entity")
            copied["type"] = node_type_map.setdefault(raw_type, _schema_identifier(raw_type, "entity"))
            if raw_type != copied["type"]:
                copied.setdefault("original_type", raw_type)
            nodes.append(copied)
        edges = []
        for edge in annotation.get("edges", []):
            if not isinstance(edge, dict):
                continue
            copied = dict(edge)
            if not copied.get("source") or not copied.get("target"):
                continue
            raw_type = str(copied.get("type") or "relation")
            copied["source"] = str(copied["source"])
            copied["target"] = str(copied["target"])
            copied["type"] = edge_type_map.setdefault(raw_type, _schema_identifier(raw_type, "relation"))
            if raw_type != copied["type"]:
                copied.setdefault("original_type", raw_type)
            edges.append(copied)
        normalized_record = dict(record)
        normalized_record["annotation"] = {"nodes": nodes, "edges": edges}
        normalized.append(normalized_record)
    return normalized


def infer_graph_schema(records: list[dict[str, Any]]) -> GraphExtractionSchema:
    node_values: dict[str, dict[str, list[Any]]] = {}
    edge_values: dict[str, dict[str, list[Any]]] = {}
    edge_sources: dict[str, set[str]] = {}
    edge_targets: dict[str, set[str]] = {}
    for record in records:
        annotation = record.get("annotation") or {}
        node_types_by_id = {
            str(node.get("id")): str(node.get("type", "entity"))
            for node in annotation.get("nodes", [])
            if isinstance(node, dict) and node.get("id")
        }
        for node in annotation.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type", "entity"))
            fields = node_values.setdefault(node_type, {})
            for key, value in node.items():
                if key not in {"id", "type"}:
                    fields.setdefault(key, []).append(value)
        for edge in annotation.get("edges", []):
            if not isinstance(edge, dict):
                continue
            edge_type = str(edge.get("type", "relation"))
            fields = edge_values.setdefault(edge_type, {})
            for key, value in edge.items():
                if key not in {"source", "target", "type"}:
                    fields.setdefault(key, []).append(value)
            source_type = node_types_by_id.get(str(edge.get("source", "")))
            target_type = node_types_by_id.get(str(edge.get("target", "")))
            if source_type:
                edge_sources.setdefault(edge_type, set()).add(source_type)
            if target_type:
                edge_targets.setdefault(edge_type, set()).add(target_type)

    def fields(values_by_name: dict[str, list[Any]]) -> list[AnnotationField]:
        return [
            AnnotationField(name=_schema_identifier(name, "field"), type=_field_type(values))
            for name, values in sorted(values_by_name.items())
        ]

    return GraphExtractionSchema(
        node_types=[
            NodeTypeDefinition(name=name, properties=fields(values))
            for name, values in sorted(node_values.items())
        ],
        edge_types=[
            EdgeTypeDefinition(
                name=name,
                source_types=sorted(edge_sources.get(name, set())),
                target_types=sorted(edge_targets.get(name, set())),
                properties=fields(values),
            )
            for name, values in sorted(edge_values.items())
        ],
    )


def validate_extracted_graph(
    annotation: dict[str, Any], definition: GraphExtractionSchema
) -> list[str]:
    warnings: list[str] = []
    nodes = annotation.get("nodes", [])
    edges = annotation.get("edges", [])
    node_types = {item.name for item in definition.node_types}
    edge_types = {item.name: item for item in definition.edge_types}
    ids: dict[str, str] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        node_type = str(node.get("type", ""))
        if node_id in ids:
            warnings.append(f"Duplicate node ID '{node_id}'")
        if node_type not in node_types:
            warnings.append(f"Node {index} has unknown type '{node_type}'")
        if node_id:
            ids[node_id] = node_type
    for index, edge in enumerate(edges):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        edge_type = str(edge.get("type", ""))
        if source not in ids or target not in ids:
            warnings.append(f"Edge {index} has unresolved reference: {source} -> {target}")
            continue
        rule = edge_types.get(edge_type)
        if not rule:
            warnings.append(f"Edge {index} has unknown type '{edge_type}'")
            continue
        if rule.source_types and ids[source] not in rule.source_types:
            warnings.append(f"Edge '{edge_type}' cannot start at node type '{ids[source]}'")
        if rule.target_types and ids[target] not in rule.target_types:
            warnings.append(f"Edge '{edge_type}' cannot end at node type '{ids[target]}'")
    return sorted(set(warnings))


def build_extracted_graph(
    records: list[dict[str, Any]], definition: GraphExtractionSchema, row_id_field: str | None = None
) -> tuple[nx.MultiDiGraph, list[str]]:
    graph = nx.MultiDiGraph()
    warnings: list[str] = []
    edge_rules = {item.name: item for item in definition.edge_types}
    pending_edges: list[tuple[str, str, dict[str, Any]]] = []
    for record in records:
        annotation = record.get("annotation") or {}
        row_id = str(record.get(row_id_field, "")) if row_id_field else ""
        warnings.extend(validate_extracted_graph(annotation, definition))
        for node in annotation.get("nodes", []):
            if not isinstance(node, dict) or not node.get("id"):
                continue
            node_id = str(node["id"])
            attributes = {key: value for key, value in node.items() if key != "id"}
            existing_rows = list(graph.nodes.get(node_id, {}).get("source_rows", []))
            if row_id and row_id not in existing_rows:
                existing_rows.append(row_id)
            graph.add_node(node_id, **attributes, source_rows=existing_rows)
        for edge in annotation.get("edges", []):
            if not isinstance(edge, dict) or not edge.get("source") or not edge.get("target"):
                continue
            source, target = str(edge["source"]), str(edge["target"])
            attributes = {key: value for key, value in edge.items() if key not in {"source", "target"}}
            attributes["source_row"] = row_id
            rule = edge_rules.get(str(edge.get("type", "")))
            if rule and not rule.directed and source != target:
                attributes["undirected"] = True
            pending_edges.append((source, target, attributes))
    for source, target, attributes in pending_edges:
        if source not in graph or target not in graph:
            continue
        graph.add_edge(source, target, **attributes)
    return graph, sorted(set(warnings))


def graph_exports(graph: nx.MultiDiGraph) -> dict[str, bytes]:
    nodes = [{"id": node, **attributes} for node, attributes in graph.nodes(data=True)]
    edges = [
        {"source": source, "target": target, "key": key, **attributes}
        for source, target, key, attributes in graph.edges(keys=True, data=True)
    ]
    graphml = io.BytesIO()
    safe_graph = nx.MultiDiGraph()
    def graphml_value(value: Any) -> str | int | float | bool:
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return json.dumps(value, ensure_ascii=False)

    for node, data in graph.nodes(data=True):
        safe_graph.add_node(node, **{key: graphml_value(value) for key, value in data.items()})
    for source, target, key, data in graph.edges(keys=True, data=True):
        safe_graph.add_edge(
            source,
            target,
            key=key,
            **{key_: graphml_value(value) for key_, value in data.items()},
        )
    nx.write_graphml(safe_graph, graphml)
    return {
        "nodes.csv": pd.DataFrame(nodes).to_csv(index=False).encode(),
        "edges.csv": pd.DataFrame(edges).to_csv(index=False).encode(),
        "graph.json": json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2).encode(),
        "graph.graphml": graphml.getvalue(),
        "graph.networkx.pkl": pickle.dumps(graph),
    }
