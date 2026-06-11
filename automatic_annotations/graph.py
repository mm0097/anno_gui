from __future__ import annotations

import io
import json
import pickle
from typing import Any

import networkx as nx
import pandas as pd

from .models import GraphExtractionSchema


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
