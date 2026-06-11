import networkx as nx

from automatic_annotations.graph import build_extracted_graph, graph_exports, validate_extracted_graph
from automatic_annotations.models import EdgeTypeDefinition, GraphExtractionSchema, NodeTypeDefinition


def graph_definition():
    return GraphExtractionSchema(
        node_types=[NodeTypeDefinition(name="person"), NodeTypeDefinition(name="organization")],
        edge_types=[EdgeTypeDefinition(
            name="works_for", source_types=["person"], target_types=["organization"]
        )],
    )


def test_builds_graph_directly_from_extraction():
    records = [{
        "doc_id": "d1",
        "annotation": {
            "nodes": [
                {"id": "alice", "label": "Alice", "type": "person"},
                {"id": "acme", "label": "Acme", "type": "organization"},
            ],
            "edges": [{"source": "alice", "target": "acme", "type": "works_for"}],
        },
    }]
    graph, warnings = build_extracted_graph(records, graph_definition(), row_id_field="doc_id")
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.nodes["alice"]["source_rows"] == ["d1"]
    assert warnings == []
    assert {"nodes.csv", "edges.csv", "graph.json", "graph.graphml", "graph.networkx.pkl"} == set(graph_exports(graph))


def test_reports_invalid_endpoint_types_and_unresolved_nodes():
    annotation = {
        "nodes": [{"id": "acme", "label": "Acme", "type": "organization"}],
        "edges": [
            {"source": "acme", "target": "missing", "type": "works_for"},
            {"source": "acme", "target": "acme", "type": "works_for"},
        ],
    }
    warnings = validate_extracted_graph(annotation, graph_definition())
    assert "Edge 0 has unresolved reference: acme -> missing" in warnings
    assert "Edge 'works_for' cannot start at node type 'organization'" in warnings
