import json

import numpy as np
import pandas as pd

from automatic_annotations.exports import csv_bytes, export_records, jsonl_bytes
from automatic_annotations.models import (
    ExtractionResponse,
    GraphExtractionSchema,
    NodeTypeDefinition,
    ReviewedAnnotation,
)
from automatic_annotations.project import ProjectStore
from automatic_annotations.providers import ExtractionProvider
from automatic_annotations.runner import ExtractionRunner


class FakeProvider(ExtractionProvider):
    def __init__(self):
        self.calls = 0

    def extract(self, text, schema, instructions, examples, settings, api_key=None):
        self.calls += 1
        parsed = {"nodes": [{"id": text.lower(), "label": text, "type": "concept"}], "edges": []}
        return ExtractionResponse(parsed=parsed, raw_output=json.dumps(parsed), model="fake")


class ErrorProvider(ExtractionProvider):
    def __init__(self, error):
        self.error = error
        self.calls = 0
        self.timeouts = []

    def extract(self, text, schema, instructions, examples, settings, api_key=None):
        self.calls += 1
        self.timeouts.append(settings.timeout_seconds)
        return ExtractionResponse(error=self.error)


def make_project(tmp_path):
    store = ProjectStore.create(tmp_path, "Test project")
    store.save_graph_schema(GraphExtractionSchema(node_types=[NodeTypeDefinition(name="concept")]))
    config = store.load_config()
    config.text_column = "text"
    config.id_column = "id"
    store.save_source(pd.DataFrame([{"id": "a", "text": "Good"}, {"id": "b", "text": "Great"}]), config)
    return store


def test_run_resumes_and_reruns_as_new_attempt(tmp_path):
    store = make_project(tmp_path)
    provider = FakeProvider()
    runner = ExtractionRunner(store, provider)
    run_id = runner.create_run(2).run_id
    runner.run(run_id)
    assert provider.calls == 2
    runner.run(run_id)
    assert provider.calls == 2
    runner.run(run_id, row_ids=["a"], rerun_completed=True)
    assert provider.calls == 3
    results = store.load_results(run_id)
    assert len(results) == 3
    assert results[0].attempt_id != results[2].attempt_id


def test_review_layer_and_exports(tmp_path):
    store = make_project(tmp_path)
    runner = ExtractionRunner(store, FakeProvider())
    run_id = runner.create_run(2).run_id
    runner.run(run_id)
    store.save_reviews(run_id, {
        "a": ReviewedAnnotation(row_id="a", annotation={"nodes": [], "edges": []})
    })
    records = export_records(store, run_id)
    assert records[0]["annotation"] == {"nodes": [], "edges": []}
    assert records[0]["review_status"] == "reviewed"
    assert "annotation_nodes" in csv_bytes(store, run_id).decode()
    parsed = [json.loads(line) for line in jsonl_bytes(store, run_id).decode().splitlines()]
    assert parsed[1]["annotation"]["nodes"][0]["label"] == "Great"


def test_rejects_null_and_duplicate_ids(tmp_path):
    store = ProjectStore.create(tmp_path, "IDs")
    config = store.load_config()
    config.text_column = "text"
    config.id_column = "id"
    for values in ([None, "b"], ["a", "a"]):
        try:
            store.save_source(pd.DataFrame({"id": values, "text": ["x", "y"]}), config)
            assert False, "Expected invalid IDs to be rejected"
        except ValueError:
            pass


def test_huggingface_values_are_json_safe_and_unselected_columns_are_dropped(tmp_path):
    store = ProjectStore.create(tmp_path, "HF values")
    config = store.load_config()
    config.id_column = "id"
    config.text_column = "text"
    config.metadata_columns = ["tokens", "score", "created"]
    frame = pd.DataFrame([{
        "id": np.int64(7),
        "text": "Example",
        "tokens": np.array([1, 2, 3]),
        "score": np.float32(0.5),
        "created": pd.Timestamp("2026-06-11T10:00:00"),
        "drop_me": np.array(["unused"]),
    }])
    store.save_source(frame, config)
    record = json.loads((store.root / "source.jsonl").read_text())
    assert record == {
        "id": "7",
        "text": "Example",
        "tokens": [1, 2, 3],
        "score": 0.5,
        "created": "2026-06-11T10:00:00",
    }


def test_test_extraction_is_single_attempt_with_short_timeout(tmp_path):
    store = make_project(tmp_path)
    provider = ErrorProvider("request timed out")
    results = ExtractionRunner(store, provider).test_rows([0])
    assert results[0].state.value == "failed"
    assert provider.calls == 1
    assert provider.timeouts == [120]


def test_batch_does_not_retry_deterministic_provider_errors(tmp_path):
    store = make_project(tmp_path)
    config = store.load_config()
    config.model.retry_limit = 3
    store.save_config(config)
    provider = ErrorProvider("Invalid schema for response_format")
    runner = ExtractionRunner(store, provider)
    run_id = runner.create_run(2).run_id
    runner.run(run_id, row_ids=["a"])
    assert provider.calls == 1
