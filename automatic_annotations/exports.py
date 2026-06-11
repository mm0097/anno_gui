from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from .project import ProjectStore


def _serialize_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def export_records(store: ProjectStore, run_id: str) -> list[dict[str, Any]]:
    config = store.load_config()
    source = store.load_source()
    results = store.latest_results(run_id)
    reviews = store.load_reviews(run_id)
    output = []
    for _, row in source.iterrows():
        base = row.to_dict()
        row_id = str(base[config.id_column])
        result = results.get(row_id)
        review = reviews.get(row_id)
        annotation = review.annotation if review else (result.annotation if result else None)
        base["run_id"] = run_id
        base["review_status"] = "reviewed" if review and review.reviewed else (result.state.value if result else "pending")
        base["annotation"] = annotation
        output.append(base)
    return output


def jsonl_bytes(store: ProjectStore, run_id: str) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in export_records(store, run_id)).encode()


def csv_bytes(store: ProjectStore, run_id: str) -> bytes:
    config = store.load_config()
    flat = []
    for record in export_records(store, run_id):
        annotation = record.pop("annotation") or {}
        for key, value in annotation.items():
            record[f"{config.annotation_prefix}{key}"] = _serialize_csv(value)
        flat.append(record)
    return pd.DataFrame(flat).to_csv(index=False).encode()


def run_bundle_bytes(store: ProjectStore, run_id: str) -> bytes:
    memory = io.BytesIO()
    run_dir = store.run_dir(run_id)
    files = [
        store.root / "project.yaml",
        store.root / "schema.json",
        store.root / "fields.json",
        store.root / "graph_schema.json",
        store.root / "instructions.md",
        store.root / "examples.jsonl",
        run_dir / "manifest.json",
        run_dir / "results.jsonl",
        run_dir / "errors.jsonl",
        run_dir / "reviewed.jsonl",
    ]
    with zipfile.ZipFile(memory, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.exists():
                archive.write(path, path.relative_to(store.root))
    return memory.getvalue()


def write_exports(store: ProjectStore, run_id: str) -> tuple[Path, Path]:
    export_dir = store.root / "exports"
    csv_path = export_dir / f"{run_id}.csv"
    jsonl_path = export_dir / f"{run_id}.jsonl"
    csv_path.write_bytes(csv_bytes(store, run_id))
    jsonl_path.write_bytes(jsonl_bytes(store, run_id))
    return csv_path, jsonl_path
