from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io import atomic_write_text, read_jsonl, replace_jsonl, write_json, write_yaml
from .models import (
    AnnotationField,
    FewShotExample,
    GraphExtractionSchema,
    ProjectConfig,
    ReviewedAnnotation,
    RunManifest,
    RunResult,
    utc_now,
)
from .schema import compile_graph_schema, compile_schema


class ProjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    @classmethod
    def create(cls, parent: str | Path, name: str) -> "ProjectStore":
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-.")
        if not slug:
            raise ValueError("Project name must contain letters or numbers")
        store = cls(Path(parent).expanduser() / slug)
        if store.root.exists() and any(store.root.iterdir()):
            raise FileExistsError(f"Project directory is not empty: {store.root}")
        for directory in ("runs", "graph", "exports"):
            (store.root / directory).mkdir(parents=True, exist_ok=True)
        store.save_config(ProjectConfig(name=name))
        store.save_fields([])
        store.save_instructions("")
        store.save_examples([])
        return store

    @property
    def exists(self) -> bool:
        return (self.root / "project.yaml").exists()

    def load_config(self) -> ProjectConfig:
        return ProjectConfig.model_validate(yaml.safe_load((self.root / "project.yaml").read_text()))

    def save_config(self, config: ProjectConfig) -> None:
        config.updated_at = utc_now()
        # model settings intentionally contain only an environment variable name, never a key.
        write_yaml(self.root / "project.yaml", config.model_dump(mode="json"))

    def load_fields(self) -> list[AnnotationField]:
        path = self.root / "fields.json"
        if path.exists():
            return [AnnotationField.model_validate(value) for value in json.loads(path.read_text())]
        return []

    def save_fields(self, fields: list[AnnotationField]) -> dict[str, Any]:
        schema = compile_schema(fields)
        write_json(self.root / "fields.json", [field.model_dump(mode="json") for field in fields])
        write_json(self.root / "schema.json", schema)
        if (self.root / "project.yaml").exists():
            config = self.load_config()
            config.extraction_mode = "annotations"
            self.save_config(config)
        return schema

    def load_schema(self) -> dict[str, Any]:
        return json.loads((self.root / "schema.json").read_text())

    def save_instructions(self, instructions: str) -> None:
        atomic_write_text(self.root / "instructions.md", instructions)

    def load_instructions(self) -> str:
        path = self.root / "instructions.md"
        return path.read_text() if path.exists() else ""

    def save_examples(self, examples: list[FewShotExample]) -> None:
        replace_jsonl(self.root / "examples.jsonl", [item.model_dump(mode="json") for item in examples])

    def load_examples(self) -> list[FewShotExample]:
        return [FewShotExample.model_validate(value) for value in read_jsonl(self.root / "examples.jsonl")]

    def save_source(self, frame: pd.DataFrame, config: ProjectConfig) -> None:
        frame = frame.copy()
        if config.id_column not in frame:
            frame.insert(0, config.id_column, [str(index) for index in range(len(frame))])
        if config.text_column not in frame:
            raise ValueError(f"Text column not found: {config.text_column}")
        selected_columns = list(dict.fromkeys([
            config.id_column,
            config.text_column,
            *[column for column in config.metadata_columns if column in frame.columns],
        ]))
        frame = frame[selected_columns].copy()
        if frame[config.id_column].isna().any():
            raise ValueError("The selected ID column must contain unique, non-null values")
        ids = frame[config.id_column].astype(str)
        if ids.duplicated().any():
            raise ValueError("The selected ID column must contain unique, non-null values")
        frame[config.id_column] = ids
        records = [
            {key: self._json_safe(value) for key, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]
        replace_jsonl(self.root / "source.jsonl", records)
        self.save_config(config)

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return [cls._json_safe(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return cls._json_safe(value.item())
        if isinstance(value, (pd.Timestamp, datetime, date)):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): cls._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._json_safe(item) for item in value]
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def load_source(self) -> pd.DataFrame:
        return pd.DataFrame(read_jsonl(self.root / "source.jsonl"))

    def list_runs(self) -> list[str]:
        runs = self.root / "runs"
        if not runs.exists():
            return []
        return sorted(
            [path.name for path in runs.iterdir() if (path / "manifest.json").exists()],
            reverse=True,
        )

    def run_dir(self, run_id: str) -> Path:
        return self.root / "runs" / run_id

    def save_manifest(self, manifest: RunManifest) -> None:
        manifest.updated_at = utc_now()
        write_json(
            self.run_dir(manifest.run_id) / "manifest.json",
            manifest.model_dump(mode="json", by_alias=True),
        )

    def load_manifest(self, run_id: str) -> RunManifest:
        return RunManifest.model_validate(json.loads((self.run_dir(run_id) / "manifest.json").read_text()))

    def load_results(self, run_id: str) -> list[RunResult]:
        return [RunResult.model_validate(value) for value in read_jsonl(self.run_dir(run_id) / "results.jsonl")]

    def latest_results(self, run_id: str) -> dict[str, RunResult]:
        latest: dict[str, RunResult] = {}
        for result in self.load_results(run_id):
            latest[result.row_id] = result
        return latest

    def load_reviews(self, run_id: str) -> dict[str, ReviewedAnnotation]:
        reviews = {}
        for value in read_jsonl(self.run_dir(run_id) / "reviewed.jsonl"):
            review = ReviewedAnnotation.model_validate(value)
            reviews[review.row_id] = review
        return reviews

    def save_reviews(self, run_id: str, reviews: dict[str, ReviewedAnnotation]) -> None:
        replace_jsonl(
            self.run_dir(run_id) / "reviewed.jsonl",
            [value.model_dump(mode="json") for value in reviews.values()],
        )

    def save_graph_schema(self, definition: GraphExtractionSchema) -> dict[str, Any]:
        schema = compile_graph_schema(definition)
        write_json(self.root / "graph_schema.json", definition.model_dump(mode="json"))
        write_json(self.root / "schema.json", schema)
        config = self.load_config()
        config.extraction_mode = "graph"
        self.save_config(config)
        return schema

    def load_graph_schema(self) -> GraphExtractionSchema:
        path = self.root / "graph_schema.json"
        if not path.exists():
            return GraphExtractionSchema()
        return GraphExtractionSchema.model_validate(json.loads(path.read_text()))

    @staticmethod
    def content_hash(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()
