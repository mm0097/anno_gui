from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class FieldType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CATEGORY = "category"
    LIST = "list"
    OBJECT = "object"


class AnnotationField(BaseModel):
    name: str
    label: str = ""
    type: FieldType = FieldType.STRING
    description: str = ""
    required: bool = False
    allowed_values: list[str] = Field(default_factory=list)
    multiple: bool = False
    examples: list[Any] = Field(default_factory=list)
    counterexamples: list[Any] = Field(default_factory=list)
    item_type: FieldType = FieldType.STRING
    children: list["AnnotationField"] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        value = value.strip()
        if not value or not value.replace("_", "a").isalnum() or value[0].isdigit():
            raise ValueError("Use a non-empty identifier containing letters, numbers, or underscores")
        return value

    @model_validator(mode="after")
    def validate_options(self) -> "AnnotationField":
        if self.type == FieldType.CATEGORY and not self.allowed_values:
            raise ValueError("Category fields require at least one allowed value")
        if self.type == FieldType.OBJECT and not self.children:
            raise ValueError("Object fields require child fields")
        return self


class ModelSettings(BaseModel):
    provider: str = "openai_compatible"
    model: str = ""
    api_base: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_output_tokens: int = Field(default=1000, ge=1)
    reasoning_effort: str = "low"
    concurrency: int = Field(default=1, ge=1, le=32)

    @field_validator("reasoning_effort")
    @classmethod
    def valid_reasoning_effort(cls, value: str) -> str:
        allowed = {"default", "none", "minimal", "low", "medium", "high", "xhigh"}
        if value not in allowed:
            raise ValueError(f"Reasoning effort must be one of: {sorted(allowed)}")
        return value
    retry_limit: int = Field(default=2, ge=0, le=10)
    timeout_seconds: float = Field(default=120, gt=0)


class ProjectConfig(BaseModel):
    name: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    id_column: str = "_row_id"
    text_column: str = ""
    metadata_columns: list[str] = Field(default_factory=list)
    annotation_prefix: str = "annotation_"
    source_kind: str = "upload"
    source_reference: dict[str, Any] = Field(default_factory=dict)
    extraction_mode: str = "annotations"
    model: ModelSettings = Field(default_factory=ModelSettings)


class FewShotExample(BaseModel):
    row_id: str
    text: str
    annotation: dict[str, Any]


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class ExtractionResponse(BaseModel):
    parsed: dict[str, Any] | None = None
    raw_output: str = ""
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float = 0
    model: str = ""
    error: str | None = None
    validation_errors: list[str] = Field(default_factory=list)


class RowState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"


class RunResult(BaseModel):
    run_id: str
    attempt_id: str
    row_id: str
    state: RowState
    source_index: int
    annotation: dict[str, Any] | None = None
    raw_output: str = ""
    validation_errors: list[str] = Field(default_factory=list)
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    latency_seconds: float = 0
    model: str = ""
    created_at: str = Field(default_factory=utc_now)


class ReviewedAnnotation(BaseModel):
    row_id: str
    annotation: dict[str, Any]
    reviewed: bool = True
    updated_at: str = Field(default_factory=utc_now)
    source_attempt_id: str | None = None


class RunManifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    status: str = "running"
    schema_hash: str
    prompt_hash: str
    schema_snapshot: dict[str, Any] = Field(alias="schema", serialization_alias="schema")
    instructions: str
    examples: list[dict[str, Any]]
    model_settings: dict[str, Any]
    total_rows: int
    completed_rows: int = 0
    failed_rows: int = 0
    cancelled: bool = False


class NodeTypeDefinition(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    properties: list[AnnotationField] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return AnnotationField.valid_name(value)


class EdgeTypeDefinition(BaseModel):
    name: str
    label: str = ""
    description: str = ""
    source_types: list[str] = Field(default_factory=list)
    target_types: list[str] = Field(default_factory=list)
    directed: bool = True
    properties: list[AnnotationField] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        return AnnotationField.valid_name(value)


class GraphExtractionSchema(BaseModel):
    node_types: list[NodeTypeDefinition] = Field(default_factory=list)
    edge_types: list[EdgeTypeDefinition] = Field(default_factory=list)
    include_evidence: bool = True
    include_confidence: bool = True

    @model_validator(mode="after")
    def validate_graph_types(self) -> "GraphExtractionSchema":
        node_names = [item.name for item in self.node_types]
        edge_names = [item.name for item in self.edge_types]
        if len(node_names) != len(set(node_names)):
            raise ValueError("Node type names must be unique")
        if len(edge_names) != len(set(edge_names)):
            raise ValueError("Edge type names must be unique")
        known = set(node_names)
        for edge in self.edge_types:
            unknown = (set(edge.source_types) | set(edge.target_types)) - known
            if unknown:
                raise ValueError(f"Edge type '{edge.name}' references unknown node types: {sorted(unknown)}")
        return self
