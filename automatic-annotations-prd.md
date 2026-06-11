# PRD: Schema-Driven Automatic Annotations

**Status:** Draft  
**Working title:** Automatic Annotations  
**Target:** Standalone repository  
**Primary users:** Researchers who need structured annotations from text without building a custom extraction pipeline

## 1. Product Summary

Automatic Annotations is a lightweight, locally runnable web application for configuring LLM-based text extraction, reviewing results, exporting annotated datasets, and optionally deriving graphs from those annotations.

A researcher should be able to:

1. Load a CSV, JSONL file, or Hugging Face dataset.
2. Describe the fields they want extracted.
3. Add instructions and optional few-shot examples.
4. Select and configure an extraction model.
5. Test the configuration on a few rows.
6. Run extraction over the dataset.
7. Review and edit annotations.
8. Download the annotated dataset as CSV or JSONL.
9. Optionally map extracted fields to graph nodes and edges and export the graph.

The product is intended to be portable and project-agnostic. It is not a hosted multi-user annotation platform in its first version.

## 2. Product Principles

- **Local first:** A user can run the complete application on a laptop.
- **No required database:** Projects and run artifacts are ordinary files.
- **Annotations first:** The reviewed tabular dataset is the source of truth. Graphs are derived outputs.
- **Schema driven:** Extraction behavior is defined by fields, types, instructions, and examples rather than custom code.
- **Reproducible:** Every output records the schema, prompt, examples, model, and run settings used to produce it.
- **Provider neutral:** Model providers are adapters behind a common interface.
- **Easy to leave:** Users can always export standard CSV, JSONL, and graph formats.

## 3. Scope

### MVP

- Single-user Streamlit application
- Local CSV and JSONL upload
- Hugging Face dataset loading
- Selection of the input text column and ID column
- Visual extraction-schema editor
- Field types: string, integer, number, boolean, category, list, and nested object
- Per-field descriptions, required flags, allowed values, and examples
- Global extraction instructions
- Few-shot examples
- Model adapters:
  - OpenAI API
  - OpenAI-compatible endpoints, including Ollama, LM Studio, and vLLM where supported
  - Hugging Face Inference Providers or dedicated endpoints
- Test extraction on selected rows
- Batch extraction with progress, retry, cancel, and resume behavior
- Structured-output validation
- Editable review table
- CSV and JSONL download
- Local project save/load
- Basic graph mapping, preview, and export

### Later

- Hugging Face Hub dataset publishing
- Multiple annotators and adjudication
- Inter-annotator agreement
- Authentication and shared hosted workspaces
- Active learning and sampling strategies
- Prompt optimization
- Automated entity resolution
- Scheduled jobs and distributed workers
- Rich span annotation directly over source text
- Image, audio, and video inputs

### Explicitly Out of Scope for MVP

- PostgreSQL, Redis, Celery, or Kubernetes
- A React frontend or separate backend service
- Real-time collaborative editing
- A permanent hosted SaaS offering
- Graph neural-network training
- Fully automatic ontology design

## 4. Primary Workflow

### 4.1 Create or Open a Project

The user creates a named project or opens an existing project directory. A project contains configuration, source metadata, run state, annotations, and exports.

### 4.2 Load Data

Supported sources:

- Upload CSV
- Upload JSONL
- Enter a Hugging Face dataset ID, configuration, and split

The application previews the data and asks the user to select:

- A stable row ID column, or generate one
- The text column passed to the model
- Optional metadata columns preserved in exports

### 4.3 Define Annotations

The user adds fields in a form or table. Each field has:

- Machine-readable name
- Human-readable label
- Data type
- Description/instructions
- Required or optional status
- Allowed values for categorical fields
- Whether multiple values are allowed
- Optional examples and counterexamples

Example:

| Field | Type | Description |
|---|---|---|
| `actors` | list[object] | People, organizations, or social groups mentioned in the text |
| `stance` | category | Overall stance toward the policy: support, oppose, mixed, or unclear |
| `claims` | list[string] | Distinct factual or normative claims made by the author |

The application compiles this definition into JSON Schema and a model prompt. Advanced users can inspect and edit the generated schema.

### 4.4 Add Few-Shot Examples

The user can select dataset rows and provide the expected structured annotations. Examples are stored with the project and included in the extraction prompt subject to a configurable prompt budget.

### 4.5 Select Model

The model screen contains:

- Provider
- Model name
- API base URL where applicable
- API key sourced from an environment variable or session-only input
- Temperature
- Maximum output tokens
- Concurrency
- Retry limit

Secrets must never be written into project files or exported run manifests.

### 4.6 Test

The user runs extraction on one or more rows. The application displays:

- Raw source text
- Parsed structured result
- Validation errors
- Raw model response when parsing fails
- Latency and estimated or reported token usage

The schema and prompt can be revised without starting a full run.

### 4.7 Run Extraction

Each source row receives one of these states:

- Pending
- Running
- Completed
- Failed
- Reviewed

Results are written incrementally after every row so interrupted runs can resume. Re-running a completed row requires explicit selection and produces a new run result rather than silently overwriting reviewed data.

### 4.8 Review

The review screen shows the source columns alongside flattened annotation fields. The user can:

- Filter by state or validation error
- Edit extracted values
- Mark rows as reviewed
- Re-run selected rows
- Inspect the model output and run provenance

Nested fields may be edited as JSON in the MVP. Specialized nested editors are a later enhancement.

### 4.9 Export

Required exports:

- **CSV:** Original selected columns plus annotation columns
- **JSONL:** One complete structured record per source row
- **Run bundle:** Schema, instructions, few-shot examples, model settings, and run summary

CSV serialization rules:

- Scalar values use native cells.
- Lists and nested objects use JSON strings in cells.
- Annotation columns use a configurable prefix, defaulting to `annotation_`.
- The export contains a run ID and review status.

## 5. Graph Builder

Graph building is optional and occurs after extraction.

The user defines mappings such as:

- An annotation object or field becomes a node.
- A field supplies the node ID, label, and type.
- A relation object becomes an edge.
- Fields supply source, target, edge type, confidence, and evidence.

MVP graph functionality:

- Configure node and edge mappings
- Exact-ID deduplication
- Small interactive preview
- Basic validation for missing node references
- Export as node CSV, edge CSV, JSON, GraphML, and NetworkX pickle only if clearly marked Python-specific

Fuzzy entity resolution is not part of the MVP. The tool should expose unresolved or duplicate candidates rather than silently merge them.

## 6. Hugging Face Support

### MVP

- Load public datasets with `datasets.load_dataset()`
- Load private datasets using a session-only token or environment variable
- Select dataset configuration and split
- Preserve Hugging Face feature metadata where practical
- Run models through a Hugging Face inference adapter

### Follow-Up

- Push reviewed annotations to a new or existing Hub dataset repository
- Generate a dataset card containing the schema, model provenance, limitations, and review status
- Optional hosted deployment guidance

Publishing to the Hub must require an explicit confirmation and show exactly which columns and rows will be uploaded.

## 7. Technical Design

### Recommended Stack

- **Language:** Python 3.11+
- **GUI:** Streamlit
- **Validation:** Pydantic 2 and JSON Schema
- **Tabular data:** pandas initially; optional Polars migration if scale demands it
- **Dataset integration:** Hugging Face `datasets`
- **Graph processing:** NetworkX
- **Graph preview:** a Streamlit-compatible Cytoscape component or PyVis
- **HTTP/model clients:** provider-specific SDKs behind a small adapter interface
- **Tests:** pytest
- **Packaging:** `pyproject.toml`, optional `uv.lock`

Streamlit is sufficient because the MVP is a single-user research tool centered on forms, tables, progress, and downloads. A separate frontend/backend architecture should only be introduced if multi-user hosting or complex browser interactions become actual requirements.

### Provider Interface

```python
class ExtractionProvider(Protocol):
    def extract(
        self,
        text: str,
        schema: dict,
        instructions: str,
        examples: list[dict],
        settings: ModelSettings,
    ) -> ExtractionResponse: ...
```

`ExtractionResponse` includes parsed data, raw output, usage, latency, model identity, and error details.

### Local Project Layout

```text
project-name/
  project.yaml
  schema.json
  instructions.md
  examples.jsonl
  source.jsonl
  runs/
    <run-id>/
      manifest.json
      results.jsonl
      errors.jsonl
      reviewed.jsonl
  graph/
    mapping.yaml
  exports/
```

Project writes should be atomic where possible. JSONL result files are append-oriented, and manifests record completion state. SQLite may be added later if review performance or indexing becomes a problem; it is not a prerequisite.

## 8. Deployment

### Local

```bash
uv sync
uv run streamlit run app.py
```

The application opens in the browser and stores projects in a user-selected local directory.

Hosted deployment, including Hugging Face Spaces, is deferred until after the local application is stable. Local execution remains the default for sensitive research data.

## 9. Non-Functional Requirements

- A new user can reach a test extraction within 10 minutes.
- No external service is required when using a local OpenAI-compatible model.
- A failed or interrupted run can resume without repeating completed rows.
- Every annotation can be traced to source row, run, model, schema, and prompt version.
- API keys and Hugging Face tokens are never persisted in project artifacts.
- A project can be archived and moved by copying one directory.
- The UI remains usable for at least 50,000 rows, while review may use pagination or sampling.
- Invalid structured outputs are retained for debugging rather than discarded.

## 10. Success Criteria

The MVP is successful when a researcher can, without writing code:

1. Import an arbitrary text dataset.
2. Define at least one scalar and one nested annotation field.
3. Test and run extraction using a selected model.
4. Correct results in the UI.
5. Download a CSV containing the original data and reviewed annotations.
6. Reopen the project and reproduce the run configuration.
7. Optionally export a graph derived from the annotations.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Model outputs do not match the schema | Prefer native structured output; validate with Pydantic; retry with error context |
| Costs grow unexpectedly | Show row count, concurrency, token usage, and an estimate before full runs |
| Interrupted runs lose progress | Append each result immediately and support resume by stable row ID |
| CSV cannot represent nested structures cleanly | Encode nested values as JSON strings and provide JSONL export |
| Researchers mistake model output for ground truth | Preserve review status and distinguish extracted from reviewed values |
| Hosted execution exposes sensitive data | Keep hosted deployment out of the MVP and make local execution the default |
| Entity resolution corrupts graph identity | Use exact IDs in MVP and require explicit resolution for ambiguous entities |

## 12. Delivery Plan

### Milestone 1: Extraction Loop

- Project model and local persistence
- CSV/JSONL import
- Schema editor
- One provider adapter
- Test extraction
- Batch run with incremental results
- CSV/JSONL export

### Milestone 2: Review and Reproducibility

- Editable review table
- Run manifests and prompt preview
- Resume and selective re-run
- Error inspection
- Local installation and startup documentation

### Milestone 3: Hugging Face and Graphs

- Hugging Face dataset import
- Hugging Face inference adapter
- Graph mapping, preview, and export
- Optional Hub publishing
- Optional hosted deployment guidance

## 13. Open Product Decisions

- Whether span-level evidence highlighting is necessary for the first research use case
- Whether nested annotations need a custom visual editor or JSON editing is acceptable initially
- Whether reviewed edits overwrite extracted values or are stored as a separate annotation layer; separate layers are recommended
- Which model provider should be the reference implementation
- Whether project directories may contain full source data or only references to external files
