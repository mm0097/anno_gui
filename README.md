# Automatic Annotations

Automatic Annotations is a local Streamlit application for schema-driven LLM extraction, review, dataset export, and graph visualization. Projects may extract ordinary structured annotations or a graph of typed nodes and edges. Project state is stored as portable files; no database or hosted service is required.

## Run

Python 3.11 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
streamlit run app.py
```

With `uv`:

```bash
uv sync --extra dev
uv run streamlit run app.py
```

## Workflow

1. Create a project from the sidebar.
2. Upload CSV/JSONL or load a Hugging Face dataset, then choose ID and text columns.
3. Choose an extraction mode:
   - **Annotations:** define arbitrary scalar, categorical, list, or nested fields.
   - **Graph:** define node types, edge types, allowed endpoint types, and custom properties.
4. Add instructions and optional few-shot examples.
5. Configure an OpenAI-compatible or Hugging Face inference endpoint. Credentials can come from an environment variable or a session-only field.
6. Test rows, then create or resume a batch run.
7. Edit JSON annotations and mark rows reviewed.
8. Export CSV, JSONL, and a reproducibility bundle. Graph projects can also visualize and export the extracted graph directly.

OpenAI, Ollama, LM Studio, and vLLM use the `openai_compatible` provider. Official `api.openai.com` requests use the Responses API with strict Structured Outputs and never silently downgrade schema enforcement. For Ollama, for example, use an OpenAI-compatible URL such as `http://localhost:11434/v1`; a key can be left blank. Non-OpenAI servers that reject strict JSON Schema may fall back to JSON-object mode.

OpenAI reasoning effort defaults to `low` to reduce extraction latency and reasoning-token use. The model screen can select the model default or another supported effort level.

## Project files

```text
project-name/
  project.yaml
  fields.json
  graph_schema.json       # graph projects only
  schema.json
  instructions.md
  examples.jsonl
  source.jsonl
  runs/<run-id>/
    manifest.json
    results.jsonl
    errors.jsonl
    reviewed.jsonl
  graph/mapping.yaml
  exports/
```

Results are appended after each row. Re-running a completed row creates a new attempt, and reviewed edits remain a separate layer. API keys and tokens are never written to project artifacts or bundles.

## Test

```bash
pytest
```
