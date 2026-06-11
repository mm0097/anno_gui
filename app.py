from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from automatic_annotations.data import load_huggingface, load_upload
from automatic_annotations.exports import csv_bytes, export_records, jsonl_bytes, run_bundle_bytes
from automatic_annotations.graph import build_extracted_graph, graph_exports
from automatic_annotations.models import (
    AnnotationField,
    EdgeTypeDefinition,
    FewShotExample,
    FieldType,
    GraphExtractionSchema,
    ModelSettings,
    NodeTypeDefinition,
    ProjectConfig,
    ReviewedAnnotation,
)
from automatic_annotations.project import ProjectStore
from automatic_annotations.providers import get_provider
from automatic_annotations.runner import ExtractionRunner
from automatic_annotations.schema import build_prompt

st.set_page_config(page_title="Automatic Annotations", page_icon="AA", layout="wide")


def init_state() -> None:
    defaults = {
        "store_path": "",
        "pending_frame": None,
        "api_key": "",
        "cancel_requested": False,
        "test_results": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def store() -> ProjectStore | None:
    path = st.session_state.store_path
    candidate = ProjectStore(path) if path else None
    return candidate if candidate and candidate.exists else None


def project_sidebar() -> ProjectStore | None:
    with st.sidebar:
        st.title("Automatic Annotations")
        st.caption("Local, schema-driven extraction")
        project_path = st.text_input("Open project directory", value=st.session_state.store_path)
        if st.button("Open", use_container_width=True):
            candidate = ProjectStore(project_path)
            if candidate.exists:
                st.session_state.store_path = str(candidate.root)
                st.rerun()
            st.error("No project.yaml found in that directory")
        with st.expander("Create project"):
            parent = st.text_input("Parent directory", value=str(Path.cwd() / "projects"))
            name = st.text_input("Project name")
            if st.button("Create", disabled=not name):
                try:
                    created = ProjectStore.create(parent, name)
                    st.session_state.store_path = str(created.root)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        current = store()
        if current:
            st.success(current.load_config().name)
            st.code(str(current.root), language=None)
        st.divider()
        st.caption("Secrets entered in this session are never saved.")
        return current


def data_page(current: ProjectStore) -> None:
    st.header("1. Data")
    upload_tab, hf_tab = st.tabs(["CSV / JSONL", "Hugging Face"])
    with upload_tab:
        uploaded = st.file_uploader("Source file", type=["csv", "jsonl", "ndjson"])
        if uploaded and st.button("Read upload"):
            try:
                st.session_state.pending_frame = load_upload(uploaded, uploaded.name)
            except Exception as exc:
                st.error(str(exc))
    with hf_tab:
        col1, col2, col3 = st.columns(3)
        dataset_id = col1.text_input("Dataset ID", placeholder="org/dataset")
        dataset_config = col2.text_input("Configuration (optional)")
        split = col3.text_input("Split", value="train")
        hf_token = st.text_input("Session-only HF token", type="password")
        if st.button("Load dataset", disabled=not dataset_id):
            with st.spinner("Loading dataset..."):
                try:
                    st.session_state.pending_frame = load_huggingface(
                        dataset_id, dataset_config, split, hf_token or os.getenv("HF_TOKEN")
                    )
                except Exception as exc:
                    st.error(str(exc))

    frame = st.session_state.pending_frame
    if frame is None and (current.root / "source.jsonl").exists():
        frame = current.load_source()
    if frame is None:
        st.info("Upload a file or load a Hugging Face dataset.")
        return
    st.caption(f"{len(frame):,} rows, {len(frame.columns)} columns")
    st.dataframe(frame.head(100), use_container_width=True)
    columns = list(map(str, frame.columns))
    config = current.load_config()
    generated = "Generate _row_id"
    id_default = columns.index(config.id_column) if config.id_column in columns else 0
    id_choice = st.selectbox("Stable row ID", [generated, *columns], index=id_default + 1 if config.id_column in columns else 0)
    text_default = columns.index(config.text_column) if config.text_column in columns else 0
    text_column = st.selectbox("Text column", columns, index=text_default)
    additional_columns = [column for column in columns if column not in {id_choice, text_column}]
    metadata = st.multiselect(
        "Additional columns to keep",
        additional_columns,
        default=[column for column in config.metadata_columns if column in additional_columns],
        help="Only the ID, text, and selected additional columns are stored in the project and included in exports.",
    )
    if st.button("Save source configuration", type="primary"):
        try:
            config.id_column = "_row_id" if id_choice == generated else id_choice
            config.text_column = text_column
            config.metadata_columns = [column for column in metadata if column not in {config.id_column, text_column}]
            current.save_source(frame, config)
            st.session_state.pending_frame = None
            st.success("Source saved to the project.")
        except Exception as exc:
            st.error(str(exc))


def parse_properties(value: Any) -> list[AnnotationField]:
    return [AnnotationField.model_validate(item) for item in json.loads(str(value or "[]"))]


def annotation_schema_editor(current: ProjectStore) -> None:
    fields = current.load_fields()
    rows = [{
        "name": field.name,
        "label": field.label,
        "type": field.type.value,
        "description": field.description,
        "required": field.required,
        "multiple": field.multiple,
        "allowed_values": ", ".join(field.allowed_values),
        "item_type": field.item_type.value,
        "children_json": json.dumps([child.model_dump(mode="json") for child in field.children], ensure_ascii=False),
    } for field in fields]
    edited = st.data_editor(
        pd.DataFrame(rows, columns=[
            "name", "label", "type", "description", "required", "multiple",
            "allowed_values", "item_type", "children_json",
        ]),
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={
            "type": st.column_config.SelectboxColumn("Type", options=[item.value for item in FieldType]),
            "item_type": st.column_config.SelectboxColumn("List item type", options=[item.value for item in FieldType]),
            "children_json": st.column_config.TextColumn("Nested fields (JSON)", width="large"),
        },
        key="annotation_field_editor",
    )
    st.caption("Lists and nested objects are supported. For object fields, children_json contains child field definitions.")
    if st.button("Save annotation schema", type="primary"):
        try:
            parsed = [AnnotationField(
                name=str(row["name"]).strip(), label=str(row.get("label") or ""),
                type=FieldType(str(row.get("type") or "string")),
                description=str(row.get("description") or ""), required=bool(row.get("required", False)),
                multiple=bool(row.get("multiple", False)),
                allowed_values=[value.strip() for value in str(row.get("allowed_values") or "").split(",") if value.strip()],
                item_type=FieldType(str(row.get("item_type") or "string")),
                children=parse_properties(row.get("children_json")),
            ) for row in edited.to_dict(orient="records") if str(row.get("name") or "").strip()]
            schema = current.save_fields(parsed)
            config = current.load_config()
            config.extraction_mode = "annotations"
            current.save_config(config)
            st.success("Annotation schema saved.")
            st.json(schema)
        except Exception as exc:
            st.error(str(exc))


def graph_schema_page(current: ProjectStore) -> None:
    st.write("Define what entities the model should extract and which relations may connect them.")
    definition = current.load_graph_schema()
    node_rows = [{
        "name": item.name,
        "label": item.label,
        "description": item.description,
        "properties_json": json.dumps([field.model_dump(mode="json") for field in item.properties], ensure_ascii=False),
    } for item in definition.node_types]
    st.subheader("Node types")
    nodes = st.data_editor(
        pd.DataFrame(node_rows, columns=["name", "label", "description", "properties_json"]),
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"properties_json": st.column_config.TextColumn("Properties (JSON)", width="large")},
        key="node_type_editor",
    )
    st.caption('Example property: [{"name":"role","type":"string","description":"Role in the text"}]')

    edge_rows = [{
        "name": item.name,
        "label": item.label,
        "description": item.description,
        "source_types": ", ".join(item.source_types),
        "target_types": ", ".join(item.target_types),
        "directed": item.directed,
        "properties_json": json.dumps([field.model_dump(mode="json") for field in item.properties], ensure_ascii=False),
    } for item in definition.edge_types]
    st.subheader("Edge types")
    edges = st.data_editor(
        pd.DataFrame(edge_rows, columns=[
            "name", "label", "description", "source_types", "target_types", "directed", "properties_json"
        ]),
        num_rows="dynamic", use_container_width=True, hide_index=True,
        column_config={"properties_json": st.column_config.TextColumn("Properties (JSON)", width="large")},
        key="edge_type_editor",
    )
    st.caption("Source and target types are comma-separated node type names. Leave either side empty to allow any node type.")
    col1, col2 = st.columns(2)
    include_evidence = col1.checkbox("Extract supporting evidence", value=definition.include_evidence)
    include_confidence = col2.checkbox("Extract relation confidence", value=definition.include_confidence)
    if st.button("Save graph schema", type="primary"):
        try:
            parsed_nodes = [NodeTypeDefinition(
                name=str(row["name"]).strip(), label=str(row.get("label") or ""),
                description=str(row.get("description") or ""), properties=parse_properties(row.get("properties_json")),
            ) for row in nodes.to_dict(orient="records") if str(row.get("name") or "").strip()]
            parsed_edges = [EdgeTypeDefinition(
                name=str(row["name"]).strip(), label=str(row.get("label") or ""),
                description=str(row.get("description") or ""),
                source_types=[value.strip() for value in str(row.get("source_types") or "").split(",") if value.strip()],
                target_types=[value.strip() for value in str(row.get("target_types") or "").split(",") if value.strip()],
                directed=bool(row.get("directed", True)), properties=parse_properties(row.get("properties_json")),
            ) for row in edges.to_dict(orient="records") if str(row.get("name") or "").strip()]
            definition = GraphExtractionSchema(
                node_types=parsed_nodes, edge_types=parsed_edges,
                include_evidence=include_evidence, include_confidence=include_confidence,
            )
            schema = current.save_graph_schema(definition)
            config = current.load_config()
            config.extraction_mode = "graph"
            current.save_config(config)
            st.success("Graph extraction schema saved and validated.")
            st.json(schema)
        except Exception as exc:
            st.error(str(exc))
    if definition.node_types and (current.root / "schema.json").exists():
        with st.expander("Generated extraction JSON Schema"):
            st.json(current.load_schema())


def extraction_schema_page(current: ProjectStore) -> None:
    st.header("2. Extraction Schema")
    config = current.load_config()
    options = ["Annotations", "Graph"]
    selected = st.radio(
        "What should be extracted?",
        options,
        index=1 if config.extraction_mode == "graph" else 0,
        horizontal=True,
    )
    if selected == "Annotations":
        st.write("Define arbitrary scalar, categorical, list, or nested annotation fields.")
        annotation_schema_editor(current)
    else:
        graph_schema_page(current)


def prompt_page(current: ProjectStore) -> None:
    st.header("3. Instructions & Examples")
    instructions = st.text_area("Global extraction instructions", current.load_instructions(), height=180)
    if st.button("Save instructions"):
        current.save_instructions(instructions)
        st.success("Instructions saved.")
    if not (current.root / "source.jsonl").exists():
        st.info("Save a source dataset before adding examples.")
        return
    source = current.load_source()
    config = current.load_config()
    examples = current.load_examples()
    options = {f"{row[config.id_column]}: {str(row[config.text_column])[:80]}": index for index, row in source.iterrows()}
    selected = st.selectbox("Example source row", options, index=None, placeholder="Select a row")
    annotation_json = st.text_area("Expected annotation JSON", "{}", height=160)
    if st.button("Add example", disabled=selected is None):
        try:
            row = source.iloc[options[selected]]
            examples.append(FewShotExample(
                row_id=str(row[config.id_column]), text=str(row[config.text_column]), annotation=json.loads(annotation_json)
            ))
            current.save_examples(examples)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if examples:
        example_frame = pd.DataFrame([
            {"row_id": example.row_id, "text": example.text, "annotation": json.dumps(example.annotation, ensure_ascii=False)}
            for example in examples
        ])
        st.dataframe(example_frame, use_container_width=True, hide_index=True)
        remove = st.multiselect("Remove examples", [example.row_id for example in examples])
        if st.button("Remove selected", disabled=not remove):
            current.save_examples([example for example in examples if example.row_id not in remove])
            st.rerun()


def model_page(current: ProjectStore) -> None:
    st.header("4. Model")
    config = current.load_config()
    model = config.model
    provider = st.selectbox("Provider", ["openai_compatible", "huggingface"], index=0 if model.provider == "openai_compatible" else 1)
    col1, col2 = st.columns(2)
    model_name = col1.text_input("Model name", model.model)
    api_base = col2.text_input("API base URL", model.api_base)
    if "api.openai.com" in api_base:
        st.caption("Official OpenAI requests use the Responses API with strict Structured Outputs. Schema failures are not downgraded to plain JSON mode.")
    else:
        st.caption("OpenAI-compatible endpoints use Chat Completions and fall back to JSON mode only if strict JSON Schema is unsupported.")
    key_env = st.text_input("API key environment variable", model.api_key_env)
    st.session_state.api_key = st.text_input("Session-only API key", type="password", value=st.session_state.api_key)
    col1, col2, col3, col4 = st.columns(4)
    temperature = col1.number_input("Temperature", 0.0, 2.0, model.temperature, 0.1)
    max_tokens = col2.number_input("Max output tokens", 1, 100000, model.max_output_tokens)
    concurrency = col3.number_input("Concurrency", 1, 32, model.concurrency)
    retries = col4.number_input("Retry limit", 0, 10, model.retry_limit)
    col1, col2 = st.columns(2)
    reasoning_options = ["default", "none", "minimal", "low", "medium", "high", "xhigh"]
    reasoning_effort = col1.selectbox(
        "Reasoning effort",
        reasoning_options,
        index=reasoning_options.index(model.reasoning_effort),
        help="Low is recommended for faster extraction. Supported values depend on the selected OpenAI model.",
    )
    timeout = col2.number_input("Timeout seconds", 1.0, 3600.0, model.timeout_seconds)
    if st.button("Save model settings", type="primary"):
        config.model = ModelSettings(
            provider=provider, model=model_name, api_base=api_base, api_key_env=key_env,
            temperature=temperature, max_output_tokens=max_tokens, reasoning_effort=reasoning_effort,
            concurrency=concurrency,
            retry_limit=retries, timeout_seconds=timeout,
        )
        current.save_config(config)
        st.success("Model settings saved without credentials.")
    if (current.root / "schema.json").exists():
        with st.expander("Compiled prompt preview"):
            prompt = build_prompt(
                current.load_schema(), current.load_instructions(),
                [example.model_dump(mode="json") for example in current.load_examples()],
            )
            st.code(prompt)


def render_graph(graph: Any, warnings: list[str], download: bool = False) -> None:
    col1, col2 = st.columns(2)
    col1.metric("Nodes", graph.number_of_nodes())
    col2.metric("Edges", graph.number_of_edges())
    if warnings:
        st.warning("\n".join(warnings[:100]))
    if graph.number_of_nodes() <= 200 and graph.number_of_edges() <= 500:
        def dot_escape(value: Any) -> str:
            return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

        dot = ["digraph extraction {", "rankdir=LR;", "node [shape=box, style=rounded];"]
        for node, data in graph.nodes(data=True):
            label = data.get("label", node)
            node_type = data.get("type", "")
            dot.append(f'"{dot_escape(node)}" [label="{dot_escape(label)}\\n{dot_escape(node_type)}"];')
        for source, target, data in graph.edges(data=True):
            dot.append(
                f'"{dot_escape(source)}" -> "{dot_escape(target)}" '
                f'[label="{dot_escape(data.get("type", ""))}"];'
            )
        dot.append("}")
        st.graphviz_chart("\n".join(dot), use_container_width=True)
    else:
        st.info("Preview is limited to 200 nodes and 500 edges. Exports contain the full graph.")
    node_rows = [{"id": node, **data} for node, data in graph.nodes(data=True)]
    edge_rows = [{"source": source, "target": target, **data} for source, target, data in graph.edges(data=True)]
    with st.expander("Node and edge tables"):
        st.dataframe(pd.DataFrame(node_rows).head(500), use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(edge_rows).head(500), use_container_width=True, hide_index=True)
    if download:
        exports = graph_exports(graph)
        columns = st.columns(len(exports))
        for column, (filename, content) in zip(columns, exports.items()):
            column.download_button(filename, content, filename, use_container_width=True)


def extract_page(current: ProjectStore) -> None:
    st.header("5. Test & Run")
    config = current.load_config()
    definition = current.load_graph_schema()
    has_schema = bool(definition.node_types) if config.extraction_mode == "graph" else bool(current.load_fields())
    if not config.text_column or not (current.root / "source.jsonl").exists() or not has_schema:
        st.warning("Configure a source dataset, text column, and extraction schema first.")
        return
    source = current.load_source()
    provider = get_provider(config.model.provider)
    runner = ExtractionRunner(current, provider, st.session_state.api_key or None)
    test_indices = st.multiselect("Rows to test", list(range(len(source))), default=[0] if len(source) else [], max_selections=10)
    st.caption("Test extraction uses one attempt per row and a 120-second request timeout. Batch runs use the saved retry and timeout settings.")
    if st.button("Test extraction", disabled=not test_indices):
        test_status = st.status("Starting test extraction...", expanded=True)

        def test_progress(position: int, total: int, row_index: int) -> None:
            test_status.write(f"Testing row {row_index} ({position}/{total}), up to 120 seconds...")

        st.session_state.test_results = runner.test_rows(test_indices, progress=test_progress)
        failed = sum(result.state.value == "failed" for result in st.session_state.test_results)
        test_status.update(
            label=f"Test complete: {len(st.session_state.test_results) - failed} succeeded, {failed} failed",
            state="error" if failed else "complete",
            expanded=bool(failed),
        )
    for result in st.session_state.test_results:
        with st.expander(f"Row {result.source_index}: {result.state.value}", expanded=True):
            st.write(source.iloc[result.source_index][config.text_column])
            st.json(result.annotation)
            if result.annotation and config.extraction_mode == "graph":
                graph, warnings = build_extracted_graph([{"annotation": result.annotation}], definition)
                render_graph(graph, warnings)
            if result.validation_errors:
                st.error("\n".join(result.validation_errors))
            if result.error:
                st.error(result.error)
            st.caption(f"{result.latency_seconds:.2f}s | model: {result.model} | usage: {result.usage.model_dump()}")
            if result.raw_output and (result.error or result.validation_errors):
                st.code(result.raw_output)

    st.subheader("Batch extraction")
    runs = current.list_runs()
    run_choice = st.selectbox("Run", ["Create new run", *runs])
    selected_run = None if run_choice == "Create new run" else run_choice
    if selected_run:
        manifest = current.load_manifest(selected_run)
        st.caption(f"Status: {manifest.status}; completed: {manifest.completed_rows}; failed: {manifest.failed_rows}")
    rerun = st.checkbox("Explicitly rerun completed rows as new attempts")
    row_filter = st.text_input("Optional row IDs (comma-separated)")
    col1, col2 = st.columns([3, 1])
    run_clicked = col1.button("Start / resume extraction", type="primary", use_container_width=True)
    if col2.button("Request cancel", use_container_width=True):
        st.session_state.cancel_requested = True
    if run_clicked:
        st.session_state.cancel_requested = False
        row_ids = [value.strip() for value in row_filter.split(",") if value.strip()] or None
        if selected_run is None:
            selected_run = runner.create_run(len(source) if row_ids is None else len(row_ids)).run_id
        progress_bar = st.progress(0)
        status = st.empty()

        def update(done: int, total: int, result: Any) -> None:
            progress_bar.progress(done / max(total, 1))
            status.write(f"{done}/{total}: row {result.row_id} -> {result.state.value}")

        with st.spinner("Extraction in progress. Results are saved after each row."):
            manifest = runner.run(
                selected_run, row_ids=row_ids, rerun_completed=rerun, progress=update,
                cancelled=lambda: st.session_state.cancel_requested,
            )
        st.success(f"Run {manifest.run_id}: {manifest.status}")


def review_page(current: ProjectStore) -> None:
    st.header("6. Review")
    runs = current.list_runs()
    if not runs:
        st.info("Run an extraction first.")
        return
    run_id = st.selectbox("Run to review", runs, key="review_run")
    source = current.load_source()
    config = current.load_config()
    results = current.latest_results(run_id)
    reviews = current.load_reviews(run_id)
    state_filter = st.multiselect("State", ["pending", "completed", "failed", "reviewed"], default=[])
    errors_only = st.checkbox("Validation errors only")
    rows = []
    for index, source_row in source.iterrows():
        row_id = str(source_row[config.id_column])
        result = results.get(row_id)
        review = reviews.get(row_id)
        state = "reviewed" if review and review.reviewed else (result.state.value if result else "pending")
        if state_filter and state not in state_filter:
            continue
        if errors_only and not (result and (result.error or result.validation_errors)):
            continue
        annotation = review.annotation if review else (result.annotation if result and result.annotation else {})
        rows.append({
            "row_id": row_id,
            "source_index": int(index),
            "state": state,
            "text": str(source_row[config.text_column]),
            "annotation_json": json.dumps(annotation, ensure_ascii=False),
            "reviewed": bool(review and review.reviewed),
            "error": result.error if result else None,
            "validation_errors": "; ".join(result.validation_errors) if result else "",
        })
    page_size = st.selectbox("Rows per page", [25, 50, 100], index=0)
    max_page = max(1, (len(rows) + page_size - 1) // page_size)
    page = st.number_input("Page", 1, max_page, 1)
    page_rows = rows[(page - 1) * page_size : page * page_size]
    edited = st.data_editor(
        pd.DataFrame(page_rows), use_container_width=True, hide_index=True,
        disabled=["row_id", "source_index", "state", "text", "error", "validation_errors"],
        column_config={"annotation_json": st.column_config.TextColumn("Annotation JSON", width="large")},
    )
    if st.button("Save page reviews", type="primary"):
        try:
            for row in edited.to_dict(orient="records"):
                if row["reviewed"] or row["row_id"] in reviews:
                    result = results.get(str(row["row_id"]))
                    reviews[str(row["row_id"])] = ReviewedAnnotation(
                        row_id=str(row["row_id"]), annotation=json.loads(row["annotation_json"]),
                        reviewed=bool(row["reviewed"]), source_attempt_id=result.attempt_id if result else None,
                    )
            current.save_reviews(run_id, reviews)
            st.success("Reviews saved as a separate annotation layer.")
        except Exception as exc:
            st.error(str(exc))
    inspect_id = st.selectbox("Inspect provenance", [row["row_id"] for row in page_rows], index=None)
    if inspect_id:
        result = results.get(inspect_id)
        if result:
            st.json(result.model_dump(mode="json"))
            annotation = reviews[inspect_id].annotation if inspect_id in reviews else result.annotation
            if annotation and config.extraction_mode == "graph":
                graph, warnings = build_extracted_graph([{"annotation": annotation}], current.load_graph_schema())
                render_graph(graph, warnings)


def export_page(current: ProjectStore) -> None:
    st.header("8. Export")
    runs = current.list_runs()
    if not runs:
        st.info("Run an extraction first.")
        return
    run_id = st.selectbox("Run to export", runs, key="export_run")
    prefix = st.text_input("Annotation column prefix", current.load_config().annotation_prefix)
    if prefix != current.load_config().annotation_prefix and st.button("Save prefix"):
        config = current.load_config()
        config.annotation_prefix = prefix
        current.save_config(config)
        st.rerun()
    col1, col2, col3 = st.columns(3)
    col1.download_button("Download CSV", csv_bytes(current, run_id), f"{run_id}.csv", "text/csv", use_container_width=True)
    col2.download_button("Download JSONL", jsonl_bytes(current, run_id), f"{run_id}.jsonl", "application/x-ndjson", use_container_width=True)
    col3.download_button("Download run bundle", run_bundle_bytes(current, run_id), f"{run_id}-bundle.zip", "application/zip", use_container_width=True)


def graph_page(current: ProjectStore) -> None:
    st.header("7. Graph Visualization")
    if current.load_config().extraction_mode != "graph":
        st.info("Graph visualization is available for graph extraction schemas. This project currently uses ordinary annotations.")
        return
    runs = current.list_runs()
    if not runs:
        st.info("Run an extraction first.")
        return
    run_id = st.selectbox("Source run", runs, key="graph_run")
    records = export_records(current, run_id)
    reviewed_only = st.checkbox("Reviewed rows only")
    if reviewed_only:
        records = [record for record in records if record.get("review_status") == "reviewed"]
    graph, warnings = build_extracted_graph(
        records, current.load_graph_schema(), row_id_field=current.load_config().id_column
    )
    st.caption("Nodes are deduplicated by exact ID across all selected rows. No fuzzy merging is performed.")
    render_graph(graph, warnings, download=True)


def main() -> None:
    init_state()
    current = project_sidebar()
    if not current:
        st.title("Automatic Annotations")
        st.write("Create a project or open an existing project directory to begin.")
        return
    pages = {
        "Data": data_page,
        "Extraction Schema": extraction_schema_page,
        "Instructions": prompt_page,
        "Model": model_page,
        "Test & Run": extract_page,
        "Review": review_page,
        "Visualize Graph": graph_page,
        "Export": export_page,
    }
    page = st.sidebar.radio("Workflow", list(pages))
    pages[page](current)


if __name__ == "__main__":
    main()
