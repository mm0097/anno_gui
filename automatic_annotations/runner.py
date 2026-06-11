from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any

from .io import append_jsonl
from .graph import validate_extracted_graph
from .models import RowState, RunManifest, RunResult
from .project import ProjectStore
from .providers import ExtractionProvider
from .schema import build_prompt

ProgressCallback = Callable[[int, int, RunResult], None]
TestProgressCallback = Callable[[int, int, int], None]
CancelCallback = Callable[[], bool]


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


class ExtractionRunner:
    def __init__(self, store: ProjectStore, provider: ExtractionProvider, api_key: str | None = None):
        self.store = store
        self.provider = provider
        self.api_key = api_key

    def create_run(self, row_count: int) -> RunManifest:
        config = self.store.load_config()
        schema = self.store.load_schema()
        instructions = self.store.load_instructions()
        examples = [example.model_dump(mode="json") for example in self.store.load_examples()]
        prompt = build_prompt(schema, instructions, examples)
        run_id = new_run_id()
        manifest = RunManifest(
            run_id=run_id,
            schema_hash=self.store.content_hash(schema),
            prompt_hash=self.store.content_hash(prompt),
            schema=schema,
            instructions=instructions,
            examples=examples,
            model_settings=config.model.model_dump(mode="json"),
            total_rows=row_count,
        )
        self.store.run_dir(run_id).mkdir(parents=True, exist_ok=False)
        self.store.save_manifest(manifest)
        return manifest

    def run(
        self,
        run_id: str,
        row_ids: Iterable[str] | None = None,
        rerun_completed: bool = False,
        progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ) -> RunManifest:
        config = self.store.load_config()
        source = self.store.load_source()
        manifest = self.store.load_manifest(run_id)
        existing = self.store.latest_results(run_id)
        selected = set(map(str, row_ids)) if row_ids is not None else None
        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, row in source.iterrows():
            record = row.to_dict()
            row_id = str(record[config.id_column])
            if selected is not None and row_id not in selected:
                continue
            previous = existing.get(row_id)
            if previous and previous.state in {RowState.COMPLETED, RowState.REVIEWED} and not rerun_completed:
                continue
            candidates.append((int(index), record))

        manifest.status = "running"
        manifest.cancelled = False
        self.store.save_manifest(manifest)
        completed = 0
        with ThreadPoolExecutor(max_workers=config.model.concurrency) as executor:
            futures = {}
            for index, record in candidates:
                if cancelled and cancelled():
                    break
                future = executor.submit(self._extract_with_retry, run_id, index, record)
                futures[future] = str(record[config.id_column])
            for future in as_completed(futures):
                result = future.result()
                append_jsonl(
                    self.store.run_dir(run_id) / "results.jsonl",
                    result.model_dump(mode="json"),
                )
                if result.state == RowState.FAILED:
                    append_jsonl(
                        self.store.run_dir(run_id) / "errors.jsonl",
                        result.model_dump(mode="json"),
                    )
                completed += 1
                if progress:
                    progress(completed, len(candidates), result)

        latest = self.store.latest_results(run_id)
        manifest.completed_rows = sum(result.state == RowState.COMPLETED for result in latest.values())
        manifest.failed_rows = sum(result.state == RowState.FAILED for result in latest.values())
        manifest.cancelled = bool(cancelled and cancelled())
        manifest.status = "cancelled" if manifest.cancelled else "completed"
        self.store.save_manifest(manifest)
        return manifest

    def test_rows(
        self,
        indices: Iterable[int],
        progress: TestProgressCallback | None = None,
        timeout_seconds: float = 120,
    ) -> list[RunResult]:
        source = self.store.load_source()
        test_id = "test-" + uuid.uuid4().hex[:8]
        selected = list(indices)
        results = []
        for position, index in enumerate(selected, start=1):
            if progress:
                progress(position, len(selected), int(index))
            results.append(
                self._extract_with_retry(
                    test_id,
                    int(index),
                    source.iloc[int(index)].to_dict(),
                    retry_limit=0,
                    timeout_seconds=timeout_seconds,
                )
            )
        return results

    def _extract_with_retry(
        self,
        run_id: str,
        source_index: int,
        record: dict[str, Any],
        retry_limit: int | None = None,
        timeout_seconds: float | None = None,
    ) -> RunResult:
        config = self.store.load_config()
        settings = config.model.model_copy(update={
            **({"timeout_seconds": timeout_seconds} if timeout_seconds is not None else {}),
        })
        attempts = settings.retry_limit if retry_limit is None else retry_limit
        schema = self.store.load_schema()
        instructions = self.store.load_instructions()
        examples = [example.model_dump(mode="json") for example in self.store.load_examples()]
        row_id = str(record[config.id_column])
        response = None
        for attempt in range(attempts + 1):
            response = self.provider.extract(
                text=str(record[config.text_column]),
                schema=schema,
                instructions=instructions,
                examples=examples,
                settings=settings,
                api_key=self.api_key,
            )
            if response.parsed and config.extraction_mode == "graph":
                response.validation_errors.extend(
                    validate_extracted_graph(response.parsed, self.store.load_graph_schema())
                )
            if not response.error and not response.validation_errors:
                break
            if response.error and not self._retryable_error(response.error):
                break
            if attempt < attempts:
                time.sleep(min(2**attempt, 8))
        assert response is not None
        failed = bool(response.error or response.validation_errors)
        return RunResult(
            run_id=run_id,
            attempt_id=uuid.uuid4().hex,
            row_id=row_id,
            state=RowState.FAILED if failed else RowState.COMPLETED,
            source_index=source_index,
            annotation=response.parsed,
            raw_output=response.raw_output,
            validation_errors=response.validation_errors,
            error=response.error,
            usage=response.usage,
            latency_seconds=response.latency_seconds,
            model=response.model,
        )

    @staticmethod
    def _retryable_error(error: str) -> bool:
        lowered = error.lower()
        return any(marker in lowered for marker in (
            "timeout", "timed out", "429", "rate limit", "500", "502", "503", "504",
            "connection reset", "connection refused", "temporarily unavailable",
        ))
