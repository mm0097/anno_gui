from __future__ import annotations

import io
import json
from pathlib import Path
from typing import BinaryIO

import pandas as pd


def load_upload(file: BinaryIO, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(file, lines=True)
    raise ValueError("Upload a .csv, .jsonl, or .ndjson file")


def load_huggingface(dataset_id: str, config: str | None, split: str, token: str | None) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the 'datasets' package to load Hugging Face datasets") from exc
    dataset = load_dataset(dataset_id, config or None, split=split, token=token or None)
    return dataset.to_pandas()
