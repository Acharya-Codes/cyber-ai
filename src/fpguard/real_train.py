"""Leakage-conscious real-data training and evaluation.

The command consumes only explicit CSV inputs accompanied by a matching
Stage 6A audit report. It never modifies synthetic experiment artifacts.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from fpguard.data import LABELS
from fpguard.real_data import (
    REPORT_FILENAME as AUDIT_REPORT_FILENAME,
    _normalize_label,
    normalize_column_name,
)
from fpguard.threshold import DEFAULT_CANDIDATE_THRESHOLDS, select_threshold

DEFAULT_OUTPUT_DIR = Path("artifacts/real_data/cicids2018")
DEFAULT_AUDIT_REPORT = DEFAULT_OUTPUT_DIR / AUDIT_REPORT_FILENAME
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_MARKER = ".fpguard-real-data-output"
CHUNK_SIZE = 100_000
DEFAULT_MAX_CATEGORICAL_LEVELS = 50
MANIFEST_FILENAME = "split_manifest.json"
COMPARISON_FILENAME = "model_comparison.json"
MODEL_FILENAME = "selected_model.joblib"
METADATA_FILENAME = "selected_model_metadata.json"
TEST_METRICS_FILENAME = "test_metrics.json"
DIAGNOSTICS_FILENAME = "diagnostics.csv"
OUTPUT_FILENAMES = (
    MANIFEST_FILENAME,
    COMPARISON_FILENAME,
    MODEL_FILENAME,
    METADATA_FILENAME,
    TEST_METRICS_FILENAME,
    DIAGNOSTICS_FILENAME,
)

_IDENTIFIER_NAMES = {
    "id",
    "uuid",
    "flow_id",
    "record_id",
    "row_id",
}
_IDENTIFIER_TOKENS = ("ip", "address", "mac", "src", "dst")
_TIMESTAMP_TOKENS = ("timestamp", "datetime", "date", "time")
_ATTACK_TOKENS = ("attack_type", "attack_category", "category", "subcategory")
_TARGET_TOKENS = (
    "attack",
    "category",
    "class",
    "outcome",
    "result",
    "prediction",
    "target",
    "severity",
    "incident",
    "malicious",
)
_CAPTURE_CONTEXT_NAMES = {
    "capture_day",
    "capture_date",
    "day",
    "session",
    "session_id",
    "capture_session",
}
_INTERNAL_COLUMNS = {"__fpguard_source_path__", "__fpguard_source_row__"}


class RealTrainingError(ValueError):
    """Raised when real-data provenance, schema, or split integrity is unsafe."""


@dataclass
class RealDataset:
    """Validated rows plus source provenance for the real-data experiment."""

    features: pd.DataFrame
    labels: pd.Series
    references: pd.DataFrame
    source_summaries: list[dict[str, Any]]
    audit_mapping: dict[str, Any]
    total_audited_rows: int
    selected_rows: int
    sampling: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise RealTrainingError(f"Could not read CSV header from '{path}': {exc}") from exc
    if not header:
        raise RealTrainingError(f"CSV '{path}' is empty or has no header.")
    normalized = [normalize_column_name(column) for column in header]
    if any(not column for column in normalized):
        raise RealTrainingError(f"CSV '{path}' contains a blank column name.")
    if any(column in _INTERNAL_COLUMNS for column in normalized):
        raise RealTrainingError(f"CSV '{path}' uses a reserved FPGuard internal column name.")
    seen: set[str] = set()
    collisions: list[str] = []
    for original, name in zip(header, normalized):
        if name in seen:
            collisions.append(f"{original!r} normalizes to duplicate name {name!r}")
        seen.add(name)
    if collisions:
        raise RealTrainingError(
            f"CSV '{path}' has duplicate or normalized-colliding headers: "
            + "; ".join(collisions)
        )
    return normalized


def _load_audit_report(
    audit_report_path: str | Path,
    input_paths: Sequence[Path],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    report_path = Path(audit_report_path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Matching Stage 6A audit report not found: {report_path}. "
            "Run `python -m fpguard.real_data` for the explicit input CSVs first."
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealTrainingError(f"Could not read Stage 6A audit report: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(report.get("sources"), list):
        raise RealTrainingError("Stage 6A report must contain a sources array.")
    mapping_section = report.get("label_mapping")
    if not isinstance(mapping_section, dict):
        raise RealTrainingError("Stage 6A report is missing its explicit label mapping.")
    alias_to_class: dict[str, str] = {}
    for class_name in LABELS:
        aliases = mapping_section.get(class_name)
        if not isinstance(aliases, list) or not aliases:
            raise RealTrainingError(
                f"Stage 6A report is missing aliases for class {class_name!r}."
            )
        for alias in aliases:
            normalized = _normalize_label(alias)
            if not normalized:
                raise RealTrainingError("Stage 6A report contains a blank label alias.")
            if normalized in alias_to_class and alias_to_class[normalized] != class_name:
                raise RealTrainingError(
                    f"Stage 6A label alias {alias!r} maps to both classes."
                )
            alias_to_class[normalized] = class_name
    if set(alias_to_class.values()) != set(LABELS):
        raise RealTrainingError("Stage 6A label mapping must define benign and attack.")

    sources: dict[str, dict[str, Any]] = {}
    for item in report["sources"]:
        if not isinstance(item, dict) or not isinstance(item.get("source_path"), str):
            raise RealTrainingError("Stage 6A report contains an invalid source entry.")
        source_path = str(Path(item["source_path"]).expanduser().resolve())
        sources[source_path] = item
    verified_sources: dict[str, dict[str, Any]] = {}
    for input_path in input_paths:
        key = str(input_path)
        audit_source = sources.get(key)
        if audit_source is None:
            raise RealTrainingError(
                f"Input '{input_path}' is not listed in the Stage 6A report. "
                "Audit the exact input file before training."
            )
        if not input_path.is_file():
            raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
        actual_hash = _sha256(input_path)
        if audit_source.get("sha256") != actual_hash:
            raise RealTrainingError(
                f"SHA-256 mismatch for '{input_path}'; the file changed since Stage 6A audit."
            )
        verified_sources[key] = audit_source
    return report, alias_to_class, verified_sources


def _load_cleaning_manifest(
    manifest_path: str | Path | None,
    input_paths: Sequence[Path],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if manifest_path is None:
        return None, {}
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Cleaning manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealTrainingError(f"Could not read cleaning manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RealTrainingError("Cleaning manifest must be a JSON object.")
    if len(input_paths) != 1:
        raise RealTrainingError(
            "A single-input cleaning manifest can only validate one CSV input."
        )
    expected_path = manifest.get("output_path")
    expected_hash = manifest.get("output_sha256")
    expected_rows = manifest.get("output_data_row_count")
    if (
        not isinstance(expected_path, str)
        or Path(expected_path).expanduser().resolve() != input_paths[0]
    ):
        raise RealTrainingError(
            "Cleaning manifest output_path does not match the training CSV."
        )
    if not isinstance(expected_hash, str) or _sha256(input_paths[0]) != expected_hash:
        raise RealTrainingError(
            "Cleaning manifest output SHA-256 does not match the training CSV."
        )
    if type(expected_rows) is not int or expected_rows < 1:
        raise RealTrainingError(
            "Cleaning manifest output_data_row_count is missing or invalid."
        )
    if manifest.get("retained_labels_explicitly_counted") is not True:
        raise RealTrainingError(
            "Cleaning manifest does not confirm explicit retained-label accounting."
        )
    if manifest.get("output_has_no_exact_repeated_header_records") is not True:
        raise RealTrainingError(
            "Cleaning manifest does not verify removal of exact repeated headers."
        )
    return manifest, {
        str(input_paths[0]): {
            "manifest_path": str(path),
            "expected_raw_labels": manifest.get("retained_raw_label_counts"),
            "expected_rows": expected_rows,
        }
    }


def _load_investigation_report(
    report_path: str | Path | None,
    input_paths: Sequence[Path],
    audit_sources: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    if report_path is None:
        return None, None
    path = Path(report_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Stage 6A.1 investigation report not found: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealTrainingError(f"Could not read Stage 6A.1 investigation report: {exc}") from exc
    if not isinstance(report, dict):
        raise RealTrainingError("Stage 6A.1 investigation report must be a JSON object.")
    if len(input_paths) != 1:
        raise RealTrainingError(
            "The CIC-IDS2018 investigation report applies to a single CSV input."
        )
    source_key = str(input_paths[0])
    audit_source = audit_sources[source_key]
    if report.get("source_sha256") != audit_source.get("sha256"):
        raise RealTrainingError(
            "Stage 6A.1 report source hash does not match the Stage 6A audit/input."
        )
    if report.get("row_count") != audit_source.get("row_count"):
        raise RealTrainingError(
            "Stage 6A.1 report row count does not match the Stage 6A audit/input."
        )
    if report.get("stage6a_reference", {}).get("conflict_counts_match") is not True:
        raise RealTrainingError(
            "Stage 6A.1 report does not confirm conflict-count reproduction."
        )
    if report.get("rounding_assessment", {}).get("status") != "unresolved":
        raise RealTrainingError(
            "Unexpected Stage 6A.1 rounding assessment; review before training."
        )
    heuristic = report.get("stage6a_name_heuristic_review", {})
    if heuristic.get("possible_attack_category_columns") or heuristic.get(
        "possible_target_or_post_event_columns"
    ):
        raise RealTrainingError(
            "Stage 6A.1 report contains unresolved attack-category or target/post-event "
            "features; training is blocked pending review."
        )
    return report, str(path)


def _read_explicit_csvs(
    input_paths: Sequence[Path],
    verified_sources: dict[str, dict[str, Any]],
    alias_to_class: dict[str, str],
    cleaning_expectations: dict[str, dict[str, Any]],
    *,
    label_column: str | None,
    max_rows: int | None,
    random_state: int,
) -> tuple[RealDataset, str]:
    normalized_headers = [_checked_header(path) for path in input_paths]
    if any(header != normalized_headers[0] for header in normalized_headers[1:]):
        raise RealTrainingError(
            "Input CSVs must have identical normalized column schemas; "
            "no columns will be silently added, dropped, or merged."
        )
    columns = normalized_headers[0]
    source_label_columns = {
        normalize_column_name(str(entry.get("label_column", "")))
        for entry in verified_sources.values()
    }
    if label_column is None:
        if len(source_label_columns) != 1 or "" in source_label_columns:
            raise RealTrainingError(
                "Input audit entries do not agree on a label column; specify --label-column."
            )
        label_name = next(iter(source_label_columns))
    else:
        label_name = normalize_column_name(label_column)
    if label_name not in columns:
        raise RealTrainingError(
            f"Configured label column {label_name!r} is absent. Available columns: {columns}"
        )

    expected_counts: list[int] = []
    for path in input_paths:
        count = verified_sources[str(path)].get("row_count")
        if type(count) is not int or count < 1:
            raise RealTrainingError(
                f"Stage 6A row count is missing or invalid for '{path}'. Re-audit this file."
            )
        expected_counts.append(count)
    total_rows = sum(expected_counts)
    if max_rows is not None and max_rows < 1:
        raise RealTrainingError("--max-rows must be a positive integer.")
    if max_rows is not None and max_rows < total_rows:
        sample_size = max_rows
        selected_global_positions = np.sort(
            np.random.default_rng(random_state).choice(
                total_rows, size=sample_size, replace=False
            )
        )
        sampling = {
            "applied": True,
            "method": "uniform seeded row sample across explicitly listed source files",
            "seed": random_state,
            "maximum_rows": max_rows,
            "population_rows": total_rows,
            "selected_rows": sample_size,
            "representativeness_warning": (
                "A row sample can omit rare classes or capture contexts; results are "
                "development-only and should be repeated on the full dataset."
            ),
        }
    else:
        selected_global_positions = None
        sampling = {
            "applied": False,
            "method": "all audited rows",
            "seed": random_state,
            "maximum_rows": max_rows,
            "population_rows": total_rows,
            "selected_rows": total_rows,
            "representativeness_warning": None,
        }

    row_frames: list[pd.DataFrame] = []
    label_values: list[pd.Series] = []
    refs: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    global_offset = 0
    for path, audited_rows in zip(input_paths, expected_counts):
        digest = _sha256(path)
        start = global_offset
        source_retained = 0
        raw_label_counts: Counter[str] = Counter()
        try:
            chunks = pd.read_csv(
                path,
                encoding="utf-8-sig",
                dtype=str,
                chunksize=CHUNK_SIZE,
                on_bad_lines="error",
            )
            source_offset = 0
            for chunk in chunks:
                chunk.columns = columns
                raw_label_counts.update(chunk[label_name].astype(str).tolist())
                if selected_global_positions is None:
                    local_positions = np.arange(len(chunk), dtype=np.int64)
                else:
                    left = int(np.searchsorted(selected_global_positions, start + source_offset))
                    right = int(
                        np.searchsorted(
                            selected_global_positions,
                            start + source_offset + len(chunk),
                        )
                    )
                    local_positions = (
                        selected_global_positions[left:right] - (start + source_offset)
                    )
                if len(local_positions):
                    kept = chunk.iloc[local_positions].copy()
                    kept_labels = kept.pop(label_name)
                    source_row_positions = source_offset + local_positions
                    row_frames.append(kept.reset_index(drop=True))
                    label_values.append(kept_labels.reset_index(drop=True))
                    refs.append(
                        pd.DataFrame(
                            {
                                "source_path": str(path),
                                "source_row_index": source_row_positions.astype(np.int64),
                            }
                        )
                    )
                    source_retained += len(kept)
                source_offset += len(chunk)
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise RealTrainingError(f"Could not read audited CSV '{path}': {exc}") from exc
        if source_offset != audited_rows:
            raise RealTrainingError(
                f"CSV row count changed for '{path}': audit recorded {audited_rows}, "
                f"current file has {source_offset}."
            )
        audit_raw_counts = verified_sources[str(path)].get("label_distribution_raw")
        observed_raw_counts = dict(sorted(raw_label_counts.items()))
        if audit_raw_counts != observed_raw_counts:
            raise RealTrainingError(
                f"Raw label counts for '{path}' differ from Stage 6A; "
                "the CSV labels were not accepted."
            )
        cleaning = cleaning_expectations.get(str(path))
        if cleaning is not None:
            if source_offset != cleaning["expected_rows"]:
                raise RealTrainingError(
                    f"CSV row count for '{path}' differs from cleaning manifest."
                )
            if cleaning["expected_raw_labels"] != observed_raw_counts:
                raise RealTrainingError(
                    f"Raw labels for '{path}' differ from cleaning manifest."
                )
        global_offset += source_offset
        summaries.append(
            {
                "source_path": str(path),
                "sha256": digest,
                "audited_rows": audited_rows,
                "selected_rows": source_retained,
                "source_size_bytes": path.stat().st_size,
                "raw_label_counts": observed_raw_counts,
            }
        )
    if global_offset != total_rows:
        raise RealTrainingError("Audited row counts do not match rows read from input CSVs.")

    raw_labels = pd.concat(label_values, ignore_index=True).astype("string")
    normalized_labels = raw_labels.map(_normalize_label)
    unknown = sorted(set(normalized_labels.dropna()) - set(alias_to_class))
    if normalized_labels.isna().any() or unknown:
        raise RealTrainingError(
            "Input labels no longer match the explicit Stage 6A mapping; "
            f"missing labels={int(normalized_labels.isna().sum())}, unknown={unknown[:10]}."
        )
    mapped_labels = normalized_labels.map(alias_to_class).astype(str)
    if set(mapped_labels.unique()) != set(LABELS):
        raise RealTrainingError(
            "Loaded data must contain both benign and attack rows after the configured "
            "development sample; increase --max-rows or use the full dataset."
        )
    features = pd.concat(row_frames, ignore_index=True)
    if not len(features):
        raise RealTrainingError("No rows were selected from the explicitly audited files.")
    reference_frame = pd.concat(refs, ignore_index=True)
    if features.columns.has_duplicates or list(features.columns) != columns[: columns.index(label_name)] + columns[columns.index(label_name) + 1 :]:
        raise RealTrainingError("Normalized feature schema is ambiguous.")
    return (
        RealDataset(
            features=features,
            labels=mapped_labels,
            references=reference_frame,
            source_summaries=summaries,
            audit_mapping={
                "benign": [
                    alias for alias, target in alias_to_class.items() if target == "benign"
                ],
                "attack": [
                    alias for alias, target in alias_to_class.items() if target == "attack"
                ],
                "raw_label_distribution_full_audit": {
                    str(path): verified_sources[str(path)].get("label_distribution_raw")
                    for path in input_paths
                },
                "mapped_label_distribution_full_audit": {
                    str(path): verified_sources[str(path)].get("label_distribution_mapped")
                    for path in input_paths
                },
            },
            total_audited_rows=total_rows,
            selected_rows=len(features),
            sampling=sampling,
        ),
        label_name,
    )


def _is_review_column(name: str) -> bool:
    lowered = name.casefold()
    if lowered in _IDENTIFIER_NAMES or lowered.endswith(("_id", "_uuid")):
        return True
    if lowered in _CAPTURE_CONTEXT_NAMES:
        return True
    if any(token in lowered for token in _TIMESTAMP_TOKENS):
        return True
    if any(token in lowered for token in _ATTACK_TOKENS):
        return True
    if re.search(r"(^|_)attack($|_)", lowered):
        return True
    if any(token in lowered for token in _TARGET_TOKENS):
        return True
    if any(
        lowered == token or lowered.endswith(f"_{token}") or lowered.startswith(f"{token}_")
        for token in _IDENTIFIER_TOKENS
    ):
        return True
    if (lowered.startswith("source_") or lowered.startswith("destination_")) and any(
        token in lowered for token in ("ip", "address", "mac", "host", "id")
    ):
        return True
    return False


def select_predictive_features(
    columns: Sequence[str],
    label_column: str,
    *,
    excluded_columns: Sequence[str] = (),
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Select columns by conservative name-based exclusions, with an audit trail."""
    explicit_exclusions = {normalize_column_name(name) for name in excluded_columns}
    features: list[str] = []
    exclusions: list[dict[str, str]] = []
    for column in columns:
        if column == label_column:
            exclusions.append({"column": column, "reason": "configured target label"})
        elif column in explicit_exclusions:
            exclusions.append({"column": column, "reason": "explicit --exclude-column"})
        elif column == "dst_port":
            # A destination service port is a behavioral feature, not a flow identifier.
            features.append(column)
        elif _is_review_column(column):
            exclusions.append(
                {
                    "column": column,
                    "reason": "possible identifier, timestamp, attack category, or target-derived field",
                }
            )
        else:
            features.append(column)
    if not features:
        raise RealTrainingError("No predictive features remain after leakage exclusions.")
    return (
        features,
        [column for column in columns if column in explicit_exclusions],
        exclusions,
    )


def _capture_context_columns(
    columns: Sequence[str],
    references: pd.DataFrame,
    features: pd.DataFrame,
) -> tuple[list[str], str]:
    eligible = [name for name in columns if name in features.columns]
    for name in eligible:
        if name in _CAPTURE_CONTEXT_NAMES or name.startswith(
            ("capture_day_", "capture_session_")
        ):
            values = features[name].fillna("__missing_capture_context__").astype(str)
            if values.nunique() >= 5:
                return [name], f"grouped by capture-context column {name!r}"
    timestamp_columns = [
        name for name in columns if any(token in name for token in _TIMESTAMP_TOKENS)
    ]
    for name in timestamp_columns:
        parsed = pd.to_datetime(features[name], errors="coerce", utc=True, format="mixed")
        dates = parsed.dt.strftime("%Y-%m-%d")
        if parsed.notna().all() and dates.nunique() >= 5:
            return [name], f"grouped by calendar day derived from timestamp {name!r}"
    source_count = references["source_path"].nunique()
    if source_count >= 5:
        return [], f"grouped by source CSV capture file ({source_count} groups)"
    if source_count > 1:
        return [], (
            f"source CSV files may be separate capture contexts ({source_count}), but fewer "
            "than five are available; context grouping is not sufficiently granular"
        )
    return [], "no usable independent capture-day/session groups were found"


def _union_find_groups(
    feature_frame: pd.DataFrame,
    references: pd.DataFrame,
    context_frame: pd.DataFrame,
    context_columns: Sequence[str],
    *,
    group_by_source_file: bool,
) -> tuple[np.ndarray, np.ndarray]:
    row_count = len(feature_frame)
    parent = np.arange(row_count, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(first: int, second: int) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    feature_hashes = pd.util.hash_pandas_object(
        feature_frame, index=False, categorize=True
    ).to_numpy(dtype=np.uint64)
    first_by_key: dict[tuple[str, Any], int] = {}
    for index, feature_hash in enumerate(feature_hashes):
        key = ("features", int(feature_hash))
        if key in first_by_key:
            union(index, first_by_key[key])
        else:
            first_by_key[key] = index

    if group_by_source_file:
        context = references["source_path"].astype(str).to_numpy()
        context_name = "source_file"
        for index, value in enumerate(context):
            key = (context_name, value)
            if key in first_by_key:
                union(index, first_by_key[key])
            else:
                first_by_key[key] = index
    for column in context_columns:
        if any(token in column for token in _TIMESTAMP_TOKENS):
            parsed = pd.to_datetime(
                context_frame[column], errors="coerce", utc=True, format="mixed"
            )
            values = parsed.dt.strftime("%Y-%m-%d").fillna(
                "__missing_capture_context__"
            )
        else:
            values = context_frame[column].fillna(
                "__missing_capture_context__"
            ).astype(str)
        for index, value in enumerate(values):
            key = (f"context:{column}", value)
            if key in first_by_key:
                union(index, first_by_key[key])
            else:
                first_by_key[key] = index

    roots = np.fromiter((find(index) for index in range(row_count)), dtype=np.int64)
    return roots, feature_hashes


def make_group_split(
    features: pd.DataFrame,
    labels: pd.Series,
    references: pd.DataFrame,
    *,
    random_state: int,
    context_frame: pd.DataFrame | None = None,
    grouping_features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Create deterministic approximate 60/20/20 partitions without splitting groups."""
    context_source = context_frame if context_frame is not None else features
    group_source = grouping_features if grouping_features is not None else features
    if len(group_source) != len(features):
        raise RealTrainingError(
            "Grouping features and predictive features must contain the same rows."
        )
    context_columns, context_note = _capture_context_columns(
        list(context_source.columns), references, context_source
    )
    group_by_source_file = (
        references["source_path"].nunique() >= 5 and not context_columns
    )
    groups, feature_hashes = _union_find_groups(
        group_source,
        references,
        context_source,
        context_columns,
        group_by_source_file=group_by_source_file,
    )
    feature_frame = features.reset_index(drop=True)
    duplicate_labels = labels.reset_index(drop=True)
    group_label_frame = pd.DataFrame(
        {"feature_hash": feature_hashes, "label": duplicate_labels.to_numpy()}
    )
    label_counts_by_group = group_label_frame.groupby(
        "feature_hash", sort=False
    )["label"].nunique()
    mixed_group_ids = set(label_counts_by_group[label_counts_by_group > 1].index)
    mixed_label_group_mask = np.isin(feature_hashes, list(mixed_group_ids))
    distinct_groups = np.unique(groups)
    try:
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=random_state,
        )
        folds = list(splitter.split(features, labels, groups))
    except ValueError as exc:
        raise RealTrainingError(
            f"Could not construct a five-fold stratified group split: {exc}. "
            "More independent duplicate/capture groups may be required."
        ) from exc
    test_indices = np.asarray(folds[0][1], dtype=np.int64)
    validation_indices = np.asarray(folds[1][1], dtype=np.int64)
    training_indices = np.sort(
        np.concatenate([folds[index][1] for index in range(2, 5)]).astype(np.int64)
    )
    split_indices = {
        "train": training_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    for split_name, indices in split_indices.items():
        class_values = set(labels.iloc[indices].tolist())
        if class_values != set(LABELS):
            raise RealTrainingError(
                f"{split_name.title()} split does not contain both classes: {class_values}. "
                "The available groups cannot support reliable 60/20/20 evaluation."
            )
    row_split = np.full(len(features), "", dtype=object)
    for split_name, indices in split_indices.items():
        row_split[indices] = split_name
    if (row_split == "").any():
        raise RealTrainingError("Group splitter failed to assign all rows.")
    group_split: dict[str, set[int]] = {
        name: set(groups[indices].tolist()) for name, indices in split_indices.items()
    }
    if (
        group_split["train"] & group_split["validation"]
        or group_split["train"] & group_split["test"]
        or group_split["validation"] & group_split["test"]
    ):
        raise RealTrainingError("A duplicate/capture group crossed a split boundary.")
    if len(set(feature_hashes[training_indices]) & set(feature_hashes[test_indices])):
        raise RealTrainingError("Identical feature groups crossed the train/test boundary.")
    mixed_group_stats: dict[str, dict[str, Any]] = {}
    for split_name, indices in split_indices.items():
        split_mixed_ids = set(feature_hashes[indices]).intersection(mixed_group_ids)
        split_mixed_mask = mixed_label_group_mask[indices]
        mixed_group_stats[split_name] = {
            "mixed_label_group_count": int(len(split_mixed_ids)),
            "affected_row_count": int(split_mixed_mask.sum()),
            "affected_rows_by_label": {
                label: int(duplicate_labels.iloc[indices[split_mixed_mask]].eq(label).sum())
                for label in LABELS
            },
        }
    context_used = bool(context_columns or group_by_source_file)
    strategy = (
        "StratifiedGroupKFold: folds 2-4 train, fold 1 validation, fold 0 test; "
        "exact groups defined by all supplied grouping features are unioned and held "
        "within one split; mixed-label groups are preserved without relabeling."
    )
    if context_used:
        strategy += f" Capture context was included: {context_note}."
    else:
        strategy += (
            " No independent capture context was available at usable granularity; "
            f"duplicate-only holdout used. Context review: {context_note}."
        )
    return {
        "indices": split_indices,
        "row_split": row_split,
        "groups": groups,
        "feature_group_hashes": feature_hashes,
        "feature_hashes": feature_hashes,
        "mixed_label_group_ids": mixed_group_ids,
        "mixed_label_group_mask": mixed_label_group_mask,
        "mixed_label_group_stats": mixed_group_stats,
        "strategy": strategy,
        "context_grouping_used": context_used,
        "context_columns": context_columns,
        "distinct_group_count": int(len(distinct_groups)),
        "split_group_counts": {
            name: int(len(group_split[name])) for name in split_indices
        },
        "class_counts": {
            name: {
                label: int(labels.iloc[indices].eq(label).sum())
                for label in LABELS
            }
            for name, indices in split_indices.items()
        },
        "row_counts": {
            name: int(len(indices)) for name, indices in split_indices.items()
        },
        "random_state": random_state,
    }


def _typed_features(
    frame: pd.DataFrame,
    training_indices: np.ndarray,
    feature_names: Sequence[str],
    *,
    max_categorical_levels: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Infer data-only feature types from training rows and validate all rows."""
    prepared = pd.DataFrame(index=frame.index)
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    for name in feature_names:
        values = frame[name]
        training_values = values.iloc[training_indices]
        training_numeric = pd.to_numeric(training_values, errors="coerce")
        training_invalid = training_values.notna() & training_numeric.isna()
        if training_invalid.any() and training_numeric.notna().any():
            bad = training_values.loc[training_invalid].astype(str).head(5).tolist()
            raise RealTrainingError(
                f"Feature {name!r} contains malformed numeric values in training rows: {bad}"
            )
        is_numeric = (
            not training_numeric.notna().empty
            and training_numeric.notna().any()
        ) or (
            pd.api.types.is_numeric_dtype(values.dtype)
            and training_values.isna().all()
        )
        if is_numeric:
            all_numeric = pd.to_numeric(values, errors="coerce")
            invalid = values.notna() & all_numeric.isna()
            if invalid.any():
                bad = values.loc[invalid].astype(str).head(5).tolist()
                raise RealTrainingError(
                    f"Numeric feature {name!r} has malformed values outside training rows: {bad}"
                )
            prepared[name] = all_numeric.replace([np.inf, -np.inf], np.nan).astype(float)
            numeric_features.append(name)
            continue

        train_text = training_values.dropna().astype(str)
        distinct_levels = train_text.nunique()
        if distinct_levels == 0:
            prepared[name] = values.astype("string")
            numeric_features.append(name)
            prepared[name] = pd.to_numeric(values, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            continue
        if distinct_levels > max_categorical_levels:
            raise RealTrainingError(
                f"Feature {name!r} appears categorical with {distinct_levels} training "
                f"levels (limit {max_categorical_levels}). Exclude it explicitly with "
                "--exclude-column or review its semantics."
            )
        prepared[name] = values.astype("string")
        categorical_features.append(name)
    if not numeric_features and not categorical_features:
        raise RealTrainingError("No usable numeric or categorical features remain.")
    return prepared.loc[:, list(feature_names)], numeric_features, categorical_features


def _replace_nonfinite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, np.nan)


def build_real_pipeline(
    feature_names: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    *,
    classifier: Any,
    scale_numeric: bool,
) -> Pipeline:
    """Build a pipeline whose imputers, scalers, and encoders fit in Pipeline.fit."""
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_features:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "finite_to_missing",
                FunctionTransformer(_replace_nonfinite, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric_features)))
    if categorical_features:
        categorical_steps = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="__missing__",
                        keep_empty_features=True,
                    ),
                ),
                ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_steps, list(categorical_features)))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _attack_probabilities(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    classes = list(model.classes_)
    if "attack" not in classes:
        raise RealTrainingError(
            f"Fitted candidate does not expose positive class 'attack'; classes are {classes}."
        )
    scores = np.asarray(probabilities[:, classes.index("attack")], dtype=float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RealTrainingError("Candidate returned invalid attack-class probabilities.")
    return scores


def evaluate_real_metrics(
    labels: Sequence[str],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Compute real-flow binary metrics with nulls for undefined measures."""
    y_true = np.asarray(labels, dtype=str)
    scores = np.asarray(probabilities, dtype=float)
    if y_true.ndim != 1 or scores.ndim != 1 or len(y_true) != len(scores):
        raise ValueError("Labels and probabilities must be one-dimensional and equal length.")
    if not len(y_true):
        raise ValueError("Cannot evaluate an empty set.")
    invalid_labels = set(y_true) - set(LABELS)
    if invalid_labels:
        raise ValueError(f"Labels must be benign/attack; found {sorted(invalid_labels)}.")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1].")
    predicted = np.where(scores >= threshold, "attack", "benign")
    matrix = confusion_matrix(y_true, predicted, labels=list(LABELS))
    tn, fp, fn, tp = map(int, (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]))
    predicted_positive = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp
    precision = tp / predicted_positive if predicted_positive else None
    recall = tp / actual_positive if actual_positive else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision == 0 and recall == 0 else None)
    )
    has_both_classes = actual_positive > 0 and actual_negative > 0
    metrics: dict[str, Any] = {
        "positive_class": "attack",
        "class_order": list(LABELS),
        "confusion_matrix": matrix.tolist(),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": fp / actual_negative if actual_negative else None,
        "false_negative_rate": fn / actual_positive if actual_positive else None,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": (
            float(roc_auc_score(y_true == "attack", scores)) if has_both_classes else None
        ),
        "pr_auc_average_precision": (
            float(average_precision_score(y_true == "attack", scores))
            if actual_positive
            else None
        ),
        "class_prevalence": {
            "benign": actual_negative / len(y_true),
            "attack": actual_positive / len(y_true),
        },
        "threshold": float(threshold),
        "undefined_metric_notes": [],
    }
    if predicted_positive == 0:
        metrics["undefined_metric_notes"].append(
            "Precision and F1 are null because no rows were predicted as attack."
        )
    if actual_positive == 0:
        metrics["undefined_metric_notes"].append(
            "Recall, FNR, ROC-AUC, and PR-AUC are null because no attack rows are present."
        )
    if actual_negative == 0:
        metrics["undefined_metric_notes"].append(
            "FPR and ROC-AUC are null because no benign rows are present."
        )
    return metrics


def _candidate_pipelines(
    feature_names: Sequence[str],
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    random_state: int,
) -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_real_pipeline(
            feature_names,
            numeric_features,
            categorical_features,
            classifier=LogisticRegression(
                C=1.0,
                max_iter=500,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            scale_numeric=True,
        ),
        "random_forest": build_real_pipeline(
            feature_names,
            numeric_features,
            categorical_features,
            classifier=RandomForestClassifier(
                n_estimators=64,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            scale_numeric=False,
        ),
        "extra_trees": build_real_pipeline(
            feature_names,
            numeric_features,
            categorical_features,
            classifier=ExtraTreesClassifier(
                n_estimators=64,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
            scale_numeric=False,
        ),
    }


def select_winning_candidate(
    validation_results: dict[str, dict[str, Any]],
) -> str | None:
    """Choose a feasible candidate using only its validation threshold metrics."""
    feasible = [
        (name, result)
        for name, result in validation_results.items()
        if result.get("feasible") and isinstance(result.get("validation_metrics"), dict)
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda entry: (
            entry[1]["validation_metrics"]["false_positive_rate"],
            -entry[1]["validation_metrics"]["precision"],
            -entry[1]["validation_metrics"]["recall"],
            entry[0],
        ),
    )[0]


def _metric_changes(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_fpr = baseline["false_positive_rate"]
    candidate_fpr = candidate["false_positive_rate"]
    absolute_fpr_change = (
        (candidate_fpr - baseline_fpr) * 100
        if candidate_fpr is not None and baseline_fpr is not None
        else None
    )
    relative_reduction = (
        (baseline_fpr - candidate_fpr) / baseline_fpr * 100
        if baseline_fpr not in (None, 0) and candidate_fpr is not None
        else None
    )
    return {
        "absolute_fpr_change_percentage_points_candidate_minus_baseline": absolute_fpr_change,
        "relative_fpr_reduction_percent": relative_reduction,
        "relative_fpr_reduction_undefined_reason": (
            "Baseline FPR is zero."
            if baseline_fpr == 0
            else "FPR is undefined because the test partition has no benign rows."
            if baseline_fpr is None
            else None
        ),
        "recall_change_percentage_points_candidate_minus_baseline": (
            (candidate["recall"] - baseline["recall"]) * 100
            if candidate["recall"] is not None and baseline["recall"] is not None
            else None
        ),
        "false_positive_count_change_candidate_minus_baseline": (
            candidate["false_positives"] - baseline["false_positives"]
        ),
        "false_negative_count_change_candidate_minus_baseline": (
            candidate["false_negatives"] - baseline["false_negatives"]
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for distribution in ("joblib",):
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_real_experiment(
    input_paths: Sequence[str | Path],
    *,
    audit_report_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    label_column: str | None = None,
    cleaning_manifest_path: str | Path | None = None,
    investigation_report_path: str | Path | None = None,
    minimum_recall: float = 0.80,
    random_state: int = 42,
    max_rows: int | None = None,
    exclude_columns: Sequence[str] = (),
    max_categorical_levels: int = DEFAULT_MAX_CATEGORICAL_LEVELS,
    force: bool = False,
    preflight_only: bool = False,
) -> dict[str, Any]:
    """Train, select on validation, and report one final untouched test comparison."""
    if not 0 < minimum_recall <= 1 or not np.isfinite(minimum_recall):
        raise RealTrainingError("--minimum-recall must be greater than 0 and at most 1.")
    if not isinstance(random_state, int):
        raise RealTrainingError("--random-state must be an integer.")
    if max_categorical_levels < 1:
        raise RealTrainingError("--max-categorical-levels must be positive.")
    paths = [Path(value).expanduser().resolve() for value in input_paths]
    if not paths:
        raise RealTrainingError("Provide one or more explicitly audited input CSV files.")
    if len(set(paths)) != len(paths):
        raise RealTrainingError("The same input CSV path was supplied more than once.")
    if any(path.suffix.casefold() != ".csv" for path in paths):
        raise RealTrainingError(
            "Only audited .csv files are accepted. Convert workbooks explicitly, "
            "then run Stage 6A on the resulting CSV before training."
        )
    report, alias_to_class, verified_sources = _load_audit_report(
        audit_report_path, paths
    )
    cleaning_manifest, cleaning_expectations = _load_cleaning_manifest(
        cleaning_manifest_path,
        paths,
    )
    investigation_report, investigation_report_resolved = _load_investigation_report(
        investigation_report_path,
        paths,
        verified_sources,
    )
    dataset, label_name = _read_explicit_csvs(
        paths,
        verified_sources,
        alias_to_class,
        cleaning_expectations,
        label_column=label_column,
        max_rows=max_rows,
        random_state=random_state,
    )
    features, explicitly_excluded, exclusions = select_predictive_features(
        list(dataset.features.columns),
        label_name,
        excluded_columns=exclude_columns,
    )
    selected_features = dataset.features.loc[:, features].copy()
    split = make_group_split(
        selected_features,
        dataset.labels,
        dataset.references,
        random_state=random_state,
        context_frame=dataset.features,
        grouping_features=dataset.features,
    )
    training_indices = split["indices"]["train"]
    prepared_features, numeric_features, categorical_features = _typed_features(
        selected_features,
        training_indices,
        features,
        max_categorical_levels=max_categorical_levels,
    )
    training_constant_features = [
        name
        for name in numeric_features
        if prepared_features.iloc[training_indices][name].nunique(dropna=False) <= 1
    ]
    if preflight_only:
        return {
            "preflight_only": True,
            "selected_model": None,
            "selected_threshold": None,
            "validation_target_feasible": None,
            "split_counts": split["row_counts"],
            "class_counts": split["class_counts"],
            "group_counts": split["split_group_counts"],
            "mixed_label_group_stats": split["mixed_label_group_stats"],
            "distinct_group_count": split["distinct_group_count"],
            "split_strategy": split["strategy"],
            "predictive_feature_names": features,
            "excluded_columns": exclusions,
            "training_constant_features": training_constant_features,
            "artifact_paths_written": [],
        }
    output_path = Path(output_dir).expanduser().resolve()
    if output_path == (PROJECT_ROOT / "artifacts").resolve():
        raise RealTrainingError(
            "The artifacts root cannot be used as the real-data output directory; "
            "choose a dedicated child directory such as artifacts/real_data."
        )
    output_collisions = [output_path / name for name in OUTPUT_FILENAMES if (output_path / name).exists()]
    if output_collisions and not force:
        raise FileExistsError(
            f"Real-data output already exists: {output_collisions[0]}. "
            "Choose another --output-dir or pass --force to replace only Stage 6B files."
        )
    if force and output_collisions and not (output_path / OUTPUT_MARKER).is_file():
        raise RealTrainingError(
            f"Refusing to overwrite unrecognized files in '{output_path}'; "
            "the directory is not marked as FPGuard Stage 6B output."
        )

    split_refs = {
        name: [
            {
                "source_path": str(dataset.references.iloc[index]["source_path"]),
                "source_row_index": int(dataset.references.iloc[index]["source_row_index"]),
            }
            for index in indices
        ]
        for name, indices in split["indices"].items()
    }
    manifest = {
        "manifest_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": (
            "locally processed CIC-IDS2018 flow CSV with matching Stage 6A audit "
            "and cleaning-manifest hashes; external source authenticity is not "
            "independently established"
        ),
        "synthetic_data": None,
        "source_authenticity_verified": False,
        "cleaning_manifest": (
            {
                "path": str(Path(cleaning_manifest_path).expanduser().resolve()),
                "sha256": _sha256(Path(cleaning_manifest_path).expanduser().resolve()),
                "output_sha256_matches_input": True,
                "input_sha256": cleaning_manifest.get("source_sha256"),
                "output_sha256": cleaning_manifest.get("output_sha256"),
                "input_data_row_count": cleaning_manifest.get("input_data_row_count"),
                "output_data_row_count": cleaning_manifest.get("output_data_row_count"),
                "removed_repeated_header_count": cleaning_manifest.get(
                    "removed_repeated_header_count"
                ),
                "removed_source_csv_record_positions": cleaning_manifest.get(
                    "removed_source_csv_record_positions"
                ),
                "removal_rule": cleaning_manifest.get("removal_rule"),
            }
            if cleaning_manifest_path is not None and cleaning_manifest is not None
            else None
        ),
        "stage6a_investigation_report": (
            {
                "path": investigation_report_resolved,
                "sha256": _sha256(Path(investigation_report_resolved)),
                "source_sha256": investigation_report.get("source_sha256"),
                "conflict_counts_match_stage6a": investigation_report.get(
                    "stage6a_reference", {}
                ).get("conflict_counts_match"),
            }
            if investigation_report is not None
            and investigation_report_resolved is not None
            else None
        ),
        "input_sources": dataset.source_summaries,
        "stage6a_audit_report": str(Path(audit_report_path).expanduser().resolve()),
        "label_column": label_name,
        "positive_class": "attack",
        "label_mapping": dataset.audit_mapping,
        "population_rows_audited": dataset.total_audited_rows,
        "rows_used": dataset.selected_rows,
        "source_rows_not_used": dataset.total_audited_rows - dataset.selected_rows,
        "source_rows_not_used_reason": (
            dataset.sampling.get("representativeness_warning")
            if dataset.total_audited_rows != dataset.selected_rows
            else None
        ),
        "sampling": dataset.sampling,
        "class_counts_used": {
            label: int(dataset.labels.eq(label).sum()) for label in LABELS
        },
        "split_strategy": split["strategy"],
        "capture_context_grouping_used": split["context_grouping_used"],
        "capture_context_columns": split["context_columns"],
        "duplicate_predictive_feature_groups_kept_together": True,
        "distinct_group_count": split["distinct_group_count"],
        "split_group_counts": split["split_group_counts"],
        "random_state": random_state,
        "split_row_counts": split["row_counts"],
        "split_class_counts": split["class_counts"],
        "group_mixed_label_statistics": split["mixed_label_group_stats"],
        "mixed_label_group_count_total": int(
            len(split["mixed_label_group_ids"])
        ),
        "rows_in_mixed_label_groups_total": int(
            split["mixed_label_group_mask"].sum()
        ),
        "group_isolation_verified": True,
        "grouping_feature_names": list(dataset.features.columns),
        "row_references": split_refs,
        "predictive_feature_names": features,
        "feature_types": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "excluded_columns": exclusions,
        "reviewed_feature_decisions": {
            "dst_port": {
                "decision": "included_as_predictive_feature",
                "reason": (
                    "Semantically a destination service port, not a unique flow "
                    "identifier; Stage 6A.1 shows the heuristic is name-based. "
                    "Retained as a candidate behavioral predictor with capture "
                    "association/generalization limitations."
                ),
            },
            "timestamp": {
                "decision": "excluded_from_predictive_features",
                "reason": (
                    "Raw timestamp is held as source metadata only; one date is "
                    "present and it is not an independent capture-day grouping key."
                ),
            },
        },
        "training_constant_features_retained": training_constant_features,
        "feature_engineering_note": (
            "Grouping uses every source feature column (all 79 non-label fields), "
            "while prediction excludes raw timestamp and includes reviewed dst_port. "
            "No synthetic values or arbitrary feature renaming. Infinities become "
            "missing; medians, scaling, and any encoding are fit on training rows only. "
            "Training-constant numeric columns are retained; no validation/test "
            "statistics are used to remove features."
        ),
        "generalization_limitations": (
            "All rows are from 2018-02-28. This split does not establish cross-day, "
            "cross-network, or unseen-attack-family generalization. Exact feature "
            "grouping does not ensure separation of related flows or collection artifacts."
        ),
    }

    y = dataset.labels.reset_index(drop=True)
    x = prepared_features.reset_index(drop=True)
    candidate_models = _candidate_pipelines(
        features,
        numeric_features,
        categorical_features,
        random_state,
    )
    validation_indices = split["indices"]["validation"]
    candidate_results: dict[str, dict[str, Any]] = {}
    validation_elapsed: dict[str, float] = {}
    for name, candidate in candidate_models.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate.fit(x.iloc[training_indices], y.iloc[training_indices])
        validation_probabilities = _attack_probabilities(
            candidate, x.iloc[validation_indices]
        )
        selected = select_threshold(
            validation_probabilities,
            y.iloc[validation_indices].to_numpy(),
            minimum_recall=minimum_recall,
            candidate_thresholds=DEFAULT_CANDIDATE_THRESHOLDS,
        )
        validation_elapsed[name] = len(caught)
        if selected["feasible"]:
            threshold = float(selected["selected_threshold"])
            validation_metrics = evaluate_real_metrics(
                y.iloc[validation_indices].to_numpy(),
                validation_probabilities,
                threshold=threshold,
            )
        else:
            validation_metrics = None
        candidate_results[name] = {
            "feasible": bool(selected["feasible"]),
            "selected_threshold": selected["selected_threshold"],
            "minimum_validation_recall": minimum_recall,
            "validation_metrics": validation_metrics,
            "selection_reason": selected["selection_reason"],
            "fit_warning_count": validation_elapsed[name],
            "fit_warnings": [
                str(warning.message) for warning in caught[:10]
            ],
        }

    winner = select_winning_candidate(candidate_results)
    experiment: dict[str, Any] = {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_type": (
            manifest["data_type"]
        ),
        "synthetic_data": None,
        "source_authenticity_verified": False,
        "minimum_validation_recall": minimum_recall,
        "validation_target_feasible": winner is not None,
        "selected_model": winner,
        "selected_threshold": (
            candidate_results[winner]["selected_threshold"] if winner is not None else None
        ),
        "selection_policy": (
            "For each candidate, lowest validation FPR among 0.00–1.00 thresholds "
            "in 0.01 increments satisfying minimum recall; ties prefer precision, "
            "then higher threshold. Winner prefers validation FPR, precision, recall, "
            "then lexical candidate name."
        ),
        "candidates": candidate_results,
        "test_evaluation": {
            "performed": False,
            "reason": (
                "No candidate met minimum validation recall; no selected alternative "
                "configuration exists for final test evaluation."
                if winner is None
                else "Test set is evaluated only after freezing the validation winner."
            ),
        },
        "configuration": {
            "random_state": random_state,
            "maximum_rows": max_rows,
            "rows_used": dataset.selected_rows,
            "row_sampling": dataset.sampling,
            "input_sources": dataset.source_summaries,
            "stage6a_audit_report": str(
                Path(audit_report_path).expanduser().resolve()
            ),
            "stage6a_audit_report_sha256": _sha256(
                Path(audit_report_path).expanduser().resolve()
            ),
            "cleaning_manifest": manifest["cleaning_manifest"],
            "stage6a_investigation_report": manifest[
                "stage6a_investigation_report"
            ],
            "minimum_validation_recall": minimum_recall,
            "candidate_thresholds": list(DEFAULT_CANDIDATE_THRESHOLDS),
            "test_used_for_selection": False,
            "baseline_model": "logistic_regression",
            "baseline_threshold": 0.5,
            "feature_names": features,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "excluded_columns": exclusions,
            "explicitly_excluded_columns": explicitly_excluded,
            "split_strategy": split["strategy"],
            "split_row_counts": split["row_counts"],
            "split_class_counts": split["class_counts"],
            "mixed_label_group_statistics": split["mixed_label_group_stats"],
            "grouping_feature_names": list(dataset.features.columns),
            "test_evaluated_only_after_validation_selection": True,
            "packages": _package_versions(),
        },
        "warnings": [
            "Data quality, feature semantics, and leakage risks require human review.",
            "Metrics do not establish generalization to other networks or unseen attack families.",
        ],
    }

    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / OUTPUT_MARKER).write_text(
        "FPGuard AI Stage 6B real-data experiment outputs.\n",
        encoding="utf-8",
    )
    manifest_path = output_path / MANIFEST_FILENAME
    comparison_path = output_path / COMPARISON_FILENAME
    _write_json(manifest_path, manifest)
    if winner is None:
        _write_json(comparison_path, experiment)
        return {
            "selected_model": None,
            "selected_threshold": None,
            "validation_target_feasible": False,
            "artifacts": [str(manifest_path), str(comparison_path)],
            "class_counts": manifest["class_counts_used"],
            "split_counts": split["row_counts"],
        }

    selected_threshold = float(candidate_results[winner]["selected_threshold"])
    baseline_model = candidate_models["logistic_regression"]
    selected_model = candidate_models[winner]
    test_indices = split["indices"]["test"]
    test_features = x.iloc[test_indices]
    test_labels = y.iloc[test_indices].to_numpy()
    baseline_probabilities = _attack_probabilities(baseline_model, test_features)
    selected_probabilities = (
        baseline_probabilities
        if winner == "logistic_regression"
        else _attack_probabilities(selected_model, test_features)
    )
    baseline_test = evaluate_real_metrics(
        test_labels,
        baseline_probabilities,
        threshold=0.5,
    )
    selected_test = evaluate_real_metrics(
        test_labels,
        selected_probabilities,
        threshold=selected_threshold,
    )
    test_mixed_mask = split["mixed_label_group_mask"][test_indices]
    test_mixed_indices = test_indices[test_mixed_mask]
    if len(test_mixed_indices):
        test_mixed_labels = y.iloc[test_mixed_indices].to_numpy()
        test_mixed_probabilities = selected_probabilities[test_mixed_mask]
        test_mixed_group_ids = set(split["feature_group_hashes"][test_mixed_indices])
        conflict_diagnostics = {
            "mixed_label_group_count": len(test_mixed_group_ids),
            "affected_row_count": int(len(test_mixed_indices)),
            "label_counts": {
                label: int(np.count_nonzero(test_mixed_labels == label))
                for label in LABELS
            },
            "baseline_metrics_at_threshold_0_50": evaluate_real_metrics(
                test_mixed_labels,
                baseline_probabilities[test_mixed_mask],
                threshold=0.5,
            ),
            "selected_policy_metrics_at_selected_threshold": evaluate_real_metrics(
                test_mixed_labels,
                test_mixed_probabilities,
                threshold=selected_threshold,
            ),
            "interpretation": (
                "Diagnostic subset of test rows belonging to mixed-label exact "
                "all-feature groups; not a replacement for overall test metrics."
            ),
        }
    else:
        conflict_diagnostics = {
            "mixed_label_group_count": 0,
            "affected_row_count": 0,
            "label_counts": {label: 0 for label in LABELS},
            "baseline_metrics_at_threshold_0_50": None,
            "selected_policy_metrics_at_selected_threshold": None,
            "interpretation": "No mixed-label feature groups were assigned to test.",
        }
    experiment["test_evaluation"] = {
        "performed": True,
        "split": "untouched test partition, used once after validation selection was frozen",
        "baseline_logistic_regression_threshold_0_50": baseline_test,
        "selected_policy": selected_test,
        "selected_policy_validation_feasible": True,
        "selected_policy_test_recall_met_target": (
            selected_test["recall"] >= minimum_recall
            if selected_test["recall"] is not None
            else False
        ),
        "comparison_selected_vs_baseline": _metric_changes(selected_test, baseline_test),
        "mixed_label_group_diagnostics": conflict_diagnostics,
        "interpretation": (
            "Measured on this dataset split only. The selected threshold is not "
            "promised to satisfy the recall target on test; no generalization claim is made."
        ),
    }
    _write_json(comparison_path, experiment)

    metadata = {
        "artifact_version": 1,
        "model_file": MODEL_FILENAME,
        "selected_model": winner,
        "selected_threshold": selected_threshold,
        "minimum_validation_recall": minimum_recall,
        "positive_class": "attack",
        "feature_names": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "input_sources": dataset.source_summaries,
        "stage6a_audit_report": str(
            Path(audit_report_path).expanduser().resolve()
        ),
        "random_state": random_state,
        "split_strategy": split["strategy"],
        "grouping_feature_names": list(dataset.features.columns),
        "training_constant_features_retained": training_constant_features,
        "reviewed_feature_decisions": manifest["reviewed_feature_decisions"],
        "training_rows": split["row_counts"]["train"],
        "validation_rows": split["row_counts"]["validation"],
        "test_rows": split["row_counts"]["test"],
        "training_pipeline_note": (
            "Non-finite conversion, imputation, scaler, and any categorical "
            "encoding are in the fitted pipeline and learned using training rows only. "
            "This artifact is separate from the synthetic baseline."
        ),
        "package_versions": _package_versions(),
    }
    joblib.dump(selected_model, output_path / MODEL_FILENAME)
    _write_json(output_path / METADATA_FILENAME, metadata)
    _write_json(
        output_path / TEST_METRICS_FILENAME,
        {
            "data_type": manifest["data_type"],
            "synthetic_data": None,
            "positive_class": "attack",
            "minimum_validation_recall": minimum_recall,
            "selected_threshold": selected_threshold,
            "test_metrics": experiment["test_evaluation"],
            "undefined_metric_policy": "Undefined metrics are JSON null with explanatory notes.",
        },
    )
    predicted = np.where(
        selected_probabilities >= selected_threshold,
        "attack",
        "benign",
    )
    diagnostics = dataset.references.iloc[test_indices].reset_index(drop=True).copy()
    diagnostics["true_label"] = test_labels
    diagnostics["predicted_label"] = predicted
    diagnostics["attack_probability"] = selected_probabilities
    diagnostics["source_csv_record_position"] = (
        diagnostics["source_row_index"].astype(np.int64) + 2
    )
    diagnostics["exact_all_feature_group_id"] = split["feature_group_hashes"][test_indices]
    diagnostics["mixed_label_feature_group"] = test_mixed_mask
    diagnostics["error_type"] = np.select(
        [
            (test_labels == "benign") & (predicted == "attack"),
            (test_labels == "attack") & (predicted == "benign"),
            (test_labels == "attack") & (predicted == "attack"),
        ],
        ["false_positive", "false_negative", "true_positive"],
        default="true_negative",
    )
    diagnostics.to_csv(output_path / DIAGNOSTICS_FILENAME, index=False)

    return {
        "selected_model": winner,
        "selected_threshold": selected_threshold,
        "validation_target_feasible": True,
        "test_metrics": experiment["test_evaluation"],
        "artifacts": [str(output_path / name) for name in OUTPUT_FILENAMES],
        "class_counts": manifest["class_counts_used"],
        "split_counts": split["row_counts"],
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate separate real-data network-flow classifiers. "
            "Requires explicit CSV input paths and a matching Stage 6A audit report."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="explicit audited CSV files")
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=DEFAULT_AUDIT_REPORT,
        help="matching Stage 6A real_data_audit.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="separate output directory (default: artifacts/real_data/cicids2018)",
    )
    parser.add_argument(
        "--label-column",
        help="override label column from Stage 6A report",
    )
    parser.add_argument(
        "--cleaning-manifest",
        type=Path,
        help="optional Stage 6A preparation manifest to verify CSV hash and labels",
    )
    parser.add_argument(
        "--investigation-report",
        type=Path,
        help="optional Stage 6A.1 conflict/leakage report whose source hash must match",
    )
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.80,
        help="minimum validation recall for threshold feasibility (default: 0.80)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="seed for sampling, group splitting, and candidate models",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="reproducible uniform development sample limit; default uses all audited rows",
    )
    parser.add_argument(
        "--exclude-column",
        action="append",
        default=[],
        help="additional source feature to exclude; repeat for multiple columns",
    )
    parser.add_argument(
        "--max-categorical-levels",
        type=int,
        default=DEFAULT_MAX_CATEGORICAL_LEVELS,
        help="maximum training-row levels accepted for categorical features",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing Stage 6B outputs in the chosen output directory only",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate provenance/schema and report deterministic grouped splits without fitting models",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real-data training and evaluation CLI."""
    args = _argument_parser().parse_args(argv)
    try:
        result = run_real_experiment(
            args.inputs,
            audit_report_path=args.audit_report,
            output_dir=args.output_dir,
            label_column=args.label_column,
            cleaning_manifest_path=args.cleaning_manifest,
            investigation_report_path=args.investigation_report,
            minimum_recall=args.minimum_recall,
            random_state=args.random_state,
            max_rows=args.max_rows,
            exclude_columns=args.exclude_column,
            max_categorical_levels=args.max_categorical_levels,
            force=args.force,
            preflight_only=args.preflight_only,
        )
    except (RealTrainingError, OSError, ValueError, TypeError) as exc:
        print(f"Real-data experiment blocked: {exc}", file=sys.stderr)
        return 2
    if result.get("preflight_only"):
        print(json.dumps(_json_safe(result), indent=2, allow_nan=False))
        print("Preflight only: no model was fitted and no artifacts were written.")
    else:
        print(
            f"Validation feasibility: {result['validation_target_feasible']}; "
            f"selected model: {result['selected_model']}; "
            f"threshold: {result['selected_threshold']}"
        )
        print("Artifacts written only to the configured real-data output directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
