"""Local-only CSV provenance and quality audits for external flow datasets.

This module audits data only. It does not split data, fit preprocessing,
train models, or modify existing experiment artifacts.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fpguard.data import FEATURE_COLUMNS, LABEL_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_METADATA = PROJECT_ROOT / "artifacts" / "baseline_metadata.json"
DEFAULT_BENIGN_LABELS = ("benign", "normal")
DEFAULT_ATTACK_LABELS = ("attack", "malicious")
REPORT_FILENAME = "real_data_audit.json"
EXAMPLE_LIMIT = 5

_IDENTIFIER_PARTS = ("ip", "address", "mac", "src", "dst")
_TIMESTAMP_PARTS = ("timestamp", "datetime", "date", "time")
_ATTACK_CATEGORY_PARTS = ("attack_type", "attack_category", "category", "subcategory")
_CAPTURE_CONTEXT_NAMES = {
    "capture_day",
    "capture_date",
    "day",
    "session",
    "session_id",
    "capture_session",
}
_LEAKAGE_PARTS = (
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


class AuditError(ValueError):
    """Raised when a source CSV cannot be safely interpreted for auditing."""


def normalize_column_name(name: str) -> str:
    """Normalize header case and whitespace to a stable snake-like form."""
    return re.sub(r"\s+", "_", name.strip().casefold()).strip("_")


def _normalize_label(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def _build_label_mapping(
    benign_labels: Sequence[str],
    attack_labels: Sequence[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, aliases in (("benign", benign_labels), ("attack", attack_labels)):
        if not aliases:
            raise AuditError(f"At least one {canonical} label mapping must be configured.")
        for alias in aliases:
            normalized = _normalize_label(alias)
            if not normalized:
                raise AuditError("Label mapping aliases cannot be blank.")
            if normalized in mapping and mapping[normalized] != canonical:
                raise AuditError(
                    f"Label alias {alias!r} is mapped to both benign and attack."
                )
            mapping[normalized] = canonical
    if set(mapping.values()) != {"benign", "attack"}:
        raise AuditError("Label mappings must define both benign and attack classes.")
    return mapping


def _read_checked_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist or is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            header = next(csv.reader(source), None)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise AuditError(f"Could not read CSV header from '{path}': {exc}") from exc
    if not header:
        raise AuditError(f"CSV '{path}' is empty or has no header.")

    normalized = [normalize_column_name(column) for column in header]
    if any(not column for column in normalized):
        raise AuditError(f"CSV '{path}' contains a blank column name.")
    owners: dict[str, str] = {}
    collisions: list[str] = []
    for original, canonical in zip(header, normalized):
        if canonical in owners:
            collisions.append(
                f"{owners[canonical]!r} and {original!r} normalize to {canonical!r}"
            )
        else:
            owners[canonical] = original
    if collisions:
        raise AuditError(
            f"CSV '{path}' contains duplicate or normalized-colliding column names: "
            + "; ".join(collisions)
        )

    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise AuditError(f"Could not parse CSV '{path}': {exc}") from exc
    if frame.shape[1] != len(normalized):
        raise AuditError(
            f"CSV '{path}' header has {len(normalized)} columns but parsed data has "
            f"{frame.shape[1]} columns."
        )
    frame.columns = normalized
    if frame.empty:
        raise AuditError(f"CSV '{path}' has a header but no data rows.")
    return frame, normalized


def _column_warnings(columns: Sequence[str], label_column: str) -> dict[str, list[str]]:
    features = [column for column in columns if column != label_column]
    identifiers = [
        column
        for column in features
        if column in {"id", "uuid", "flow_id", "record_id", "row_id"}
        or column.endswith(("_id", "_uuid"))
        or any(
            column == part
            or column.endswith(f"_{part}")
            or column.startswith(f"{part}_")
            for part in _IDENTIFIER_PARTS
        )
        or (
            (column.startswith("source_") or column.startswith("destination_"))
            and any(token in column for token in ("ip", "address", "mac", "host", "id"))
        )
    ]
    timestamps = [
        column
        for column in features
        if any(part in column for part in _TIMESTAMP_PARTS)
    ]
    attack_categories = [
        column
        for column in features
        if any(part in column for part in _ATTACK_CATEGORY_PARTS)
        or re.search(r"(^|_)attack($|_)", column)
    ]
    capture_context = [
        column
        for column in features
        if column in _CAPTURE_CONTEXT_NAMES
        or column.startswith("capture_day_")
        or column.startswith("capture_session_")
    ]
    leakage_risks = [
        column
        for column in features
        if any(part in column for part in _LEAKAGE_PARTS)
    ]
    return {
        "possible_identifiers": identifiers,
        "possible_timestamps": timestamps,
        "possible_attack_category_columns": attack_categories,
        "possible_capture_context_columns": capture_context,
        "possible_target_or_post_event_columns": leakage_risks,
    }


def _feature_conflicts(
    frame: pd.DataFrame,
    label_column: str,
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    feature_columns = list(feature_columns)
    if not feature_columns:
        return {
            "compared_feature_columns": [],
            "conflicting_group_count": 0,
            "rows_involved": 0,
            "examples": [],
        }

    groups = frame.groupby(feature_columns, dropna=False, sort=False).ngroup()
    group_label_counts = pd.DataFrame(
        {"group": groups, "label": frame[label_column]}
    ).groupby("group", sort=False)["label"].nunique()
    conflicting_groups = set(group_label_counts[group_label_counts > 1].index)
    conflict_rows = groups.isin(conflicting_groups)
    examples: list[dict[str, Any]] = []
    for group_id in list(conflicting_groups)[:EXAMPLE_LIMIT]:
        mask = groups == group_id
        examples.append(
            {
                "source_row_indices": [int(index) for index in frame.index[mask][:EXAMPLE_LIMIT]],
                "labels": sorted(frame.loc[mask, label_column].dropna().unique().tolist()),
            }
        )
    return {
        "compared_feature_columns": feature_columns,
        "conflicting_group_count": len(conflicting_groups),
        "rows_involved": int(conflict_rows.sum()),
        "examples": examples,
    }


def audit_csv_file(
    input_path: str | Path,
    *,
    label_column: str = LABEL_COLUMN,
    benign_labels: Sequence[str] = DEFAULT_BENIGN_LABELS,
    attack_labels: Sequence[str] = DEFAULT_ATTACK_LABELS,
    model_features: Sequence[str] = FEATURE_COLUMNS,
) -> dict[str, Any]:
    """Audit one CSV and return its provenance, data-quality, and schema report."""
    path = Path(input_path).expanduser().resolve()
    normalized_label_column = normalize_column_name(label_column)
    frame, columns = _read_checked_csv(path)
    if normalized_label_column not in frame.columns:
        raise AuditError(
            f"CSV '{path}' is missing configured label column "
            f"{normalized_label_column!r}. Available columns: {', '.join(columns)}"
        )

    mapping = _build_label_mapping(benign_labels, attack_labels)
    raw_labels = frame[normalized_label_column]
    if raw_labels.isna().any():
        missing_count = int(raw_labels.isna().sum())
        raise AuditError(
            f"Label column {normalized_label_column!r} in '{path}' contains "
            f"{missing_count} missing value(s)."
        )
    raw_distribution = Counter(str(value).strip() for value in raw_labels)
    normalized_labels = raw_labels.map(_normalize_label)
    unknown = sorted(set(normalized_labels) - set(mapping))
    if unknown:
        raise AuditError(
            f"Unknown label value(s) in {normalized_label_column!r} for '{path}': "
            f"{unknown[:EXAMPLE_LIMIT]}. Configure --benign-label and --attack-label "
            "explicitly; unknown labels are never assumed benign."
        )
    mapped_labels = normalized_labels.map(mapping)
    mapped_distribution = Counter(mapped_labels)

    missing_by_column = {
        column: int(count)
        for column, count in frame.isna().sum().items()
    }
    numeric_issues: list[dict[str, Any]] = []
    infinities_by_column: dict[str, int] = {}
    expected_features = [normalize_column_name(name) for name in model_features]
    for column in columns:
        if column == normalized_label_column:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid_mask = frame[column].notna() & converted.isna()
        invalid_count = int(invalid_mask.sum())
        if invalid_count and (converted.notna().any() or column in expected_features):
            numeric_issues.append(
                {
                    "column": column,
                    "invalid_value_count": invalid_count,
                    "examples": [
                        {
                            "row_index": int(index),
                            "value": str(value),
                        }
                        for index, value in frame.loc[invalid_mask, column]
                        .head(EXAMPLE_LIMIT)
                        .items()
                    ],
                }
            )
        infinity_count = int(np.isinf(converted.to_numpy(dtype=float, na_value=np.nan)).sum())
        if infinity_count:
            infinities_by_column[column] = infinity_count

    constant_columns = [
        column
        for column in columns
        if frame[column].nunique(dropna=False) <= 1
    ]
    duplicate_mask = frame.duplicated(keep=False)
    duplicate_count = int(frame.duplicated().sum())
    duplicated_distinct_rows = int(frame.loc[duplicate_mask].drop_duplicates().shape[0])

    canonical_frame = frame.copy()
    canonical_frame[normalized_label_column] = mapped_labels
    review = _column_warnings(columns, normalized_label_column)
    non_predictive_review_columns = {
        column for flagged_columns in review.values() for column in flagged_columns
    }
    conflict_features = [
        column
        for column in columns
        if column != normalized_label_column and column not in non_predictive_review_columns
    ]
    conflicts = _feature_conflicts(
        canonical_frame,
        normalized_label_column,
        conflict_features,
    )
    normalized_model_features = [normalize_column_name(name) for name in model_features]
    source_features = [column for column in columns if column != normalized_label_column]
    missing_model_features = [
        name for name in normalized_model_features if name not in source_features
    ]
    extra_source_features = [
        name for name in source_features if name not in normalized_model_features
    ]
    schema_compatibility = {
        "compatible_exact_feature_schema": not missing_model_features and not extra_source_features,
        "reference_model_features": normalized_model_features,
        "matched_features": [
            name for name in normalized_model_features if name in source_features
        ],
        "missing_model_features": missing_model_features,
        "additional_source_features": extra_source_features,
        "comparison_note": (
            "Name-level feature comparison only; value semantics and model-artifact "
            "behavioral compatibility have not been verified."
        ),
    }
    review_messages = [
        {
            "category": category,
            "columns": found,
            "message": f"Review {category.replace('_', ' ')} before selecting predictive features.",
        }
        for category, found in review.items()
        if found
    ]
    review_messages.append(
        {
            "category": "target_exclusion",
            "columns": [normalized_label_column],
            "message": "The configured label is the target and must never be a predictive input.",
        }
    )
    if missing_model_features or extra_source_features:
        review_messages.append(
            {
                "category": "feature_schema_mismatch",
                "columns": [*missing_model_features, *extra_source_features],
                "message": "Source feature names do not exactly match the baseline schema.",
            }
        )
    if numeric_issues:
        review_messages.append(
            {
                "category": "numeric_parsing",
                "columns": [item["column"] for item in numeric_issues],
                "message": "Numeric-looking or model-input columns contain unparseable values.",
            }
        )

    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "source_path": str(path),
        "file_size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
        "row_count": int(len(frame)),
        "column_count": int(len(columns)),
        "columns": columns,
        "label_column": normalized_label_column,
        "label_distribution_raw": dict(sorted(raw_distribution.items())),
        "label_distribution_mapped": dict(sorted(mapped_distribution.items())),
        "missing_values": {
            "total_cells": int(sum(missing_by_column.values())),
            "by_column": missing_by_column,
        },
        "infinities": {
            "total_values": int(sum(infinities_by_column.values())),
            "by_column": infinities_by_column,
        },
        "numeric_parsing_issues": numeric_issues,
        "constant_columns": constant_columns,
        "duplicate_rows": {
            "duplicate_excess_count": duplicate_count,
            "rows_in_duplicate_groups": int(duplicate_mask.sum()),
            "duplicate_group_count": duplicated_distinct_rows,
        },
        "conflicting_labels": conflicts,
        "schema_compatibility": schema_compatibility,
        "columns_requiring_review": review,
        "warnings": review_messages,
    }


def load_reference_model_schema(
    metadata_path: str | Path = DEFAULT_MODEL_METADATA,
) -> tuple[list[str], dict[str, Any]]:
    """Read baseline feature names from metadata without deserializing a model."""
    path = Path(metadata_path).expanduser()
    if path.is_file():
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return list(FEATURE_COLUMNS), {
                "source": "fpguard.data.FEATURE_COLUMNS fallback",
                "metadata_path": str(path.resolve()),
                "warning": f"Could not read baseline metadata: {exc}",
            }
        features = metadata.get("feature_names") if isinstance(metadata, dict) else None
        if (
            isinstance(features, list)
            and features
            and all(isinstance(feature, str) and feature.strip() for feature in features)
            and len(set(features)) == len(features)
        ):
            model_file = metadata.get("model_file")
            return features, {
                "source": "saved baseline metadata",
                "metadata_path": str(path.resolve()),
                "model_file": model_file,
                "model_file_available": bool(
                    isinstance(model_file, str) and (path.parent / model_file).is_file()
                ),
                "warning": None,
            }
        return list(FEATURE_COLUMNS), {
            "source": "fpguard.data.FEATURE_COLUMNS fallback",
            "metadata_path": str(path.resolve()),
            "warning": "Baseline metadata has no valid feature_names list.",
        }
    return list(FEATURE_COLUMNS), {
        "source": "fpguard.data.FEATURE_COLUMNS fallback",
        "metadata_path": str(path.resolve()),
        "warning": "Baseline metadata is unavailable; using the documented baseline schema.",
    }


def write_audit_report(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    label_column: str = LABEL_COLUMN,
    benign_labels: Sequence[str] = DEFAULT_BENIGN_LABELS,
    attack_labels: Sequence[str] = DEFAULT_ATTACK_LABELS,
    model_metadata_path: str | Path = DEFAULT_MODEL_METADATA,
) -> Path:
    """Audit explicitly supplied CSVs and write one JSON report under output_dir."""
    if not input_paths:
        raise AuditError("Provide at least one explicit CSV input path.")
    model_features, model_metadata = load_reference_model_schema(model_metadata_path)
    audits = [
        audit_csv_file(
            input_path,
            label_column=label_column,
            benign_labels=benign_labels,
            attack_labels=attack_labels,
            model_features=model_features,
        )
        for input_path in input_paths
    ]
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / REPORT_FILENAME
    if any(Path(audit["source_path"]) == report_path for audit in audits):
        raise AuditError("Audit report path would overwrite one of the source CSV files.")
    report = {
        "report_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": "local_csv_quality_and_provenance_only",
        "training_performed": False,
        "splits_created": False,
        "synthetic_data_regenerated": False,
        "label_mapping": {
            "benign": list(benign_labels),
            "attack": list(attack_labels),
            "matching": "case-insensitive with surrounding/repeated whitespace normalized",
        },
        "reference_model_schema": {
            **model_metadata,
            "feature_names": model_features,
            "compatibility_note": "Name-level comparison only; this audit does not certify model suitability.",
        },
        "sources": audits,
        "leakage_assessment": {
            "status": "review_required",
            "statement": (
                "Potential identifiers, timestamps, target-derived fields, and attack categories "
                "are flagged for human review. This report does not establish that the data is "
                "leakage-free."
            ),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report_path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit explicitly provided network-flow CSV files locally. "
            "No data is downloaded, split, or used to train a model."
        ),
        epilog=(
            "Default label aliases are benign/normal and attack/malicious. "
            "Use repeatable --benign-label and --attack-label options for "
            "dataset-specific values; unconfigured labels are rejected."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="one or more input CSV paths")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory where real_data_audit.json will be written",
    )
    parser.add_argument(
        "--label-column",
        default=LABEL_COLUMN,
        help="label column name (whitespace and case are normalized; default: label)",
    )
    parser.add_argument(
        "--benign-label",
        action="append",
        dest="benign_labels",
        help="source label mapped to benign; may be supplied more than once",
    )
    parser.add_argument(
        "--attack-label",
        action="append",
        dest="attack_labels",
        help="source label mapped to attack; may be supplied more than once",
    )
    parser.add_argument(
        "--model-metadata",
        type=Path,
        default=DEFAULT_MODEL_METADATA,
        help="saved baseline metadata JSON used for feature-name comparison",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line audit."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    benign_labels = [*DEFAULT_BENIGN_LABELS, *(args.benign_labels or [])]
    attack_labels = [*DEFAULT_ATTACK_LABELS, *(args.attack_labels or [])]
    try:
        report_path = write_audit_report(
            args.inputs,
            args.output_dir,
            label_column=args.label_column,
            benign_labels=benign_labels,
            attack_labels=attack_labels,
            model_metadata_path=args.model_metadata,
        )
    except (AuditError, OSError) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"Audited {len(args.inputs)} CSV file(s). Report: {report_path}")
    print("No training, splitting, downloading, or experiment-artifact changes were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
