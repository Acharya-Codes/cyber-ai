"""Investigate Stage 6A feature conflicts and leakage risks without training."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from fpguard.real_data import (
    _column_warnings,
    _normalize_label,
    normalize_column_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2018"
    / "cicids2018_audit_ready.csv"
)
DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "audit_reports"
    / "cicids2018_audit_ready"
    / "real_data_audit.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "audit_reports" / "cicids2018_investigation"
)
REPORT_NAME = "cicids2018_investigation.json"
SUMMARY_NAME = "cicids2018_investigation.md"
REPORT_VERSION = 1
EXAMPLE_GROUPS = 8
EXAMPLE_ROWS = 8


class InvestigationError(ValueError):
    """Raised when the audit evidence does not match the source being investigated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(audit_report: dict[str, Any]) -> dict[str, str]:
    section = audit_report.get("label_mapping")
    if not isinstance(section, dict):
        raise InvestigationError("Stage 6A report has no explicit label_mapping.")
    mapping: dict[str, str] = {}
    for canonical in ("benign", "attack"):
        aliases = section.get(canonical)
        if not isinstance(aliases, list) or not aliases:
            raise InvestigationError(
                f"Stage 6A report has no aliases for {canonical!r}."
            )
        for alias in aliases:
            mapping[_normalize_label(alias)] = canonical
    return mapping


def _feature_fingerprint(values: Sequence[Any]) -> str:
    serializable = [
        None if pd.isna(value) else str(value)
        for value in values
    ]
    payload = json.dumps(serializable, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _group_label_table(group_ids: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    return pd.crosstab(
        pd.Series(group_ids, name="group_id"),
        labels.reset_index(drop=True).rename("label"),
        dropna=False,
    )


def _group_summary(
    group_ids: np.ndarray,
    labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_table = _group_label_table(group_ids, labels)
    group_rows = pd.Series(group_ids).value_counts(sort=False).rename("rows")
    summary = pd.concat([group_rows, label_table], axis=1).fillna(0)
    for name in ("benign", "attack"):
        if name not in summary:
            summary[name] = 0
    summary["label_count"] = (summary[["benign", "attack"]] > 0).sum(axis=1)
    return summary, label_table


def analyze_frame(
    frame: pd.DataFrame,
    *,
    label_column: str,
    label_mapping: dict[str, str],
    source_path: str = "<fixture>",
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Reproduce exact Stage 6A grouping and classify conflict evidence."""
    normalized = frame.copy()
    normalized.columns = [normalize_column_name(str(name)) for name in normalized.columns]
    label_name = normalize_column_name(label_column)
    if normalized.columns.has_duplicates:
        raise InvestigationError("Input has duplicate normalized headers.")
    if label_name not in normalized.columns:
        raise InvestigationError(f"Label column {label_name!r} is missing.")

    original_values = normalized.copy()
    raw_labels = normalized[label_name].copy()
    if raw_labels.isna().any():
        raise InvestigationError("Missing labels prevent conflict analysis.")
    normalized_aliases = raw_labels.map(_normalize_label)
    unknown = sorted(set(normalized_aliases) - set(label_mapping))
    if unknown:
        raise InvestigationError(f"Unmapped labels found: {unknown[:10]}.")
    labels = normalized_aliases.map(label_mapping).reset_index(drop=True)
    raw_label_counts = {
        str(value): int(count)
        for value, count in raw_labels.value_counts(dropna=False).items()
    }
    normalized[label_name] = labels

    columns = list(normalized.columns)
    warning_columns = _column_warnings(columns, label_name)
    excluded = sorted(
        {column for flagged in warning_columns.values() for column in flagged}
    )
    all_features = [column for column in columns if column != label_name]
    comparison_features = [
        column for column in all_features if column not in set(excluded)
    ]
    if not comparison_features:
        raise InvestigationError("No Stage 6A comparison features remain.")

    base_group_ids = normalized.groupby(
        comparison_features, dropna=False, sort=False
    ).ngroup().to_numpy()
    base_summary, base_labels = _group_summary(base_group_ids, labels)
    conflicts = base_summary[base_summary["label_count"] > 1].copy()
    conflict_ids = set(conflicts.index.tolist())
    conflict_row_mask = pd.Series(base_group_ids).isin(conflict_ids).to_numpy()
    conflict_rows = normalized.loc[conflict_row_mask].copy()
    conflict_labels = labels.loc[conflict_row_mask]
    class_totals = {
        name: int(conflict_labels.eq(name).sum()) for name in ("benign", "attack")
    }
    raw_conflict_label_counts = {
        str(value): int(count)
        for value, count in raw_labels.loc[conflict_row_mask]
        .value_counts(dropna=False)
        .items()
    }

    variation_columns = [
        column for column in ("dst_port", "timestamp") if column in normalized.columns
    ]
    variation_records: dict[int, dict[str, int]] = {}
    if variation_columns and len(conflicts):
        group_variation = (
            normalized.loc[conflict_row_mask, [*variation_columns]]
            .assign(__group_id__=base_group_ids[conflict_row_mask])
            .groupby("__group_id__", sort=False)[variation_columns]
            .nunique(dropna=False)
        )
        for group_id, row in group_variation.iterrows():
            variation_records[int(group_id)] = {
                column: int(row[column]) for column in variation_columns
            }

    category_group_counts: Counter[str] = Counter()
    category_row_counts: Counter[str] = Counter()
    for group_id, row in conflicts.iterrows():
        varied = [
            column
            for column, count in variation_records.get(int(group_id), {}).items()
            if count > 1
        ]
        category = "+".join(varied) if varied else "no_excluded_field_variation"
        category_group_counts[category] += 1
        category_row_counts[category] += int(row["rows"])

    by_extra_keys: dict[str, dict[str, int]] = {}
    for added in (["dst_port"], ["timestamp"], variation_columns):
        added = [column for column in added if column in normalized.columns]
        if not added:
            continue
        key_name = "+".join(added)
        group_ids = normalized.groupby(
            [*comparison_features, *added], dropna=False, sort=False
        ).ngroup().to_numpy()
        grouped, _ = _group_summary(group_ids, labels)
        grouped_conflicts = grouped[grouped["label_count"] > 1]
        by_extra_keys[key_name] = {
            "conflicting_group_count": int(len(grouped_conflicts)),
            "rows_in_conflicting_groups": int(grouped_conflicts["rows"].sum()),
        }

    exact_feature_group_ids = normalized.groupby(
        all_features, dropna=False, sort=False
    ).ngroup().to_numpy()
    exact_feature_summary, _ = _group_summary(exact_feature_group_ids, labels)
    exact_feature_conflicts = exact_feature_summary[
        exact_feature_summary["label_count"] > 1
    ]

    exact_duplicate_rows = original_values.duplicated(keep=False)
    duplicate_excess = int(original_values.duplicated().sum())
    duplicate_group_count = int(
        original_values.loc[exact_duplicate_rows].drop_duplicates().shape[0]
    )

    examples: list[dict[str, Any]] = []
    largest_groups = conflicts.sort_values(
        ["rows", "benign", "attack"], ascending=[False, False, False]
    ).head(EXAMPLE_GROUPS)
    for group_id, group in largest_groups.iterrows():
        positions = np.flatnonzero(base_group_ids == group_id)
        first_position = int(positions[0])
        varied = variation_records.get(int(group_id), {})
        examples.append(
            {
                "feature_group_fingerprint_sha256": _feature_fingerprint(
                    normalized.iloc[first_position][comparison_features].tolist()
                ),
                "rows_in_group": int(group["rows"]),
                "label_counts": {
                    "benign": int(group["benign"]),
                    "attack": int(group["attack"]),
                },
                "source_csv_record_positions_1_based": [
                    int(position + 2) for position in positions[:EXAMPLE_ROWS]
                ],
                "record_position_note": (
                    "1-based CSV record position including header at position 1."
                ),
                "excluded_column_distinct_value_counts": varied,
            }
        )

    timestamp_summary: dict[str, Any] = {
        "column_present": "timestamp" in normalized.columns,
        "parsed_values": False,
        "parse_failures": None,
        "distinct_capture_dates": None,
        "date_counts_by_label": None,
        "source_timezone_information": None,
    }
    if "timestamp" in normalized.columns:
        timestamp_strings = normalized["timestamp"].astype("string")
        explicit_timezone = timestamp_strings.str.contains(
            r"(?:Z|[+-]\d{2}:?\d{2})$", case=False, na=False
        )
        timezone_encoded = bool(explicit_timezone.any())
        parsed = pd.to_datetime(
            normalized["timestamp"],
            errors="coerce",
            format="mixed",
            utc=timezone_encoded,
        )
        dates = parsed.dt.strftime("%Y-%m-%d")
        date_table = pd.crosstab(dates, labels)
        timestamp_summary = {
            "column_present": True,
            "parsed_values": bool(parsed.notna().all()),
            "parse_failures": int(parsed.isna().sum()),
            "distinct_capture_dates": int(dates.nunique(dropna=True)),
            "minimum_timestamp_value": (
                parsed.min().isoformat() if parsed.notna().any() else None
            ),
            "maximum_timestamp_value": (
                parsed.max().isoformat() if parsed.notna().any() else None
            ),
            "source_timezone_information": (
                "explicit timezone offset present in at least one value; parsed dates "
                "are normalized to UTC"
                if timezone_encoded
                else "no explicit timezone offset found; parsed values retain their "
                "naive wall-clock dates"
            ),
            "date_counts_by_label": {
                str(date): {
                    class_name: int(date_table.loc[date].get(class_name, 0))
                    for class_name in ("benign", "attack")
                }
                for date in date_table.index
            },
        }

    port_summary: dict[str, Any] = {"column_present": "dst_port" in normalized.columns}
    if "dst_port" in normalized.columns:
        port_counts = normalized["dst_port"].value_counts(dropna=False)
        port_summary.update(
            {
                "distinct_values_including_missing": int(
                    normalized["dst_port"].nunique(dropna=False)
                ),
                "most_common_values_and_counts": [
                    {
                        "value": None if pd.isna(value) else str(value),
                        "count": int(count),
                    }
                    for value, count in port_counts.head(10).items()
                ],
            }
        )

    protocol_summary: dict[str, Any] = {"column_present": "protocol" in normalized.columns}
    if "protocol" in normalized.columns:
        protocol_counts = normalized["protocol"].value_counts(dropna=False)
        protocol_summary["counts"] = {
            str(value): int(count) for value, count in protocol_counts.items()
        }

    constant_columns = [
        column
        for column in normalized.columns
        if normalized[column].nunique(dropna=False) <= 1
    ]
    return {
        "source_path": source_path,
        "source_sha256": source_sha256,
        "row_count": int(len(normalized)),
        "column_count": int(len(columns)),
        "label_counts_raw": raw_label_counts,
        "label_counts_mapped": {
            label: int(labels.eq(label).sum()) for label in ("benign", "attack")
        },
        "stage6a_grouping": {
            "feature_columns": comparison_features,
            "excluded_by_stage6a_name_heuristics": excluded,
            "operation": (
                "frame.groupby(feature_columns, dropna=False, sort=False).ngroup(); "
                "label is mapped before comparing group label uniqueness; no rounding."
            ),
            "conflicting_group_count": int(len(conflicts)),
            "rows_in_conflicting_groups": int(conflicts["rows"].sum()),
            "conflicting_rows_by_label": class_totals,
            "conflicting_rows_by_raw_label": raw_conflict_label_counts,
            "conflict_group_size_distribution": {
                str(size): int(count)
                for size, count in conflicts["rows"].value_counts().sort_index().items()
            },
            "conflict_variation_by_excluded_fields": {
                category: {
                    "group_count": int(category_group_counts[category]),
                    "rows": int(category_row_counts[category]),
                }
                for category in sorted(category_group_counts)
            },
            "representative_groups": examples,
        },
        "independent_reproduction": {
            "method": (
                "Re-applied the audited CSV to the same documented 77-column "
                "pandas exact-value grouping operation."
            ),
            "conflicting_group_count": int(len(conflicts)),
            "rows_in_conflicting_groups": int(conflicts["rows"].sum()),
        },
        "stage6a_name_heuristic_review": warning_columns,
        "exact_duplicate_rows": {
            "duplicate_excess_count": duplicate_excess,
            "rows_in_duplicate_groups": int(exact_duplicate_rows.sum()),
            "duplicate_group_count": duplicate_group_count,
            "definition": (
                f"All {len(columns)} parsed source columns equal, including raw label."
            ),
        },
        "same_all_feature_values_conflicting_labels": {
            "group_count": int(len(exact_feature_conflicts)),
            "rows": int(exact_feature_conflicts["rows"].sum()),
            "definition": (
                f"All {len(all_features)} feature columns equal, including timestamp "
                "and dst_port when present, but mapped labels differ."
            ),
        },
        "conflicts_after_adding_excluded_columns": by_extra_keys,
        "rounding_assessment": {
            "rounding_applied_by_stage6a": False,
            "near_duplicate_rounding_analysis_performed": False,
            "status": "unresolved",
            "reason": (
                "Stage 6A compares parsed values exactly. This audit cannot establish "
                "whether upstream extraction rounded values before these CSVs were created."
            ),
        },
        "capture_context": {
            "timestamp": timestamp_summary,
            "destination_port": port_summary,
            "protocol": protocol_summary,
        },
        "evaluation_design": {
            "capture_days_observed": timestamp_summary["distinct_capture_dates"],
            "cross_day_holdout_supported": (
                timestamp_summary["distinct_capture_dates"] is not None
                and timestamp_summary["distinct_capture_dates"] > 1
            ),
            "group_holdout_supported_for_exact_duplicates": True,
            "recommended_primary_design": (
                "If Stage 6B proceeds, keep exact all-feature groups (all 79 "
                "non-label fields) indivisible across train/validation/test. "
                "Allocate whole groups with class balance checked after assignment; "
                "mixed-label groups make ordinary row-level stratification invalid. "
                "Report achieved partition prevalence and fail if a partition lacks "
                "a class needed for its metrics."
            ),
            "limitations": [
                "A single observed date cannot support an independent cross-day holdout.",
                "No IP, flow ID, or independent capture-session key is present for grouping near-duplicate or related flows.",
                "Exact duplicate grouping cannot guarantee separation of all related flows or capture artifacts.",
                "A within-day chronological holdout would test only later records from this same observed capture context.",
            ],
        },
        "preprocessing_review": {
            "missing_values_total": int(normalized.isna().sum().sum()),
            "missing_by_feature": {
                column: int(count)
                for column, count in normalized.isna().sum().items()
                if count
            },
            "infinities_by_numeric_feature": {
                column: int(np.isinf(normalized[column].to_numpy(dtype=float)).sum())
                for column in normalized.columns
                if pd.api.types.is_numeric_dtype(normalized[column])
                and np.isinf(normalized[column].to_numpy(dtype=float)).any()
            },
            "constant_columns_observed_on_full_audit_input": constant_columns,
            "recommendation": (
                "Convert +/- infinity to missing and fit median imputers on training "
                "rows only. If dropping constants, detect them using training rows only "
                "inside the fitted pipeline; do not select them using validation/test."
            ),
        },
    }


def _validate_inputs(
    input_path: Path,
    audit_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str], str, dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Audit-ready CSV not found: {input_path}")
    if not audit_path.is_file():
        raise FileNotFoundError(f"Stage 6A audit report not found: {audit_path}")
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("sources"), list):
        raise InvestigationError("Stage 6A report has invalid sources structure.")
    resolved_input = input_path.resolve()
    audit_sources = [
        item
        for item in report["sources"]
        if isinstance(item, dict)
        and Path(str(item.get("source_path", ""))).resolve() == resolved_input
    ]
    if len(audit_sources) != 1:
        raise InvestigationError("Stage 6A report does not uniquely identify the input CSV.")
    actual_hash = _sha256(resolved_input)
    if audit_sources[0].get("sha256") != actual_hash:
        raise InvestigationError("Input CSV SHA-256 does not match Stage 6A audit.")
    frame = pd.read_csv(resolved_input, encoding="utf-8-sig", low_memory=False)
    if len(frame) != audit_sources[0].get("row_count"):
        raise InvestigationError("Input row count does not match Stage 6A audit.")
    if frame.shape[1] != audit_sources[0].get("column_count"):
        raise InvestigationError("Input column count does not match Stage 6A audit.")
    return frame, report, _load_mapping(report), actual_hash, audit_sources[0]


def build_summary(report: dict[str, Any]) -> str:
    conflicts = report["stage6a_grouping"]
    exact = report["exact_duplicate_rows"]
    timestamp = report["capture_context"]["timestamp"]
    ports = report["capture_context"]["destination_port"]
    preprocessing = report["preprocessing_review"]
    if timestamp["distinct_capture_dates"] == 1:
        date_assessment = (
            f"Only one date ({next(iter(timestamp['date_counts_by_label']))}) is present; "
            "a cross-day temporal holdout is not supported by this file."
        )
    else:
        date_assessment = (
            f"{timestamp['distinct_capture_dates']} dates are visible; date holdout is "
            "possible to investigate but still requires scenario/capture provenance."
        )
    return f"""# CIC-IDS2018 Stage 6A.1 investigation

## Scope and provenance

- Source CSV: `{report['source_path']}`
- SHA-256: `{report['source_sha256']}`
- Rows / columns: {report['row_count']:,} / {report['column_count']}
- Diagnostic only: no labels changed, no splits created, no model trained.

## Feature-conflict investigation

Stage 6A groups rows by **{len(conflicts['feature_columns'])} exact parsed feature columns**:
all columns except `label`, plus `dst_port` and `timestamp`, which its name-based
review flags. It uses pandas `groupby(..., dropna=False)` and does not round.
Independent reproduction found {conflicts['conflicting_group_count']:,} conflicting
groups involving {conflicts['rows_in_conflicting_groups']:,} rows: benign
{conflicts['conflicting_rows_by_label']['benign']:,}; attack
{conflicts['conflicting_rows_by_label']['attack']:,}.

- Exact full-row duplicates (including label): {exact['duplicate_excess_count']:,}
  excess rows in {exact['duplicate_group_count']:,} groups.
- Identical values across all 79 feature columns but different labels:
  {report['same_all_feature_values_conflicting_labels']['group_count']:,} groups,
  {report['same_all_feature_values_conflicting_labels']['rows']:,} rows.
- Conflict groups with variation in excluded columns:
  `{json.dumps(conflicts['conflict_variation_by_excluded_fields'], ensure_ascii=False)}`.
- Rounded-feature causation remains unresolved: Stage 6A did no rounding, but
  upstream extraction precision is not established by these files.

Representative examples are in the JSON as SHA-256 feature
fingerprints, source CSV record positions, label counts, and excluded-field
variation counts. Raw feature rows are not copied into the report. These
conflicts are observations, not proof of label error; do not drop or relabel
them without provenance review.

## Leakage and capture-context review

- `timestamp` is an actual timestamp field; {timestamp['parse_failures']:,} parse
  failures across {report['row_count']:,} rows and
  {timestamp['distinct_capture_dates']} date(s) were observed.
  {date_assessment}
- `dst_port` is a destination service port with
  {ports.get('distinct_values_including_missing', 'unknown')} distinct observed
  values. Stage 6A flags it because `dst` matches a broad identifier-name
  heuristic; that heuristic does not prove it is a unique identifier.
- `protocol` counts: `{json.dumps(report['capture_context']['protocol'].get('counts', {}), ensure_ascii=False)}`.
- Stage 6A name heuristics flagged
  `{json.dumps(report['stage6a_name_heuristic_review'], ensure_ascii=False)}`.
  No source/destination IP or flow-ID, attack-category, capture-session, or
  target/post-event columns appear by these checks. This is schema evidence,
  not proof that flow statistics are free of scenario or collection artifacts.

Recommended: exclude raw `timestamp` from predictive inputs; consider parsed
capture-day solely for chronology/grouping if trustworthy. Treat `dst_port`
and `protocol` as candidate behavioral predictors only after documenting
deployment relevance and checking scenario/capture association. Group exact
duplicates or identical predictive-feature vectors in one partition. Do not
group solely on port.

## Preprocessing and evaluation recommendations

- Full-file audit observed {preprocessing['missing_values_total']:,} missing cells;
  missing feature details and infinities are listed in the JSON.
- Convert infinities to missing, then fit imputers on training rows only.
- Constant features observed on the whole file:
  `{', '.join(preprocessing['constant_columns_observed_on_full_audit_input'])}`.
  If dropping them, make that decision in training-fitted preprocessing only.
- The recommended primary split, if proceeding, keeps exact groups made from
  **all {report['column_count'] - 1} non-label columns** indivisible across train/validation/test and
  balances class counts only at the group level. Because groups can contain
  both labels, ordinary row-level stratification is inappropriate; verify class
  presence and actual prevalence after assignment.
- Timestamp evidence supports no cross-day holdout. Grouping all 79 fields
  prevents exact-feature duplicates crossing partitions, not near-duplicates,
  related flows, or all capture artifacts. No IP/flow/session identifiers exist
  for stronger grouping. A within-day chronological holdout would assess only
  later records from this same capture context—not another day, network, or
  attack family.
- Keep conflicts intact for investigation; report their effect and limitations
  in any later experiment. Use train-only preprocessing, validation-only
  selection, and untouched test evaluation if Stage 6B proceeds.

## Status

This report is diagnostic evidence only. Upstream rounding, capture/session
independence, and the cause of conflicting labels remain unresolved. No Stage
6B run or model-performance claim is included.
"""


def _write_exclusive_pair(
    output_dir: Path,
    report: dict[str, Any],
) -> tuple[Path, Path]:
    json_path = output_dir / REPORT_NAME
    markdown_path = output_dir / SUMMARY_NAME
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"Investigation output already exists in {output_dir}; refusing overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temps: list[Path] = []
    published: list[Path] = []
    try:
        for path, content in (
            (json_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
            (markdown_path, build_summary(report)),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=output_dir,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                temps.append(temporary)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in zip(temps, (json_path, markdown_path)):
            os.link(temporary, destination)
            published.append(destination)
        return json_path, markdown_path
    except Exception:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for temporary in temps:
            temporary.unlink(missing_ok=True)


def investigate(
    input_path: str | Path = DEFAULT_INPUT,
    audit_path: str | Path = DEFAULT_AUDIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path, dict[str, Any]]:
    source = Path(input_path).expanduser().resolve()
    audit_file = Path(audit_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    frame, audit_report, label_mapping, source_hash, recorded = _validate_inputs(
        source, audit_file
    )
    report = analyze_frame(
        frame,
        label_column="label",
        label_mapping=label_mapping,
        source_path=str(source),
        source_sha256=source_hash,
    )
    recorded_conflicts = recorded["conflicting_labels"]
    conflict_counts_match = (
        report["stage6a_grouping"]["conflicting_group_count"]
        == recorded_conflicts["conflicting_group_count"]
        and report["stage6a_grouping"]["rows_in_conflicting_groups"]
        == recorded_conflicts["rows_involved"]
    )
    recorded_duplicates = recorded["duplicate_rows"]
    duplicate_counts_match = (
        report["exact_duplicate_rows"]["duplicate_excess_count"]
        == recorded_duplicates["duplicate_excess_count"]
        and report["exact_duplicate_rows"]["duplicate_group_count"]
        == recorded_duplicates["duplicate_group_count"]
    )
    report["independent_reproduction"].update(
        {
            "matches_stage6a_conflict_counts": conflict_counts_match,
            "duplicate_excess_count": report["exact_duplicate_rows"][
                "duplicate_excess_count"
            ],
            "duplicate_group_count": report["exact_duplicate_rows"][
                "duplicate_group_count"
            ],
            "matches_stage6a_duplicate_counts": duplicate_counts_match,
        }
    )
    report["metadata"] = {
        "report_version": REPORT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_mode": "diagnostic_only_no_training_no_splits",
        "implementation": Path(__file__).name,
        "memory_note": (
            "The source CSV is loaded into a pandas DataFrame for exact pandas "
            "groupby reproduction; this is not a streaming-memory analysis."
        ),
    }
    report["stage6a_reference"] = {
        "report_path": str(audit_file),
        "recorded_conflicting_group_count": recorded_conflicts[
            "conflicting_group_count"
        ],
        "recorded_conflicting_rows": recorded_conflicts["rows_involved"],
        "conflict_counts_match": conflict_counts_match,
        "duplicate_counts_match": duplicate_counts_match,
    }
    if not conflict_counts_match or not duplicate_counts_match:
        raise InvestigationError(
            "Independent conflict or duplicate counts differ from Stage 6A report; "
            "no investigation files were published."
        )
    paths = _write_exclusive_pair(destination, report)
    return paths[0], paths[1], report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostic-only analysis of Stage 6A feature conflicts and "
            "capture/leakage risks. Does not train, split, or modify source data."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        json_path, summary_path, report = investigate(
            args.input, args.audit_report, args.output_dir
        )
    except (InvestigationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Investigation failed: {exc}", file=sys.stderr)
        return 2
    print(f"JSON report: {json_path}")
    print(f"Markdown summary: {summary_path}")
    print(
        "Conflicting feature groups: "
        f"{report['stage6a_grouping']['conflicting_group_count']:,}; "
        "rows: "
        f"{report['stage6a_grouping']['rows_in_conflicting_groups']:,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
