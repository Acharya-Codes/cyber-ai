"""Stream-compare local CIC-IDS2018 CSVs and enumerate raw labels."""

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = (
    PROJECT_ROOT / "data" / "processed" / "cicids2018" / "cicids2018.csv",
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2018"
    / "cicids2018_streamed.csv",
)
LABEL_COLUMN = "Label"
CHUNK_SIZE = 1024 * 1024
SAMPLE_LIMIT = 10


class InspectionError(RuntimeError):
    """Raised when a CSV cannot be safely inspected."""


def _file_metadata(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "created_utc": datetime.fromtimestamp(
            stat.st_ctime, tz=timezone.utc
        ).isoformat(),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "sha256": digest.hexdigest(),
    }


def _open_csv(path: Path):
    stream = path.open("r", encoding="utf-8-sig", newline="")
    reader = csv.reader(stream)
    header = next(reader, None)
    if header is None:
        stream.close()
        raise InspectionError(f"CSV is empty and has no header: {path}")
    return stream, reader, header


def _initialize_duplicate_db(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute(
        "CREATE TABLE row_hashes ("
        "digest BLOB PRIMARY KEY, row_json BLOB NOT NULL, occurrences INTEGER NOT NULL)"
    )


def _insert_hash_batch(
    connection: sqlite3.Connection,
    batch: Sequence[tuple[bytes, bytes]],
) -> None:
    previous_changes = connection.total_changes
    connection.executemany(
        "INSERT INTO row_hashes(digest, row_json, occurrences) VALUES (?, ?, 1) "
        "ON CONFLICT(digest) DO UPDATE SET occurrences = occurrences + 1 "
        "WHERE row_hashes.row_json = excluded.row_json",
        batch,
    )
    if connection.total_changes - previous_changes != len(batch):
        raise InspectionError("SHA-256 collision detected while counting duplicate rows.")
    connection.commit()


def inspect_csv(path: str | Path, duplicate_db_path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    metadata = _file_metadata(path)
    stream, reader, header = _open_csv(path)
    try:
        if len(header) != len(set(header)):
            raise InspectionError(f"CSV has duplicate exact header names: {path}")
        if LABEL_COLUMN not in header:
            raise InspectionError(
                f"CSV is missing the expected {LABEL_COLUMN!r} column: {path}"
            )
        label_index = header.index(LABEL_COLUMN)
        label_counts: Counter[str] = Counter()
        missing_label_count = 0
        malformed_row_count = 0
        malformed_examples: list[dict[str, Any]] = []
        duplicate_connection = sqlite3.connect(str(duplicate_db_path))
        try:
            _initialize_duplicate_db(duplicate_connection)
            row_count = 0
            batch: list[tuple[bytes, bytes]] = []
            for row in reader:
                row_count += 1
                if len(row) != len(header):
                    malformed_row_count += 1
                    if len(malformed_examples) < SAMPLE_LIMIT:
                        malformed_examples.append(
                            {
                                "data_row_number": row_count,
                                "actual_columns": len(row),
                                "expected_columns": len(header),
                            }
                        )
                if label_index >= len(row) or row[label_index] == "":
                    missing_label_count += 1
                    if label_index < len(row):
                        label_counts[row[label_index]] += 1
                else:
                    label_counts[row[label_index]] += 1

                encoded = json.dumps(
                    row, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                batch.append((hashlib.sha256(encoded).digest(), encoded))
                if len(batch) >= 2_000:
                    _insert_hash_batch(duplicate_connection, batch)
                    batch.clear()
            if batch:
                _insert_hash_batch(duplicate_connection, batch)
            duplicate_groups, duplicate_excess, duplicate_group_rows = (
                duplicate_connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(occurrences - 1), 0), "
                    "COALESCE(SUM(occurrences), 0) FROM row_hashes "
                    "WHERE occurrences > 1"
                ).fetchone()
            )
        finally:
            duplicate_connection.close()
    finally:
        stream.close()

    return {
        **metadata,
        "header": header,
        "column_count": len(header),
        "data_row_count": row_count,
        "label_column": LABEL_COLUMN,
        "label_value_counts_exact": dict(sorted(label_counts.items())),
        "missing_label_count": missing_label_count,
        "unexpected_column_count_rows": malformed_row_count,
        "unexpected_column_count_examples": malformed_examples,
        "exact_duplicate_rows": {
            "duplicate_group_count": int(duplicate_groups),
            "duplicate_excess_count": int(duplicate_excess),
            "rows_in_duplicate_groups": int(duplicate_group_rows),
            "method": (
                "SHA-256 of each parsed CSV record with disk-backed SQLite counts; "
                "matching digests are checked against serialized records."
            ),
        },
    }


def compare_shared_rows(
    left_path: str | Path,
    right_path: str | Path,
    left_header: Sequence[str],
    right_header: Sequence[str],
) -> dict[str, Any]:
    """Compare the shared row-index prefix on identically named columns."""
    left_path = Path(left_path).resolve()
    right_path = Path(right_path).resolve()
    shared_columns = [name for name in left_header if name in set(right_header)]
    left_positions = [left_header.index(name) for name in shared_columns]
    right_positions = [right_header.index(name) for name in shared_columns]
    left_stream, left_reader, current_left_header = _open_csv(left_path)
    right_stream, right_reader, current_right_header = _open_csv(right_path)
    if current_left_header != list(left_header) or current_right_header != list(right_header):
        left_stream.close()
        right_stream.close()
        raise InspectionError("CSV headers changed after inspection.")

    compared = 0
    matching_rows = 0
    mismatching_rows = 0
    mismatch_examples: list[dict[str, Any]] = []
    try:
        while True:
            left_row = next(left_reader, None)
            right_row = next(right_reader, None)
            if left_row is None or right_row is None:
                break
            compared += 1
            differences = [
                name
                for name, left_index, right_index in zip(
                    shared_columns, left_positions, right_positions
                )
                if left_index >= len(left_row)
                or right_index >= len(right_row)
                or left_row[left_index] != right_row[right_index]
            ]
            if differences:
                mismatching_rows += 1
                if len(mismatch_examples) < SAMPLE_LIMIT:
                    mismatch_examples.append(
                        {
                            "data_row_number": compared,
                            "different_shared_columns": differences[:SAMPLE_LIMIT],
                        }
                    )
            else:
                matching_rows += 1
    finally:
        left_stream.close()
        right_stream.close()

    return {
        "comparison": "same row-index prefix; exact raw cell values",
        "shared_columns_in_left_order": shared_columns,
        "shared_column_count": len(shared_columns),
        "row_pairs_compared": compared,
        "matching_shared_rows": matching_rows,
        "mismatching_shared_rows": mismatching_rows,
        "mismatch_examples": mismatch_examples,
        "limitation": (
            "This compares rows at the same ordinal position only; it does not "
            "attempt fuzzy matching or reorder records."
        ),
    }


def inspect_pair(paths: Sequence[str | Path]) -> dict[str, Any]:
    if len(paths) != 2:
        raise ValueError("Exactly two CSV paths are required.")
    resolved = [Path(path).expanduser().resolve() for path in paths]
    with tempfile.TemporaryDirectory(prefix="fpguard_csv_inspection_") as temporary_dir:
        reports = [
            inspect_csv(path, Path(temporary_dir) / f"duplicates_{index}.sqlite")
            for index, path in enumerate(resolved)
        ]
        comparison = compare_shared_rows(
            resolved[0],
            resolved[1],
            reports[0]["header"],
            reports[1]["header"],
        )
    return {
        "files": reports,
        "headers_equal_in_name_and_order": reports[0]["header"]
        == reports[1]["header"],
        "shared_row_comparison": comparison,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-compare two CIC-IDS2018 CSVs, enumerate exact raw labels, "
            "and count exact duplicate rows with a disk-backed index."
        )
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        help="two CSV paths (defaults to the project's existing and streamed files)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = args.csv_files or DEFAULT_FILES
    try:
        report = inspect_pair(paths)
    except (InspectionError, FileNotFoundError, OSError, ValueError, csv.Error) as exc:
        print(f"Inspection failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
