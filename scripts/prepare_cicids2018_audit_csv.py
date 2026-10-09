"""Create a traceable audit CSV by removing only exact repeated headers."""

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2018"
    / "cicids2018_streamed.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cicids2018"
    / "cicids2018_audit_ready.csv"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_name("cicids2018_cleaning_manifest.json")
EXPECTED_COLUMN_COUNT = 80
LABEL_COLUMN = "Label"
SCRIPT_VERSION = "1.0.0"
HASH_CHUNK_SIZE = 1024 * 1024
REMOVAL_RULE = (
    "Remove a data record only if all 80 parsed CSV fields exactly equal the "
    "80 original header fields."
)


class PreparationError(RuntimeError):
    """Raised when source data fails audit-input preparation validation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_source_header(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), None)
    if header is None:
        raise PreparationError(f"Source CSV is empty and has no header: {path}")
    if len(header) != EXPECTED_COLUMN_COUNT:
        raise PreparationError(
            f"Expected {EXPECTED_COLUMN_COUNT} header columns; found {len(header)}."
        )
    if header[-1] != LABEL_COLUMN:
        raise PreparationError(
            f"Expected final header field {LABEL_COLUMN!r}; found {header[-1]!r}."
        )
    return tuple(header)


def _validate_output(
    path: Path,
    header: tuple[str, ...],
) -> tuple[int, Counter[str]]:
    labels: Counter[str] = Counter()
    row_count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        output_header = next(reader, None)
        if output_header != list(header):
            raise PreparationError("Temporary output header differs from source header.")
        for row_count, row in enumerate(reader, start=1):
            if len(row) != EXPECTED_COLUMN_COUNT:
                raise PreparationError(
                    f"Output data row {row_count} has {len(row)} fields; "
                    f"expected {EXPECTED_COLUMN_COUNT}."
                )
            if tuple(row) == header:
                raise PreparationError(
                    f"Output still contains an exact repeated header at data row {row_count}."
                )
            label = row[-1]
            if not label:
                raise PreparationError(
                    f"Retained output row {row_count} has an empty Label value."
                )
            labels[label] += 1
    if sum(labels.values()) != row_count:
        raise PreparationError("Not every retained row's label was counted.")
    return row_count, labels


def prepare_audit_csv(
    source_path: str | Path = DEFAULT_SOURCE,
    output_path: str | Path = DEFAULT_OUTPUT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Prepare output and manifest via temporary files, refusing all overwrites."""
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source CSV does not exist: {source}")
    if source in (output, manifest) or output == manifest:
        raise PreparationError("Source, output CSV, and manifest must be distinct files.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output CSV: {output}")
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest}")

    header = _read_source_header(source)
    source_hash_before = _sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    temporary_manifest: Path | None = None
    output_published = False
    manifest_published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_output = Path(destination.name)
            writer = csv.writer(destination)
            writer.writerow(header)
            removed_positions: list[int] = []
            input_data_rows = 0
            with source.open("r", encoding="utf-8-sig", newline="") as input_stream:
                reader = csv.reader(input_stream)
                current_header = next(reader, None)
                if current_header != list(header):
                    raise PreparationError(
                        "Source header changed between validation and processing."
                    )
                for source_record_number, row in enumerate(reader, start=2):
                    input_data_rows += 1
                    if len(row) != EXPECTED_COLUMN_COUNT:
                        raise PreparationError(
                            f"Source CSV record {source_record_number} has "
                            f"{len(row)} fields; expected {EXPECTED_COLUMN_COUNT}."
                        )
                    if tuple(row) == header:
                        removed_positions.append(source_record_number)
                        continue
                    if row[-1] == LABEL_COLUMN:
                        raise PreparationError(
                            f"Unresolved anomaly at source CSV record "
                            f"{source_record_number}: Label is {LABEL_COLUMN!r}, "
                            "but the other fields do not all match the original header. "
                            "No output was published."
                        )
                    writer.writerow(row)
            destination.flush()
            os.fsync(destination.fileno())

        output_data_rows, retained_labels = _validate_output(temporary_output, header)
        expected_rows = input_data_rows - len(removed_positions)
        if output_data_rows != expected_rows:
            raise PreparationError(
                f"Output row count {output_data_rows} does not match expected "
                f"retained row count {expected_rows}."
            )
        output_hash = _sha256(temporary_output)
        source_hash_after = _sha256(source)
        if source_hash_after != source_hash_before:
            raise PreparationError("Source file changed during preparation.")

        source_stat = source.stat()
        manifest_content = {
            "manifest_version": 1,
            "script": "scripts/prepare_cicids2018_audit_csv.py",
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_path": str(source),
            "source_size_bytes": source_stat.st_size,
            "source_sha256": source_hash_before,
            "output_path": str(output),
            "output_size_bytes": temporary_output.stat().st_size,
            "output_sha256": output_hash,
            "input_data_row_count": input_data_rows,
            "output_data_row_count": output_data_rows,
            "removed_repeated_header_count": len(removed_positions),
            "removed_source_csv_record_positions": removed_positions,
            "removal_rule": REMOVAL_RULE,
            "retained_raw_label_counts": dict(sorted(retained_labels.items())),
            "retained_labels_explicitly_counted": (
                sum(retained_labels.values()) == output_data_rows
            ),
            "output_has_no_exact_repeated_header_records": True,
            "source_hash_unchanged_during_preparation": (
                source_hash_after == source_hash_before
            ),
        }
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=manifest.parent,
            prefix=f".{manifest.name}.",
            suffix=".tmp",
            delete=False,
        ) as manifest_file:
            temporary_manifest = Path(manifest_file.name)
            json.dump(manifest_content, manifest_file, ensure_ascii=False, indent=2)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        try:
            os.link(temporary_output, output)
            output_published = True
            os.link(temporary_manifest, manifest)
            manifest_published = True
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite a destination created during preparation: {exc.filename}"
            ) from None

        return manifest_content
    except Exception:
        if output_published and not manifest_published and output.exists():
            output.unlink()
        raise
    finally:
        for temporary in (temporary_output, temporary_manifest):
            if temporary is not None and temporary.exists():
                temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove only exact repeated header records from the streamed "
            "CIC-IDS2018 CSV and write a separate audit-ready CSV plus manifest."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare_audit_csv(args.source, args.output, args.manifest)
    except (PreparationError, OSError, csv.Error, UnicodeError, ValueError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Source: {result['source_path']}")
    print(f"Output: {result['output_path']}")
    print(f"Manifest: {args.manifest.expanduser().resolve()}")
    print(f"Input data rows: {result['input_data_row_count']:,}")
    print(f"Output data rows: {result['output_data_row_count']:,}")
    print(f"Removed exact repeated headers: {result['removed_repeated_header_count']:,}")
    print(f"Retained label counts: {result['retained_raw_label_counts']}")
    print(f"Output SHA-256: {result['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
