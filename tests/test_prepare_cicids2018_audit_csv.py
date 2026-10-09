"""Tests for preparing a traceable CIC-IDS2018 audit input."""

import csv
import json
from pathlib import Path

import pytest

from scripts.prepare_cicids2018_audit_csv import (
    PreparationError,
    prepare_audit_csv,
)


HEADER = tuple([f"Feature {index}" for index in range(79)] + ["Label"])


def _write_source(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)
    return path


def _read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream))


def test_removes_only_exact_repeated_header_and_records_provenance(tmp_path):
    source = _write_source(
        tmp_path / "source.csv",
        [
            [f"v{index}" for index in range(79)] + ["Benign"],
            list(HEADER),
            [f"w{index}" for index in range(79)] + ["Infilteration"],
        ],
    )
    output = tmp_path / "audit_ready.csv"
    manifest_path = tmp_path / "cleaning_manifest.json"

    result = prepare_audit_csv(source, output, manifest_path)

    rows = _read_rows(output)
    assert rows == [
        list(HEADER),
        [f"v{index}" for index in range(79)] + ["Benign"],
        [f"w{index}" for index in range(79)] + ["Infilteration"],
    ]
    assert result["input_data_row_count"] == 3
    assert result["output_data_row_count"] == 2
    assert result["removed_repeated_header_count"] == 1
    assert result["removed_source_csv_record_positions"] == [3]
    assert result["retained_raw_label_counts"] == {
        "Benign": 1,
        "Infilteration": 1,
    }
    assert result["retained_labels_explicitly_counted"] is True
    assert result["output_has_no_exact_repeated_header_records"] is True
    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted == result
    assert len(result["source_sha256"]) == 64
    assert len(result["output_sha256"]) == 64


def test_label_equal_to_label_with_other_different_fields_fails_without_publish(
    tmp_path,
):
    anomaly = ["not a header"] + [f"value{index}" for index in range(78)] + ["Label"]
    source = _write_source(tmp_path / "source.csv", [anomaly])
    output = tmp_path / "audit_ready.csv"
    manifest_path = tmp_path / "cleaning_manifest.json"

    with pytest.raises(PreparationError, match="Unresolved anomaly.*source CSV record 2"):
        prepare_audit_csv(source, output, manifest_path)

    assert not output.exists()
    assert not manifest_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_retained_row_order_and_exact_values_are_preserved(tmp_path):
    first = [f"first-{index}" for index in range(79)] + ["Benign"]
    second = [f"second-{index}" for index in range(79)] + ["Infilteration"]
    source = _write_source(tmp_path / "source.csv", [first, second])
    output = tmp_path / "audit_ready.csv"

    prepare_audit_csv(source, output, tmp_path / "manifest.json")

    assert _read_rows(output) == [list(HEADER), first, second]


@pytest.mark.parametrize("existing_target", ["output", "manifest"])
def test_existing_outputs_are_not_overwritten(tmp_path, existing_target):
    source = _write_source(
        tmp_path / "source.csv",
        [[f"v{index}" for index in range(79)] + ["Benign"]],
    )
    output = tmp_path / "audit_ready.csv"
    manifest_path = tmp_path / "cleaning_manifest.json"
    target = output if existing_target == "output" else manifest_path
    target.write_text("preserve this", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_audit_csv(source, output, manifest_path)

    assert target.read_text(encoding="utf-8") == "preserve this"
    assert not output.exists() if existing_target == "manifest" else True
    assert not manifest_path.exists() if existing_target == "output" else True


def test_source_rows_with_wrong_column_count_are_rejected(tmp_path):
    source = _write_source(tmp_path / "source.csv", [["only", "two"]])
    output = tmp_path / "audit_ready.csv"

    with pytest.raises(PreparationError, match="record 2 has 2 fields"):
        prepare_audit_csv(source, output, tmp_path / "manifest.json")

    assert not output.exists()
