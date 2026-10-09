"""Small-workbook tests for the streaming CIC-IDS2018 converter."""

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from openpyxl import Workbook

from scripts.convert_cicids2018 import (
    ConversionError,
    convert_workbook,
)


def _workbook(
    path: Path,
    *,
    sheet_name: str = "cicids2018",
    header: tuple[str, ...] = ("Flow Duration", "Label"),
) -> Path:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet_name
    worksheet.append(header)
    worksheet.append((123, "Benign"))
    worksheet.append((456.5, "Infilteration"))
    workbook.save(path)
    workbook.close()
    return path


def test_conversion_streams_rows_and_preserves_header_values_and_source(tmp_path):
    source = _workbook(tmp_path / "source.xlsx")
    output = tmp_path / "processed" / "flows.csv"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    count = convert_workbook(source, output)

    assert count == 3
    assert output.read_bytes().startswith(b"\xef\xbb\xbf")
    with output.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows == [
        ["Flow Duration", "Label"],
        ["123", "Benign"],
        ["456.5", "Infilteration"],
    ]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_existing_destination_is_never_overwritten(tmp_path):
    source = _workbook(tmp_path / "source.xlsx")
    output = tmp_path / "flows.csv"
    output.write_text("keep this\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        convert_workbook(source, output)

    assert output.read_text(encoding="utf-8") == "keep this\n"


def test_dry_run_checks_workbook_without_creating_output(tmp_path):
    source = _workbook(tmp_path / "source.xlsx")
    output = tmp_path / "not-created" / "flows.csv"

    assert convert_workbook(source, output, dry_run=True) == 0
    assert not output.exists()
    assert not output.parent.exists()


def test_wrong_final_header_stops_without_creating_output(tmp_path):
    source = _workbook(
        tmp_path / "wrong-header.xlsx",
        header=("Flow Duration", "target"),
    )
    output = tmp_path / "flows.csv"

    with pytest.raises(ConversionError, match="final worksheet header.*Label"):
        convert_workbook(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_missing_expected_sheet_is_reported(tmp_path):
    source = _workbook(tmp_path / "wrong-sheet.xlsx", sheet_name="Sheet1")

    with pytest.raises(ConversionError, match="Expected worksheet"):
        convert_workbook(source, tmp_path / "flows.csv")


def test_failed_row_write_removes_temporary_output(tmp_path, monkeypatch):
    import scripts.convert_cicids2018 as converter

    source = _workbook(tmp_path / "source.xlsx")
    output = tmp_path / "flows.csv"
    real_writer = csv.writer

    class FailingWriter:
        def __init__(self, stream):
            self._writer = real_writer(stream)
            self._calls = 0

        def writerow(self, row):
            self._calls += 1
            if self._calls == 2:
                raise OSError("simulated disk write failure")
            self._writer.writerow(row)

    monkeypatch.setattr(converter.csv, "writer", FailingWriter)

    with pytest.raises(OSError, match="simulated disk write failure"):
        convert_workbook(source, output)

    assert not output.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_cli_help_is_available():
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_cicids2018.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--source" in result.stdout
    assert "--output" in result.stdout
