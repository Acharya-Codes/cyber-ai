"""Tests for the local-only real-data CSV audit."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fpguard.real_data import (
    AuditError,
    audit_csv_file,
    main,
    write_audit_report,
)


def _csv(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    path.write_text(content, encoding=encoding)
    return path


def test_cli_help_does_not_require_an_input_dataset():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "fpguard.real_data", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "one or more input CSV paths" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--attack-label" in result.stdout


def test_csv_bom_whitespace_and_configurable_label_mappings_are_audited(tmp_path):
    source = _csv(
        tmp_path / "flows.csv",
        "\ufeff Flow Duration , Total Fwd Packets , Label \n"
        "1.5,2,BENIGN\n"
        "3.5,4,DoS traffic\n",
    )

    report = audit_csv_file(
        source,
        model_features=("flow_duration", "total_fwd_packets"),
        benign_labels=("benign", "normal"),
        attack_labels=("attack", "DoS traffic"),
    )

    assert report["columns"] == ["flow_duration", "total_fwd_packets", "label"]
    assert report["label_distribution_raw"] == {"BENIGN": 1, "DoS traffic": 1}
    assert report["label_distribution_mapped"] == {"attack": 1, "benign": 1}
    assert report["schema_compatibility"]["compatible_exact_feature_schema"] is True
    assert report["columns_requiring_review"]["possible_identifiers"] == []
    assert report["warnings"]


def test_unknown_labels_are_rejected_instead_of_defaulting_to_benign(tmp_path):
    source = _csv(
        tmp_path / "unknown.csv",
        "flow_duration,label\n1.0,benign\n2.0,unseen-category\n",
    )

    with pytest.raises(AuditError, match="Unknown label.*never assumed benign"):
        audit_csv_file(source, model_features=("flow_duration",))


@pytest.mark.parametrize(
    "header",
    [
        "flow_duration,flow_duration,label",
        " Flow Duration ,flow_duration,label",
        "flow_duration, FLOW   DURATION ,label",
    ],
)
def test_exact_or_normalized_duplicate_headers_are_rejected(tmp_path, header):
    source = _csv(tmp_path / "duplicates.csv", f"{header}\n1,2,benign\n")

    with pytest.raises(AuditError, match="duplicate or normalized-colliding"):
        audit_csv_file(source, model_features=("flow_duration",))


def test_numeric_parsing_issues_infinities_and_missing_values_are_reported(tmp_path):
    source = _csv(
        tmp_path / "numeric.csv",
        "flow_duration,total_fwd_packets,label\n"
        "not-a-number,inf,benign\n"
        ",2,attack\n",
    )

    report = audit_csv_file(
        source,
        model_features=("flow_duration", "total_fwd_packets"),
    )

    assert report["numeric_parsing_issues"][0]["column"] == "flow_duration"
    assert report["numeric_parsing_issues"][0]["invalid_value_count"] == 1
    assert report["infinities"]["by_column"] == {"total_fwd_packets": 1}
    assert report["missing_values"]["total_cells"] == 1


def test_exact_duplicate_rows_and_conflicting_feature_labels_are_counted(tmp_path):
    source = _csv(
        tmp_path / "duplicates-and-conflicts.csv",
        "flow_duration,total_fwd_packets,label\n"
        "1,1,benign\n"
        "1,1,benign\n"
        "1,1,attack\n"
        "2,2,attack\n",
    )

    report = audit_csv_file(
        source,
        model_features=("flow_duration", "total_fwd_packets"),
    )

    assert report["duplicate_rows"] == {
        "duplicate_excess_count": 1,
        "rows_in_duplicate_groups": 2,
        "duplicate_group_count": 1,
    }
    assert report["conflicting_labels"]["conflicting_group_count"] == 1
    assert report["conflicting_labels"]["rows_involved"] == 3
    assert report["conflicting_labels"]["examples"][0]["labels"] == ["attack", "benign"]


def test_conflicts_are_detected_even_when_flow_identifiers_differ(tmp_path):
    source = _csv(
        tmp_path / "identifier-conflicts.csv",
        "flow_id,flow_duration,total_fwd_packets,label\n"
        "row-a,1,1,benign\n"
        "row-b,1,1,attack\n",
    )

    report = audit_csv_file(
        source,
        model_features=("flow_duration", "total_fwd_packets"),
    )

    assert report["duplicate_rows"]["duplicate_excess_count"] == 0
    assert report["conflicting_labels"]["conflicting_group_count"] == 1
    assert report["conflicting_labels"]["rows_involved"] == 2
    assert "flow_id" not in report["conflicting_labels"]["compared_feature_columns"]


def test_missing_input_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        audit_csv_file(tmp_path / "missing.csv")


def test_audit_report_records_provenance_and_is_written_only_to_requested_directory(
    tmp_path,
):
    source = _csv(
        tmp_path / "source.csv",
        "flow_duration,label\n1,benign\n2,attack\n",
    )
    output_dir = tmp_path / "chosen-output"
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    report_path = write_audit_report(
        [source],
        output_dir,
        model_metadata_path=tmp_path / "missing-metadata.json",
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    source_report = report["sources"][0]
    assert report_path == output_dir / "real_data_audit.json"
    assert source_report["source_path"] == str(source.resolve())
    assert source_report["file_size_bytes"] == source.stat().st_size
    assert source_report["sha256"] == expected_hash
    assert report["training_performed"] is False
    assert report["splits_created"] is False
    assert report["leakage_assessment"]["status"] == "review_required"
    assert not (tmp_path / "real_data_audit.json").exists()


def test_cli_audits_explicit_local_csv_and_writes_report_to_requested_directory(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    source = _csv(
        tmp_path / "flows.csv",
        "flow_duration,label\n1.0,benign\n2.0,attack\n",
    )
    output_dir = tmp_path / "cli-output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fpguard.real_data",
            str(source),
            "--output-dir",
            str(output_dir),
            "--model-metadata",
            str(tmp_path / "missing-metadata.json"),
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Audited 1 CSV file(s)." in result.stdout
    report = json.loads((output_dir / "real_data_audit.json").read_text(encoding="utf-8"))
    assert report["sources"][0]["row_count"] == 2


def test_schema_compatibility_reports_missing_and_additional_features(tmp_path):
    source = _csv(
        tmp_path / "schema.csv",
        "flow_duration,flow_id,protocol,label\n1,id-1,tcp,benign\n"
        "2,id-2,udp,attack\n",
    )

    report = audit_csv_file(
        source,
        model_features=("flow_duration", "total_fwd_packets"),
    )

    compatibility = report["schema_compatibility"]
    assert compatibility["compatible_exact_feature_schema"] is False
    assert compatibility["missing_model_features"] == ["total_fwd_packets"]
    assert compatibility["additional_source_features"] == ["flow_id", "protocol"]
    assert report["columns_requiring_review"]["possible_identifiers"] == ["flow_id"]
    assert report["columns_requiring_review"]["possible_target_or_post_event_columns"] == []


def test_cli_returns_nonzero_for_missing_input_and_does_not_create_report(tmp_path):
    output_dir = tmp_path / "audit-output"

    result = main(
        [
            str(tmp_path / "missing.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result == 2
    assert not output_dir.exists()
