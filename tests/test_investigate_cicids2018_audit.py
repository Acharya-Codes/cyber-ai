"""Tests for diagnostic-only CIC-IDS2018 conflict investigation."""

import json

import pandas as pd
import pytest

from scripts.investigate_cicids2018_audit import (
    InvestigationError,
    _write_exclusive_pair,
    analyze_frame,
)


@pytest.fixture
def conflict_frame():
    return pd.DataFrame(
        [
            {"dst_port": 80, "timestamp": "2018-02-28 01:00:00", "Flow Duration": 10, "Label": "Benign"},
            {"dst_port": 443, "timestamp": "2018-02-28 01:00:01", "Flow Duration": 10, "Label": "Infilteration"},
            {"dst_port": 22, "timestamp": "2018-02-28 01:00:02", "Flow Duration": 20, "Label": "Benign"},
            {"dst_port": 22, "timestamp": "2018-02-28 01:00:02", "Flow Duration": 20, "Label": "Infilteration"},
            {"dst_port": 53, "timestamp": "2018-02-28 01:00:03", "Flow Duration": 30, "Label": "Benign"},
            {"dst_port": 53, "timestamp": "2018-02-28 01:00:03", "Flow Duration": 30, "Label": "Benign"},
        ]
    )


def _analyze(frame):
    return analyze_frame(
        frame,
        label_column="Label",
        label_mapping={"benign": "benign", "infilteration": "attack"},
    )


def test_reproduces_exact_conflicts_and_distinguishes_duplicate_rows(conflict_frame):
    report = _analyze(conflict_frame)

    assert (
        report["capture_context"]["timestamp"]["source_timezone_information"]
        == "no explicit timezone offset found; parsed values retain their naive wall-clock dates"
    )
    assert report["stage6a_grouping"]["feature_columns"] == ["flow_duration"]
    assert report["stage6a_name_heuristic_review"]["possible_identifiers"] == [
        "dst_port"
    ]
    assert report["stage6a_name_heuristic_review"]["possible_timestamps"] == [
        "timestamp"
    ]
    assert report["stage6a_grouping"]["conflicting_group_count"] == 2
    assert report["stage6a_grouping"]["rows_in_conflicting_groups"] == 4
    assert report["stage6a_grouping"]["conflicting_rows_by_raw_label"] == {
        "Benign": 2,
        "Infilteration": 2,
    }
    assert report["exact_duplicate_rows"] == {
        "duplicate_excess_count": 1,
        "rows_in_duplicate_groups": 2,
        "duplicate_group_count": 1,
        "definition": "All 4 parsed source columns equal, including raw label.",
    }
    assert report["same_all_feature_values_conflicting_labels"]["group_count"] == 1
    assert report["same_all_feature_values_conflicting_labels"]["rows"] == 2
    assert report["stage6a_grouping"]["conflict_variation_by_excluded_fields"] == {
        "dst_port+timestamp": {"group_count": 1, "rows": 2},
        "no_excluded_field_variation": {"group_count": 1, "rows": 2},
    }


def test_examples_are_bounded_and_do_not_include_raw_feature_values(conflict_frame):
    examples = _analyze(conflict_frame)["stage6a_grouping"]["representative_groups"]

    assert len(examples) == 2
    assert len(examples[0]["feature_group_fingerprint_sha256"]) == 64
    assert examples[0]["source_csv_record_positions_1_based"]
    assert "features" not in examples[0]
    assert "Flow Duration" not in json.dumps(examples)


def test_unmapped_labels_are_rejected(conflict_frame):
    conflict_frame.loc[0, "Label"] = "unreviewed label"

    with pytest.raises(InvestigationError, match="Unmapped labels"):
        _analyze(conflict_frame)


def test_exact_duplicate_count_uses_raw_labels_not_mapped_aliases():
    frame = pd.DataFrame(
        [
            {"flow": 1, "label": "Benign"},
            {"flow": 1, "label": "benign"},
        ]
    )

    report = analyze_frame(
        frame,
        label_column="label",
        label_mapping={"benign": "benign"},
    )

    assert report["exact_duplicate_rows"]["duplicate_excess_count"] == 0
    assert (
        report["same_all_feature_values_conflicting_labels"]["group_count"] == 0
    )
    assert report["evaluation_design"]["cross_day_holdout_supported"] is False


def test_normalized_header_collisions_are_rejected(conflict_frame):
    conflict_frame["flow_duration"] = conflict_frame["Flow Duration"]

    with pytest.raises(InvestigationError, match="duplicate normalized headers"):
        _analyze(conflict_frame)


def test_output_pair_refuses_to_overwrite(tmp_path, conflict_frame):
    report = _analyze(conflict_frame)
    output_dir = tmp_path / "report"
    output_dir.mkdir()
    target = output_dir / "cicids2018_investigation.json"
    target.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing overwrite"):
        _write_exclusive_pair(output_dir, report)

    assert target.read_text(encoding="utf-8") == "preserve"
    assert not (output_dir / "cicids2018_investigation.md").exists()
