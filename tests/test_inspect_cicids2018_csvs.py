"""Tests for streaming CIC-IDS2018 CSV inspection."""

import csv

from scripts.inspect_cicids2018_csvs import inspect_csv, inspect_pair


def _write_csv(path, rows, *, header=("Feature", "Label")):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_inspection_preserves_exact_labels_and_counts_missing_and_malformed_rows(
    tmp_path,
):
    path = _write_csv(
        tmp_path / "labels.csv",
        [
            ["1", "Benign"],
            ["2", " benign "],
            ["3", "BENIGN"],
            ["4", ""],
            ["5"],
            ["6", "Attack", "extra"],
        ],
    )

    report = inspect_csv(path, tmp_path / "duplicates.sqlite")

    assert report["data_row_count"] == 6
    assert report["label_value_counts_exact"] == {
        "": 1,
        " benign ": 1,
        "Attack": 1,
        "BENIGN": 1,
        "Benign": 1,
    }
    assert report["missing_label_count"] == 2
    assert report["unexpected_column_count_rows"] == 2
    assert report["unexpected_column_count_examples"] == [
        {
            "data_row_number": 5,
            "actual_columns": 1,
            "expected_columns": 2,
        },
        {
            "data_row_number": 6,
            "actual_columns": 3,
            "expected_columns": 2,
        },
    ]


def test_exact_duplicates_are_counted_using_disk_backed_store(tmp_path):
    path = _write_csv(
        tmp_path / "duplicates.csv",
        [
            ["1", "Benign"],
            ["1", "Benign"],
            ["2", "Attack"],
            ["1", "Benign"],
        ],
    )

    report = inspect_csv(path, tmp_path / "duplicate-index.sqlite")

    assert report["exact_duplicate_rows"]["duplicate_group_count"] == 1
    assert report["exact_duplicate_rows"]["duplicate_excess_count"] == 2
    assert report["exact_duplicate_rows"]["rows_in_duplicate_groups"] == 3
    assert "disk-backed SQLite" in report["exact_duplicate_rows"]["method"]


def test_pair_comparison_reports_header_order_and_rowwise_shared_values(tmp_path):
    left = _write_csv(
        tmp_path / "left.csv",
        [["1", "Benign"], ["2", "Attack"]],
        header=("Feature", "Label"),
    )
    right = _write_csv(
        tmp_path / "right.csv",
        [["Benign", "1"], ["Attack", "different"]],
        header=("Label", "Feature"),
    )

    report = inspect_pair([left, right])

    assert report["headers_equal_in_name_and_order"] is False
    comparison = report["shared_row_comparison"]
    assert comparison["shared_columns_in_left_order"] == ["Feature", "Label"]
    assert comparison["row_pairs_compared"] == 2
    assert comparison["matching_shared_rows"] == 1
    assert comparison["mismatching_shared_rows"] == 1
    assert comparison["mismatch_examples"][0]["data_row_number"] == 2


def test_pair_comparison_includes_row_count_difference_in_compared_prefix(tmp_path):
    left = _write_csv(tmp_path / "left.csv", [["1", "Benign"], ["2", "Attack"]])
    right = _write_csv(tmp_path / "right.csv", [["1", "Benign"]])

    report = inspect_pair([left, right])

    assert report["files"][0]["data_row_count"] == 2
    assert report["files"][1]["data_row_count"] == 1
    assert report["shared_row_comparison"]["row_pairs_compared"] == 1
    assert report["shared_row_comparison"]["matching_shared_rows"] == 1
