"""Tests for the initial project scaffold."""

import pandas as pd
import pytest

from fpguard.data import load_csv


def test_load_csv_checks_required_columns(tmp_path):
    csv_path = tmp_path / "flows.csv"
    pd.DataFrame({"duration": [1.5], "label": [0]}).to_csv(csv_path, index=False)

    frame = load_csv(csv_path, required_columns=("duration", "label"))

    assert frame.shape == (1, 2)


def test_load_csv_reports_missing_required_columns(tmp_path):
    csv_path = tmp_path / "flows.csv"
    pd.DataFrame({"duration": [1.5]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        load_csv(csv_path, required_columns=("duration", "label"))


def test_load_csv_rejects_duplicate_source_columns(tmp_path):
    csv_path = tmp_path / "flows.csv"
    csv_path.write_text("duration,label,label\n1.5,0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_csv(csv_path, required_columns=("duration", "label"))
