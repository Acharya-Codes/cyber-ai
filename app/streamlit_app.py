"""Interactive local dashboard for the FPGuard AI synthetic flow prototype."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from fpguard.dashboard_data import (
    MAX_UPLOAD_BYTES,
    MAX_UPLOAD_ROWS,
    predict_uploaded_flows,
    read_uploaded_flow_csv,
    validate_selected_model_artifact,
)
from fpguard.data import DEFAULT_DATA_PATH
from fpguard.train import DEFAULT_ARTIFACT_DIR, MODEL_FILENAME

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / DEFAULT_ARTIFACT_DIR
DATASET_PATH = ROOT / DEFAULT_DATA_PATH
BASELINE_MODEL_PATH = ARTIFACTS / MODEL_FILENAME
SELECTED_MODEL_PATH = ARTIFACTS / "stage4_selected_model.joblib"

st.set_page_config(
    page_title="FPGuard AI | Network Flow Lab",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
      --ink: #e8f0f6; --muted: #9aabba; --panel: #111e2b;
      --panel2: #162637; --line: #24384a; --cyan: #55d6d2;
      --amber: #f4b860; --red: #ff7373; --blue: #78aaf7;
    }
    html, body, [class*="css"] { font-family: 'Segoe UI', Arial, sans-serif; }
    .stApp { background: #0a131d; color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(10,19,29,.92); }
    [data-testid="stSidebar"] { background: #0e1a26; border-right: 1px solid var(--line); }
    .block-container { padding-top: 1.5rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: -.035em; }
    h1 { font-weight: 800; }
    .hero {
      border: 1px solid #284254; border-radius: 18px; padding: 1.5rem 1.7rem;
      background: linear-gradient(120deg, #142b3b 0%, #122331 55%, #171f2d 100%);
      margin: .2rem 0 1.25rem 0;
    }
    .eyebrow { color: var(--cyan); text-transform: uppercase; letter-spacing: .16em;
      font: 600 .72rem Consolas, monospace; margin-bottom: .55rem; }
    .hero-title { color: var(--ink); font-size: 2rem; line-height: 1.15; font-weight: 800; }
    .hero-copy { color: #b5c5d1; margin-top: .65rem; max-width: 870px; }
    .metric-card {
      border: 1px solid var(--line); border-radius: 13px; padding: .9rem 1rem;
      background: linear-gradient(145deg, #132231, #101b27); min-height: 105px;
    }
    .metric-label { color: var(--muted); font: 500 .7rem Consolas, monospace;
      letter-spacing: .08em; text-transform: uppercase; }
    .metric-value { color: var(--ink); font-size: 1.65rem; font-weight: 800; margin-top: .25rem; }
    .metric-note { color: #8294a4; font-size: .75rem; }
    .status-badge {
      border: 1px solid #86544a; background: #382923; color: #ffd0ae; border-radius: 999px;
      display: inline-block; padding: .28rem .65rem; font: 600 .72rem Consolas, monospace;
    }
    .info-badge {
      border: 1px solid #28666a; background: #112f35; color: #99eddf; border-radius: 999px;
      display: inline-block; padding: .28rem .65rem; font: 600 .72rem Consolas, monospace;
    }
    .section-kicker { color: var(--cyan); font: 500 .72rem Consolas, monospace;
      text-transform: uppercase; letter-spacing: .14em; }
    div[data-testid="stMetric"] {
      background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
      padding: .8rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stTabs"] [role="tablist"] {
      gap: .3rem; border-bottom: 1px solid var(--line);
      flex-wrap: nowrap;
    }
    [data-testid="stTabs"] [data-testid="stTab"] {
      color: #aab9c6 !important; padding: .65rem .8rem; min-height: 3rem;
      flex-shrink: 0 !important; white-space: nowrap; line-height: 1.25;
      text-decoration: none !important;
    }
    [data-testid="stTabs"] [data-testid="stTab"] [data-testid="stMarkdownContainer"] p {
      margin: 0; color: inherit; font-family: "Segoe UI", Arial, sans-serif;
      font-size: .9rem; font-weight: 600; line-height: 1.25;
      white-space: nowrap; text-decoration: none;
    }
    [data-testid="stTabs"] [aria-selected="true"] { color: var(--cyan) !important; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    .small-note { color: var(--muted); font-size: .83rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "Not available"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Not available"
    return f"{number:.{digits}f}" if pd.notna(number) else "Not available"


def _metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def _plot_layout(fig: go.Figure, *, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Manrope, sans-serif", "color": "#dce7ef"},
        margin={"l": 20, "r": 20, "t": 115, "b": 20},
        title={
            "y": 1.0,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 16},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.12,
            "x": 0,
            "font": {"size": 11},
            "title": {"font": {"size": 10}},
        },
        xaxis={"gridcolor": "#213342", "zerolinecolor": "#334b5e"},
        yaxis={"gridcolor": "#213342", "zerolinecolor": "#334b5e"},
    )
    return fig


@st.cache_resource(show_spinner="Loading trusted local models…")
def _load_local_models() -> dict[str, Any]:
    from fpguard.dashboard_data import load_trusted_pipeline

    models: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for key, path in (
        ("baseline", BASELINE_MODEL_PATH),
        ("selected", SELECTED_MODEL_PATH),
    ):
        if path.is_file():
            try:
                models[key] = load_trusted_pipeline(path)
            except (OSError, ValueError, TypeError) as exc:
                errors[key] = str(exc)
    models["_errors"] = errors
    return models


@st.cache_data(show_spinner=False)
def _load_selected_model_metadata() -> dict[str, Any] | None:
    metadata_path = ARTIFACTS / "stage4_selected_model_metadata.json"
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Selected-model metadata must be a JSON object.")
    return payload


@st.cache_data(show_spinner=False)
def _load_reports() -> dict[str, Any]:
    from fpguard.dashboard_data import REPORT_FILES, load_json_report

    reports: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for name, filename in REPORT_FILES.items():
        try:
            reports[name] = load_json_report(ARTIFACTS / filename)
        except ValueError as exc:
            reports[name] = None
            errors[name] = str(exc)
    reports["_errors"] = errors
    return reports


@st.cache_data(show_spinner="Loading local demo data…")
def _load_demo_frame() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            "Synthetic demonstration CSV is missing. Run "
            "`python -m fpguard.data`, `python -m fpguard.train`, and "
            "`python -m fpguard.evaluate` from the repository root."
        )
    return pd.read_csv(DATASET_PATH)


def _policy_metrics(report: dict[str, Any] | None, selected: bool) -> dict[str, Any] | None:
    if not report:
        return None
    test_metrics = report.get("test")
    if not isinstance(test_metrics, dict):
        test_metrics = {}
    if selected:
        metrics = report.get("selected_candidate_test_metrics") or test_metrics.get(
            "alternative_policy"
        )
    else:
        metrics = report.get("baseline_test_metrics_threshold_0_50") or test_metrics.get(
            "baseline"
        )
    return metrics if isinstance(metrics, dict) else None


def _render_overview(
    reports: dict[str, dict[str, Any] | None],
    models: dict[str, Any],
    frame: pd.DataFrame | None,
) -> None:
    stage4 = reports.get("stage4_comparison")
    threshold_report = reports.get("stage3_threshold")
    baseline_metrics = (
        stage4.get("baseline_test_metrics_threshold_0_50")
        if stage4
        else (reports.get("baseline_metrics") or {}).get("test_baseline")
    )
    selected_metrics = _policy_metrics(stage4, True)
    selected_name = stage4.get("selected_model") if stage4 else None
    selected_threshold = stage4.get("selected_threshold") if stage4 else None
    if selected_metrics is None and threshold_report:
        selected_metrics = _policy_metrics(threshold_report, True)
        selected_threshold = threshold_report.get("selected_threshold")
        selected_name = "Baseline + validation threshold"

    st.markdown(
        '<div class="hero"><div class="eyebrow">Offline flow analytics · FPGuard AI</div>'
        '<div class="hero-title">Understand the alert trade-off.</div>'
        '<div class="hero-copy">Explore saved network-flow model experiments, inspect error '
        'patterns, and score compatible CSVs locally. This prototype is flow-based; it does '
        'not capture packets or connect to a network.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="status-badge">SYNTHETIC DATA · NOT REAL-WORLD VALIDATION</span>',
        unsafe_allow_html=True,
    )
    st.write("")
    if not any(name in models for name in ("baseline", "selected")):
        st.warning(
            "No compatible trusted model artifact is available. Run the baseline setup "
            "commands in the README. The dashboard will not train a model automatically."
        )
    metric_values = [
        ("BASELINE FPR", _fmt(baseline_metrics.get("false_positive_rate") if baseline_metrics else None), "test partition"),
        ("BASELINE RECALL", _fmt(baseline_metrics.get("recall") if baseline_metrics else None), "attack class"),
        ("SELECTED FPR", _fmt(selected_metrics.get("false_positive_rate") if selected_metrics else None), "test partition"),
        ("SELECTED RECALL", _fmt(selected_metrics.get("recall") if selected_metrics else None), "attack class"),
        ("THRESHOLD", _fmt(selected_threshold, 2), selected_name or "no selection report"),
        ("SELECTED FP", str(selected_metrics.get("false_positives", "Not available")) if selected_metrics else "Not available", "test partition"),
        ("SELECTED FN", str(selected_metrics.get("false_negatives", "Not available")) if selected_metrics else "Not available", "test partition"),
    ]
    for row_start in range(0, len(metric_values), 4):
        row_metrics = metric_values[row_start : row_start + 4]
        cols = st.columns(len(row_metrics))
        for column, (label, value, note) in zip(cols, row_metrics):
            with column:
                _metric_card(label, value, note)
    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.subheader("Policy snapshot")
        if baseline_metrics and selected_metrics:
            plot_data = pd.DataFrame(
                [
                    {
                        "Policy": "Baseline · 0.50",
                        "FPR": baseline_metrics.get("false_positive_rate"),
                        "Recall": baseline_metrics.get("recall"),
                        "Precision": baseline_metrics.get("precision"),
                    },
                    {
                        "Policy": f"Selected · {_fmt(selected_threshold, 2)}",
                        "FPR": selected_metrics.get("false_positive_rate"),
                        "Recall": selected_metrics.get("recall"),
                        "Precision": selected_metrics.get("precision"),
                    },
                ]
            ).melt(id_vars="Policy", var_name="Metric", value_name="Value")
            fig = px.bar(
                plot_data,
                x="Policy",
                y="Value",
                color="Metric",
                barmode="group",
                color_discrete_map={
                    "FPR": "#f4b860",
                    "Recall": "#55d6d2",
                    "Precision": "#78aaf7",
                },
                range_y=[0, 1],
                title="Test-set rates · synthetic experiment",
            )
            st.plotly_chart(_plot_layout(fig, height=350), width="stretch")
            baseline_matrix = baseline_metrics.get("confusion_matrix")
            selected_matrix = selected_metrics.get("confusion_matrix")
            if baseline_matrix and selected_matrix:
                matrix_fig = make_subplots(
                    rows=1,
                    cols=2,
                    subplot_titles=("Baseline · threshold 0.50", f"Selected · threshold {_fmt(selected_threshold, 2)}"),
                    horizontal_spacing=0.16,
                )
                for column, matrix in enumerate((baseline_matrix, selected_matrix), start=1):
                    matrix_fig.add_trace(
                        go.Heatmap(
                            z=matrix,
                            x=["Predicted benign", "Predicted attack"],
                            y=["Actual benign", "Actual attack"],
                            text=matrix,
                            texttemplate="%{text}",
                            colorscale=[[0, "#142333"], [1, "#55d6d2"]],
                            showscale=column == 2,
                            colorbar={"title": "Rows"},
                            hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>",
                        ),
                        row=1,
                        col=column,
                    )
                matrix_fig.update_layout(title="Confusion matrices · synthetic test set")
                st.plotly_chart(_plot_layout(matrix_fig, height=380), width="stretch")
            if selected_metrics.get("false_positive_rate") is not None:
                delta = selected_metrics["false_positive_rate"] - baseline_metrics["false_positive_rate"]
                if delta > 0:
                    st.error(f"Selected policy FPR is higher by {delta * 100:.2f} percentage points.")
                elif delta < 0:
                    st.info(f"Selected policy FPR is lower by {abs(delta) * 100:.2f} percentage points on this synthetic test set.")
                else:
                    st.info("Selected policy and baseline have the same measured FPR.")
        else:
            st.info("A saved comparison report with both policy metrics is not available.")
    with right:
        st.subheader("Experiment status")
        if stage4:
            feasible = stage4.get("selection_feasible")
            st.write(f"**Selected model:** {selected_name or 'No eligible model'}")
            st.write(f"**Validation selection:** {'Feasible' if feasible else 'Infeasible'}")
            st.write(f"**Candidate set:** {', '.join(stage4.get('candidate_validation_results', {}).keys()) or 'Not reported'}")
            change = (stage4.get("test_comparison_selected_candidate_vs_baseline") or {}).get(
                "absolute_fpr_change_percentage_points_candidate_minus_baseline"
            )
            if change is not None:
                st.caption(f"Measured test FPR change: {change:+.2f} percentage points.")
        else:
            st.warning("Stage 4 comparison report is missing or unavailable.")
        st.write(f"**Baseline model:** {'Loaded' if 'baseline' in models else 'Unavailable'}")
        st.write(f"**Selected model artifact:** {'Loaded' if 'selected' in models else 'Unavailable'}")
        st.write(f"**Demo rows:** {len(frame):,}" if frame is not None else "**Demo rows:** unavailable")
        st.caption("Probability outputs are model scores, not calibrated guarantees of attack likelihood.")


def _render_detection_demo(
    models: dict[str, Any],
    stage4: dict[str, Any] | None,
    demo_frame: pd.DataFrame | None,
) -> None:
    st.markdown('<div class="section-kicker">Local inference</div>', unsafe_allow_html=True)
    st.header("Detection Demo")
    st.caption("The selected fitted pipeline is used when available; otherwise the saved baseline is used.")
    model_key = "selected" if "selected" in models else "baseline"
    model = models.get(model_key)
    if model is None:
        st.warning("A trusted local model is unavailable. Run the baseline setup or Stage 4 comparison commands.")
        return
    default_threshold = (
        float(stage4["selected_threshold"])
        if model_key == "selected" and stage4 and stage4.get("selected_threshold") is not None
        else 0.5
    )
    threshold = st.number_input(
        "Prediction threshold",
        min_value=0.0,
        max_value=1.0,
        value=default_threshold,
        step=0.01,
        format="%.2f",
        key="detection_threshold",
    )
    mode = st.radio("Input mode", ["Single flow", "CSV batch"], horizontal=True)
    if mode == "Single flow":
        st.caption("Numeric defaults use the saved synthetic demo feature medians when available.")
        columns = st.columns(2)
        values: dict[str, float] = {}
        defaults = {
            "flow_duration": 100.0,
            "total_fwd_packets": 8.0,
            "total_backward_packets": 5.0,
            "total_fwd_bytes": 4_000.0,
            "total_backward_bytes": 2_200.0,
            "flow_bytes_per_second": 62_000.0,
            "packet_length_mean": 430.0,
            "packet_length_std": 110.0,
            "syn_flag_count": 1.0,
            "destination_port": 443.0,
        }
        if demo_frame is not None:
            for feature in model.feature_names_in_:
                if feature in demo_frame:
                    values_in_demo = pd.to_numeric(demo_frame[feature], errors="coerce")
                    median = values_in_demo.replace([np.inf, -np.inf], np.nan).median()
                    if pd.notna(median):
                        defaults[feature] = float(median)
        for index, feature in enumerate(model.feature_names_in_):
            with columns[index % 2]:
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    value=defaults.get(feature, 0.0),
                    key=f"single_{feature}",
                )
        if st.button("Classify flow", type="primary", key="classify_single"):
            try:
                from fpguard.predict import predict_probabilities

                row = pd.DataFrame([values], columns=model.feature_names_in_)
                probability = float(predict_probabilities(model, row).iloc[0])
                prediction = "attack" if probability >= threshold else "benign"
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Predicted class", prediction.upper())
                with col_b:
                    st.metric("Attack score", f"{probability:.1%}")
                st.caption(f"Temporary decision threshold: {threshold:.2f}. Score is not a calibrated probability.")
            except (ValueError, TypeError, KeyError) as exc:
                st.error(f"Could not classify this flow: {exc}")
    else:
        st.caption(
            f"Local processing only · max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB · "
            f"max {MAX_UPLOAD_ROWS:,} rows. Compatible features: "
            + ", ".join(model.feature_names_in_)
        )
        uploaded = st.file_uploader(
            "Upload flow CSV",
            type=["csv"],
            key="batch_csv",
            max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
        )
        if uploaded is not None:
            try:
                content = uploaded.getvalue()
                preview, metadata_columns = read_uploaded_flow_csv(
                    content, list(model.feature_names_in_)
                )
                st.success(
                    f"Schema valid: {len(preview):,} rows; "
                    f"{len(metadata_columns)} metadata column(s) excluded from inference."
                )
                output = predict_uploaded_flows(content, model, threshold=threshold)
                st.dataframe(output.head(100), width="stretch", hide_index=True)
                st.download_button(
                    "Download predictions CSV",
                    data=output.to_csv(index=False).encode("utf-8"),
                    file_name="fpguard_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )
                st.caption("Input row order and metadata columns are preserved; metadata is not passed to the model.")
            except (ValueError, TypeError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                st.error(f"Upload could not be processed: {exc}")
    st.info(
        "This is a synthetic-data demonstration. Uploaded flows may have a different distribution; "
        "predictions are not validated real-world intrusion detections."
    )


def _render_threshold_lab(
    reports: dict[str, dict[str, Any] | None],
    models: dict[str, Any],
) -> None:
    st.markdown('<div class="section-kicker">Decision policy exploration</div>', unsafe_allow_html=True)
    st.header("Threshold Lab")
    st.warning(
        "Retrospective exploration on saved validation rows. These interactive metrics are exploratory, "
        "not a new unbiased final evaluation; changing this slider does not alter experiment artifacts."
    )
    if "baseline" not in models:
        st.warning("The saved baseline model is unavailable.")
        return
    try:
        from fpguard.dashboard_data import validation_threshold_curve

        curve = validation_threshold_curve(DATASET_PATH, ARTIFACTS, model=models["baseline"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        st.error(f"Threshold curve unavailable: {exc}")
        return
    stage4 = reports.get("stage4_comparison")
    stage3 = reports.get("stage3_threshold")
    frozen_threshold = (
        stage4.get("selected_threshold")
        if stage4 and stage4.get("selected_threshold") is not None
        else (stage3 or {}).get("selected_threshold")
    )
    if frozen_threshold is not None:
        st.caption(f"Frozen experiment threshold: {float(frozen_threshold):.2f} · Slider value below is temporary.")
    threshold = st.slider(
        "Temporary threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(frozen_threshold if frozen_threshold is not None else 0.5),
        step=0.01,
        key="threshold_lab_value",
    )
    selected = curve.iloc[int(round(threshold * 100))]
    columns = st.columns(6)
    for col, label, value in zip(
        columns,
        ["FPR", "Recall", "Precision", "F1", "False positives", "False negatives"],
        [
            selected["false_positive_rate"],
            selected["recall"],
            selected["precision"],
            selected["f1_score"],
            selected["false_positives"],
            selected["false_negatives"],
        ],
    ):
        with col:
            st.metric(label, _fmt(value, 3) if isinstance(value, (float, np.floating)) else str(int(value)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["false_positive_rate"], name="FPR", line={"color": "#f4b860", "width": 2.5}))
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["recall"], name="Recall", line={"color": "#55d6d2", "width": 2.5}))
    fig.add_vline(x=threshold, line_dash="dot", line_color="#f1f5f8", annotation_text=f"Temporary {threshold:.2f}")
    st.plotly_chart(_plot_layout(fig, height=420), width="stretch")
    st.caption("Lower thresholds generally increase sensitivity and may also increase false positives. Curve source: saved validation partition.")


def _render_error_analysis(reports: dict[str, dict[str, Any] | None]) -> None:
    st.markdown('<div class="section-kicker">Post-hoc diagnosis</div>', unsafe_allow_html=True)
    st.header("Error Analysis")
    st.warning("Baseline diagnostic records are from the synthetic test set and are post-hoc—not model-selection inputs. Feature differences are exploratory associations, not causal explanations.")
    diagnosis = reports.get("stage4_diagnosis")
    records_path = ARTIFACTS / "stage4_baseline_diagnostics.csv"
    features_path = ARTIFACTS / "stage4_feature_diagnostics.csv"
    if diagnosis:
        counts = diagnosis.get("error_type_counts", {})
        metric_cols = st.columns(4)
        for col, key, label in zip(metric_cols, ["false_positive", "false_negative", "true_positive", "true_negative"], ["False positives", "False negatives", "True positives", "True negatives"]):
            with col:
                st.metric(label, counts.get(key, "Not reported"))
        matrix = diagnosis.get("metrics", {}).get("confusion_matrix")
        if matrix:
            fig = px.imshow(
                matrix,
                text_auto=True,
                x=["Predicted benign", "Predicted attack"],
                y=["Actual benign", "Actual attack"],
                color_continuous_scale=[[0, "#142333"], [1, "#55d6d2"]],
                title="Baseline confusion matrix · class order benign, attack",
                aspect="auto",
            )
            st.plotly_chart(_plot_layout(fig, height=360), width="stretch")
    else:
        st.info("Stage 4 baseline diagnosis report is unavailable.")
    if records_path.is_file():
        try:
            records = pd.read_csv(records_path)
            if "error_type" not in records.columns:
                st.error("Diagnostic CSV is missing its error_type column.")
            else:
                error_filter = st.multiselect(
                    "Filter error types",
                    options=["false_positive", "false_negative", "true_positive", "true_negative"],
                    default=["false_positive", "false_negative"],
                )
                filtered = records[records["error_type"].isin(error_filter)]
                st.dataframe(filtered.head(250), width="stretch", hide_index=True)
                st.download_button(
                    "Download filtered diagnostic subset",
                    filtered.to_csv(index=False).encode("utf-8"),
                    "fpguard_error_subset.csv",
                    "text/csv",
                )
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read baseline diagnostic CSV: {exc}")
    else:
        st.info("Per-row diagnostic CSV is missing. Run the Stage 4 comparison command to create it.")
    if features_path.is_file():
        try:
            feature_summary = pd.read_csv(features_path)
            st.subheader("Exploratory feature-group summaries")
            st.dataframe(feature_summary.head(80), width="stretch", hide_index=True)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Could not read feature diagnostic CSV: {exc}")


def _render_model_comparison(reports: dict[str, dict[str, Any] | None]) -> None:
    st.markdown('<div class="section-kicker">Validation selection · frozen test evaluation</div>', unsafe_allow_html=True)
    st.header("Model Comparison")
    report = reports.get("stage4_comparison")
    if not report:
        st.warning("Stage 4 comparison report is unavailable. Run `python -m fpguard.compare_models --minimum-recall 0.80`.")
        return
    config = report.get("configuration", {})
    st.caption(f"Minimum validation attack recall: {_fmt(config.get('minimum_validation_recall'))} · Positive class: {config.get('positive_class', 'Not reported')}")
    candidate_rows = []
    for name, result in report.get("candidate_validation_results", {}).items():
        metrics = result.get("validation_metrics") or {}
        candidate_rows.append(
            {
                "Candidate": name,
                "Feasible": result.get("feasible", False),
                "Selected threshold": result.get("selected_threshold"),
                "Validation FPR": metrics.get("false_positive_rate"),
                "Validation recall": metrics.get("recall"),
                "Validation precision": metrics.get("precision"),
                "Validation F1": metrics.get("f1_score"),
            }
        )
    if candidate_rows:
        st.dataframe(pd.DataFrame(candidate_rows), width="stretch", hide_index=True)
    if not report.get("selection_feasible"):
        st.error("No candidate met the validation recall constraint; no selected model is reported.")
        return
    selected = report.get("selected_candidate_test_metrics") or {}
    baseline = report.get("baseline_test_metrics_threshold_0_50") or {}
    comparison = report.get("test_comparison_selected_candidate_vs_baseline") or {}
    if not selected or not baseline:
        st.info("Final test metrics are missing from the saved report.")
        return
    delta = comparison.get("absolute_fpr_change_percentage_points_candidate_minus_baseline")
    if delta is not None and delta > 0:
        st.error(f"Measured FPR increased by {delta:.2f} percentage points on synthetic test data.")
    elif delta is not None and delta < 0:
        st.info(f"Measured FPR decreased by {abs(delta):.2f} percentage points on this synthetic test data.")
    left, right = st.columns(2)
    with left:
        st.subheader("Frozen test comparison")
        metric_rows = []
        for label, metrics in [("Baseline · 0.50", baseline), (f"{report.get('selected_model')} · {_fmt(report.get('selected_threshold'), 2)}", selected)]:
            metric_rows.append(
                {
                    "Policy": label,
                    "FPR": metrics.get("false_positive_rate"),
                    "Recall": metrics.get("recall"),
                    "Precision": metrics.get("precision"),
                    "F1": metrics.get("f1_score"),
                    "False positives": metrics.get("false_positives"),
                    "False negatives": metrics.get("false_negatives"),
                    "Accuracy": metrics.get("accuracy"),
                }
            )
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
        st.write(f"Relative FPR reduction: {_fmt(comparison.get('relative_fpr_reduction_percent'), 2)}%")
        st.write(f"Recall change: {_fmt(comparison.get('recall_change_percentage_points_candidate_minus_baseline'), 2)} percentage points")
        st.write(f"False-negative count change: {comparison.get('false_negative_count_change_candidate_minus_baseline', 'Not reported')}")
    with right:
        plot_frame = pd.DataFrame(
            [
                {"Policy": "Baseline", "FPR": baseline.get("false_positive_rate"), "Recall": baseline.get("recall"), "Precision": baseline.get("precision")},
                {"Policy": "Selected", "FPR": selected.get("false_positive_rate"), "Recall": selected.get("recall"), "Precision": selected.get("precision")},
            ]
        ).melt(id_vars="Policy", var_name="Metric", value_name="Value")
        chart = px.bar(
            plot_frame, x="Policy", y="Value", color="Metric", barmode="group",
            color_discrete_map={"FPR": "#f4b860", "Recall": "#55d6d2", "Precision": "#78aaf7"},
            range_y=[0, 1], title="Synthetic test metrics",
        )
        st.plotly_chart(_plot_layout(chart), width="stretch")
    st.caption("Test results are final evaluation only; they were not used to select the model or threshold. Synthetic results are not real-world NIDS evidence.")


def _render_about() -> None:
    st.markdown('<div class="section-kicker">Scope and interpretation</div>', unsafe_allow_html=True)
    st.header("About & Limitations")
    st.warning("FPGuard AI is an offline, synthetic-data prototype. It does not capture packets, monitor a live network, or establish real-world cybersecurity effectiveness.")
    st.markdown(
        """
        - **Flow-based:** inputs are tabular flow records; there is no live packet capture.
        - **Synthetic training data:** saved scores and reports illustrate the pipeline only.
        - **FPR:** false positives divided by benign rows; a false positive is a benign flow predicted as attack.
        - **Recall:** attacks correctly detected divided by all attack rows; a false negative is an attack flow predicted benign.
        - **Threshold:** the attack-score cutoff for the temporary or frozen policy. Lowering it usually detects more attacks but can flag more benign flows.
        - **Scores:** model outputs are not calibrated guarantees of attack likelihood and may not transfer to another dataset distribution.
        - **Evaluation discipline:** validation was used for threshold/model selection. Test metrics are final retrospective measurements and must not drive further tuning.
        - **Local-only uploads:** uploaded rows are processed in this Streamlit process and are not sent to an external service.
        """
    )
    st.code(".\\.venv\\Scripts\\python.exe -m streamlit run app\\streamlit_app.py", language="powershell")


def main() -> None:
    st.sidebar.markdown("### 🛡 FPGuard AI")
    st.sidebar.caption("Offline network-flow analysis")
    st.sidebar.markdown(
        '<span class="status-badge">SYNTHETIC DEMO</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.write("")
    st.sidebar.caption("Trusted local artifacts only · no automatic retraining")
    try:
        models = _load_local_models()
    except (OSError, ValueError, TypeError) as exc:
        models = {}
        st.sidebar.error(f"Model loading issue: {exc}")
    try:
        reports = _load_reports()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        reports = {}
        st.sidebar.error(f"Report loading issue: {exc}")
    for name, message in models.get("_errors", {}).items():
        st.sidebar.error(f"{name.title()} model unavailable: {message}")
    for name, message in reports.get("_errors", {}).items():
        st.sidebar.error(f"{name.replace('_', ' ').title()} report unavailable: {message}")
    try:
        frame = _load_demo_frame()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        frame = None
        st.sidebar.caption(f"Demo dataset unavailable: {exc}")

    tabs = st.tabs(
        [
            "Overview",
            "Detection Demo",
            "Threshold Lab",
            "Error Analysis",
            "Model Comparison",
            "About & Limitations",
        ]
    )
    with tabs[0]:
        _render_overview(reports, models, frame)
    if "selected" in models:
        try:
            metadata = _load_selected_model_metadata()
            validate_selected_model_artifact(
                models["selected"],
                ARTIFACTS / "stage4_selected_model_metadata.json",
                reports.get("stage4_comparison"),
                dataset_path=DATASET_PATH,
                manifest_path=ARTIFACTS / "split_indices.json",
            )
            if metadata is not None:
                st.sidebar.caption(
                    f"Selected artifact · {metadata.get('selected_model')} · "
                    f"threshold {_fmt(metadata.get('selected_threshold'), 2)}"
                )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            models.pop("selected", None)
            models.setdefault("_errors", {})["selected"] = str(exc)
            st.sidebar.error(f"Selected model unavailable: {exc}")
    with tabs[1]:
        _render_detection_demo(models, reports.get("stage4_comparison"), frame)
    with tabs[2]:
        _render_threshold_lab(reports, models)
    with tabs[3]:
        _render_error_analysis(reports)
    with tabs[4]:
        _render_model_comparison(reports)
    with tabs[5]:
        _render_about()


main()
