# 🛡️ FPGuard-AI
### Machine Learning for Network Intrusion Detection & False-Positive Analysis

FPGuard-AI is a modular machine-learning-based Network Intrusion Detection System (NIDS) prototype designed to explore the challenge of detecting malicious network traffic while analyzing false positives and detection performance.

The project combines a reproducible ML pipeline, model comparison, validation-based threshold selection, dataset auditing, and an interactive Streamlit dashboard to make intrusion-detection experiments easier to inspect and understand.

> **Project status:** Research and development prototype. The real-world detection performance of the model must be established through controlled evaluation on appropriately prepared real network traffic data.

---

## ✨ Features

- **Modular ML pipeline** — Separate components for data preparation, preprocessing, training, evaluation, and prediction.
- **Multiple model comparison** — Compare Random Forest, Logistic Regression, and Extra Trees baselines.
- **Threshold optimization** — Select a classification threshold using validation data and a configurable recall target.
- **False-positive analysis** — Examine false-positive rates, precision, recall, and classification errors.
- **Interactive dashboard** — Explore model metrics, prediction results, threshold experiments, and diagnostic reports through Streamlit.
- **Single-flow prediction** — Submit a network-flow feature record for a prediction.
- **CSV prediction support** — Upload a CSV containing the expected model features for batch prediction.
- **Dataset auditing** — Inspect real-data quality issues, label mappings, schema differences, and potential leakage concerns.
- **Automated tests** — Test data validation, preprocessing, model workflows, threshold selection, and dashboard functionality.
- **Reproducibility safeguards** — Track model metadata, feature schemas, split information, and experiment reports.

## 🎯 Project Goals

FPGuard-AI focuses on four key questions:

1. Can machine learning distinguish benign traffic from potentially malicious traffic?
2. How does changing the classification threshold affect false positives and recall?
3. How do different baseline classifiers compare under a consistent evaluation process?
4. How can data-quality problems and evaluation risks be identified before drawing conclusions about a model?

The objective is not simply to maximize accuracy. In intrusion detection, false alarms can overwhelm analysts, while missed attacks can create security risks. Both need to be evaluated.

## 🧠 How It Works

```text
Network Flow Dataset
        |
        v
Data Validation & Quality Auditing
        |
        v
Preprocessing Pipeline
        |
        v
Model Training
        |
        v
Validation-Based Model & Threshold Selection
        |
        v
Held-Out Test Evaluation
        |
        v
Prediction & Diagnostic Reports
        |
        v
Streamlit Dashboard
```

The intended workflow separates training, validation, and test data. Validation data is used for model and threshold selection, while the held-out test set is reserved for final evaluation.

## 🧪 Machine Learning Approach

The project includes three baseline classifiers:

| Model | Purpose |
|---|---|
| Random Forest | Nonlinear classification using an ensemble of decision trees |
| Logistic Regression | Linear baseline for comparison |
| Extra Trees | Tree-ensemble baseline with randomized split selection |

The pipeline includes preprocessing and model artifacts so predictions can use the expected feature schema and feature order.

### Threshold Selection

A classification threshold determines when a predicted malicious-traffic probability is treated as positive.

FPGuard-AI includes a validation-based threshold experiment that:

- Evaluates candidate thresholds from `0.00` to `1.00` in increments of `0.01`.
- Applies a configurable minimum recall requirement.
- Selects the feasible threshold with the lowest validation false-positive rate.
- Uses documented tie-breaking rules.
- Reports when the target cannot be achieved instead of silently weakening the requirement.

Threshold selection is an experimental procedure, not a guarantee of low false-positive rates on unseen traffic.

## 📊 Interactive Dashboard

The Streamlit application provides an interface for exploring the available experiment results and making predictions.

Dashboard sections include:

- Overview and metrics
- Single-flow prediction
- CSV-based prediction
- Threshold analysis
- Error analysis
- Model comparison

The displayed metrics depend on the datasets and saved reports available to the application. Synthetic-baseline results and real-dataset results should be clearly distinguished.

## 🗂️ Project Structure

The repository contains the following major components:

```text
FPGuard-AI/
├── app/
│   └── streamlit_app.py
├── data/
│   └── processed/
│       └── cicids2018/
├── src/
│   └── fpguard/
│       ├── data.py
│       ├── preprocessing.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── dashboard_data.py
├── tests/
├── README.md
├── pyproject.toml
└── .gitignore
```

*Note: This is a high-level overview. The exact directories, test filenames, configuration files, and generated artifacts may differ in your current checkout.*

## ⚙️ Getting Started

### Prerequisites

- Python version compatible with the project's `pyproject.toml`
- Git
- pip
- Windows, Linux, or another supported Python environment

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/FPGuard-AI.git
cd FPGuard-AI
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

If the repository uses the provided `pyproject.toml`:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Install any additional development or dashboard dependencies documented by the project if they are not included in the base installation.

### 4. Launch the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit, typically:

`http://localhost:8501`

### 5. Run the tests

If pytest is installed and the repository's tests are available:

```bash
python -m pytest -q
```

## 🧾 Dataset and Evaluation

The project includes a synthetic baseline dataset and tooling to audit CIC-IDS2018-derived data.

Dataset auditing can help identify:

- Missing and infinite values
- Constant or duplicate features
- Duplicate records and conflicting labels
- Label-mapping issues
- Potential feature leakage
- Feature-schema mismatches
- Dataset provenance and integrity concerns

**Important:** Dataset auditing does not establish model effectiveness. Reliable claims about real-world intrusion detection require documented data preparation, appropriate train/validation/test separation, leakage checks, reproducible evaluation, and analysis of false-positive and false-negative behavior.

Do not assume that the synthetic baseline's metrics represent performance on real network traffic.

## 🔐 Security and Responsible Use

FPGuard-AI is intended for defensive cybersecurity research, education, and experimentation.

- Use network data you are authorized to process.
- Do not upload confidential traffic captures, credentials, or private network information to a public repository.
- Keep API keys, tokens, and secrets out of source code and Git history.
- Review uploaded CSV files before processing them.
- Treat model predictions as indicators for investigation, not definitive proof of malicious activity.
- Do not deploy the prototype as a production security control without additional validation, monitoring, and security review.

## 🚧 Current Limitations and Future Work

Potential next steps include:

- Complete controlled evaluation on the audited CIC-IDS2018 data.
- Investigate dataset imbalance, duplicates, conflicting labels, and leakage risks.
- Measure false-positive and false-negative rates on an untouched test set.
- Evaluate performance across attack categories and benign traffic.
- Assess generalization to traffic collected under different conditions.
- Improve robustness, explainability, and operational monitoring.
- Benchmark inference latency and resource consumption.
- Package the application for reproducible deployment.

These are development goals, not claims that every item has already been completed.

## 🛠️ Technology Stack

- **Language:** Python
- **Machine learning:** scikit-learn
- **Data processing:** pandas, NumPy
- **Dashboard:** Streamlit
- **Testing:** pytest
- **Version control:** Git and GitHub

See the project's dependency configuration for the exact installed packages and versions.

## 👨‍💻 Author

Developed as an independent cybersecurity and machine-learning project to explore network intrusion detection, model evaluation, and false-positive analysis.

## 📄 License

No license is specified here. Before adding one, decide whether you want others to be able to reuse, modify, and redistribute your code. If you choose an open-source license, add the appropriate `LICENSE` file and update this section.

---

**FPGuard-AI — Building and evaluating more useful intrusion-detection models, one experiment at a time.**

 
 
