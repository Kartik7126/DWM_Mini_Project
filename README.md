# CardioInsight — Cardiac Patient Profiling & Hospital Analytics Using K-Means Clustering

**CardioInsight** is an end-to-end clinical analytics and unsupervised machine learning web application developed as a **Data Warehousing and Mining (DWM)** academic mini-project. It employs K-Means clustering, standardized multivariate data pipelines, and interactive visual analytics to discover clinically meaningful patient phenotypes without diagnostic supervision.

---

## 🫀 Clinical & Academic Disclaimer

> **Important Notice:** CardioInsight is an educational research and analytical prototype. It is **not** a certified medical diagnostic device or clinical decision support system. Cluster memberships reflect statistical similarity within the analyzed cohort and do **not** establish a medical diagnosis, predict heart attacks, or prescribe clinical interventions. All patient care decisions must be made by qualified healthcare professionals.

---

## 🏛️ DWM Concepts Demonstrated

| Concept | Implementation in CardioInsight |
| :--- | :--- |
| **Data Warehousing & ETL** | Ingestion of raw cardiovascular observations, column schema validation, automated missing-value auditing, median/mode imputation, and anonymous record indexing (`REC-001`). |
| **Data Preprocessing & Scaling** | Strict separation of diagnostic labels (`target` excluded), one-hot encoding for multiclass features, and Z-score standardization via `StandardScaler` to prevent feature dominance in Euclidean space. |
| **Model Reproducibility & Leakage Prevention** | Stateful transformer pipelines (`CardiacPreprocessor`) fit strictly on cohort data and applied consistently to hypothetical patients. |
| **Unsupervised Clustering** | K-Means clustering with multiple initializations (`n_init=10`) and fixed random seeds. |
| **Cluster Evaluation & Validation** | Multi-criteria evaluation combining the **Elbow Method** (Sum of Squared Errors / Inertia) and **Silhouette Analysis** (cohesion vs. separation) across candidate values of $K \in [2, 8]$. |
| **Dimensionality Reduction** | Principal Component Analysis (PCA) projecting high-dimensional clinical feature vectors into 2D orthogonal components for spatial validation and centroid tracking. |
| **Cluster Profiling & Phenotyping** | Generation of neutral, feature-based clinical labels, population Z-score deviation heatmaps, and distinguishing multivariate characteristics. |
| **Out-of-Distribution (OOD) Guardrails** | Anomaly detection flagging patient entries exceeding cohort bounds or exhibiting extreme $|z| > 3.0$ deviations. |

---

## 📁 Project Architecture

```
DWM_Mini_Project/
├── app.py                      # Main Streamlit web application & navigation
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & academic summary
├── data/
│   └── heart_disease_uci.csv   # Authentic UCI Cleveland cardiac cohort (303 records)
├── src/
│   ├── __init__.py
│   ├── config.py               # Feature schemas, units, metadata, clinical definitions & styling
│   ├── data_loader.py          # Data ingestion, schema validation, synthetic generator
│   ├── preprocessing.py        # Imputation, categorical encoding, StandardScaler, OOD checks
│   ├── clustering.py           # K-Means, Elbow method, Silhouette scoring, PCA, prediction
│   ├── cluster_analysis.py     # Cluster profiling, ANOVA feature differentiation, comparisons
│   └── visualizations.py       # Clinical Plotly interactive visualizations
└── tests/
    ├── __init__.py
    ├── test_data_loader.py     # Ingestion, validation, and demo data tests
    ├── test_preprocessing.py    # Leakage prevention, imputation, and OOD tests
    ├── test_clustering.py       # K-Means evaluation, fitting, and prediction tests
    └── test_cluster_analysis.py # Profiling and comparative analytics tests
```

---

## 📊 Dataset Attribution & Schema

- **Source:** [UCI Machine Learning Repository: Heart Disease Dataset (Cleveland Foundation)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Principal Investigators:** Robert Detrano, M.D., Ph.D., et al. (1988)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Cohort Size:** 303 patient records, 13 clinical attributes + 1 angiographic target.

### Supported Features
1. `age`: Age in years (29 – 77)
2. `sex`: Biological sex (1 = Male, 0 = Female)
3. `cp`: Chest pain type (1: Typical angina, 2: Atypical angina, 3: Non-anginal, 4: Asymptomatic)
4. `trestbps`: Resting blood pressure upon admission (94 – 200 mm Hg)
5. `chol`: Total serum cholesterol (126 – 564 mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
7. `restecg`: Resting ECG (0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy)
8. `thalach`: Maximum heart rate achieved during exercise (71 – 202 bpm)
9. `exang`: Exercise-induced angina (1 = Yes, 0 = No)
10. `oldpeak`: ST depression induced by exercise relative to rest (0.0 – 6.2 mm)
11. `slope`: Peak exercise ST segment slope (1: Upsloping, 2: Flat, 3: Downsloping)
12. `ca`: Number of major coronary vessels colored by fluoroscopy (0 – 3)
13. `thal`: Thallium myocardial scintigraphy (3: Normal, 6: Fixed defect, 7: Reversible defect)

> **Integrity Note:** The angiographic status column (`target`) is strictly **excluded** from all clustering features to ensure true unsupervised discovery. Attributes not present in the public dataset (such as BMI, smoking history, or longitudinal visits) are not fabricated.

---

## 🖥️ Application Navigation & Features

### 1. Overview Dashboard
- Executive KPI summary cards: Total patients, Discovered clusters, Selected K, Cohort Silhouette Score, Data completeness.
- Automated descriptive cohort summary identifying dominant clusters and distinguishing clinical features.
- Patient distribution bar chart and proportion donut chart.
- Age distribution box plots across clusters.
- Clinical feature cross-cluster comparison.
- Interactive 2D PCA projection with marked cluster centroids.

### 2. Patient Explorer
- Searchable and filterable patient database with anonymous identifiers (`REC-001`).
- Filter by cluster membership, biological sex, and age range.
- Interactive deep-dive card for inspecting individual clinical metrics.
- One-click CSV export of filtered records or the complete clustered cohort.

### 3. Patient Profile Analysis
- **Mode A (Existing Patient):** Select any patient record to display their assigned cluster, distance to centroid, relative similarity to all discovered clusters, multivariate radar chart, and feature-by-feature percentage deviations from the cluster average.
- **Mode B (Hypothetical Patient):** Interactive form accepting physiological and stress test metrics. Features automated Out-of-Distribution (OOD) checks, pipeline transformation, nearest centroid assignment, and plain-language similarity insights.

### 4. Cluster Insights
- Neutral, feature-based phenotypic labels (e.g., *"Cluster 0: Patient Cohort with comparatively lower peak exercise heart rate and higher resting blood pressure"*).
- Population-relative Z-score heatmap displaying normalized multivariate elevations across clusters.
- Statistical summary cards: mean, median, standard deviation, and categorical frequency breakdowns.
- Side-by-side comparative analytics between any two selected clusters.

### 5. Model Evaluation
- Interactive **Elbow Curve** plotting Inertia (SSE) vs. K with active selection markers.
- **Silhouette Analysis** comparing cluster separation scores across candidate values ($K = 2 \dots 8$).
- Automated recommendation rationale balancing silhouette scores, SSE drops, and cluster size stability.
- Diagnostic evaluation table and PCA explained variance ratios.

### 6. Dataset Management
- Data provenance, license citation, and dataset dimensions.
- Missing-value audit and completeness ratios.
- Custom CSV upload workflow with automated schema and feature compatibility validation.
- Quick switch to synthetic demo cohort or reset to default Cleveland dataset.
- Export clustered dataset with cluster labels.

---

## 🚀 Installation & Local Execution

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git (optional)

### 2. Setup Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd CardioInsight

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

A comprehensive unit test suite built with `pytest` verifies data ingestion, schema validation, preprocessor leakage prevention, K-Means clustering, out-of-distribution detection, and patient profiling:

```bash
python -m pytest tests/ -v
```

All 10 tests should pass cleanly.
#   D W M _ M i n i _ P r o j e c t  
 