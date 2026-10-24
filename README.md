# Cardiac Phenotype Profiling & Hospital Resource Platform

An end-to-end clinical analytics and unsupervised machine learning platform designed for cardiology departments and hospital administrators. It translates multidimensional cardiovascular patient profiles into actionable clinical phenotypes, automated hospital resource allocations (CCU, Telemetry, and Ward beds), diagnostic facility scheduling, and standardized care pathways.

---

## 🏥 Clinical Decision Support Advisory

> **Notice:** This system is an analytical decision support tool designed for clinical cohort segmentation, risk factor assessment, and hospital operational resource planning. Cluster memberships and risk profiles represent empirical cohort patterns and must not be used as a standalone substitute for individualized clinical judgment. All patient diagnostic and treatment decisions must be made by licensed healthcare professionals.

---

## 🏛️ Clinical & Analytical Capabilities

| Module | Operational & Clinical Impact |
| :--- | :--- |
| **Cohort Phenotyping** | Unsupervised segmentation of multidimensional cardiovascular vitals into distinct clinical clusters using standardized Euclidean feature spaces. |
| **Risk Factor Deconvolution** | Quantitative stratification of ischemic stress markers (ST depression, exercise-induced angina), Stage 2 systolic hypertension, multi-vessel coronary calcification, perfusion deficits, and chronotropic competence. |
| **Hospital Capacity & Bed Allocation** | Dynamic projection of Coronary Care Unit (CCU/ICU) beds, Step-Down Telemetry beds, and General Ward beds based on active cluster acuity. |
| **Diagnostic Scheduling** | Predictive estimation of procedural facility demand: Cardiac Catheterization Lab (Cath Lab) angiography, Echocardiography (TTE), Nuclear SPECT/CT scans, and 24-hr Holter ECG monitors. |
| **Staffing Shift Projections** | Calculation of specialized critical care nursing ratios (1:1/1:2 for CCU, 1:3 for Telemetry, 1:5 for Ward) and attending cardiologist coverage hours. |
| **Actionable Care Pathways** | Standardized, evidence-based hospital clinical protocols detailing emergency triage, diagnostic angiography urgency, guideline-directed pharmacotherapy (GDMT), and post-discharge Phase II Cardiac Rehabilitation enrollment. |
| **Admissions Sizing Simulator** | Interactive capacity slider enabling hospital directors and nursing supervisors to simulate variable monthly inpatient intake volumes. |
| **Individual Patient Triage & Intake** | Immediate similarity profiling of admitted patients, flagging physiological out-of-distribution values and generating patient-specific bed and diagnostic recommendations. |

---

## 📁 System Architecture

```
DWM_Mini_Project/
├── app.py                      # Main Streamlit web application & 6-view clinical navigation
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & clinical workflows
├── data/
│   └── heart_disease_uci.csv   # Cardiovascular benchmark cohort (303 patient records)
├── src/
│   ├── __init__.py
│   ├── config.py               # Feature schemas, units, metadata, clinical definitions & styling
│   ├── data_loader.py          # Data ingestion, schema validation, synthetic demo generator
│   ├── preprocessing.py        # Imputation, categorical encoding, StandardScaler, OOD checks
│   ├── clustering.py           # K-Means, Elbow method, Silhouette scoring, PCA, prediction
│   ├── cluster_analysis.py     # Cluster profiling, ANOVA feature differentiation, comparisons
│   ├── resource_allocation.py  # Hospital bed modeling, staffing, risk factors & care pathways
│   └── visualizations.py       # Clinical Plotly interactive visualizations
└── tests/
    ├── __init__.py
    ├── test_data_loader.py     # Ingestion & validation test suite
    ├── test_preprocessing.py    # Leakage prevention & scaling tests
    ├── test_clustering.py       # K-Means & prediction tests
    ├── test_cluster_analysis.py # Phenotypic profiling & comparison tests
    └── test_resource_allocation.py # Bed allocation, risk factors & pathways tests
```

---

## 📊 Dataset Schema & Clinical Variables

- **Cohort Size:** 303 patient records, 13 clinical attributes + 1 angiographic target.
- **Integrity Guarantee:** The diagnosis column (`target`) is strictly **excluded** from all clustering features to ensure true unsupervised discovery. Attributes not present in the benchmark dataset (such as BMI, smoking history, or longitudinal visits) are not fabricated.

### Clinical Parameters
1. `age`: Age in years (29 – 77)
2. `sex`: Biological sex (1 = Male, 0 = Female)
3. `cp`: Chest pain presentation (1: Typical angina, 2: Atypical angina, 3: Non-anginal, 4: Asymptomatic)
4. `trestbps`: Resting systolic blood pressure upon admission (94 – 200 mm Hg)
5. `chol`: Total serum cholesterol (126 – 564 mg/dl)
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)
7. `restecg`: Resting ECG (0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy)
8. `thalach`: Maximum heart rate achieved during exercise (71 – 202 bpm)
9. `exang`: Exercise-induced angina (1 = Yes, 0 = No)
10. `oldpeak`: ST depression induced by exercise relative to rest (0.0 – 6.2 mm)
11. `slope`: Peak exercise ST segment slope (1: Upsloping, 2: Flat, 3: Downsloping)
12. `ca`: Number of major coronary vessels colored by fluoroscopy (0 – 3)
13. `thal`: Thallium myocardial scintigraphy (3: Normal, 6: Fixed defect, 7: Reversible defect)

---

## 🖥️ Clinical Views & Workflows

### 1. Clinical Cohort Overview
- Executive summary metrics: Active census, discovered phenotypes, total beds required, cath lab demand, silhouette cohesion.
- Automated clinical summary highlighting dominant clusters and primary physiological differentiators.
- Visual distribution of patient volumes, age distributions, and 2D Principal Component projections.

### 2. Hospital Resource Allocation & Care Pathways
- Inpatient bed allocation across units: CCU/ICU, Monitored Telemetry Step-Down, and General Cardiology Ward.
- Cath Lab angiography slots, Echocardiograms, Nuclear SPECT/CT scans, and nurse shifts per day.
- **Interactive Admissions Simulator:** Adjust projected monthly patient inflow to instantly recalculate bed requirements and facility load.
- Detailed **"How Hospitals Will Help"** Care Pathway cards per phenotype cluster covering emergency triage, diagnostic pipelines, guideline-directed pharmacotherapy, and post-acute cardiac rehab.

### 3. Patient Registry & Triage Explorer
- Searchable patient database with anonymized identifiers (`REC-001`, `REC-002`, ...).
- Filter by cluster, clinical acuity tier, resting SBP threshold (e.g. Stage 2 HTN ≥140 mm Hg), biological sex, and age.
- Deep-dive clinical card with vitals, ECG stress parameters, and recommended hospital unit.
- One-click CSV export of filtered records or full cohort.

### 4. Patient Clinical Profile & Intake
- **Mode A (Existing Patient):** Inspect any cohort record, view distance to centroid, relative similarity to all clusters, radar chart, and delta bar charts showing percentage deviations from cluster averages.
- **Mode B (New Patient Intake):** Input physiological and stress test metrics to receive instant phenotype matching, out-of-distribution alerts, and immediate bed and care pathway recommendations.

### 5. Cluster Phenotypes & Risk Factors
- Multi-axis risk radar chart: Ischemic stress burden, Stage 2 hypertension, multi-vessel CAD, perfusion defect rate, and metabolic risk.
- Standardized Z-score deviation heatmap.
- Statistical summary cards and side-by-side comparative analytics between any two selected clusters.

### 6. System Diagnostics & Data Management
- Elbow Curve (Inertia vs. K) and Silhouette Score Curve.
- Candidate K diagnostic table and PCA explained variance ratios.
- Custom CSV upload with automated schema and column compatibility validation.
- Dataset reset and synthetic cohort generation options.

---

## 🚀 Running the Platform Locally

### 1. Requirements
- Python 3.10+ (tested on Python 3.12)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Run the automated test suite with `pytest`:

```bash
python -m pytest tests/ -v
```

All 14 unit tests verify data loading, preprocessing, K-Means clustering, hospital resource allocation, risk stratification, and patient triage.
