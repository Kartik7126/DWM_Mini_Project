"""
Configuration, metadata, and styling constants for CardioInsight.
"""

from typing import Dict, List, Any

# Dataset Attribution
DATASET_METADATA = {
    "name": "UCI Heart Disease Dataset (Cleveland Clinic Foundation)",
    "creators": "Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation)",
    "year": "1988",
    "source_url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
    "citation": "Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64(5), 304-310.",
    "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    "description": (
        "This benchmark cardiac dataset comprises 303 patient records collected at the Cleveland Clinic. "
        "It contains 13 clinical and demographic attributes (such as age, resting blood pressure, "
        "cholesterol, and ECG findings) alongside angiographic disease status."
    )
}

# Supported Features and Clinical Definitions
FEATURE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "age": {
        "display_name": "Age",
        "unit": "years",
        "type": "numerical",
        "description": "Patient age in completed years.",
        "min": 18.0,
        "max": 100.0,
        "default": 55.0,
        "step": 1.0,
    },
    "sex": {
        "display_name": "Biological Sex",
        "unit": "category",
        "type": "categorical",
        "description": "Biological sex of the patient.",
        "options": {1.0: "Male (1)", 0.0: "Female (0)"},
        "default": 1.0,
    },
    "cp": {
        "display_name": "Chest Pain Type",
        "unit": "category",
        "type": "categorical",
        "description": "Type of chest pain experienced by the patient.",
        "options": {
            1.0: "Typical Angina (1)",
            2.0: "Atypical Angina (2)",
            3.0: "Non-anginal Pain (3)",
            4.0: "Asymptomatic (4)"
        },
        "default": 4.0,
    },
    "trestbps": {
        "display_name": "Resting Blood Pressure",
        "unit": "mm Hg",
        "type": "numerical",
        "description": "Resting systolic blood pressure upon admission to hospital.",
        "min": 80.0,
        "max": 220.0,
        "default": 130.0,
        "step": 1.0,
    },
    "chol": {
        "display_name": "Serum Cholesterol",
        "unit": "mg/dl",
        "type": "numerical",
        "description": "Serum total cholesterol level.",
        "min": 100.0,
        "max": 600.0,
        "default": 240.0,
        "step": 1.0,
    },
    "fbs": {
        "display_name": "Fasting Blood Sugar > 120 mg/dl",
        "unit": "binary",
        "type": "categorical",
        "description": "Fasting blood sugar greater than 120 mg/dl (indicators of elevated glycaemia).",
        "options": {1.0: "True (> 120 mg/dl) (1)", 0.0: "False (<= 120 mg/dl) (0)"},
        "default": 0.0,
    },
    "restecg": {
        "display_name": "Resting ECG Results",
        "unit": "category",
        "type": "categorical",
        "description": "Resting electrocardiographic results.",
        "options": {
            0.0: "Normal (0)",
            1.0: "ST-T Wave Abnormality (1)",
            2.0: "Left Ventricular Hypertrophy (2)"
        },
        "default": 0.0,
    },
    "thalach": {
        "display_name": "Maximum Heart Rate Achieved",
        "unit": "bpm",
        "type": "numerical",
        "description": "Highest heart rate achieved during exercise stress testing.",
        "min": 60.0,
        "max": 220.0,
        "default": 150.0,
        "step": 1.0,
    },
    "exang": {
        "display_name": "Exercise-Induced Angina",
        "unit": "binary",
        "type": "categorical",
        "description": "Presence of angina chest discomfort during exercise.",
        "options": {1.0: "Yes (1)", 0.0: "No (0)"},
        "default": 0.0,
    },
    "oldpeak": {
        "display_name": "ST Depression (Oldpeak)",
        "unit": "mm",
        "type": "numerical",
        "description": "ST depression induced by exercise relative to resting baseline.",
        "min": 0.0,
        "max": 7.0,
        "default": 1.0,
        "step": 0.1,
    },
    "slope": {
        "display_name": "Slope of Peak Exercise ST",
        "unit": "category",
        "type": "categorical",
        "description": "The slope of the peak exercise ST segment.",
        "options": {
            1.0: "Upsloping (1)",
            2.0: "Flat (2)",
            3.0: "Downsloping (3)"
        },
        "default": 1.0,
    },
    "ca": {
        "display_name": "Major Vessels Colored by Fluoroscopy",
        "unit": "vessels (0-3)",
        "type": "numerical",
        "description": "Number of major coronary vessels (0-3) visualized by fluoroscopy.",
        "min": 0.0,
        "max": 3.0,
        "default": 0.0,
        "step": 1.0,
    },
    "thal": {
        "display_name": "Thallium Stress Test Result",
        "unit": "category",
        "type": "categorical",
        "description": "Thallium scintigraphy myocardial perfusion result.",
        "options": {
            3.0: "Normal Perfusion (3)",
            6.0: "Fixed Defect (6)",
            7.0: "Reversible Defect (7)"
        },
        "default": 3.0,
    }
}

# The Target column is strictly EXCLUDED from clustering
EXCLUDED_COLUMNS = ["target", "num", "id", "patient_id"]

NUMERICAL_FEATURES: List[str] = [
    col for col, meta in FEATURE_DEFINITIONS.items() if meta["type"] == "numerical"
]

CATEGORICAL_FEATURES: List[str] = [
    col for col, meta in FEATURE_DEFINITIONS.items() if meta["type"] == "categorical"
]

ALL_CLUSTERING_FEATURES: List[str] = list(FEATURE_DEFINITIONS.keys())

# Default Model Hyperparameters
DEFAULT_RANDOM_STATE = 42
DEFAULT_K_MIN = 2
DEFAULT_K_MAX = 8
DEFAULT_K = 3
N_INIT = 10

# UI Theme Color Palette (Clinical Navy & Teal)
THEME_COLORS = {
    "primary": "#0F766E",       # Deep Teal
    "primary_dark": "#0F2942",  # Clinical Navy
    "secondary": "#0284C7",    # Medical Blue
    "accent": "#0D9488",       # Teal accent
    "background": "#F8FAFC",   # Slate-50 light background
    "card_bg": "#FFFFFF",      # Crisp White
    "border": "#E2E8F0",       # Slate-200
    "text": "#1E293B",         # Slate-800
    "text_muted": "#64748B",   # Slate-500
    "warning": "#D97706",      # Amber-600
    "danger": "#E11D48",       # Rose-600
    "success": "#059669",      # Emerald-600
    "cluster_palette": [
        "#0284C7",  # Sky Blue (Cluster 0)
        "#0D9488",  # Teal (Cluster 1)
        "#F59E0B",  # Amber (Cluster 2)
        "#8B5CF6",  # Violet (Cluster 3)
        "#EC4899",  # Pink (Cluster 4)
        "#10B981",  # Emerald (Cluster 5)
        "#6366F1",  # Indigo (Cluster 6)
        "#F97316"   # Orange (Cluster 7)
    ]
}

# Medical Disclaimer Text
MEDICAL_DISCLAIMER = (
    "**Academic Analytical Prototype Notice:** CardioInsight is an educational Data Warehousing & Mining (DWM) "
    "prototype developed for unsupervised patient profiling and hospital analytics using K-Means clustering. "
    "It is **not** a certified medical diagnostic device or clinical decision support system. "
    "Cluster memberships indicate similarity groupings within the analyzed cohort and do not establish medical diagnoses, "
    "prognoses, or treatment recommendations. All clinical decisions must be made by qualified healthcare professionals."
)
