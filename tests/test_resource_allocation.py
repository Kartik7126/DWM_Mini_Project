"""
Unit tests for hospital resource allocation, clinical risk stratification, and care pathways.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessing import CardiacPreprocessor
from src.clustering import CardiacClusteringModel
from src.resource_allocation import (
    analyze_cluster_risk_factors,
    compute_hospital_resource_allocation,
    get_hospital_care_pathway,
    get_patient_hospital_recommendation,
    ACUITY_TIERS
)


@pytest.fixture
def clustered_data():
    df, _ = load_dataset()
    preprocessor = CardiacPreprocessor()
    X = preprocessor.fit_transform(df)
    model = CardiacClusteringModel(random_state=42)
    model.fit(X, k=3)
    df_clustered = df.copy()
    df_clustered["cluster"] = model.cluster_labels
    return df_clustered


def test_analyze_cluster_risk_factors(clustered_data):
    for c_id in [0, 1, 2]:
        risk = analyze_cluster_risk_factors(clustered_data, c_id)
        assert risk["cluster_id"] == c_id
        assert risk["patient_count"] > 0
        assert risk["acuity_tier_key"] in ACUITY_TIERS
        assert "risk_indicators" in risk
        assert "ischemic_burden_pct" in risk["risk_indicators"]
        assert "htn_stage2_pct" in risk["risk_indicators"]
        assert "vessels_1plus_pct" in risk["risk_indicators"]
        assert "perfusion_defect_pct" in risk["risk_indicators"]


def test_compute_hospital_resource_allocation(clustered_data):
    res = compute_hospital_resource_allocation(clustered_data)
    assert res["total_patients"] == len(clustered_data)
    tot = res["total_resources"]
    assert tot["ccu_beds"] >= 0
    assert tot["telemetry_beds"] >= 0
    assert tot["ward_beds"] >= 0
    assert tot["total_inpatient_beds"] > 0
    assert tot["cath_lab_slots"] > 0
    assert tot["total_nurse_shifts_per_day"] > 0

    # Test scaling simulation
    sim_res = compute_hospital_resource_allocation(clustered_data, simulated_intake_volume=500)
    assert sim_res["total_patients"] == 500
    assert sim_res["is_simulated"] is True
    assert sim_res["total_resources"]["total_inpatient_beds"] > tot["total_inpatient_beds"]


def test_get_hospital_care_pathways():
    for tier in ["TIER_1", "TIER_2", "TIER_3"]:
        pathway = get_hospital_care_pathway(tier)
        assert len(pathway["immediate_actions"]) > 0
        assert len(pathway["diagnostic_interventions"]) > 0
        assert len(pathway["pharmacological_strategy"]) > 0
        assert len(pathway["hospital_discharge_followup"]) > 0


def test_patient_hospital_recommendation(clustered_data):
    patient_record = clustered_data.iloc[0].to_dict()
    assigned_c = int(patient_record["cluster"])
    risk = analyze_cluster_risk_factors(clustered_data, assigned_c)

    rec = get_patient_hospital_recommendation(patient_record, assigned_c, risk)
    assert "recommended_bed" in rec
    assert "triage_urgency" in rec
    assert "immediate_actions" in rec
    assert len(rec["immediate_actions"]) > 0
