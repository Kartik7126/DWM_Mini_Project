"""
Hospital Resource Allocation, Clinical Risk Stratification, and Care Pathways Module.
Provides dynamic hospital operational planning and clinical management guidelines based on cohort phenotypes.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


# Acuity Tier Constants
ACUITY_TIERS = {
    "TIER_1": {
        "name": "High Clinical Acuity (Intensive / Interventional)",
        "badge_color": "#E11D48",  # Rose-600
        "bg_color": "#FFE4E6",
        "description": "Marked exercise-induced ischemia, multi-vessel anatomical involvement, or hemodynamic volatility.",
        "bed_type": "Coronary Care Unit (CCU) / Monitored Cardiac ICU",
        "nurse_ratio": "1:1 or 1:2 (Specialized Critical Care Nurse)",
        "triage_urgency": "Immediate / Urgent (Workup within 24–48 hours)"
    },
    "TIER_2": {
        "name": "Moderate Acuity (Intermediate / Monitored Telemetry)",
        "badge_color": "#D97706",  # Amber-600
        "bg_color": "#FEF3C7",
        "description": "Uncontrolled systolic hypertension, borderline exercise ischemia, or isolated vascular disease.",
        "bed_type": "Step-Down Cardiac Telemetry Bed",
        "nurse_ratio": "1:3 or 1:4 (Telemetry Certified Nurse)",
        "triage_urgency": "Priority (Clinical stabilization & scheduled diagnostics)"
    },
    "TIER_3": {
        "name": "Standard Acuity (Ambulatory & Secondary Prevention)",
        "badge_color": "#0D9488",  # Teal-600
        "bg_color": "#CCFBF1",
        "description": "Preserved exercise tolerance, minimal ST changes, suitable for medical therapy and lifestyle optimization.",
        "bed_type": "General Cardiology Ward / Outpatient Clinic Review",
        "nurse_ratio": "1:5 (Cardiology Nurse) or Outpatient Clinic Staff",
        "triage_urgency": "Elective / Routine Outpatient Management"
    }
}


def analyze_cluster_risk_factors(
    df_clustered: pd.DataFrame, cluster_id: int
) -> Dict[str, Any]:
    """
    Compute objective cardiovascular risk factor prevalence within a cluster.
    """
    sub_df = df_clustered[df_clustered["cluster"] == cluster_id]
    n = len(sub_df)
    if n == 0:
        return {}

    # 1. Ischemic Stress Risk
    # oldpeak >= 1.5 mm indicates clinically significant exercise ST depression
    st_depression_count = int((sub_df["oldpeak"] >= 1.5).sum()) if "oldpeak" in sub_df.columns else 0
    angina_count = int((sub_df["exang"] == 1.0).sum()) if "exang" in sub_df.columns else 0
    downsloping_count = int((sub_df["slope"] == 3.0).sum()) if "slope" in sub_df.columns else 0
    ischemic_burden_pct = round(((st_depression_count + angina_count) / (2 * n)) * 100.0, 1)

    # 2. Hemodynamic / Hypertensive Risk
    # Stage 2 HTN: SBP >= 140 mm Hg; Severe: SBP >= 160 mm Hg
    htn_stage2_count = int((sub_df["trestbps"] >= 140.0).sum()) if "trestbps" in sub_df.columns else 0
    htn_severe_count = int((sub_df["trestbps"] >= 160.0).sum()) if "trestbps" in sub_df.columns else 0
    htn_stage2_pct = round((htn_stage2_count / n) * 100.0, 1)
    htn_severe_pct = round((htn_severe_count / n) * 100.0, 1)

    # 3. Coronary Vessel Involvement (Anatomical Burden)
    vessels_1plus = int((sub_df["ca"] >= 1.0).sum()) if "ca" in sub_df.columns else 0
    vessels_multivessel = int((sub_df["ca"] >= 2.0).sum()) if "ca" in sub_df.columns else 0
    vessels_pct = round((vessels_1plus / n) * 100.0, 1)
    multivessel_pct = round((vessels_multivessel / n) * 100.0, 1)

    # 4. Perfusion Defect Burden (Thallium)
    # 7.0 = reversible defect (ischemia), 6.0 = fixed defect (prior infarction)
    rev_defect_count = int((sub_df["thal"] == 7.0).sum()) if "thal" in sub_df.columns else 0
    fixed_defect_count = int((sub_df["thal"] == 6.0).sum()) if "thal" in sub_df.columns else 0
    perfusion_defect_pct = round(((rev_defect_count + fixed_defect_count) / n) * 100.0, 1)

    # 5. Metabolic & Glycemic Risk
    fbs_elevated_count = int((sub_df["fbs"] == 1.0).sum()) if "fbs" in sub_df.columns else 0
    chol_high_count = int((sub_df["chol"] >= 240.0).sum()) if "chol" in sub_df.columns else 0
    metabolic_pct = round(((fbs_elevated_count + chol_high_count) / (2 * n)) * 100.0, 1)

    # 6. Chronotropic Incompetence / Impaired Exercise Capacity
    # Peak HR < 130 bpm or < 80% age predicted (220 - age)
    if "thalach" in sub_df.columns and "age" in sub_df.columns:
        expected_hr = 220.0 - sub_df["age"]
        chronotropic_incomp = int((sub_df["thalach"] < 0.80 * expected_hr).sum())
        chronotropic_pct = round((chronotropic_incomp / n) * 100.0, 1)
    elif "thalach" in sub_df.columns:
        chronotropic_incomp = int((sub_df["thalach"] < 130.0).sum())
        chronotropic_pct = round((chronotropic_incomp / n) * 100.0, 1)
    else:
        chronotropic_pct = 0.0

    # Composite Acuity Scoring
    # Composite score based on normalized clinical indicators (0 to 100)
    acuity_score = (
        (ischemic_burden_pct * 0.30) +
        (vessels_pct * 0.25) +
        (perfusion_defect_pct * 0.20) +
        (htn_stage2_pct * 0.15) +
        (metabolic_pct * 0.10)
    )

    if acuity_score >= 45.0 or (st_depression_count / n >= 0.40 and vessels_pct >= 40.0):
        tier_key = "TIER_1"
    elif acuity_score >= 25.0 or htn_stage2_pct >= 40.0 or vessels_pct >= 25.0:
        tier_key = "TIER_2"
    else:
        tier_key = "TIER_3"

    tier_info = ACUITY_TIERS[tier_key]

    return {
        "cluster_id": cluster_id,
        "patient_count": n,
        "acuity_tier_key": tier_key,
        "acuity_tier_name": tier_info["name"],
        "acuity_score": round(acuity_score, 1),
        "badge_color": tier_info["badge_color"],
        "bg_color": tier_info["bg_color"],
        "bed_type": tier_info["bed_type"],
        "nurse_ratio": tier_info["nurse_ratio"],
        "triage_urgency": tier_info["triage_urgency"],
        "risk_indicators": {
            "ischemic_burden_pct": ischemic_burden_pct,
            "st_depression_count": st_depression_count,
            "st_depression_pct": round((st_depression_count / n) * 100.0, 1),
            "angina_count": angina_count,
            "angina_pct": round((angina_count / n) * 100.0, 1),
            "htn_stage2_pct": htn_stage2_pct,
            "htn_severe_pct": htn_severe_pct,
            "vessels_1plus_pct": vessels_pct,
            "multivessel_pct": multivessel_pct,
            "perfusion_defect_pct": perfusion_defect_pct,
            "reversible_defect_count": rev_defect_count,
            "fixed_defect_count": fixed_defect_count,
            "metabolic_pct": metabolic_pct,
            "fbs_elevated_pct": round((fbs_elevated_count / n) * 100.0, 1),
            "chol_high_pct": round((chol_high_count / n) * 100.0, 1),
            "chronotropic_pct": chronotropic_pct
        }
    }


def compute_hospital_resource_allocation(
    df_clustered: pd.DataFrame,
    simulated_intake_volume: Optional[int] = None
) -> Dict[str, Any]:
    """
    Calculate dynamic hospital capacity, bed allocation, staffing shifts,
    and procedural slot requirements for the entire cohort.
    """
    total_records = len(df_clustered)
    effective_n = simulated_intake_volume if simulated_intake_volume is not None else total_records
    scale_factor = effective_n / total_records if total_records > 0 else 1.0

    unique_clusters = sorted(df_clustered["cluster"].unique())
    cluster_allocations = {}

    total_ccu_beds = 0
    total_telemetry_beds = 0
    total_ward_beds = 0
    total_ambulatory_slots = 0

    total_cath_lab_slots = 0
    total_echo_slots = 0
    total_spect_ct_slots = 0
    total_holter_slots = 0

    total_nurse_shifts = 0.0
    total_cardiologist_hours = 0.0

    for c_id in unique_clusters:
        risk = analyze_cluster_risk_factors(df_clustered, c_id)
        c_count = risk["patient_count"]
        scaled_c_count = int(round(c_count * scale_factor))
        tier = risk["acuity_tier_key"]

        # Bed distribution based on clinical tier
        if tier == "TIER_1":
            # High acuity: 45% CCU beds, 40% Telemetry, 15% Ward
            ccu_beds = int(np.ceil(scaled_c_count * 0.45))
            telemetry_beds = int(np.ceil(scaled_c_count * 0.40))
            ward_beds = int(scaled_c_count - ccu_beds - telemetry_beds)
            ward_beds = max(0, ward_beds)
            ambulatory = int(np.ceil(scaled_c_count * 0.15))

            # Procedures: High cath lab need, high echo, high SPECT
            cath_slots = int(np.ceil(scaled_c_count * 0.65))
            echo_slots = int(np.ceil(scaled_c_count * 0.85))
            spect_slots = int(np.ceil(scaled_c_count * 0.50))
            holter_slots = int(np.ceil(scaled_c_count * 0.30))

            # Staffing
            nurse_shifts = round((ccu_beds * 1.5) + (telemetry_beds * 0.35) + (ward_beds * 0.20), 1)
            cardiologist_hours = round(scaled_c_count * 2.5, 1)

        elif tier == "TIER_2":
            # Moderate acuity: 10% CCU, 55% Telemetry, 35% Ward
            ccu_beds = int(np.ceil(scaled_c_count * 0.10))
            telemetry_beds = int(np.ceil(scaled_c_count * 0.55))
            ward_beds = int(scaled_c_count - ccu_beds - telemetry_beds)
            ward_beds = max(0, ward_beds)
            ambulatory = int(np.ceil(scaled_c_count * 0.35))

            cath_slots = int(np.ceil(scaled_c_count * 0.30))
            echo_slots = int(np.ceil(scaled_c_count * 0.70))
            spect_slots = int(np.ceil(scaled_c_count * 0.35))
            holter_slots = int(np.ceil(scaled_c_count * 0.45))

            nurse_shifts = round((ccu_beds * 1.5) + (telemetry_beds * 0.35) + (ward_beds * 0.20), 1)
            cardiologist_hours = round(scaled_c_count * 1.5, 1)

        else:
            # Standard / Ambulatory acuity: 0% CCU, 20% Telemetry, 80% Ward/Outpatient
            ccu_beds = 0
            telemetry_beds = int(np.ceil(scaled_c_count * 0.20))
            ward_beds = int(scaled_c_count - telemetry_beds)
            ambulatory = scaled_c_count

            cath_slots = int(np.ceil(scaled_c_count * 0.10))
            echo_slots = int(np.ceil(scaled_c_count * 0.50))
            spect_slots = int(np.ceil(scaled_c_count * 0.15))
            holter_slots = int(np.ceil(scaled_c_count * 0.30))

            nurse_shifts = round((telemetry_beds * 0.35) + (ward_beds * 0.20), 1)
            cardiologist_hours = round(scaled_c_count * 0.8, 1)

        total_ccu_beds += ccu_beds
        total_telemetry_beds += telemetry_beds
        total_ward_beds += ward_beds
        total_ambulatory_slots += ambulatory

        total_cath_lab_slots += cath_slots
        total_echo_slots += echo_slots
        total_spect_ct_slots += spect_slots
        total_holter_slots += holter_slots

        total_nurse_shifts += nurse_shifts
        total_cardiologist_hours += cardiologist_hours

        # Clinical Care Pathway & Action Plan
        pathway = get_hospital_care_pathway(tier)

        cluster_allocations[int(c_id)] = {
            "cluster_id": int(c_id),
            "patient_count": scaled_c_count,
            "raw_patient_count": c_count,
            "risk_profile": risk,
            "bed_allocation": {
                "ccu_beds": ccu_beds,
                "telemetry_beds": telemetry_beds,
                "ward_beds": ward_beds,
                "ambulatory_slots": ambulatory,
                "total_beds": ccu_beds + telemetry_beds + ward_beds
            },
            "diagnostic_slots": {
                "cath_lab_slots": cath_slots,
                "echo_slots": echo_slots,
                "spect_ct_slots": spect_slots,
                "holter_slots": holter_slots
            },
            "staffing_requirements": {
                "nurse_shifts_per_day": nurse_shifts,
                "cardiologist_hours": cardiologist_hours
            },
            "care_pathway": pathway
        }

    return {
        "total_patients": effective_n,
        "is_simulated": simulated_intake_volume is not None,
        "clusters": cluster_allocations,
        "total_resources": {
            "ccu_beds": total_ccu_beds,
            "telemetry_beds": total_telemetry_beds,
            "ward_beds": total_ward_beds,
            "total_inpatient_beds": total_ccu_beds + total_telemetry_beds + total_ward_beds,
            "ambulatory_slots": total_ambulatory_slots,
            "cath_lab_slots": total_cath_lab_slots,
            "echo_slots": total_echo_slots,
            "spect_ct_slots": total_spect_ct_slots,
            "holter_slots": total_holter_slots,
            "total_nurse_shifts_per_day": round(total_nurse_shifts, 1),
            "total_cardiologist_hours": round(total_cardiologist_hours, 1)
        }
    }


def get_hospital_care_pathway(acuity_tier_key: str) -> Dict[str, Any]:
    """
    Detailed hospital assistance protocols, triage pathways, and post-discharge plans.
    """
    if acuity_tier_key == "TIER_1":
        return {
            "tier_title": "Critical / Interventional Pathway",
            "triage_response": "Urgent Emergency / Inpatient CCU Triage",
            "immediate_actions": [
                "Direct admission to Coronary Care Unit (CCU) with continuous continuous 12-lead ECG telemetry.",
                "Venous access and immediate baseline cardiac biomarkers (high-sensitivity Troponin, BNP, Lipid Panel, eGFR).",
                "Continuous automated non-invasive blood pressure monitoring every 15–30 minutes."
            ],
            "diagnostic_interventions": [
                "Urgent diagnostic Coronary Angiography (Cath Lab scheduling within 24 hours if hemodynamically unstable or marked ST depression).",
                "Urgent Transthoracic Echocardiogram (TTE) to evaluate regional wall motion abnormalities and Left Ventricular Ejection Fraction (LVEF).",
                "Cardiology consultation for revascularization evaluation (PCI vs. CABG review)."
            ],
            "pharmacological_strategy": [
                "Anti-ischemic medical optimization: Titrated IV or oral beta-blockers (if no bradycardia) and sublingual/transdermal nitrates.",
                "Dual Antiplatelet Therapy (DAPT) protocol preparation (Aspirin + P2Y12 inhibitor).",
                "High-intensity statin therapy (Atorvastatin 80mg or Rosuvastatin 40mg).",
                "Tight blood pressure control with ACE inhibitor or ARB titration."
            ],
            "hospital_discharge_followup": [
                "Enrollment into Phase II Outpatient Cardiac Rehabilitation prior to hospital discharge.",
                "Structured cardiology outpatient consultation within 7 to 10 days post-discharge.",
                "Home blood pressure monitoring program and direct 24/7 cardiac nurse helpline contact."
            ]
        }
    elif acuity_tier_key == "TIER_2":
        return {
            "tier_title": "Intermediate / Monitored Telemetry Pathway",
            "triage_response": "Step-Down Cardiac Ward with Telemetry Monitoring",
            "immediate_actions": [
                "Admission to intermediate care / telemetry-monitored bed.",
                "Serial ECG monitoring and blood pressure titration protocol.",
                "Cardiovascular risk panel: fasting lipid profile, HbA1c, renal function, and electrolytes."
            ],
            "diagnostic_interventions": [
                "Transthoracic Echocardiogram (TTE) within 24–48 hours to assess cardiac structure and diastolic function.",
                "Stress imaging evaluation (Stress Echocardiography or Myocardial Perfusion SPECT scan) to delineate ischemia vs. baseline defect.",
                "Elective Cath Lab hold pending stress test findings."
            ],
            "pharmacological_strategy": [
                "Guideline-Directed Medical Therapy (GDMT): Combination antihypertensive regimen (ACEi/ARB + Calcium Channel Blocker / Thiazide).",
                "Moderate-to-high intensity statin initiation to achieve target LDL-C reduction.",
                "Glycemic control optimization and lifestyle counseling."
            ],
            "hospital_discharge_followup": [
                "Outpatient cardiology clinic review in 2 to 4 weeks.",
                "Enrollment in supervised exercise conditioning and cardiovascular risk education.",
                "Automated digital medication adherence reminder enrollment."
            ]
        }
    else:
        return {
            "tier_title": "Ambulatory & Secondary Prevention Pathway",
            "triage_response": "General Cardiology / Outpatient Care Pathway",
            "immediate_actions": [
                "Short-stay observation or same-day outpatient cardiology assessment.",
                "Comprehensive baseline vitals, resting 12-lead ECG, and metabolic screening."
            ],
            "diagnostic_interventions": [
                "Elective outpatient Echocardiogram or Treadmill Exercise Stress Test if symptoms evolve.",
                "Ambulatory 24-hour Holter monitoring if palpitations or intermittent symptoms reported."
            ],
            "pharmacological_strategy": [
                "Primary / secondary preventive pharmacotherapy: Low-dose antihypertensive or lipid-lowering medication as indicated.",
                "Cardiovascular risk factor lifestyle counseling (Mediterranean diet, smoking cessation, salt restriction)."
            ],
            "hospital_discharge_followup": [
                "Routine outpatient follow-up at 3 to 6 months.",
                "Cardiometabolic health coaching and remote lifestyle progress tracking."
            ]
        }


def get_patient_hospital_recommendation(
    patient_record: Dict[str, Any],
    assigned_cluster_id: int,
    cluster_risk_profile: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate an instant clinical triage and hospital resource recommendation
    for a specific individual patient based on their features and phenotype cluster.
    """
    tier_key = cluster_risk_profile.get("acuity_tier_key", "TIER_2")
    tier_info = ACUITY_TIERS[tier_key]
    pathway = get_hospital_care_pathway(tier_key)

    # Check for individual red-flag escalations
    st_dep = float(patient_record.get("oldpeak", 0.0) or 0.0)
    has_angina = float(patient_record.get("exang", 0.0) or 0.0) == 1.0
    vessels = float(patient_record.get("ca", 0.0) or 0.0)
    sbp = float(patient_record.get("trestbps", 120.0) or 120.0)

    escalations = []
    if st_dep >= 2.0:
        escalations.append(f"Significant ST Depression ({st_dep:.1f} mm) — High ischemic risk indicator")
    if has_angina:
        escalations.append("Active Exercise-Induced Angina reported")
    if vessels >= 2.0:
        escalations.append(f"Multi-vessel coronary involvement ({int(vessels)} vessels on fluoroscopy)")
    if sbp >= 160.0:
        escalations.append(f"Severe systolic elevation ({int(sbp)} mm Hg) — Requires urgent titration")

    # If severe individual red flags present, escalate triage priority
    effective_bed = tier_info["bed_type"]
    effective_triage = tier_info["triage_urgency"]
    if escalations and tier_key != "TIER_1":
        effective_bed = "Urgent Telemetry or CCU Evaluation (Escalated due to individual markers)"
        effective_triage = "Urgent Specialist Evaluation (Within 24 Hours)"

    return {
        "assigned_cluster": assigned_cluster_id,
        "cluster_acuity": tier_info["name"],
        "badge_color": tier_info["badge_color"],
        "bg_color": tier_info["bg_color"],
        "recommended_bed": effective_bed,
        "nursing_ratio": tier_info["nurse_ratio"],
        "triage_urgency": effective_triage,
        "escalation_flags": escalations,
        "immediate_actions": pathway["immediate_actions"],
        "diagnostic_plan": pathway["diagnostic_interventions"],
        "pharmacotherapy": pathway["pharmacological_strategy"],
        "discharge_plan": pathway["hospital_discharge_followup"]
    }
