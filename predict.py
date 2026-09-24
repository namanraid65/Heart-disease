"""
Heterogeneous Federated Heart Disease Risk Predictor (CLI Inference & Clinical Explainability)

Executes patient-level risk estimation and native feature explainability (SHAP & LIME)
strictly following the heterogeneous federated learning architecture:
  select hospital
        ↓
  load hospital schema
        ↓
  load hospital preprocessing
        ↓
  load private encoder (D_i -> Z)
        ↓
  load global shared predictor (Z -> 1)
        ↓
  transform input
        ↓
  encoder -> predictor -> probability
        ↓
  SHAP / LIME attribution on native features

Supports:
  - Hospital 1 (Cleveland): D1 = 25 native processed features
  - Hospital 2 (Hungarian): D2 = 25 native processed features
  - Hospital 3 (Switzerland): D3 = 25 native processed features
  - Hospital 4 (Synthetic): D4 = 30 native processed features (25 base + bmi, hba1c, crp, ldl, hdl)
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output encoding on Windows terminals if supported
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import argparse
import json
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
import torch

from preprocessing.heterogeneous_schema import (
    get_client_schema,
    ClientFeatureSchema,
    CLIENT_SCHEMAS,
    PROCESSED_FEATURE_NAMES,
    H4_30_FEATURE_NAMES,
    H4_ADDITIONAL_FEATURES
)
from preprocessing.feature_schema import INPUT_FEATURES, RAW_FEATURE_NAMES
from preprocessing.preprocess_client import ClientPreprocessor, load_client_preprocessor
from models.heterogeneous.encoder import HospitalEncoder
from models.heterogeneous.predictor import SharedPredictor
from models.heterogeneous.composite import HeterogeneousCompositeModel
from federated.heterogeneous.xai_compat import HeterogeneousLocalXAIWrapper
from federated.heterogeneous.config import LATENT_DIM, HETEROGENEOUS_CHECKPOINTS_DIR
from xai.config import FEATURE_DISPLAY_NAMES


def format_progress_bar(probability: float, length: int = 20) -> str:
    """Generates an ASCII risk gauge."""
    filled = int(round(probability * length))
    empty = length - filled
    return f"[{'#' * filled}{'-' * empty}]"


def load_heterogeneous_inference_pipeline(
    hospital_id: str = 'hospital_1',
    device: str = 'cpu'
) -> Tuple[HeterogeneousCompositeModel, ClientFeatureSchema, Optional[ClientPreprocessor]]:
    """
    Loads the complete inference pipeline for a hospital:
      1. Schema (input dimension D_i and feature names)
      2. Fitted client preprocessor (if available)
      3. Hospital-local private encoder (D_i -> Z)
      4. Global shared predictor (Z -> 1)
    """
    schema = get_client_schema(hospital_id)
    input_dim = schema.input_dimension
    latent_dim = LATENT_DIM

    # Load Client Preprocessor
    preprocessor = None
    prep_path = PROJECT_ROOT / 'data' / 'processed' / hospital_id / 'preprocessor.joblib'
    if prep_path.exists():
        preprocessor = load_client_preprocessor(prep_path)

    dropout_rate = schema.encoder_config.get('dropout_rate', 0.2)
    encoder = HospitalEncoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=schema.encoder_config.get('hidden_dims', [64]),
        dropout_rate=dropout_rate
    )
    predictor = SharedPredictor(
        latent_dim=latent_dim,
        hidden_dims=[32],
        dropout_rate=dropout_rate
    )

    # 1. Attempt loading client-paired checkpoint
    client_ckpt_path = HETEROGENEOUS_CHECKPOINTS_DIR / f"{hospital_id}_heterogeneous_final.pt"
    global_ckpt_path = HETEROGENEOUS_CHECKPOINTS_DIR / "global_heterogeneous_predictor_final.pt"

    encoder_loaded = False
    predictor_loaded = False

    if client_ckpt_path.exists():
        ckpt = torch.load(client_ckpt_path, map_location=device, weights_only=False)
        if 'encoder_state_dict' in ckpt:
            encoder.load_state_dict(ckpt['encoder_state_dict'])
            encoder_loaded = True
        if 'shared_predictor_state_dict' in ckpt:
            predictor.load_state_dict(ckpt['shared_predictor_state_dict'])
            predictor_loaded = True

    # 2. If global shared predictor checkpoint exists, prioritize it for the predictor head
    if global_ckpt_path.exists():
        g_ckpt = torch.load(global_ckpt_path, map_location=device, weights_only=False)
        if 'shared_predictor_state_dict' in g_ckpt:
            predictor.load_state_dict(g_ckpt['shared_predictor_state_dict'])
            predictor_loaded = True

    # 3. Synthetic Hospital 4 handling: initialize and cache if absent
    if not encoder_loaded and 'synthetic' in hospital_id.lower():
        print(f" [SYNTHETIC TEST] Initialized synthetic private encoder for '{hospital_id}' (D={input_dim} -> Z={latent_dim})")
        torch.manual_seed(42)
        # Initialize reasonable weights
        encoder = HospitalEncoder(input_dim=input_dim, latent_dim=latent_dim, hidden_dims=[64])
        # Save synthetic client checkpoint so it's persisted
        HETEROGENEOUS_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        synthetic_payload = {
            'hospital_id': hospital_id,
            'hospital_name': schema.hospital_name,
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'encoder_state_dict': encoder.state_dict(),
            'shared_predictor_state_dict': predictor.state_dict(),
            'feature_names': schema.feature_names,
            'feature_types': schema.feature_types,
            'is_synthetic': True
        }
        torch.save(synthetic_payload, client_ckpt_path)
        encoder_loaded = True

    model = HeterogeneousCompositeModel(encoder=encoder, predictor=predictor).to(device)
    model.eval()
    return model, schema, preprocessor


def preprocess_patient_input(
    patient_data: Dict[str, Any],
    schema: ClientFeatureSchema,
    preprocessor: Optional[ClientPreprocessor]
) -> np.ndarray:
    """
    Transforms user input into the expected native feature tensor of shape [1, D_i].
    Supports both raw clinical inputs (13 UCI features + optional biomarkers)
    and directly preprocessed feature dictionaries/arrays.
    """
    input_dim = schema.input_dimension
    feature_names = schema.feature_names

    # Case A: Input dictionary already provides all D_i feature names directly
    if all(feat in patient_data for feat in feature_names):
        feature_vals = [float(patient_data[feat]) for feat in feature_names]
        return np.array(feature_vals, dtype=np.float32).reshape(1, -1)

    # Case B: Input is raw UCI clinical features (13 attributes)
    raw_df = pd.DataFrame([{k: patient_data.get(k, np.nan) for k in INPUT_FEATURES}])
    if 'chol' in raw_df.columns and raw_df['chol'].iloc[0] == 0:
        raw_df['chol'] = np.nan
    if 'oldpeak' in raw_df.columns and not pd.isna(raw_df['oldpeak'].iloc[0]):
        raw_df['oldpeak'] = raw_df['oldpeak'].clip(lower=0.0)

    # Transform through client preprocessor if available, otherwise use hospital_1 fallback preprocessor
    if preprocessor is not None and preprocessor.is_fitted_:
        base_processed = preprocessor.transform(raw_df)
    else:
        fallback_path = PROJECT_ROOT / 'data' / 'processed' / 'hospital_1' / 'preprocessor.joblib'
        if fallback_path.exists():
            h1_prep = load_client_preprocessor(fallback_path)
            base_processed = h1_prep.transform(raw_df)
        else:
            raise RuntimeError("No fitted ClientPreprocessor found to process raw patient attributes.")

    # If schema requires standard 25 features:
    if input_dim == 25:
        return base_processed.values.astype(np.float32)

    # If schema is Hospital 4 (30 features): base 25 + 5 biomarkers
    if input_dim == 30 and schema.hospital_id == 'hospital_4_synthetic':
        biomarker_defaults = {
            'bmi': (26.5, 25.0, 5.0),       # (default, mean, std)
            'hba1c': (5.7, 5.5, 1.0),
            'crp': (1.5, 1.0, 1.2),
            'ldl': (130.0, 115.0, 30.0),
            'hdl': (48.0, 50.0, 15.0)
        }
        bio_vals = []
        for bio in H4_ADDITIONAL_FEATURES:
            raw_v = float(patient_data.get(bio, biomarker_defaults[bio][0]))
            mean_v, std_v = biomarker_defaults[bio][1], biomarker_defaults[bio][2]
            scaled_v = (raw_v - mean_v) / (std_v + 1e-6)
            bio_vals.append(scaled_v)

        full_arr = np.hstack([base_processed.values.astype(np.float32), np.array(bio_vals, dtype=np.float32).reshape(1, -1)])
        return full_arr

    # Default fallback: zero-pad or trim to schema input dimension
    curr = base_processed.values.astype(np.float32)
    if curr.shape[1] < input_dim:
        pad = np.zeros((1, input_dim - curr.shape[1]), dtype=np.float32)
        return np.hstack([curr, pad])
    return curr[:, :input_dim]


def explain_prediction(
    wrapper: HeterogeneousLocalXAIWrapper,
    x_native: np.ndarray,
    schema: ClientFeatureSchema,
    hospital_id: str,
    explainer_type: str = "shap",
    num_background: int = 15
) -> Dict[str, Any]:
    """
    Computes local feature explanations directly on the hospital's native feature space.
    """
    results = {}
    feature_names = schema.feature_names
    num_features = len(feature_names)

    # Background dataset for KernelExplainer / LIME
    # Synthesize small local background around zero (standardized distribution)
    np.random.seed(42)
    bg_data = np.random.normal(loc=0.0, scale=0.8, size=(num_background, num_features)).astype(np.float32)

    if explainer_type in ('shap', 'both'):
        try:
            from xai.shap_explainer import DeepLearningShapExplainer
            shap_explainer = DeepLearningShapExplainer(
                predict_fn=wrapper.predict_proba,
                background_data=bg_data,
                feature_names=feature_names,
                n_background=num_background,
                random_state=42
            )
            shap_res = shap_explainer.explain_instance(x_native, nsamples=100)
            results['shap'] = shap_res
        except Exception as e:
            results['shap_error'] = str(e)

    if explainer_type in ('lime', 'both'):
        try:
            from xai.lime_explainer import LimeTabularExplainerWrapper
            lime_explainer = LimeTabularExplainerWrapper(
                training_data=bg_data,
                feature_names=feature_names,
                class_names=['Healthy (0)', 'Heart Disease (1)'],
                random_state=42
            )
            lime_res = lime_explainer.explain_instance(
                sample=x_native,
                predict_fn=wrapper.predict_proba,
                num_features=10,
                num_samples=150
            )
            results['lime'] = lime_res
        except Exception as e:
            results['lime_error'] = str(e)

    return results


def run_heterogeneous_inference(
    patient_data: Dict[str, Any],
    hospital_id: str = 'hospital_1',
    explainer_type: str = 'shap',
    device: str = 'cpu',
    compare_legacy_baselines: bool = False
) -> Dict[str, Any]:
    """
    Executes the full heterogeneous prediction and explainability workflow.
    """
    # 1. Load pipeline
    model, schema, preprocessor = load_heterogeneous_inference_pipeline(hospital_id=hospital_id, device=device)

    # 2. Transform patient input to native feature space [1, D_i]
    x_native = preprocess_patient_input(patient_data, schema, preprocessor)
    x_tensor = torch.as_tensor(x_native, dtype=torch.float32, device=device)

    # 3. Forward pass through heterogeneous pipeline: x_native -> encoder -> z -> predictor -> logits
    with torch.no_grad():
        latent_z = model.encode(x_tensor)
        logits = model.predict_from_latent(latent_z)
        risk_prob = float(torch.sigmoid(logits).cpu().item())

    pred_class = 1 if risk_prob >= 0.50 else 0
    status_label = "HIGH RISK (Heart Disease Indicated)" if pred_class == 1 else "LOW RISK (Healthy / Unlikely Disease)"

    # 4. Generate Local Explanations in Native Feature Space
    xai_wrapper = HeterogeneousLocalXAIWrapper(model, device=device)
    explanations = {}
    if explainer_type != 'none':
        explanations = explain_prediction(
            wrapper=xai_wrapper,
            x_native=x_native,
            schema=schema,
            hospital_id=hospital_id,
            explainer_type=explainer_type
        )

    # 5. Diagnostic observations
    drivers = []
    if patient_data.get('oldpeak', 0) >= 1.5:
        drivers.append(f"Significant ST Depression ({patient_data['oldpeak']:.1f} mm) strongly elevates ischemic risk.")
    if patient_data.get('cp', 0) == 4:
        drivers.append("Asymptomatic chest pain presentation is statistically correlated with advanced ischemia.")
    if patient_data.get('ca', 0) > 0:
        drivers.append(f"Fluoroscopy detected {int(patient_data['ca'])} vessel(s) with >50% narrowing.")
    if patient_data.get('thal', 0) == 7:
        drivers.append("Reversible thallium defect indicates stress-induced myocardial ischemia.")
    if patient_data.get('exang', 0) == 1:
        drivers.append("Positive exercise-induced angina indicates low coronary reserve.")
    if patient_data.get('thalach', 200) < 130:
        drivers.append(f"Reduced chronotropic capacity (Max HR: {patient_data['thalach']:.0f} bpm).")

    # Additional Hospital 4 biomarkers
    if hospital_id == 'hospital_4_synthetic':
        if patient_data.get('bmi', 0) >= 30.0:
            drivers.append(f"Obesity class indicator (BMI: {patient_data['bmi']:.1f} kg/m²).")
        if patient_data.get('hba1c', 0) >= 6.5:
            drivers.append(f"Diabetic glycated hemoglobin marker (HbA1c: {patient_data['hba1c']:.1f}%).")
        if patient_data.get('crp', 0) >= 3.0:
            drivers.append(f"Elevated inflammatory biomarker (hs-CRP: {patient_data['crp']:.1f} mg/L).")

    # 6. Legacy baselines (optional, strictly non-heterogeneous comparison)
    legacy_results = {}
    if compare_legacy_baselines:
        from xai.model_loader import load_federated_model
        raw_df = pd.DataFrame([{k: patient_data.get(k, np.nan) for k in INPUT_FEATURES}])
        if preprocessor is not None:
            proc_df = preprocessor.transform(raw_df)
        else:
            h1_prep = load_client_preprocessor(PROJECT_ROOT / 'data' / 'processed' / 'hospital_1' / 'preprocessor.joblib')
            proc_df = h1_prep.transform(raw_df)

        for m_name in ['Federated_1D_AlexNet', 'Federated_1D_ResNet', 'Local_XGBoost_Ensemble']:
            try:
                m_wrap = load_federated_model(m_name)
                if 'XGBoost' in m_name:
                    p = float(m_wrap.predict_proba(raw_df)[0, 1])
                else:
                    p = float(m_wrap.predict_proba(proc_df)[0, 1])
                legacy_results[m_name] = p
            except Exception as e:
                legacy_results[m_name] = f"Error: {e}"

    report = {
        'hospital_id': hospital_id,
        'hospital_name': schema.hospital_name,
        'input_dim': schema.input_dimension,
        'latent_dim': LATENT_DIM,
        'predicted_risk_probability': risk_prob,
        'predicted_class': pred_class,
        'status_label': status_label,
        'diagnostic_drivers': drivers,
        'explanations': explanations,
        'legacy_baselines': legacy_results,
        'is_synthetic': 'synthetic' in hospital_id.lower()
    }
    return report


def print_clinical_report(report: Dict[str, Any], patient_data: Dict[str, Any]):
    """Prints a structured clinical risk report."""
    print("\n" + "=" * 80)
    print("      HETEROGENEOUS FEDERATED HEART DISEASE RISK INFERENCE REPORT")
    print("=" * 80)
    if report['is_synthetic']:
        print(" [SYNTHETIC TEST NOTICE] This inference utilizes a synthetic hospital cohort.")

    print(f" [i] Hospital Domain:        {report['hospital_name']} (ID: {report['hospital_id']})")
    h_num = ''.join(c for c in report['hospital_id'] if c.isdigit()) or 'i'
    print(f" [i] Architecture Pipeline:  Native D_{h_num} = {report['input_dim']} -> Private Encoder -> Latent Z = {report['latent_dim']} -> Shared Predictor")
    print(f" [i] Patient Profile:")
    if 'age' in patient_data:
        print(f"     * Age: {patient_data.get('age', 'N/A')} | Sex: {'Male' if patient_data.get('sex') == 1 else 'Female'}")
        cp_map = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic'}
        print(f"     * Chest Pain: {cp_map.get(patient_data.get('cp'), str(patient_data.get('cp')))} | BP: {patient_data.get('trestbps', 'N/A')} mmHg | Chol: {patient_data.get('chol', 'N/A')} mg/dL")
        print(f"     * Max HR: {patient_data.get('thalach', 'N/A')} bpm | Angina: {'Yes' if patient_data.get('exang') == 1 else 'No'} | ST Depr: {patient_data.get('oldpeak', 'N/A')}")
    if report['input_dim'] == 30:
        print(f"     * Extended Biomarkers: BMI={patient_data.get('bmi', 'N/A')}, HbA1c={patient_data.get('hba1c', 'N/A')}%, CRP={patient_data.get('crp', 'N/A')} mg/L, LDL={patient_data.get('ldl', 'N/A')}, HDL={patient_data.get('hdl', 'N/A')}")

    print("-" * 80)
    prob = report['predicted_risk_probability']
    pct = f"{prob * 100:>5.1f}%"
    gauge = format_progress_bar(prob, length=20)
    print(f" [*] MODEL-ESTIMATED RISK:    {pct}  {gauge}")
    print(f" [*] PREDICTED RISK STATUS:   {report['status_label']}")
    print(" [*] Statistical machine learning estimate for research only; not clinical diagnosis.")
    print("-" * 80)

    # Diagnostic drivers
    if report['diagnostic_drivers']:
        print(" [!] Clinical Diagnostic Triggers:")
        for d in report['diagnostic_drivers']:
            print(f"     * {d}")
    else:
        print(" [+] Physiological indicators remain within expected baseline bounds.")

    # SHAP Explanations
    if 'shap' in report['explanations']:
        shap_res = report['explanations']['shap']
        print("-" * 80)
        print(f" [XAI] SHAP Local Attribution (Top 5 Native Features out of {report['input_dim']}):")
        sorted_feats = shap_res.get('sorted_features', [])[:5]
        for feat, val in sorted_feats:
            disp_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            direction = "(+ Increases Risk)" if val > 0 else "(- Decreases Risk)"
            print(f"     * {disp_name:<45} : {val:+.4f}  {direction}")

    # LIME Explanations
    if 'lime' in report['explanations']:
        lime_res = report['explanations']['lime']
        print("-" * 80)
        print(f" [XAI] LIME Local Attribution (Top 5 Native Features out of {report['input_dim']}):")
        sorted_feats = lime_res.get('sorted_features', [])[:5]
        for feat, val in sorted_feats:
            disp_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
            direction = "(+ Increases Risk)" if val > 0 else "(- Decreases Risk)"
            print(f"     * {disp_name:<45} : {val:+.4f}  {direction}")

    # Legacy Baselines
    if report['legacy_baselines']:
        print("-" * 80)
        print(" [Ref] Legacy Baseline Comparisons (Homogeneous 25-dim):")
        for m_name, p in report['legacy_baselines'].items():
            if isinstance(p, float):
                print(f"     * {m_name:<28} : {p*100:>5.1f}%  {format_progress_bar(p, length=15)}")
            else:
                print(f"     * {m_name:<28} : {p}")

    print("=" * 80 + "\n")


def get_demo_patient(demo_type: str, hospital_id: str = 'hospital_1') -> Dict[str, Any]:
    """Returns preset patient profiles."""
    if demo_type.lower() == 'high_risk':
        patient = {
            'age': 63.0,
            'sex': 1.0,        # Male
            'cp': 4.0,         # Asymptomatic chest pain
            'trestbps': 150.0, # High BP
            'chol': 285.0,     # High cholesterol
            'fbs': 1.0,        # Diabetic FBS > 120
            'restecg': 2.0,    # LV Hypertrophy
            'thalach': 120.0,  # Low max heart rate
            'exang': 1.0,      # Exercise angina present
            'oldpeak': 2.6,    # High ST depression
            'slope': 2.0,      # Flat slope
            'ca': 2.0,         # 2 vessels blocked
            'thal': 7.0        # Reversible defect
        }
        if hospital_id == 'hospital_4_synthetic':
            patient.update({
                'bmi': 32.4,
                'hba1c': 7.8,
                'crp': 4.5,
                'ldl': 165.0,
                'hdl': 38.0
            })
        return patient

    elif demo_type.lower() == 'h4_synthetic':
        # Explicit 30-feature demonstration patient
        return {
            'age': 61.0,
            'sex': 1.0,
            'cp': 4.0,
            'trestbps': 148.0,
            'chol': 270.0,
            'fbs': 1.0,
            'restecg': 1.0,
            'thalach': 125.0,
            'exang': 1.0,
            'oldpeak': 2.2,
            'slope': 2.0,
            'ca': 1.0,
            'thal': 7.0,
            'bmi': 31.8,
            'hba1c': 7.2,
            'crp': 3.8,
            'ldl': 155.0,
            'hdl': 40.0
        }

    else:  # healthy
        patient = {
            'age': 44.0,
            'sex': 0.0,        # Female
            'cp': 2.0,         # Atypical angina
            'trestbps': 118.0, # Normal BP
            'chol': 195.0,     # Normal cholesterol
            'fbs': 0.0,        # Normal sugar
            'restecg': 0.0,    # Normal ECG
            'thalach': 175.0,  # High exercise tolerance
            'exang': 0.0,      # No exercise angina
            'oldpeak': 0.0,    # No ST depression
            'slope': 1.0,      # Upsloping
            'ca': 0.0,         # 0 vessels blocked
            'thal': 3.0        # Normal
        }
        if hospital_id == 'hospital_4_synthetic':
            patient.update({
                'bmi': 22.8,
                'hba1c': 5.2,
                'crp': 0.8,
                'ldl': 98.0,
                'hdl': 62.0
            })
        return patient


def get_interactive_input(hospital_id: str) -> Dict[str, Any]:
    """Interactively prompts user for patient clinical attributes."""
    print("=" * 80)
    print(f"       FEDERATED HEART DISEASE RISK PREDICTOR: {hospital_id.upper()}")
    print("       Enter patient clinical parameters below (press Enter for default)")
    print("=" * 80)

    prompts = [
        ('age', 'Patient Age in years', 55.0, float),
        ('sex', 'Biological Sex (1 = Male, 0 = Female)', 1.0, float),
        ('cp', 'Chest Pain Type (1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic)', 4.0, float),
        ('trestbps', 'Resting Blood Pressure (mm Hg, e.g. 120-180)', 130.0, float),
        ('chol', 'Serum Cholesterol (mg/dl, e.g. 150-350)', 240.0, float),
        ('fbs', 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', 0.0, float),
        ('restecg', 'Resting ECG (0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy)', 0.0, float),
        ('thalach', 'Maximum Heart Rate Achieved (bpm, e.g. 90-200)', 145.0, float),
        ('exang', 'Exercise Induced Angina (1 = Yes, 0 = No)', 0.0, float),
        ('oldpeak', 'ST Depression Induced by Exercise (e.g. 0.0 - 5.0)', 1.0, float),
        ('slope', 'ST Segment Slope (1 = Upsloping, 2 = Flat, 3 = Downsloping)', 2.0, float),
        ('ca', 'Fluoroscopy Major Vessels (0, 1, 2, 3)', 0.0, float),
        ('thal', 'Thallium Stress Test (3 = Normal, 6 = Fixed, 7 = Reversible)', 3.0, float),
    ]

    if hospital_id == 'hospital_4_synthetic':
        prompts.extend([
            ('bmi', 'Body Mass Index (BMI in kg/m², e.g. 20-40)', 27.0, float),
            ('hba1c', 'Glycated Hemoglobin HbA1c (%, e.g. 4.5-10.0)', 5.8, float),
            ('crp', 'C-Reactive Protein hs-CRP (mg/L, e.g. 0.5-10.0)', 1.5, float),
            ('ldl', 'Low-Density Lipoprotein LDL (mg/dl, e.g. 70-190)', 125.0, float),
            ('hdl', 'High-Density Lipoprotein HDL (mg/dl, e.g. 35-80)', 50.0, float),
        ])

    patient = {}
    for key, desc, default_val, val_type in prompts:
        while True:
            try:
                user_val = input(f" >> {desc} [Default: {default_val}]: ").strip()
                if not user_val:
                    patient[key] = default_val
                else:
                    patient[key] = val_type(user_val)
                break
            except ValueError:
                print(f"    [!] Invalid input. Please enter a valid numeric value.")

    return patient


def main():
    parser = argparse.ArgumentParser(description="Heterogeneous Federated Heart Disease Risk Inference")
    parser.add_argument(
        '--hospital',
        type=str,
        default='hospital_1',
        choices=['hospital_1', 'hospital_2', 'hospital_3', 'hospital_4_synthetic'],
        help="Hospital clinical standard / client schema to evaluate (default: hospital_1 / Cleveland)"
    )
    parser.add_argument('--demo', type=str, choices=['high_risk', 'healthy', 'h4_synthetic'], help="Run preset demonstration")
    parser.add_argument('--explain', type=str, default='shap', choices=['shap', 'lime', 'both', 'none'], help="XAI explainer to run (default: shap)")
    parser.add_argument('--legacy-baselines', action='store_true', help="Also run non-heterogeneous baselines (AlexNet, ResNet, XGBoost) for comparison")
    parser.add_argument('--patient-json', type=str, default=None, help="JSON string or path to JSON file containing patient attributes")

    # Clinical features
    parser.add_argument('--age', type=float, default=None)
    parser.add_argument('--sex', type=float, default=None)
    parser.add_argument('--cp', type=float, default=None)
    parser.add_argument('--trestbps', type=float, default=None)
    parser.add_argument('--chol', type=float, default=None)
    parser.add_argument('--fbs', type=float, default=None)
    parser.add_argument('--restecg', type=float, default=None)
    parser.add_argument('--thalach', type=float, default=None)
    parser.add_argument('--exang', type=float, default=None)
    parser.add_argument('--oldpeak', type=float, default=None)
    parser.add_argument('--slope', type=float, default=None)
    parser.add_argument('--ca', type=float, default=None)
    parser.add_argument('--thal', type=float, default=None)

    # Hospital 4 Extended Biomarkers
    parser.add_argument('--bmi', type=float, default=None)
    parser.add_argument('--hba1c', type=float, default=None)
    parser.add_argument('--crp', type=float, default=None)
    parser.add_argument('--ldl', type=float, default=None)
    parser.add_argument('--hdl', type=float, default=None)

    args = parser.parse_args()

    # Determine hospital
    hospital_id = args.hospital
    if args.demo == 'h4_synthetic':
        hospital_id = 'hospital_4_synthetic'

    # Determine patient input
    if args.patient_json:
        if Path(args.patient_json).exists():
            with open(args.patient_json, 'r') as f:
                patient = json.load(f)
        else:
            patient = json.loads(args.patient_json)
    elif args.demo:
        patient = get_demo_patient(args.demo, hospital_id=hospital_id)
    elif args.age is not None:
        patient = {
            'age': args.age,
            'sex': args.sex if args.sex is not None else 1.0,
            'cp': args.cp if args.cp is not None else 4.0,
            'trestbps': args.trestbps if args.trestbps is not None else 130.0,
            'chol': args.chol if args.chol is not None else 240.0,
            'fbs': args.fbs if args.fbs is not None else 0.0,
            'restecg': args.restecg if args.restecg is not None else 0.0,
            'thalach': args.thalach if args.thalach is not None else 145.0,
            'exang': args.exang if args.exang is not None else 0.0,
            'oldpeak': args.oldpeak if args.oldpeak is not None else 0.0,
            'slope': args.slope if args.slope is not None else 2.0,
            'ca': args.ca if args.ca is not None else 0.0,
            'thal': args.thal if args.thal is not None else 3.0,
        }
        if hospital_id == 'hospital_4_synthetic':
            patient['bmi'] = args.bmi if args.bmi is not None else 27.0
            patient['hba1c'] = args.hba1c if args.hba1c is not None else 5.8
            patient['crp'] = args.crp if args.crp is not None else 1.5
            patient['ldl'] = args.ldl if args.ldl is not None else 125.0
            patient['hdl'] = args.hdl if args.hdl is not None else 50.0
    else:
        patient = get_interactive_input(hospital_id=hospital_id)

    report = run_heterogeneous_inference(
        patient_data=patient,
        hospital_id=hospital_id,
        explainer_type=args.explain,
        compare_legacy_baselines=args.legacy_baselines
    )

    print_clinical_report(report, patient)


if __name__ == '__main__':
    main()
