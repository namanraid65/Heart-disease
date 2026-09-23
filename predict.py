"""
Interactive Terminal Inference & Prediction CLI for Federated Heart Disease Models
Run single-patient predictions using the trained Federated 1D AlexNet, 1D ResNet, and Local XGBoost Ensemble models.

Usage:
  Interactive Mode:
    python predict.py

  Preset Demo Examples:
    python predict.py --demo high_risk
    python predict.py --demo healthy

  Command-Line Arguments:
    python predict.py --age 58 --sex 1 --cp 4 --trestbps 140 --chol 260 --fbs 0 --restecg 0 --thalach 145 --exang 1 --oldpeak 2.5 --slope 2 --ca 1 --thal 7
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
from typing import Dict, Any
import numpy as np
import pandas as pd

from preprocessing.preprocess_client import ClientPreprocessor, load_client_preprocessor
from xai.model_loader import load_federated_model


def load_all_models():
    """Loads all trained final federated and ensemble model wrappers and all client preprocessors."""
    models = {}
    
    # 1. Federated 1D AlexNet
    try:
        models['Federated 1D AlexNet'] = load_federated_model('Federated_1D_AlexNet', device='cpu')
    except Exception as e:
        print(f"[-] Could not load Federated AlexNet: {e}")

    # 2. Federated 1D ResNet
    try:
        models['Federated 1D ResNet'] = load_federated_model('Federated_1D_ResNet', device='cpu')
    except Exception as e:
        print(f"[-] Could not load Federated ResNet: {e}")

    # 3. Sample-Weighted Local XGBoost Ensemble
    try:
        models['Local XGBoost Ensemble'] = load_federated_model('Local_XGBoost_Ensemble')
    except Exception as e:
        print(f"[-] Could not load Local XGBoost Ensemble: {e}")

    # Preprocessors (Hospital-specific fitted preprocessors for all 3 clinical silos)
    preprocessors = {}
    for cid in ['hospital_1', 'hospital_2', 'hospital_3']:
        prep_path = PROJECT_ROOT / 'data' / 'processed' / cid / 'preprocessor.joblib'
        if prep_path.exists():
            preprocessors[cid] = load_client_preprocessor(prep_path)

    return models, preprocessors


def get_interactive_input() -> dict:
    """Interactively prompts user for patient clinical attributes with validation."""
    print("=" * 75)
    print("       FEDERATED HEART DISEASE RISK PREDICTOR (CLI INFERENCE)")
    print("       Enter patient clinical parameters below (press Enter for default)")
    print("=" * 75)

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


def get_demo_patient(demo_type: str) -> dict:
    """Returns sample pre-configured patient profiles."""
    if demo_type.lower() == 'high_risk':
        return {
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
    else:  # healthy
        return {
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


def format_progress_bar(probability: float, length: int = 20) -> str:
    """Generates an ASCII risk gauge."""
    filled = int(round(probability * length))
    empty = length - filled
    return f"[{'#' * filled}{'-' * empty}]"


def predict_patient(
    patient_dict: dict,
    models: dict,
    preprocessors: Dict[str, ClientPreprocessor],
    target_hospital: str = 'hospital_1'
):
    """Preprocesses input and executes model inference with strict preprocessing consistency."""
    raw_df = pd.DataFrame([patient_dict])
    
    # Clean and clip
    clean_raw_df = raw_df.copy()
    if 'chol' in clean_raw_df.columns and clean_raw_df['chol'].iloc[0] == 0:
        clean_raw_df['chol'] = np.nan
    if 'oldpeak' in clean_raw_df.columns:
        clean_raw_df['oldpeak'] = clean_raw_df['oldpeak'].clip(lower=0.0)

    # Get target hospital preprocessor for neural models
    target_prep = preprocessors.get(target_hospital, preprocessors.get('hospital_1'))
    if target_prep is not None:
        processed_df = target_prep.transform(clean_raw_df)
    else:
        processed_df = None

    print("\n" + "=" * 75)
    print("                    CLINICAL INFERENCE REPORT")
    print("=" * 75)
    print(" [i] Patient Profile Summary:")
    print(f"     * Age: {int(patient_dict['age'])} | Sex: {'Male' if patient_dict['sex'] == 1 else 'Female'}")
    cp_map = {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal', 4: 'Asymptomatic'}
    print(f"     * Chest Pain: {cp_map.get(int(patient_dict['cp']), str(patient_dict['cp']))} | BP: {patient_dict['trestbps']:.0f} mmHg | Chol: {patient_dict['chol']:.0f} mg/dL")
    print(f"     * Max HR: {patient_dict['thalach']:.0f} bpm | Exercise Angina: {'Yes' if patient_dict['exang'] == 1 else 'No'} | ST Depression: {patient_dict['oldpeak']:.1f}")
    print(f"     * Major Vessels (ca): {int(patient_dict['ca'])} | Thallium (thal): {int(patient_dict['thal'])}")
    print(f"     * Neural Reference Standard: {target_hospital} (Cleveland/Hungarian/Swiss scaler)")
    print("-" * 75)
    print(" [*] Multi-Model Predictions:")

    probabilities = {}
    for name, wrapper in models.items():
        if 'XGBoost' in name:
            # Preprocessing Consistency: Local XGBoost Ensemble transforms raw_df
            # using each constituent model's native client preprocessor
            probs = wrapper.predict_proba(clean_raw_df)
        else:
            if processed_df is None:
                continue
            probs = wrapper.predict_proba(processed_df)

        prob = float(probs[0, 1])
        probabilities[name] = prob
        status = "HIGH RISK (1)" if prob >= 0.50 else "LOW RISK (0)"
        pct = f"{prob * 100:>5.1f}%"
        gauge = format_progress_bar(prob, length=20)
        print(f"     * {name:<26} : {pct}  {gauge} -> {status}")

    # Ensemble Consensus Average
    if probabilities:
        avg_prob = float(np.mean(list(probabilities.values())))
        consensus_status = "HIGH RISK (Heart Disease Indicated)" if avg_prob >= 0.50 else "LOW RISK (Healthy / Unlikely Disease)"

        print("-" * 75)
        print(f" [>] CONSENSUS ENSEMBLE RISK : {avg_prob * 100:.1f}%")
        print(f" [>] MODEL CONSENSUS OUTPUT : {consensus_status}")
        print(" [>] Note: Statistical machine learning estimate for research only; not clinical diagnosis.")
    print("=" * 75)

    # Key Risk Drivers Breakdown
    print("\n [!] Key Diagnostic Observations:")
    drivers = []
    if patient_dict.get('oldpeak', 0) >= 1.5:
        drivers.append(f"Significant ST Depression ({patient_dict['oldpeak']:.1f} mm) strongly elevates ischemic risk.")
    if patient_dict.get('cp', 0) == 4:
        drivers.append("Asymptomatic chest pain presentation is statistically correlated with advanced silent ischemia.")
    if patient_dict.get('ca', 0) > 0:
        drivers.append(f"Fluoroscopy detected {int(patient_dict['ca'])} vessel(s) with >50% narrowing.")
    if patient_dict.get('thal', 0) == 7:
        drivers.append("Reversible thallium defect indicates stress-induced myocardial ischemia.")
    if patient_dict.get('exang', 0) == 1:
        drivers.append("Positive exercise-induced angina indicates low coronary reserve.")
    if patient_dict.get('thalach', 200) < 130:
        drivers.append(f"Reduced chronotropic capacity (Max HR: {patient_dict['thalach']:.0f} bpm).")

    if drivers:
        for d in drivers:
            print(f"     [!] {d}")
    else:
        print("     [+] No high-acuity ischemic triggers identified; physiological markers remain within healthy bounds.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Federated Heart Disease Prediction CLI")
    parser.add_argument('--demo', type=str, choices=['high_risk', 'healthy'], help="Run preset patient demonstration")
    parser.add_argument(
        '--hospital',
        type=str,
        default='hospital_1',
        choices=['hospital_1', 'hospital_2', 'hospital_3'],
        help="Hospital clinical standard to apply for neural preprocessor (default: hospital_1 / Cleveland)"
    )
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

    args = parser.parse_args()

    models, preprocessors = load_all_models()
    if not models:
        print("[ERROR] Could not load model checkpoints.")
        sys.exit(1)

    if args.demo:
        patient = get_demo_patient(args.demo)
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
    else:
        patient = get_interactive_input()

    predict_patient(patient, models, preprocessors, target_hospital=args.hospital)


if __name__ == '__main__':
    main()
