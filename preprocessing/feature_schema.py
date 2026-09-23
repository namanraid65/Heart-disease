"""
Common Feature Schema Definition for Federated Heart Disease Prediction
Defines the unified feature order, types, clinical reference ranges,
and category definitions across all three hospital clients.
"""

from typing import List, Dict, Any

# Raw 14 UCI Attribute Names and Order
RAW_FEATURE_NAMES: List[str] = [
    'age',       # Age in years (Continuous)
    'sex',       # Biological sex: 1 = Male, 0 = Female (Binary)
    'cp',        # Chest pain type: 1 = typical, 2 = atypical, 3 = non-anginal, 4 = asymptomatic (Categorical)
    'trestbps',  # Resting blood pressure in mm Hg (Continuous)
    'chol',      # Serum cholesterol in mg/dl (Continuous)
    'fbs',       # Fasting blood sugar > 120 mg/dl: 1 = True, 0 = False (Binary)
    'restecg',   # Resting ECG: 0 = normal, 1 = ST-T wave abnormality, 2 = LV hypertrophy (Categorical)
    'thalach',   # Maximum heart rate achieved (Continuous)
    'exang',     # Exercise induced angina: 1 = Yes, 0 = No (Binary)
    'oldpeak',   # ST depression induced by exercise relative to rest (Continuous)
    'slope',     # Slope of the peak exercise ST segment: 1 = up, 2 = flat, 3 = down (Categorical)
    'ca',        # Number of major vessels colored by fluoroscopy: 0, 1, 2, 3 (Categorical / Count)
    'thal',      # Thalassemia status: 3 = normal, 6 = fixed defect, 7 = reversible defect (Categorical)
    'num'        # Target: coronary artery disease status (0: <50% stenosis, 1-4: >50% stenosis)
]

# Raw 13 input features (excluding target)
INPUT_FEATURES: List[str] = RAW_FEATURE_NAMES[:-1]
TARGET_COLUMN: str = 'num'

# Feature Type Groupings
CONTINUOUS_FEATURES: List[str] = [
    'age',
    'trestbps',
    'chol',
    'thalach',
    'oldpeak'
]

BINARY_FEATURES: List[str] = [
    'sex',
    'fbs',
    'exang'
]

CATEGORICAL_FEATURES: List[str] = [
    'cp',
    'restecg',
    'slope',
    'ca',
    'thal'
]

# Defined Domain Categories for One-Hot Encoding across all clients
# This ensures that even if a small client fold lacks a specific category,
# the generated one-hot encoded matrix maintains an identical feature dimension and order.
CATEGORICAL_CATEGORIES: Dict[str, List[float]] = {
    'cp': [1.0, 2.0, 3.0, 4.0],
    'restecg': [0.0, 1.0, 2.0],
    'slope': [1.0, 2.0, 3.0],
    'ca': [0.0, 1.0, 2.0, 3.0],
    'thal': [3.0, 6.0, 7.0]
}

# Clinical Fallback Reference Values for imputation when a client's training fold
# contains 100% missing values for a specific test (e.g. unrecorded cholesterol in Switzerland)
CLINICAL_FALLBACKS: Dict[str, float] = {
    'age': 55.0,
    'sex': 1.0,
    'cp': 4.0,          # Mode in clinical cohorts
    'trestbps': 130.0,   # Population median
    'chol': 240.0,       # Clinical median reference
    'fbs': 0.0,          # Majority non-diabetic
    'restecg': 0.0,      # Normal ECG
    'thalach': 140.0,    # Normal peak heart rate
    'exang': 0.0,        # Majority no exercise angina
    'oldpeak': 0.0,      # Normal ST segment
    'slope': 2.0,        # Flat slope (most frequent)
    'ca': 0.0,           # 0 major vessels (normal baseline)
    'thal': 3.0          # Normal thallium scan
}

# Generates the deterministic list of all 25 processed feature column names
def get_processed_feature_names() -> List[str]:
    names: List[str] = []
    # 1. Scaled Continuous Features (5)
    names.extend(CONTINUOUS_FEATURES)
    # 2. Binary Features (3)
    names.extend(BINARY_FEATURES)
    # 3. One-Hot Encoded Categorical Features (17)
    for feat, cats in CATEGORICAL_CATEGORIES.items():
        for cat in cats:
            names.append(f"{feat}_{int(cat)}")
    return names

# Final ordered 25 feature schema for all Federated Clients
PROCESSED_FEATURE_NAMES: List[str] = get_processed_feature_names()
NUM_PROCESSED_FEATURES: int = len(PROCESSED_FEATURE_NAMES)  # 25
