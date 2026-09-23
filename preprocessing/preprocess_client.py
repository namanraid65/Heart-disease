"""
Client-Specific Preprocessing Pipeline for Federated Learning
Handles data cleaning, target preparation, stratified splitting,
leakage-free preprocessing fitting, and transformation for individual hospital clients.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from preprocessing.feature_schema import (
    RAW_FEATURE_NAMES,
    INPUT_FEATURES,
    TARGET_COLUMN,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_CATEGORIES,
    CLINICAL_FALLBACKS,
    PROCESSED_FEATURE_NAMES,
    NUM_PROCESSED_FEATURES
)


class ClientPreprocessor:
    """
    Encapsulates preprocessing statistics fitted strictly on a client's training fold.
    Guarantees no data leakage between train, validation, and test splits.
    """

    def __init__(self):
        self.imputation_values_: Dict[str, float] = {}
        self.scaler_: Optional[StandardScaler] = None
        self.is_fitted_: bool = False

    def fit(self, X_train: pd.DataFrame) -> 'ClientPreprocessor':
        """
        Learns imputation values and scaling parameters strictly from training data.
        """
        X = X_train.copy()
        
        # 1. Compute imputation values per feature
        self.imputation_values_ = {}
        
        for col in INPUT_FEATURES:
            valid_series = X[col].dropna()
            
            if len(valid_series) == 0:
                # 100% missing in this client's training fold (e.g. unrecorded chol in Switzerland)
                # Fallback to domain clinical reference standard
                fallback = CLINICAL_FALLBACKS[col]
                self.imputation_values_[col] = float(fallback)
            else:
                if col in CONTINUOUS_FEATURES:
                    # Median for continuous features
                    self.imputation_values_[col] = float(valid_series.median())
                else:
                    # Mode (most frequent value) for categorical / binary features
                    mode_val = valid_series.mode()
                    if len(mode_val) > 0:
                        self.imputation_values_[col] = float(mode_val.iloc[0])
                    else:
                        self.imputation_values_[col] = float(CLINICAL_FALLBACKS[col])

        # 2. Impute training set continuous columns to fit StandardScaler
        X_imputed = X.copy()
        for col in CONTINUOUS_FEATURES:
            X_imputed[col] = X_imputed[col].fillna(self.imputation_values_[col])

        # 3. Fit StandardScaler strictly on training continuous features
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_imputed[CONTINUOUS_FEATURES])
        
        self.is_fitted_ = True
        return self

    def transform(self, X_input: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms any dataset split (train, validation, test, or future inference)
        using the parameters learned during fit().
        """
        if not self.is_fitted_ or self.scaler_ is None:
            raise RuntimeError("ClientPreprocessor must be fitted before transforming data.")

        X = X_input.copy()
        
        # 1. Apply learned imputation values
        for col in INPUT_FEATURES:
            X[col] = X[col].fillna(self.imputation_values_[col])

        # 2. Scale continuous features using fitted StandardScaler
        scaled_cont = self.scaler_.transform(X[CONTINUOUS_FEATURES])
        scaled_cont_df = pd.DataFrame(
            scaled_cont,
            columns=CONTINUOUS_FEATURES,
            index=X.index
        )

        # 3. Binary features (ensure 0.0 or 1.0 float format)
        binary_df = pd.DataFrame(index=X.index)
        for col in BINARY_FEATURES:
            binary_df[col] = X[col].astype(float)

        # 4. One-Hot Encode categorical features across fixed domain categories
        ohe_dfs = []
        for feat, categories in CATEGORICAL_CATEGORIES.items():
            for cat in categories:
                col_name = f"{feat}_{int(cat)}"
                # 1.0 if feature equals category value, else 0.0
                col_series = (X[feat].round() == cat).astype(float)
                col_series.name = col_name
                ohe_dfs.append(col_series)
        
        ohe_df = pd.concat(ohe_dfs, axis=1)

        # 5. Assemble into the exact unified schema order
        processed_df = pd.concat([scaled_cont_df, binary_df, ohe_df], axis=1)
        processed_df = processed_df[PROCESSED_FEATURE_NAMES]

        return processed_df

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fits on training data and returns the transformed training set.
        """
        self.fit(X_train)
        return self.transform(X_train)


def load_client_raw(file_path: Path) -> pd.DataFrame:
    """
    Loads a raw client dataset without modifying the original file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Raw client dataset not found at: {file_path}")

    df = pd.read_csv(
        file_path,
        header=None,
        names=RAW_FEATURE_NAMES,
        na_values='?',
        sep=',',
        skipinitialspace=True
    )
    return df


def clean_client_data(df: pd.DataFrame, client_name: str) -> pd.DataFrame:
    """
    Cleans raw client data according to domain analysis:
      1. Converts chol == 0 to NaN (biologically impossible, unrecorded in Swiss cohort).
      2. Rectifies negative oldpeak to 0.0 (ST depression is non-negative; negative indicates ST elevation).
      3. Drops duplicate records if present (e.g. Hungarian duplicate pair at rows 101-102).
      4. Converts all feature columns to float representations.
    """
    cleaned = df.copy()

    # 1. Handle unrecorded cholesterol (chol == 0)
    cleaned.loc[cleaned['chol'] == 0, 'chol'] = np.nan

    # 2. Rectify negative oldpeak values (clip lower bound at 0.0)
    cleaned['oldpeak'] = cleaned['oldpeak'].clip(lower=0.0)

    # 3. Deduplicate exact duplicate records
    dup_count = cleaned.duplicated().sum()
    if dup_count > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 4. Cast numeric types
    for col in INPUT_FEATURES:
        cleaned[col] = cleaned[col].astype(float)
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)

    return cleaned


def prepare_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepares the binary classification target:
      0 = No heart disease (< 50% stenosis)
      1 = Heart disease present (> 50% stenosis, original values 1, 2, 3, 4)
    """
    X = df[INPUT_FEATURES].copy()
    y = (df[TARGET_COLUMN] > 0).astype(int)
    y.name = 'target'
    return X, y


def split_client_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs stratified train/validation/test split for a single client.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    temp_ratio = val_ratio + test_ratio
    
    # First split: Train vs (Validation + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=y
    )

    # Second split: Validation vs Test
    val_proportion_of_temp = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_proportion_of_temp,
        random_state=random_state,
        stratify=y_temp
    )

    return {
        'X_train': X_train.reset_index(drop=True),
        'y_train': y_train.reset_index(drop=True),
        'X_val': X_val.reset_index(drop=True),
        'y_val': y_val.reset_index(drop=True),
        'X_test': X_test.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True)
    }


def preprocess_single_client(
    raw_file_path: Path,
    client_name: str,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Complete end-to-end preprocessing workflow for a single client without data leakage:
      1. Load raw client data.
      2. Clean client data (replace 0-chol with NaN, clip negative oldpeak, deduplicate).
      3. Prepare binary target (y = 1 if num > 0 else 0).
      4. Split into train, val, and test partitions with stratification.
      5. Fit preprocessor strictly on train partition.
      6. Transform train, val, and test partitions.
      7. Save processed partitions and serialized preprocessor object.
    """
    # 1. Load raw
    raw_df = load_client_raw(raw_file_path)

    # 2. Clean
    cleaned_df = clean_client_data(raw_df, client_name)

    # 3. Target
    X, y = prepare_target(cleaned_df)

    # 4. Split
    splits = split_client_data(
        X, y,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )

    # 5. Fit Preprocessor STRICTLY on Training Split
    preprocessor = ClientPreprocessor()
    preprocessor.fit(splits['X_train'])

    # 6. Transform all splits using training-fitted preprocessor
    X_train_proc = preprocessor.transform(splits['X_train'])
    X_val_proc = preprocessor.transform(splits['X_val'])
    X_test_proc = preprocessor.transform(splits['X_test'])

    # 7. Save outputs
    save_client_processed_data(
        output_dir=output_dir,
        splits_proc={
            'X_train': X_train_proc,
            'y_train': splits['y_train'],
            'X_val': X_val_proc,
            'y_val': splits['y_val'],
            'X_test': X_test_proc,
            'y_test': splits['y_test']
        },
        preprocessor=preprocessor
    )

    return {
        'client_name': client_name,
        'raw_count': len(raw_df),
        'cleaned_count': len(cleaned_df),
        'train_count': len(X_train_proc),
        'val_count': len(X_val_proc),
        'test_count': len(X_test_proc),
        'train_target_dist': splits['y_train'].value_counts().to_dict(),
        'val_target_dist': splits['y_val'].value_counts().to_dict(),
        'test_target_dist': splits['y_test'].value_counts().to_dict(),
        'imputation_values': preprocessor.imputation_values_,
        'scaler_mean': dict(zip(CONTINUOUS_FEATURES, preprocessor.scaler_.mean_)),
        'scaler_scale': dict(zip(CONTINUOUS_FEATURES, preprocessor.scaler_.scale_)),
        'feature_count': X_train_proc.shape[1],
        'processed_splits': {
            'X_train': X_train_proc,
            'y_train': splits['y_train'],
            'X_val': X_val_proc,
            'y_val': splits['y_val'],
            'X_test': X_test_proc,
            'y_test': splits['y_test']
        }
    }


def save_client_processed_data(
    output_dir: Path,
    splits_proc: Dict[str, Any],
    preprocessor: ClientPreprocessor
) -> None:
    """
    Saves processed splits and the fitted preprocessor to disk.
    """
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'validation'
    test_dir = output_dir / 'test'

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    splits_proc['X_train'].to_csv(train_dir / 'X_train.csv', index=False)
    splits_proc['y_train'].to_csv(train_dir / 'y_train.csv', index=False)

    splits_proc['X_val'].to_csv(val_dir / 'X_val.csv', index=False)
    splits_proc['y_val'].to_csv(val_dir / 'y_val.csv', index=False)

    splits_proc['X_test'].to_csv(test_dir / 'X_test.csv', index=False)
    splits_proc['y_test'].to_csv(test_dir / 'y_test.csv', index=False)

    # Save serialized preprocessor
    joblib.dump(preprocessor, output_dir / 'preprocessor.joblib')


def load_client_preprocessor(joblib_path: Path) -> ClientPreprocessor:
    """
    Loads a saved client preprocessor object for inference or validation.
    """
    if not joblib_path.exists():
        raise FileNotFoundError(f"Preprocessor object not found at: {joblib_path}")
    return joblib.load(joblib_path)
