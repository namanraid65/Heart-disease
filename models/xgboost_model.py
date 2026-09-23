"""
Local XGBoost Model Wrapper for Tabular Heart Disease Prediction
Encapsulates XGBClassifier with class weighting, early stopping, probability generation,
and JSON serialization.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Optional
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from models.config import RANDOM_SEED, XGBOOST_PARAMS


class LocalXGBoostModel:
    """
    Wrapper for XGBoost binary classifier on tabular clinical client data.
    """

    def __init__(
        self,
        n_estimators: int = XGBOOST_PARAMS['n_estimators'],
        max_depth: int = XGBOOST_PARAMS['max_depth'],
        learning_rate: float = XGBOOST_PARAMS['learning_rate'],
        subsample: float = XGBOOST_PARAMS['subsample'],
        colsample_bytree: float = XGBOOST_PARAMS['colsample_bytree'],
        gamma: float = XGBOOST_PARAMS['gamma'],
        reg_alpha: float = XGBOOST_PARAMS['reg_alpha'],
        reg_lambda: float = XGBOOST_PARAMS['reg_lambda'],
        scale_pos_weight: float = 1.0,
        random_state: int = RANDOM_SEED,
        eval_metric: str = 'logloss',
        early_stopping_rounds: Optional[int] = 15
    ):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'eval_metric': eval_metric,
            'early_stopping_rounds': early_stopping_rounds
        }
        self.clf = XGBClassifier(**self.params)
        self.is_fitted = False
        self.best_iteration = 0

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        verbose: bool = False
    ) -> 'LocalXGBoostModel':
        """
        Fits the XGBoost classifier. If validation data is provided, utilizes early stopping.
        """
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            if self.params['early_stopping_rounds'] is not None:
                self.clf.set_params(early_stopping_rounds=self.params['early_stopping_rounds'])
        else:
            self.clf.set_params(early_stopping_rounds=None)

        self.clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        self.is_fitted = True
        self.best_iteration = getattr(self.clf, 'best_iteration', self.params['n_estimators'])
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns discrete binary predictions {0, 1}."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict().")
        return self.clf.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns prediction probabilities of shape (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return self.clf.predict_proba(X)

    def get_feature_importances(self, feature_names: Optional[list] = None) -> pd.Series:
        """Returns feature importances sorted in descending order."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before retrieving feature importances.")
        importances = self.clf.feature_importances_
        if feature_names is not None and len(feature_names) == len(importances):
            return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        return pd.Series(importances).sort_values(ascending=False)

    def save_model(self, file_path: Path) -> None:
        """Saves model to JSON format."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.clf.save_model(str(file_path))

    def load_model(self, file_path: Path) -> 'LocalXGBoostModel':
        """Loads model from JSON format."""
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found at: {file_path}")
        self.clf.load_model(str(file_path))
        self.is_fitted = True
        return self


def build_xgboost_model(
    scale_pos_weight: float = 1.0,
    **kwargs
) -> LocalXGBoostModel:
    """Factory function for LocalXGBoostModel."""
    params = dict(XGBOOST_PARAMS)
    params['scale_pos_weight'] = scale_pos_weight
    params.update(kwargs)
    return LocalXGBoostModel(**params)


if __name__ == '__main__':
    model = build_xgboost_model()
    X_dummy = np.random.randn(50, INPUT_FEATURES)
    y_dummy = np.random.randint(0, 2, size=50)
    model.fit(X_dummy, y_dummy)
    preds = model.predict(X_dummy)
    probs = model.predict_proba(X_dummy)
    print(f"XGBoost instantiated & tested successfully! Predictions shape: {preds.shape}, Probs shape: {probs.shape}")
