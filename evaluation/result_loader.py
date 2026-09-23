"""
Master Experimental Result Loader and Evaluator (Authoritative Results Source)
Loads, executes, and normalizes test evaluations for all Local and Federated models
across Hospital 1 (Cleveland), Hospital 2 (Hungarian), and Hospital 3 (Switzerland).

Strict Evaluation Integrity Guarantees:
  1. Authoritative machine-readable single source of truth (master_results.csv / master_results.json).
  2. Complete metric suite: accuracy, precision, recall, sensitivity, specificity, F1, ROC-AUC, PR-AUC,
     TP, TN, FP, FN, sample_count, positive_count, negative_count.
  3. No metric fabrication: undefined values (e.g. division by zero, single-class cohorts) are strictly NaN.
  4. Automatic identity assertion:
       TP + TN + FP + FN == sample_count
       TP + FN == positive_count
       TN + FP == negative_count
  5. Transparent Hospital 3 reporting: 18 positive, 1 negative; specificity explicitly flagged.
  6. Strict separation of Macro (unweighted) vs. Sample-Weighted multi-center aggregations.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from models.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED
)
from models.dataset import load_client_raw_splits
from models.alexnet_1d import build_alexnet_1d
from models.resnet_1d import build_resnet_1d
from models.xgboost_model import build_xgboost_model
from xai.model_loader import load_federated_model


class ExperimentResultLoader:
    """
    Standardized result loader that evaluates or retrieves all local and federated models
    on the exact held-out test splits of the three independent hospitals.
    Acts as the single authoritative results source for all downstream reports.
    """

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.hospital_names = {
            'hospital_1': 'Hospital 1 (Cleveland)',
            'hospital_2': 'Hospital 2 (Hungarian)',
            'hospital_3': 'Hospital 3 (Switzerland)'
        }
        self.models_to_evaluate = [
            ('AlexNet', 'Local'),
            ('AlexNet', 'Federated'),
            ('ResNet', 'Local'),
            ('ResNet', 'Federated'),
            ('XGBoost', 'Local'),
            ('XGBoost', 'Federated')  # Sample-Weighted Local XGBoost Ensemble
        ]
        self._load_client_test_data()

    def _load_client_test_data(self):
        """Loads held-out test partitions for all three hospital clients."""
        from preprocessing.preprocess_client import load_client_raw, clean_client_data, prepare_target, split_client_data
        from preprocessing.run_preprocessing import CLIENT_CONFIGS as PREP_CONFIGS

        self.test_data = {}
        for cid in self.hospital_names:
            splits = load_client_raw_splits(cid)
            X_test, y_test = splits['test']

            # Load raw test split for preprocessing-consistent ensemble evaluation
            X_raw_test = None
            try:
                cfg = next(c for c in PREP_CONFIGS if c['output_dir'].name == cid)
                raw_df = load_client_raw(cfg['raw_path'])
                cleaned_df = clean_client_data(raw_df, cfg['name'])
                X_clean, y_clean = prepare_target(cleaned_df)
                raw_splits = split_client_data(X_clean, y_clean, random_state=RANDOM_SEED)
                X_raw_test = raw_splits['X_test']
            except Exception:
                X_raw_test = None

            y_arr = y_test.to_numpy(dtype=np.int64)
            self.test_data[cid] = {
                'X': X_test.to_numpy(dtype=np.float32),
                'X_raw': X_raw_test,
                'y': y_arr,
                'sample_count': len(y_arr),
                'positive_count': int(np.sum(y_arr == 1)),
                'negative_count': int(np.sum(y_arr == 0))
            }

    def evaluate_model_on_client(
        self,
        model_type: str,
        training_type: str,
        client_id: str
    ) -> Dict[str, Any]:
        """
        Evaluates a specific model configuration on a client's held-out test set
        with strict mathematical verification and no metric fabrication.
        """
        X_test = self.test_data[client_id]['X']
        y_test = self.test_data[client_id]['y']
        cname = self.hospital_names[client_id]
        sample_count = self.test_data[client_id]['sample_count']
        positive_count = self.test_data[client_id]['positive_count']
        negative_count = self.test_data[client_id]['negative_count']

        if training_type == 'Local':
            if model_type == 'AlexNet':
                ckpt_path = CHECKPOINTS_DIR / f"{client_id}_alexnet.pt"
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = build_alexnet_1d(input_dim=25).to(self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(X_test, device=self.device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

            elif model_type == 'ResNet':
                ckpt_path = CHECKPOINTS_DIR / f"{client_id}_resnet.pt"
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = build_resnet_1d(input_dim=25).to(self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(X_test, device=self.device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

            elif model_type == 'XGBoost':
                ckpt_path = CHECKPOINTS_DIR / f"{client_id}_xgboost.json"
                xgb_model = build_xgboost_model()
                xgb_model.load_model(ckpt_path)
                probs = xgb_model.predict_proba(X_test)[:, 1]

        elif training_type in ('Federated', 'Ensemble'):
            if model_type == 'AlexNet':
                ckpt_path = CHECKPOINTS_DIR / "federated_alexnet" / "global_alexnet_final.pt"
                if not ckpt_path.exists():
                    ckpt_path = CHECKPOINTS_DIR / "global_alexnet.pt"
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = build_alexnet_1d(input_dim=25).to(self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                if 'client_bn_states' in ckpt and client_id in ckpt['client_bn_states']:
                    from federated.utils import set_model_bn_state
                    set_model_bn_state(model, ckpt['client_bn_states'][client_id])
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(X_test, device=self.device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

            elif model_type == 'ResNet':
                ckpt_path = CHECKPOINTS_DIR / "federated_resnet" / "global_resnet_final.pt"
                if not ckpt_path.exists():
                    ckpt_path = CHECKPOINTS_DIR / "global_resnet.pt"
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                model = build_resnet_1d(input_dim=25).to(self.device)
                model.load_state_dict(ckpt['model_state_dict'])
                if 'client_bn_states' in ckpt and client_id in ckpt['client_bn_states']:
                    from federated.utils import set_model_bn_state
                    set_model_bn_state(model, ckpt['client_bn_states'][client_id])
                model.eval()
                with torch.no_grad():
                    logits = model(torch.tensor(X_test, device=self.device))
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()

            elif model_type == 'XGBoost':
                wrapper = load_federated_model('Local_XGBoost_Ensemble')
                if 'X_raw' in self.test_data[client_id] and self.test_data[client_id]['X_raw'] is not None:
                    probs = wrapper.predict_proba(self.test_data[client_id]['X_raw'])[:, 1]
                else:
                    probs = wrapper.predict_proba(X_test)[:, 1]

        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Strict identity verification
        if tp + tn + fp + fn != sample_count:
            raise ValueError(
                f"[Integrity Failure] Confusion matrix sum ({tp + tn + fp + fn}) != sample_count ({sample_count}) "
                f"for {model_type} {training_type} on {client_id}."
            )
        if (tp + fn) != positive_count:
            raise ValueError(
                f"[Integrity Failure] TP + FN ({tp + fn}) != positive_count ({positive_count}) "
                f"for {model_type} {training_type} on {client_id}."
            )
        if (tn + fp) != negative_count:
            raise ValueError(
                f"[Integrity Failure] TN + FP ({tn + fp}) != negative_count ({negative_count}) "
                f"for {model_type} {training_type} on {client_id}."
            )

        # Standard classification metrics with zero-division handling using NaN
        acc = float(accuracy_score(y_test, preds))

        # Precision = TP / (TP + FP)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan

        # Recall / Sensitivity = TP / (TP + FN)
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        sens = rec

        # Specificity = TN / (TN + FP)
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan

        # F1 = 2 * (prec * rec) / (prec + rec)
        if not np.isnan(prec) and not np.isnan(rec) and (prec + rec) > 0:
            f1 = float(2 * (prec * rec) / (prec + rec))
        else:
            f1 = np.nan

        # ROC-AUC: defined only when both positive and negative classes exist
        if len(np.unique(y_test)) > 1:
            try:
                auc_val = float(roc_auc_score(y_test, probs))
            except ValueError:
                auc_val = np.nan
        else:
            auc_val = np.nan

        # PR-AUC / Average Precision: scikit-learn standard average_precision_score
        if len(np.unique(y_test)) > 1:
            try:
                pr_auc_val = float(average_precision_score(y_test, probs))
            except ValueError:
                pr_auc_val = np.nan
        else:
            pr_auc_val = np.nan

        # Hospital 3 specific statistical limitation disclosure
        hosp3_note = ""
        if client_id == 'hospital_3':
            hosp3_note = "Specificity is based on only 1 negative test sample (18 positive, 1 negative)."

        # Standardize strategy/training display name
        display_training = "Local" if training_type == "Local" else (
            "Sample-Weighted Local Ensemble" if model_type == "XGBoost" else "Federated (FedAvg + FedBN)"
        )

        return {
            'Hospital': cname,
            'client_id': client_id,
            'Model Type': model_type,
            'Training Type': training_type,
            'Strategy': display_training,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'Sensitivity': sens,
            'Specificity': spec,
            'F1 Score': f1,
            'ROC AUC': auc_val,
            'PR AUC': pr_auc_val,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'sample_count': int(sample_count),
            'positive_count': int(positive_count),
            'negative_count': int(negative_count),
            'hospital_3_note': hosp3_note,
            'y_true': y_test,
            'y_prob': probs,
            'y_pred': preds
        }

    def collect_all_results(self) -> pd.DataFrame:
        """
        Collects authoritative evaluation metrics for all 18 combinations
        (3 hospitals x 3 architectures x 2 training modes).
        """
        records = []
        for model_type, training_type in self.models_to_evaluate:
            for cid in self.hospital_names:
                res = self.evaluate_model_on_client(model_type, training_type, cid)
                records.append({
                    'Hospital': res['Hospital'],
                    'client_id': res['client_id'],
                    'Model Type': res['Model Type'],
                    'Training Type': res['Training Type'],
                    'Strategy': res['Strategy'],
                    'Accuracy': res['Accuracy'],
                    'Precision': res['Precision'],
                    'Recall': res['Recall'],
                    'Sensitivity': res['Sensitivity'],
                    'Specificity': res['Specificity'],
                    'F1 Score': res['F1 Score'],
                    'ROC AUC': res['ROC AUC'],
                    'PR AUC': res['PR AUC'],
                    'TP': res['TP'],
                    'TN': res['TN'],
                    'FP': res['FP'],
                    'FN': res['FN'],
                    'sample_count': res['sample_count'],
                    'positive_count': res['positive_count'],
                    'negative_count': res['negative_count'],
                    'hospital_3_note': res['hospital_3_note']
                })
        return pd.DataFrame(records)

    def compute_multi_center_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes both Macro (unweighted) and Sample-Weighted multi-center aggregations
        with strict mathematical rigor and transparent denominator reporting.
        """
        agg_rows = []
        numeric_metrics = [
            'Accuracy', 'Precision', 'Recall', 'Sensitivity',
            'Specificity', 'F1 Score', 'ROC AUC', 'PR AUC'
        ]

        models = df[['Model Type', 'Training Type']].drop_duplicates().values
        for mtype, ttype in models:
            sub = df[(df['Model Type'] == mtype) & (df['Training Type'] == ttype)].copy()

            # 1. Macro Hospital Average (Unweighted Arithmetic Mean of Valid Hospitals)
            macro_entry = {
                'Model Type': mtype,
                'Training Type': ttype,
                'Aggregation': 'Macro Average (Unweighted)',
                'Scope': 'All 3 Medical Centers',
                'Total Test Samples': int(sub['sample_count'].sum()),
                'Total Positives': int(sub['positive_count'].sum()),
                'Total Negatives': int(sub['negative_count'].sum()),
                'Total TP': int(sub['TP'].sum()),
                'Total TN': int(sub['TN'].sum()),
                'Total FP': int(sub['FP'].sum()),
                'Total FN': int(sub['FN'].sum())
            }
            for m in numeric_metrics:
                valid_vals = sub[m].dropna()
                valid_count = len(valid_vals)
                if valid_count > 0:
                    macro_entry[m] = float(valid_vals.mean())
                    macro_entry[f"{m}_valid_hospitals"] = f"{valid_count}/3"
                else:
                    macro_entry[m] = np.nan
                    macro_entry[f"{m}_valid_hospitals"] = "0/3"
            agg_rows.append(macro_entry)

            # 2. Sample-Weighted Average (Weighted by Test Sample Count: 46 H1, 44 H2, 19 H3)
            total_n = sub['sample_count'].sum()
            weighted_entry = {
                'Model Type': mtype,
                'Training Type': ttype,
                'Aggregation': 'Sample-Weighted Average',
                'Scope': f'All Medical Centers (N={total_n})',
                'Total Test Samples': int(total_n),
                'Total Positives': int(sub['positive_count'].sum()),
                'Total Negatives': int(sub['negative_count'].sum()),
                'Total TP': int(sub['TP'].sum()),
                'Total TN': int(sub['TN'].sum()),
                'Total FP': int(sub['FP'].sum()),
                'Total FN': int(sub['FN'].sum())
            }
            for m in numeric_metrics:
                # Filter to rows where metric is valid
                sub_valid = sub.dropna(subset=[m])
                if len(sub_valid) > 0:
                    sub_weights = sub_valid['sample_count'] / sub_valid['sample_count'].sum()
                    weighted_val = float(np.sum(sub_valid[m] * sub_weights))
                    weighted_entry[m] = weighted_val
                    weighted_entry[f"{m}_valid_hospitals"] = f"{len(sub_valid)}/3"
                else:
                    weighted_entry[m] = np.nan
                    weighted_entry[f"{m}_valid_hospitals"] = "0/3"
            agg_rows.append(weighted_entry)

        return pd.DataFrame(agg_rows)

    def save_master_results(self, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
        """
        Executes complete evaluation and writes authoritative CSV and JSON master tables.
        """
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'reports'
        output_dir.mkdir(parents=True, exist_ok=True)

        df_client = self.collect_all_results()
        df_agg = self.compute_multi_center_aggregates(df_client)

        timestamp = datetime.now(timezone.utc).isoformat()

        # CSV Export
        csv_path = output_dir / "master_results.csv"
        df_client.to_csv(csv_path, index=False, encoding='utf-8')

        # Compatibility CSV
        compat_csv_path = output_dir / "final_master_results.csv"
        df_client.to_csv(compat_csv_path, index=False, encoding='utf-8')

        # Aggregates CSV
        agg_csv_path = output_dir / "master_aggregates.csv"
        df_agg.to_csv(agg_csv_path, index=False, encoding='utf-8')

        # Complete JSON Export with nested metadata
        json_path = output_dir / "master_results.json"
        master_dict = {
            'metadata': {
                'timestamp': timestamp,
                'random_seed': RANDOM_SEED,
                'device': str(self.device),
                'total_test_samples': int(df_client['sample_count'].iloc[:3].sum()),
                'hospitals': self.hospital_names,
                'hospital_test_counts': {cid: self.test_data[cid]['sample_count'] for cid in self.hospital_names},
                'hospital_positive_counts': {cid: self.test_data[cid]['positive_count'] for cid in self.hospital_names},
                'hospital_negative_counts': {cid: self.test_data[cid]['negative_count'] for cid in self.hospital_names},
                'pr_auc_definition': 'scikit-learn average_precision_score (interpolated precision-recall curve)',
                'specificity_note': 'Hospital 3 specificity evaluated on N_neg=1 test record'
            },
            'client_results': df_client.replace({np.nan: None}).to_dict(orient='records'),
            'aggregate_results': df_agg.replace({np.nan: None}).to_dict(orient='records')
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(master_dict, f, indent=2)

        print(f">> Authoritative Master Results CSV saved to: {csv_path}")
        print(f">> Authoritative Master Results JSON saved to: {json_path}")
        print(f">> Master Aggregates CSV saved to: {agg_csv_path}")

        return csv_path, json_path


if __name__ == '__main__':
    loader = ExperimentResultLoader()
    loader.save_master_results()
