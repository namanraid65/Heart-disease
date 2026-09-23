"""
Dataset Inspection and Analysis Script
Federated Learning for Heart Disease Prediction
Simulated Hospital Clients:
  - Hospital 1: Cleveland Dataset (processed.cleveland.data)
  - Hospital 2: Hungarian Dataset (processed.hungarian.data)
  - Hospital 3: Switzerland Dataset (processed.switzerland.data)

This script loads each hospital dataset separately, computes descriptive statistics,
identifies missing/invalid values, examines target distributions, and analyzes
non-IID characteristics across the clients.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Standard UCI Heart Disease 14 attributes
FEATURE_NAMES = [
    'age',       # Age in years
    'sex',       # Sex (1 = male, 0 = female)
    'cp',        # Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal, 4: asymptomatic)
    'trestbps',  # Resting blood pressure (mm Hg)
    'chol',      # Serum cholesterol (mg/dl)
    'fbs',       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    'restecg',   # Resting ECG (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)
    'thalach',   # Maximum heart rate achieved
    'exang',     # Exercise induced angina (1 = yes, 0 = no)
    'oldpeak',   # ST depression induced by exercise relative to rest
    'slope',     # Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
    'ca',        # Number of major vessels colored by fluoroscopy (0-3)
    'thal',      # Thalassemia (3: normal, 6: fixed defect, 7: reversible defect)
    'num'        # Target: angiographic disease status (0: <50% stenosis, 1-4: >50% stenosis)
]

CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
TARGET_COL = 'num'

DATASET_PATHS = {
    'Hospital 1 (Cleveland)': Path('dataset/hospital_1/processed.cleveland.data'),
    'Hospital 2 (Hungarian)': Path('dataset/hospital_2/processed.hungarian.data'),
    'Hospital 3 (Switzerland)': Path('dataset/hospital_3/processed.switzerland.data'),
}


def load_hospital_dataset(file_path: Path) -> pd.DataFrame:
    """
    Loads a single hospital dataset with standard UCI 14-attribute column names.
    Treats '?' as NaN. Does NOT modify the underlying file.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    # Read CSV with '?' as missing values
    df = pd.read_csv(
        file_path,
        header=None,
        names=FEATURE_NAMES,
        na_values='?',
        sep=',',
        skipinitialspace=True
    )
    return df


def analyze_single_client(name: str, df: pd.DataFrame) -> dict:
    """
    Performs comprehensive single-client analysis.
    """
    total_records, total_cols = df.shape
    duplicates = df.duplicated().sum()
    
    # Missing values
    missing_counts = df.isnull().sum()
    missing_pcts = (missing_counts / total_records) * 100
    
    # Target analysis
    target_raw_counts = df[TARGET_COL].value_counts().sort_index().to_dict()
    # Binary target: 0 = No disease, >=1 = Disease
    binary_target = (df[TARGET_COL] > 0).astype(int)
    binary_counts = binary_target.value_counts().to_dict()
    
    # Categorical unique values (ignoring NaN)
    cat_uniques = {}
    for col in CATEGORICAL_FEATURES:
        unique_vals = sorted([v for v in df[col].dropna().unique().tolist()])
        cat_uniques[col] = unique_vals
    
    # Continuous feature statistics
    cont_stats = df[CONTINUOUS_FEATURES].describe().to_dict()
    
    # Check potential anomalous/invalid values
    anomalies = []
    if (df['chol'] == 0).sum() > 0:
        anomalies.append(f"chol has {(df['chol'] == 0).sum()} zero-values (biologically implausible, represents missing data)")
    if (df['trestbps'] == 0).sum() > 0:
        anomalies.append(f"trestbps has {(df['trestbps'] == 0).sum()} zero-values")
    if (df['oldpeak'] < 0).sum() > 0:
        anomalies.append(f"oldpeak has {(df['oldpeak'] < 0).sum()} negative values (ST depression is conventionally non-negative, but negative values represent ST elevation)")
    
    # Specific categorical bounds check
    for col in CATEGORICAL_FEATURES:
        vals = df[col].dropna()
        if col == 'sex' and not vals.isin([0, 1]).all():
            anomalies.append(f"sex has unexpected values: {vals[~vals.isin([0, 1])].unique()}")
        elif col == 'cp' and not vals.isin([1, 2, 3, 4]).all():
            anomalies.append(f"cp has unexpected values: {vals[~vals.isin([1, 2, 3, 4])].unique()}")
        elif col == 'fbs' and not vals.isin([0, 1]).all():
            anomalies.append(f"fbs has unexpected values: {vals[~vals.isin([0, 1])].unique()}")
        elif col == 'restecg' and not vals.isin([0, 1, 2]).all():
            anomalies.append(f"restecg has unexpected values: {vals[~vals.isin([0, 1, 2])].unique()}")
        elif col == 'exang' and not vals.isin([0, 1]).all():
            anomalies.append(f"exang has unexpected values: {vals[~vals.isin([0, 1])].unique()}")
        elif col == 'slope' and not vals.isin([1, 2, 3]).all():
            anomalies.append(f"slope has unexpected values: {vals[~vals.isin([1, 2, 3])].unique()}")
        elif col == 'ca' and not vals.isin([0, 1, 2, 3]).all():
            anomalies.append(f"ca has unexpected values: {vals[~vals.isin([0, 1, 2, 3])].unique()}")
        elif col == 'thal' and not vals.isin([3, 6, 7]).all():
            anomalies.append(f"thal has unexpected values: {vals[~vals.isin([3, 6, 7])].unique()}")

    return {
        'name': name,
        'records': total_records,
        'columns': total_cols,
        'duplicates': duplicates,
        'missing_counts': missing_counts,
        'missing_pcts': missing_pcts,
        'target_raw_counts': target_raw_counts,
        'binary_counts': binary_counts,
        'cat_uniques': cat_uniques,
        'cont_stats': cont_stats,
        'anomalies': anomalies,
        'dtypes': df.dtypes.to_dict()
    }


def print_client_summary(stats: dict, df: pd.DataFrame):
    """
    Prints a detailed readable summary for one client.
    """
    print("=" * 80)
    print(f" CLIENT ANALYSIS: {stats['name']}")
    print("=" * 80)
    print(f"Total Records: {stats['records']}")
    print(f"Total Columns: {stats['columns']}")
    print(f"Duplicate Rows: {stats['duplicates']}")
    print("\n--- Missing Values Per Column ---")
    for col in FEATURE_NAMES:
        c = stats['missing_counts'][col]
        pct = stats['missing_pcts'][col]
        print(f"  {col:<10}: {c:>3} missing ({pct:>6.2f}%) | dtype: {str(df[col].dtype):<10}")
    
    print("\n--- Categorical Unique Values (Observed) ---")
    for col, vals in stats['cat_uniques'].items():
        print(f"  {col:<10}: {vals}")
        
    print("\n--- Target Distribution (num: 0=Absence, 1-4=Presence Severity) ---")
    for val, count in stats['target_raw_counts'].items():
        pct = (count / stats['records']) * 100
        print(f"  Class {int(val)}: {count:>3} ({pct:>5.1f}%)")
    
    no_disease = stats['binary_counts'].get(0, 0)
    has_disease = stats['binary_counts'].get(1, 0)
    print(f"  Binary: No Disease (0) = {no_disease} ({no_disease/stats['records']*100:.1f}%), Disease (1) = {has_disease} ({has_disease/stats['records']*100:.1f}%)")
    
    print("\n--- Continuous Features Descriptive Statistics ---")
    stats_df = pd.DataFrame(stats['cont_stats']).T[['mean', 'std', 'min', '50%', 'max']]
    stats_df.columns = ['Mean', 'Std', 'Min', 'Median', 'Max']
    print(stats_df.round(2).to_string())

    if stats['anomalies']:
        print("\n--- Notable Data Observations / Potential Anomalies ---")
        for a in stats['anomalies']:
            print(f"  * {a}")
    print("\n")


def compare_clients(results: dict, dfs: dict):
    """
    Compares all three hospital clients to evaluate heterogeneity and non-IID properties.
    """
    print("=" * 80)
    print(" FEDERATED CLIENT COMPARISON & HETEROGENEITY (NON-IID) ANALYSIS")
    print("=" * 80)
    
    # 1. Sample Size & Balance
    print("\n1. Sample Size & Class Distribution Across Clients:")
    header = f"{'Client':<28} | {'Samples':<8} | {'No Disease (0)':<16} | {'Disease (1)':<16} | {'Disease %':<10}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        n = r['records']
        no_d = r['binary_counts'].get(0, 0)
        has_d = r['binary_counts'].get(1, 0)
        pct_d = (has_d / n) * 100
        print(f"{name:<28} | {n:<8} | {no_d:<5} ({no_d/n*100:>4.1f}%)     | {has_d:<5} ({has_d/n*100:>4.1f}%)     | {pct_d:>6.1f}%")
        
    # 2. Missing Value Rates Comparison
    print("\n2. Missing Value Rate Comparison (%) Across Clients:")
    missing_table = pd.DataFrame({
        name: r['missing_pcts'] for name, r in results.items()
    })
    print(missing_table.round(2).to_string())

    # 3. Continuous Feature Means Comparison
    print("\n3. Continuous Feature Mean +/- Std Comparison:")
    means_comparison = {}
    for name, r in results.items():
        col_summary = {}
        for col in CONTINUOUS_FEATURES:
            m = r['cont_stats'][col]['mean']
            s = r['cont_stats'][col]['std']
            col_summary[col] = f"{m:.1f} +/- {s:.1f}" if not np.isnan(m) else "N/A"
        means_comparison[name] = col_summary
    print(pd.DataFrame(means_comparison).to_string())
    
    # 4. Sex Distribution Comparison
    print("\n4. Sex Ratio Comparison (Male % vs Female %):")
    for name, df in dfs.items():
        sex_counts = df['sex'].value_counts(dropna=False)
        m_pct = (sex_counts.get(1.0, 0) / len(df)) * 100
        f_pct = (sex_counts.get(0.0, 0) / len(df)) * 100
        print(f"  {name:<28}: Male = {m_pct:.1f}%, Female = {f_pct:.1f}%")


def main():
    dfs = {}
    analysis_results = {}
    
    for name, path in DATASET_PATHS.items():
        df = load_hospital_dataset(path)
        dfs[name] = df
        res = analyze_single_client(name, df)
        analysis_results[name] = res
        print_client_summary(res, df)
        
    compare_clients(analysis_results, dfs)


if __name__ == '__main__':
    main()
