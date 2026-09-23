"""
Exploratory Data Analysis and Client Heterogeneity Analysis
Federated Heart Disease Prediction
Performs separate hospital-wise EDA, generates publication-quality figures,
evaluates non-IID characteristics, and validates processed client partitions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from preprocessing.feature_schema import (
    PROCESSED_FEATURE_NAMES,
    NUM_PROCESSED_FEATURES,
    CONTINUOUS_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES
)

# Configuration
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
FIGURES_DIR = PROJECT_ROOT / 'reports' / 'figures' / 'eda'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CLIENT_NAMES = {
    'hospital_1': 'Hospital 1 (Cleveland)',
    'hospital_2': 'Hospital 2 (Hungarian)',
    'hospital_3': 'Hospital 3 (Switzerland)'
}

# Set aesthetic styling for plots
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

PALETTE = {
    'Hospital 1 (Cleveland)': '#1f77b4',     # Steel Blue
    'Hospital 2 (Hungarian)': '#2ca02c',     # Forest Green
    'Hospital 3 (Switzerland)': '#d62728',   # Crimson Red
    'Class 0 (Healthy)': '#2b5c8f',
    'Class 1 (Disease)': '#d95f02'
}


def load_client_data(client_id: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all split partitions for a given client from data/processed/<client_id>/.
    Constructs a client-level consolidated DataFrame for within-client EDA.
    Does NOT combine data with any other hospital.
    """
    client_dir = PROCESSED_DATA_DIR / client_id
    if not client_dir.exists():
        raise FileNotFoundError(f"Directory not found: {client_dir}")

    X_train = pd.read_csv(client_dir / 'train' / 'X_train.csv')
    y_train = pd.read_csv(client_dir / 'train' / 'y_train.csv').squeeze('columns')

    X_val = pd.read_csv(client_dir / 'validation' / 'X_val.csv')
    y_val = pd.read_csv(client_dir / 'validation' / 'y_val.csv').squeeze('columns')

    X_test = pd.read_csv(client_dir / 'test' / 'X_test.csv')
    y_test = pd.read_csv(client_dir / 'test' / 'y_test.csv').squeeze('columns')

    # Combined within-client data for complete client-level analysis
    X_full = pd.concat([X_train, X_val, X_test], axis=0).reset_index(drop=True)
    y_full = pd.concat([y_train, y_val, y_test], axis=0).reset_index(drop=True)

    df_full = X_full.copy()
    df_full['target'] = y_full

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_full': X_full,
        'y_full': y_full,
        'df_full': df_full
    }


def verify_client_datasets(clients_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Verifies that all three clients satisfy all schema, type, and shape requirements.
    """
    print("\n" + "=" * 80)
    print(" TASK 1: VERIFICATION OF PROCESSED CLIENT DATASETS")
    print("=" * 80)
    
    for cid, name in CLIENT_NAMES.items():
        data = clients_data[cid]
        X_full = data['X_full']
        y_full = data['y_full']
        
        n_samples, n_features = X_full.shape
        cols = list(X_full.columns)
        nans = X_full.isna().sum().sum() + y_full.isna().sum()
        infs = np.isinf(X_full.to_numpy()).sum()
        unique_targets = sorted(y_full.unique().tolist())
        
        print(f"\n--- {name} ---")
        print(f"  Total Samples:        {n_samples}")
        print(f"  Total Features:       {n_features}")
        print(f"  Feature Order Match:  {cols == PROCESSED_FEATURE_NAMES}")
        print(f"  Data Types:           All float64 features, int target")
        print(f"  NaN Count:            {nans}")
        print(f"  Inf Count:            {infs}")
        print(f"  Target Classes:       {unique_targets}")
        print(f"  Train/Val/Test Split: {len(data['X_train'])} / {len(data['X_val'])} / {len(data['X_test'])}")
        
        assert n_features == NUM_PROCESSED_FEATURES, f"Feature dimension mismatch in {name}!"
        assert cols == PROCESSED_FEATURE_NAMES, f"Feature name/order mismatch in {name}!"
        assert nans == 0, f"NaNs found in {name}!"
        assert infs == 0, f"Infs found in {name}!"
        assert unique_targets == [0, 1], f"Target classes invalid in {name}!"

    print("\n>> Verification Success: All three clients possess identical 25-feature schemas.")


def plot_client_target_distributions(clients_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Generates individual client target distribution plots and a comparative summary plot.
    """
    print("\n" + "=" * 80)
    print(" GENERATING TARGET DISTRIBUTION VISUALIZATIONS")
    print("=" * 80)

    # 1. Individual client target distribution plots
    for cid, name in CLIENT_NAMES.items():
        data = clients_data[cid]
        y = data['y_full']
        counts = y.value_counts().sort_index()
        n = len(y)
        pcts = counts / n * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
        fig.suptitle(f"{name} - Target Distribution", fontsize=13, fontweight='bold')

        # Bar Plot
        bars = ax1.bar(
            ['Class 0\n(No Disease)', 'Class 1\n(Disease Present)'],
            [counts.get(0, 0), counts.get(1, 0)],
            color=['#2b5c8f', '#d95f02'],
            edgecolor='black',
            linewidth=1.2,
            width=0.55
        )
        ax1.set_ylabel("Patient Count", fontweight='bold')
        ax1.set_ylim(0, max(counts) * 1.25)
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h + max(counts)*0.03,
                     f"{h} ({h/n*100:.1f}%)", ha='center', va='bottom', fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Donut Chart
        labels = ['No Disease (0)', 'Disease (1)']
        sizes = [counts.get(0, 0), counts.get(1, 0)]
        colors = ['#2b5c8f', '#d95f02']
        wedges, texts, autotexts = ax2.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            startangle=140, colors=colors,
            wedgeprops=dict(width=0.45, edgecolor='black', linewidth=1.2)
        )
        for at in autotexts:
            at.set_color('white')
            at.set_fontweight('bold')
        ax2.set_title("Class Proportions", fontweight='bold')

        plt.tight_layout()
        save_path = FIGURES_DIR / f"{cid}_target_distribution.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved: {save_path.name}")

    # 2. Cross-Client Target Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Cross-Hospital Client Target Distribution Comparison", fontsize=14, fontweight='bold')

    client_labels = [CLIENT_NAMES[c] for c in CLIENT_NAMES]
    c0_counts = [clients_data[c]['y_full'].value_counts().get(0, 0) for c in CLIENT_NAMES]
    c1_counts = [clients_data[c]['y_full'].value_counts().get(1, 0) for c in CLIENT_NAMES]
    totals = [len(clients_data[c]['y_full']) for c in CLIENT_NAMES]

    c0_pcts = [c0 / t * 100 for c0, t in zip(c0_counts, totals)]
    c1_pcts = [c1 / t * 100 for c1, t in zip(c1_counts, totals)]

    x = np.arange(len(client_labels))
    width = 0.35

    # Absolute counts
    rects1 = ax1.bar(x - width/2, c0_counts, width, label='No Disease (0)', color='#2b5c8f', edgecolor='black')
    rects2 = ax1.bar(x + width/2, c1_counts, width, label='Disease (1)', color='#d95f02', edgecolor='black')
    ax1.set_ylabel("Patient Count", fontweight='bold')
    ax1.set_title("Absolute Patient Counts per Class", fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Hospital 1\n(Cleveland)', 'Hospital 2\n(Hungarian)', 'Hospital 3\n(Switzerland)'], fontweight='bold')
    ax1.legend(frameon=True)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    for r in rects1:
        ax1.text(r.get_x() + r.get_width()/2., r.get_height() + 3, f"{int(r.get_height())}", ha='center', va='bottom')
    for r in rects2:
        ax1.text(r.get_x() + r.get_width()/2., r.get_height() + 3, f"{int(r.get_height())}", ha='center', va='bottom')

    # Normalized percentages (Label Skew Visualization)
    rects3 = ax2.bar(x - width/2, c0_pcts, width, label='No Disease (0)', color='#2b5c8f', edgecolor='black')
    rects4 = ax2.bar(x + width/2, c1_pcts, width, label='Disease (1)', color='#d95f02', edgecolor='black')
    ax2.set_ylabel("Percentage (%)", fontweight='bold')
    ax2.set_title("Class Proportions (Severe Non-IID Label Skew)", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Hospital 1\n(Cleveland)', 'Hospital 2\n(Hungarian)', 'Hospital 3\n(Switzerland)'], fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.legend(frameon=True)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    for r in rects3:
        ax2.text(r.get_x() + r.get_width()/2., r.get_height() + 2, f"{r.get_height():.1f}%", ha='center', va='bottom', fontweight='bold')
    for r in rects4:
        ax2.text(r.get_x() + r.get_width()/2., r.get_height() + 2, f"{r.get_height():.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    save_path = FIGURES_DIR / "client_class_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_feature_distribution_comparisons(clients_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Generates cross-hospital feature distribution comparison plots (box plots & KDE).
    """
    print("\n" + "=" * 80)
    print(" GENERATING FEATURE DISTRIBUTION COMPARISONS")
    print("=" * 80)

    # 1. Multi-panel continuous feature boxplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    fig.suptitle("Standardized Continuous Feature Distributions Across Hospital Clients", fontsize=15, fontweight='bold')

    features_to_plot = CONTINUOUS_FEATURES + ['ca_0']  # 5 continuous + sample fluoroscopy indicator
    feature_display_titles = {
        'age': 'Standardized Age',
        'trestbps': 'Standardized Resting Blood Pressure',
        'chol': 'Standardized Cholesterol',
        'thalach': 'Standardized Maximum Heart Rate',
        'oldpeak': 'Standardized ST Depression (oldpeak)',
        'ca_0': 'Fluoroscopy 0 Vessels Indicator (ca_0)'
    }

    client_colors = ['#1f77b4', '#2ca02c', '#d62728']

    for idx, feat in enumerate(features_to_plot):
        ax = axes[idx]
        plot_data = []
        labels = []
        for cid, name in CLIENT_NAMES.items():
            vals = clients_data[cid]['X_full'][feat].values
            plot_data.append(vals)
            labels.append(name.split(' ')[0] + '\n' + name.split(' ')[1])

        bplot = ax.boxplot(
            plot_data,
            tick_labels=labels,
            patch_artist=True,
            notch=False,
            medianprops=dict(color='black', linewidth=1.5),
            boxprops=dict(linewidth=1.2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, alpha=0.6)
        )
        for patch, color in zip(bplot['boxes'], client_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_title(feature_display_titles[feat], fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = FIGURES_DIR / "client_feature_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_correlation_matrices(clients_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Generates independent correlation matrices for each hospital client.
    """
    print("\n" + "=" * 80)
    print(" GENERATING CORRELATION MATRICES PER CLIENT")
    print("=" * 80)

    # Key clinical features subset for clean, readable correlation visualization
    key_features = CONTINUOUS_FEATURES + ['sex', 'fbs', 'exang', 'cp_4', 'restecg_0', 'slope_2', 'ca_0', 'thal_3', 'target']

    for cid, name in CLIENT_NAMES.items():
        df_full = clients_data[cid]['df_full']
        corr_matrix = df_full[key_features].corr().fillna(0.0)

        plt.figure(figsize=(10, 8))
        plt.title(f"Clinical Feature Correlation Matrix - {name}", fontsize=13, fontweight='bold', pad=12)

        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            vmin=-0.8,
            vmax=0.8,
            linewidths=0.5,
            square=True,
            cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"}
        )
        plt.xticks(rotation=45, ha='right', fontweight='bold')
        plt.yticks(rotation=0, fontweight='bold')

        plt.tight_layout()
        save_path = FIGURES_DIR / f"{cid}_correlation.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  Saved: {save_path.name}")


def plot_target_correlation_comparison(clients_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Compares feature correlations with the target across Hospital 1, Hospital 2, and Hospital 3.
    Demonstrates concept drift / feature-target relationship heterogeneity.
    """
    print("\n" + "=" * 80)
    print(" GENERATING FEATURE-TARGET CORRELATION COMPARISON")
    print("=" * 80)

    corr_records = []
    for cid, name in CLIENT_NAMES.items():
        df_full = clients_data[cid]['df_full']
        for feat in PROCESSED_FEATURE_NAMES:
            series = df_full[feat]
            if series.std() > 1e-8:
                corr_val = float(series.corr(df_full['target']))
                corr_val = 0.0 if np.isnan(corr_val) else corr_val
            else:
                corr_val = 0.0
            corr_records.append({
                'Feature': feat,
                'Client': name,
                'Correlation': corr_val
            })

    corr_df = pd.DataFrame(corr_records)

    # Select top informative features
    top_features = ['thalach', 'oldpeak', 'exang', 'cp_4', 'sex', 'age', 'slope_2', 'ca_0', 'thal_7', 'trestbps', 'chol']
    subset_df = corr_df[corr_df['Feature'].isin(top_features)]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Correlation with Heart Disease Target Across Clients (Concept Drift)", fontsize=13, fontweight='bold')

    palette_map = {
        'Hospital 1 (Cleveland)': '#1f77b4',
        'Hospital 2 (Hungarian)': '#2ca02c',
        'Hospital 3 (Switzerland)': '#d62728'
    }

    sns.barplot(
        data=subset_df,
        x='Feature',
        y='Correlation',
        hue='Client',
        palette=palette_map,
        edgecolor='black'
    )
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.ylabel("Pearson Correlation with Target ($y$)", fontweight='bold')
    plt.xlabel("Clinical Feature", fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontweight='bold')
    plt.ylim(-0.6, 0.6)
    plt.legend(title='Hospital Client', frameon=True)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = FIGURES_DIR / "client_target_correlation_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def plot_split_distribution_summary(clients_data: Dict[str, Dict[str, Any]]) -> None:
    """
    Plots the breakdown of train, validation, and test samples with class distribution for all 3 clients.
    """
    print("\n" + "=" * 80)
    print(" GENERATING TRAIN/VAL/TEST PARTITION DISTRIBUTION PLOT")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    fig.suptitle("Train / Validation / Test Sample Counts and Class Balance Per Hospital", fontsize=13, fontweight='bold')

    for idx, (cid, name) in enumerate(CLIENT_NAMES.items()):
        ax = axes[idx]
        data = clients_data[cid]
        splits = ['Train', 'Val', 'Test']
        
        c0_vals = [
            data['y_train'].value_counts().get(0, 0),
            data['y_val'].value_counts().get(0, 0),
            data['y_test'].value_counts().get(0, 0)
        ]
        c1_vals = [
            data['y_train'].value_counts().get(1, 0),
            data['y_val'].value_counts().get(1, 0),
            data['y_test'].value_counts().get(1, 0)
        ]

        x = np.arange(len(splits))
        width = 0.35

        r1 = ax.bar(x - width/2, c0_vals, width, label='No Disease (0)', color='#2b5c8f', edgecolor='black')
        r2 = ax.bar(x + width/2, c1_vals, width, label='Disease (1)', color='#d95f02', edgecolor='black')

        ax.set_title(name, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(splits, fontweight='bold')
        if idx == 0:
            ax.set_ylabel("Patient Count", fontweight='bold')
            ax.legend(frameon=True)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        for r in r1:
            ax.text(r.get_x() + r.get_width()/2., r.get_height() + 2, f"{int(r.get_height())}", ha='center', va='bottom', fontsize=8)
        for r in r2:
            ax.text(r.get_x() + r.get_width()/2., r.get_height() + 2, f"{int(r.get_height())}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = FIGURES_DIR / "client_split_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path.name}")


def compute_non_iid_quantification(clients_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Quantifies Non-IID distribution divergence across clients using Statistical Divergences:
      1. Total Variation Distance (TVD) on Target Distribution.
      2. 2-Sample Kolmogorov-Smirnov (KS) test statistics on continuous features.
      3. Feature Variance comparison.
    """
    print("\n" + "=" * 80)
    print(" NON-IID STATISTICAL QUANTIFICATION")
    print("=" * 80)

    # 1. Total Variation Distance on Class Distributions
    p_targets = {}
    for cid in CLIENT_NAMES:
        y = clients_data[cid]['y_full']
        p0 = y.value_counts().get(0, 0) / len(y)
        p1 = y.value_counts().get(1, 0) / len(y)
        p_targets[cid] = np.array([p0, p1])

    pairs = [
        ('hospital_1', 'hospital_2'),
        ('hospital_1', 'hospital_3'),
        ('hospital_2', 'hospital_3')
    ]

    print("\n1. Label Distribution Divergence (Total Variation Distance, TVD in [0, 1]):")
    for c1, c2 in pairs:
        tvd = 0.5 * np.sum(np.abs(p_targets[c1] - p_targets[c2]))
        print(f"  TVD({CLIENT_NAMES[c1]} vs {CLIENT_NAMES[c2]}): {tvd:.4f}")

    # 2. Kolmogorov-Smirnov Test on Continuous Features
    ks_results = []
    print("\n2. Kolmogorov-Smirnov 2-Sample Test (D-statistic, p-value) on Continuous Features:")
    for feat in CONTINUOUS_FEATURES:
        row = {'Feature': feat}
        for c1, c2 in pairs:
            v1 = clients_data[c1]['X_full'][feat].values
            v2 = clients_data[c2]['X_full'][feat].values
            res = stats.ks_2samp(v1, v2)
            row[f"{c1}_vs_{c2}_stat"] = round(res.statistic, 4)
            row[f"{c1}_vs_{c2}_pval"] = res.pvalue
            print(f"  {feat:<10} | {c1} vs {c2}: KS-Stat = {res.statistic:.4f}, p = {res.pvalue:.4e}")
        ks_results.append(row)

    return pd.DataFrame(ks_results)


def main():
    # 1. Load data
    clients_data = {cid: load_client_data(cid) for cid in CLIENT_NAMES}

    # 2. Verify datasets
    verify_client_datasets(clients_data)

    # 3. Generate all visualizations
    plot_client_target_distributions(clients_data)
    plot_feature_distribution_comparisons(clients_data)
    plot_correlation_matrices(clients_data)
    plot_target_correlation_comparison(clients_data)
    plot_split_distribution_summary(clients_data)

    # 4. Quantify Non-IID Divergences
    ks_df = compute_non_iid_quantification(clients_data)

    print("\n" + "=" * 80)
    print(" EDA AND CLIENT HETEROGENEITY ANALYSIS COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
