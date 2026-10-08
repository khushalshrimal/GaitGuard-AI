"""
GaitGuard AI - Phase 5 Master Feature Engineering Script
Loads Phase 3 cleaned dataset, extracts torso-normalized gait features, performs QA plotting, and exports feature tables.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gaitguard.config import RANDOM_SEED
from gaitguard.features.extractor import GaitFeatureExtractor

def generate_phase5_figures(df_features, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # Figure 1: Back Arch Curvature Index by Class
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='back_arch_curvature', hue='binary_target', data=df_features, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Back Arch Curvature Index by Lameness Class', fontweight='bold', fontsize=12)
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Torso-Normalized Elevation Index')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_back_arch_curvature.png'), dpi=150)
    plt.close()

    # Figure 2: Head Nodding Amplitude by Class
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='head_nodding_amplitude', hue='binary_target', data=df_features, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Head Nodding Amplitude Index by Lameness Class', fontweight='bold', fontsize=12)
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Torso-Normalized Peak-to-Peak Amplitude')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_head_nodding_amplitude.png'), dpi=150)
    plt.close()

    # Figure 3: Torso-Normalized Stride Length by Class
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='torso_normalized_stride_length', hue='binary_target', data=df_features, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Torso-Normalized Stride Length by Lameness Class', fontweight='bold', fontsize=12)
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Normalized Hoof Displacement / Torso Length')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_normalized_stride_length.png'), dpi=150)
    plt.close()

    # Figure 4: Stance-Phase Timing Asymmetry Ratio by Class
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='stance_timing_asymmetry', hue='binary_target', data=df_features, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Stance-Phase Timing Asymmetry Ratio by Lameness Class', fontweight='bold', fontsize=12)
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Stance Time Asymmetry Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_stance_timing_asymmetry.png'), dpi=150)
    plt.close()

    # Figure 5: Torso-Normalized Walking Speed by Class
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='normalized_walking_speed', hue='binary_target', data=df_features, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Torso-Normalized Walking Speed by Lameness Class', fontweight='bold', fontsize=12)
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Forward Displacement per Frame / Torso Length')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_walking_speed.png'), dpi=150)
    plt.close()

    # Figure 6: Feature Correlation Heatmap
    plt.figure(figsize=(9, 7))
    feature_cols = [
        'back_arch_curvature', 'head_nodding_amplitude', 'torso_normalized_stride_length',
        'stance_timing_asymmetry', 'normalized_walking_speed', 'knee_flexion_range', 'binary_target'
    ]
    corr_matrix = df_features[feature_cols].corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title('Spearman Correlation Matrix of Engineered Gait Features', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'feature_correlation_heatmap.png'), dpi=150)
    plt.close()

    # Figure 7: Pairplot of Core Biomarkers
    plt.figure(figsize=(10, 8))
    pair_df = df_features[['back_arch_curvature', 'head_nodding_amplitude', 'torso_normalized_stride_length', 'binary_target']]
    g = sns.pairplot(pair_df, hue='binary_target', palette=['#2ca02c', '#d62728'], corner=True, diag_kind='kde')
    g.fig.suptitle('Pairwise Distribution of Primary Gait Biomarkers', y=1.02, fontweight='bold')
    plt.savefig(os.path.join(fig_dir, 'feature_pairplot_summary.png'), dpi=150)
    plt.close()

def run_phase5_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("GAITGUARD AI — PHASE 5: GAIT FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # 1. Load Phase 3 Cleaned Data
    dataset_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_cleaned_dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Phase 3 dataset not found at {dataset_path}")

    data = np.load(dataset_path)
    sample_ids = data['sample_ids']
    animal_ids = data['animal_ids']
    raw_seq_lens = data['raw_sequence_lengths']
    padded_kps = data['padded_keypoints']
    raw_labels = data['raw_labels']
    binary_targets = data['binary_targets']

    samples = []
    for i in range(len(sample_ids)):
        samples.append({
            'sample_id': sample_ids[i],
            'animal_id': animal_ids[i],
            'raw_sequence_length': raw_seq_lens[i],
            'padded_keypoints': padded_kps[i],
            'raw_label': raw_labels[i],
            'binary_target': binary_targets[i]
        })

    print(f"\n[STEP 1] Ingested {len(samples)} Phase 3 cleaned samples.")

    # 2. Extract Gait Features
    print("\n[STEP 2] Extracting torso-normalized gait features...")
    extractor = GaitFeatureExtractor()
    df_features = extractor.extract_dataset_features(samples)
    print(f"  - Extracted feature matrix of shape: {df_features.shape}")

    # 3. Feature Validation & Integrity Checks
    print("\n[STEP 3] Validating feature quality and integrity...")
    feature_cols = [c for c in df_features.columns if c not in ['sample_id', 'animal_id', 'binary_target', 'raw_label']]
    
    nans = df_features[feature_cols].isna().sum().sum()
    infs = np.isinf(df_features[feature_cols].values).sum()
    zero_torso = (df_features['torso_length'] <= 0).sum()

    print(f"  - Total Feature NaNs: {nans}")
    print(f"  - Total Feature Infs: {infs}")
    print(f"  - Torso Lengths <= 0: {zero_torso}")

    if nans > 0 or infs > 0 or zero_torso > 0:
        raise ValueError("Feature validation failed: Invalid numeric values detected.")

    print("  - PASS: All engineered features passed quality and non-zero torso validation.")

    # 4. Statistical Tests & Effect Size
    print("\n[STEP 4] Evaluating Feature Group Separation (Mann-Whitney U & Cohen's d):")
    norm_df = df_features[df_features['binary_target'] == 0]
    lame_df = df_features[df_features['binary_target'] == 1]
    
    feature_stats = []
    for f_name in feature_cols:
        x0 = norm_df[f_name].values
        x1 = lame_df[f_name].values
        
        u_stat, u_p = stats.mannwhitneyu(x0, x1, alternative='two-sided')
        d_val = (np.mean(x1) - np.mean(x0)) / np.sqrt(0.5 * (np.std(x0)**2 + np.std(x1)**2) + 1e-8)
        
        feature_stats.append({
            'feature': f_name,
            'normal_mean': float(np.mean(x0)),
            'lame_mean': float(np.mean(x1)),
            'p_value': float(u_p),
            'cohen_d': float(d_val),
            'stat_sig': bool(u_p < 0.05)
        })
        print(f"  - {f_name:32s} | Norm: {np.mean(x0):.4f} | Lame: {np.mean(x1):.4f} | p-value: {u_p:.4e} | Cohen's d: {d_val:+.4f} | Sig: {u_p < 0.05}")

    # 5. Export Feature Tables & Metadata
    print("\n[STEP 5] Exporting feature dataset tables...")
    out_dir = os.path.join(ROOT_DIR, "datasets", "processed")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "gaitguard_engineered_features.csv")
    npz_path = os.path.join(out_dir, "gaitguard_engineered_features.npz")
    json_path = os.path.join(out_dir, "engineered_features_metadata.json")

    df_features.to_csv(csv_path, index=False)
    
    feature_matrix = df_features[feature_cols].values.astype(np.float32)
    np.savez_compressed(
        npz_path,
        sample_ids=sample_ids,
        animal_ids=animal_ids,
        feature_names=np.array(feature_cols, dtype=str),
        feature_matrix=feature_matrix,
        binary_targets=binary_targets,
        raw_labels=raw_labels
    )

    metadata = {
        'phase': 5,
        'pipeline_status': 'FEATURES_ENGINEERED',
        'total_samples': len(df_features),
        'feature_names': feature_cols,
        'feature_matrix_shape': list(feature_matrix.shape),
        'statistical_evaluation': feature_stats
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # 6. Generate Figures
    print("\n[STEP 6] Generating Phase 5 QA figures in docs/figures/phase5/...")
    fig_dir = os.path.join(ROOT_DIR, "docs", "figures", "phase5")
    generate_phase5_figures(df_features, fig_dir)
    print("  - Generated 7 diagnostic QA figures in docs/figures/phase5/.")

    elapsed = time.time() - start_time
    file_size_kb = os.path.getsize(csv_path) / 1024

    print("\n" + "=" * 70)
    print("PHASE 5 FEATURE ENGINEERING PIPELINE PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Feature CSV: {csv_path} ({file_size_kb:.2f} KB)")
    print(f"Feature NPZ: {npz_path}")
    print(f"Summary JSON: {json_path}")
    print("STATUS: SUCCESS")
    print("=" * 70)

    return df_features, feature_stats

if __name__ == "__main__":
    run_phase5_pipeline()
