"""
GaitGuard AI - Phase 4 Exploratory Data Analysis (EDA) Script
Executes statistical pattern discovery, movement analysis, symmetry exploration, correlation analysis, and QA plotting.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from gaitguard.config import KEYPOINT_COUNT, TARGET_SEQUENCE_LENGTH, RANDOM_SEED

KEYPOINT_NAMES = [
    'LFHoof', 'LFAnkle', 'LFKnee',
    'RFHoof', 'RFAnkle', 'RFKnee',
    'LHHoof', 'LHAnkle', 'LHKnee',
    'RHHoof', 'RHAnkle', 'RHKnee',
    'Nose', 'HeadTop',
    'Spine1', 'Spine2', 'Spine3'
]

def cohen_d(x, y):
    """Calculate Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled_std)

def run_eda():
    start_time = time.time()
    print("=" * 70)
    print("GAITGUARD AI — PHASE 4: EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)

    # 1. Load Phase 3 Cleaned Dataset
    dataset_path = os.path.join(ROOT_DIR, "datasets", "processed", "gaitguard_cleaned_dataset.npz")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {dataset_path}")

    data = np.load(dataset_path)
    sample_ids = data['sample_ids']
    animal_ids = data['animal_ids']
    raw_seq_lens = data['raw_sequence_lengths']
    padded_kps = data['padded_keypoints'] # (272, 128, 17, 2)
    seq_masks = data['sequence_masks']    # (272, 128)
    raw_labels = data['raw_labels']
    binary_targets = data['binary_targets']

    n_samples = len(sample_ids)
    unique_cows = len(np.unique(animal_ids))

    print(f"\n[PART 1] Dataset Inventory & Overview:")
    print(f"  - Total Samples: {n_samples}")
    print(f"  - Unique Cow IDs: {unique_cows}")
    print(f"  - Padded Keypoint Shape: {padded_kps.shape}")
    print(f"  - Target Distribution: Normal (0) = {np.sum(binary_targets == 0)} ({np.mean(binary_targets == 0)*100:.1f}%), Lame Risk (1) = {np.sum(binary_targets == 1)} ({np.mean(binary_targets == 1)*100:.1f}%)")

    # 2. Keypoint Statistics & Exploratory Motion Features
    print("\n[PART 2] Computing Keypoint Motion Statistics & Exploratory Gait Variables...")
    
    eda_records = []
    for i in range(n_samples):
        valid_len = int(raw_seq_lens[i])
        # Valid un-padded frames: shape (valid_len, 17, 2)
        kp_seq = padded_kps[i, :valid_len, :, :] 

        # Keypoint displacements: frame-to-frame displacement sqrt(dx^2 + dy^2)
        dx = np.diff(kp_seq[:, :, 0], axis=0) # (valid_len-1, 17)
        dy = np.diff(kp_seq[:, :, 1], axis=0) # (valid_len-1, 17)
        disp = np.sqrt(dx**2 + dy**2)          # (valid_len-1, 17)
        
        total_disp_per_kp = np.sum(disp, axis=0) # (17,)
        mean_disp_per_kp = np.mean(disp, axis=0)  # (17,)
        std_disp_per_kp = np.std(disp, axis=0)    # (17,)

        # 1. Back Arching / Spine Curvature (Y-coordinate of Spine2 relative to Spine1-Spine3 baseline)
        # Y increases downwards; smaller Y means higher elevation / arched back
        spine1_y = kp_seq[:, 14, 1]
        spine2_y = kp_seq[:, 15, 1]
        spine3_y = kp_seq[:, 16, 1]
        spine_baseline_y = 0.5 * (spine1_y + spine3_y)
        spine_arch_elevation = float(np.mean(spine_baseline_y - spine2_y))

        # 2. Head Nodding / Head Oscillation Variance (Y variance of Nose and HeadTop)
        head_y_var = float(np.var(kp_seq[:, 13, 1])) # HeadTop Y variance
        nose_y_var = float(np.var(kp_seq[:, 12, 1])) # Nose Y variance

        # 3. Limb Movement Displacements
        lf_disp = float(total_disp_per_kp[0]) # LFHoof
        rf_disp = float(total_disp_per_kp[3]) # RFHoof
        lh_disp = float(total_disp_per_kp[6]) # LHHoof
        rh_disp = float(total_disp_per_kp[9]) # RHHoof

        # 4. Gait Asymmetry Ratios
        front_asym = float(abs(lf_disp - rf_disp) / (lf_disp + rf_disp + 1e-8))
        hind_asym = float(abs(lh_disp - rh_disp) / (lh_disp + rh_disp + 1e-8))
        total_limb_asym = float((front_asym + hind_asym) / 2.0)

        rec = {
            'sample_id': sample_ids[i],
            'animal_id': int(animal_ids[i]),
            'raw_seq_len': valid_len,
            'binary_target': int(binary_targets[i]),
            'raw_label': int(raw_labels[i]),
            'spine_arch_elevation': spine_arch_elevation,
            'head_nod_var': head_y_var,
            'nose_nod_var': nose_y_var,
            'lf_hoof_disp': lf_disp,
            'rf_hoof_disp': rf_disp,
            'lh_hoof_disp': lh_disp,
            'rh_hoof_disp': rh_disp,
            'front_gait_asym': front_asym,
            'hind_gait_asym': hind_asym,
            'total_gait_asym': total_limb_asym,
            'avg_hoof_disp': float(np.mean([lf_disp, rf_disp, lh_disp, rh_disp]))
        }
        eda_records.append(rec)

    df_eda = pd.DataFrame(eda_records)

    # 3. Statistical Testing (Normal vs Lame Risk)
    print("\n[PART 3] Statistical Significance Tests (Mann-Whitney U & Student's T-test):")
    stats_results = []
    
    test_vars = ['spine_arch_elevation', 'head_nod_var', 'total_gait_asym', 'avg_hoof_disp', 'raw_seq_len']
    norm_df = df_eda[df_eda['binary_target'] == 0]
    lame_df = df_eda[df_eda['binary_target'] == 1]

    for var in test_vars:
        x_norm = norm_df[var].values
        x_lame = lame_df[var].values

        u_stat, u_p = stats.mannwhitneyu(x_norm, x_lame, alternative='two-sided')
        t_stat, t_p = stats.ttest_ind(x_norm, x_lame, equal_var=False)
        d_val = cohen_d(x_lame, x_norm)

        stats_results.append({
            'variable': var,
            'normal_mean': float(np.mean(x_norm)),
            'normal_std': float(np.std(x_norm)),
            'lame_mean': float(np.mean(x_lame)),
            'lame_std': float(np.std(x_lame)),
            'mann_whitney_p': float(u_p),
            't_test_p': float(t_p),
            'cohen_d': float(d_val),
            'significant_p005': bool(u_p < 0.05)
        })
        print(f"  - Variable: {var:20s} | Normal Mean: {np.mean(x_norm):.4f} | Lame Mean: {np.mean(x_lame):.4f} | p-value: {u_p:.4e} | Cohen's d: {d_val:+.4f} | Sig: {u_p < 0.05}")

    # 4. Generate Figures in docs/figures/phase4/
    print("\n[PART 4] Generating Phase 4 EDA Figures in docs/figures/phase4/...")
    fig_dir = os.path.join(ROOT_DIR, "docs", "figures", "phase4")
    os.makedirs(fig_dir, exist_ok=True)

    # Figure 1: Target and Animal Sample Distribution
    plt.figure(figsize=(10, 4.5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='binary_target', hue='binary_target', data=df_eda, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Binary Target Sample Counts', fontweight='bold')
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    cows_per_class = df_eda.groupby('binary_target')['animal_id'].nunique()
    plt.bar(['0: Normal', '1: Lame Risk'], cows_per_class.values, color=['#2ca02c', '#d62728'], alpha=0.85)
    plt.title('Unique Cows Represented per Class', fontweight='bold')
    plt.xlabel('Target Class')
    plt.ylabel('Unique Cow Count')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_target_and_animal_distribution.png'), dpi=150)
    plt.close()

    # Figure 2: Sequence Length Distribution by Class
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=df_eda, x='raw_seq_len', hue='binary_target', common_norm=False, palette=['#2ca02c', '#d62728'], fill=True, alpha=0.4)
    plt.title('Sequence Length Distribution by Lameness Class', fontweight='bold')
    plt.xlabel('Sequence Length (Frames)')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_sequence_length_by_class.png'), dpi=150)
    plt.close()

    # Figure 3: Keypoint Movement Variance across 17 Keypoints
    kp_std_list = []
    for k_idx, kp_name in enumerate(KEYPOINT_NAMES):
        std_val = float(np.mean([np.std(padded_kps[i, :raw_seq_lens[i], k_idx, :]) for i in range(n_samples)]))
        kp_std_list.append((kp_name, std_val))

    kp_std_df = pd.DataFrame(kp_std_list, columns=['Keypoint', 'StdDev'])
    plt.figure(figsize=(12, 5))
    sns.barplot(x='Keypoint', y='StdDev', hue='Keypoint', data=kp_std_df, palette='Blues_d', legend=False)
    plt.xticks(rotation=45)
    plt.title('Spatial Movement Variance Across 17 Anatomical Keypoints', fontweight='bold')
    plt.ylabel('Mean Standard Deviation')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_keypoint_movement_variance.png'), dpi=150)
    plt.close()

    # Figure 4: Back Arching Elevation Comparison (Boxplot)
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='spine_arch_elevation', hue='binary_target', data=df_eda, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Exploratory Back Arch Elevation by Lameness Class', fontweight='bold')
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Spine Arch Elevation (Normalized Units)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_back_arch_comparison.png'), dpi=150)
    plt.close()

    # Figure 5: Limb Asymmetry Analysis (Boxplot)
    plt.figure(figsize=(7, 5))
    sns.boxplot(x='binary_target', y='total_gait_asym', hue='binary_target', data=df_eda, palette=['#2ca02c', '#d62728'], legend=False)
    plt.title('Exploratory Limb Displacement Asymmetry Ratio', fontweight='bold')
    plt.xlabel('Target Class (0: Normal, 1: Lame Risk)')
    plt.ylabel('Limb Asymmetry Ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_limb_symmetry_analysis.png'), dpi=150)
    plt.close()

    # Figure 6: Feature Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr_vars = ['spine_arch_elevation', 'head_nod_var', 'total_gait_asym', 'avg_hoof_disp', 'raw_seq_len', 'binary_target']
    corr_matrix = df_eda[corr_vars].corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title('Spearman Rank Correlation Matrix of Exploratory Variables', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'eda_feature_correlation_matrix.png'), dpi=150)
    plt.close()

    # 5. Export Summary JSON
    stats_json_path = os.path.join(ROOT_DIR, "docs", "phase4_eda_summary.json")
    summary_dict = {
        'total_samples': n_samples,
        'unique_cows': unique_cows,
        'binary_counts': {
            '0_normal': int(np.sum(binary_targets == 0)),
            '1_lame_risk': int(np.sum(binary_targets == 1))
        },
        'sequence_lengths': {
            'min': int(np.min(raw_seq_lens)),
            'max': int(np.max(raw_seq_lens)),
            'mean': float(np.mean(raw_seq_lens)),
            'median': float(np.median(raw_seq_lens)),
            'std': float(np.std(raw_seq_lens))
        },
        'statistical_tests': stats_results
    }
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_dict, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PHASE 4 EDA PIPELINE PERFORMANCE & SUMMARY")
    print("=" * 70)
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Summary JSON: {stats_json_path}")
    print(f"Figures Directory: {fig_dir}")
    print("STATUS: SUCCESS")
    print("=" * 70)

    return df_eda, stats_results

if __name__ == "__main__":
    run_eda()
