# -*- coding: utf-8 -*-
'''
Author: Rui Wang
Date: 2024-10-11
Description: Extract persistent features (all filtrations) + static features from one .npy file and save to CSV with proper column names.
'''

import os
import sys
import numpy as np
import pandas as pd
import argparse

# ================= Configurations ==================
rips_topo_bin_size = 33
rips_betti_stats_size = 6
rips_spectra_size = 231
alpha_topo_stats_size_1 = 18
alpha_topo_stats_size_2 = 18
total_atom_combination_features = (
    rips_topo_bin_size +
    rips_betti_stats_size +
    rips_spectra_size +
    alpha_topo_stats_size_1 +
    alpha_topo_stats_size_2
)

feature_npy_shape = (1, 2754)
atom_types = ['C', 'N', 'O']
atom_combinations = list(range(len(atom_types)**2))
filtration_list = np.round(np.arange(2.0, 10.25, 0.25), 2)

# ================= Feature Extractor ==================
class ExtractFeatures:
    def __init__(self, atom_types, total_atom_combination_features): 
        self.atom_types = atom_types
        self.total_atom_combination_features = total_atom_combination_features

    def get_feature_indices(self, atom_combination_index):
        start_idx = atom_combination_index * self.total_atom_combination_features
        end_idx = start_idx + self.total_atom_combination_features
        count = 0
        for a in self.atom_types:
            for b in self.atom_types:
                if count == atom_combination_index:
                    return start_idx, end_idx, a, b
                count += 1

    def extract_features_by_filtration(self, features, filtration_value, filtration, atom_combination_index):
        start_idx, end_idx, _, _ = self.get_feature_indices(atom_combination_index)
        atom_combination_features = features[:, start_idx:end_idx]
        rips_topo_bin_feature = atom_combination_features[:, 0:rips_topo_bin_size]
        rips_spectra_feature = atom_combination_features[:, rips_topo_bin_size + rips_betti_stats_size:
                                                         rips_topo_bin_size + rips_betti_stats_size + rips_spectra_size]
        filtration_index = np.where(filtration == filtration_value)[0][0]
        rips_spectra_filtration = rips_spectra_feature[:, filtration_index*7:(filtration_index+1)*7]
        persistent_features = np.concatenate(
            (rips_topo_bin_feature[:, filtration_index:filtration_index+1], rips_spectra_filtration), axis=1)
        return persistent_features

    def extract_static_features(self, features, atom_combination_index):
        start_idx, end_idx, _, _ = self.get_feature_indices(atom_combination_index)
        atom_combination_features = features[:, start_idx:end_idx]
        rips_betti_stats_feature = atom_combination_features[:, rips_topo_bin_size:rips_topo_bin_size + rips_betti_stats_size]
        alpha_topo_stats_feature_betti1 = atom_combination_features[:, -(alpha_topo_stats_size_1 + alpha_topo_stats_size_2):-alpha_topo_stats_size_2]
        alpha_topo_stats_feature_betti2 = atom_combination_features[:, -alpha_topo_stats_size_2:]
        return np.concatenate((rips_betti_stats_feature, alpha_topo_stats_feature_betti1, alpha_topo_stats_feature_betti2), axis=1)

# ================= Main Execution ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all persistent + static features from one .npy file.")
    parser.add_argument("--npy_file", type=str, required=True,
                        help="Path to input feature .npy file")
    parser.add_argument("--output_file", type=str, default="final_combined_features.csv",
                        help="Output CSV path (features only, no IDs)")
    args = parser.parse_args()

    if not os.path.exists(args.npy_file):
        raise FileNotFoundError(f"❌ File not found: {args.npy_file}")
    
    features = np.load(args.npy_file)
    if features.shape != feature_npy_shape:
        raise ValueError(f"❌ Expected shape {feature_npy_shape}, but got {features.shape}")

    extractor = ExtractFeatures(atom_types, total_atom_combination_features)
    filtration = np.round(np.arange(2, 10.25, 0.25), 2)

    # Step 1: Extract persistent features for all filtrations
    persistent_blocks = []
    persistent_colnames = []

    for filtration_value in filtration_list:
        pf_all = np.zeros((1, 8 * len(atom_combinations)))
        for i, atom_combination_index in enumerate(atom_combinations):
            pf = extractor.extract_features_by_filtration(features, filtration_value, filtration, atom_combination_index)
            pf_all[:, i*8:(i+1)*8] = pf
        persistent_blocks.append(pf_all)

        # Generate column names for this filtration
        for i in range(1, 8 * len(atom_combinations) + 1):
            colname = f"persistent_{filtration_value}_{str(i).zfill(2)}"
            persistent_colnames.append(colname)

    persistent_full = np.concatenate(persistent_blocks, axis=1)

    # Step 2: Extract static features
    static_all = np.zeros((1, (rips_betti_stats_size + alpha_topo_stats_size_1 + alpha_topo_stats_size_2) * len(atom_combinations)))
    for i, atom_combination_index in enumerate(atom_combinations):
        sf = extractor.extract_static_features(features, atom_combination_index)
        static_all[:, i*(rips_betti_stats_size + alpha_topo_stats_size_1 + alpha_topo_stats_size_2):(i+1)*(rips_betti_stats_size + alpha_topo_stats_size_1 + alpha_topo_stats_size_2)] = sf

    # static_colnames = [f"static_{str(i).zfill(2)}" for i in range(1, static_all.shape[1] + 1)]
    static_colnames = [f"static_{str(i).zfill(2) if i < 100 else str(i)}" for i in range(1, static_all.shape[1] + 1)]
    # Step 3: Combine and save
    final_feature_matrix = np.concatenate([persistent_full, static_all], axis=1)
    final_colnames = persistent_colnames + static_colnames

    df = pd.DataFrame(final_feature_matrix, columns=final_colnames)
    df.to_csv(args.output_file, index=False)

    print(f"✅ Saved combined features to: {args.output_file}")
    print(f"🔢 Final shape: {df.shape}")
