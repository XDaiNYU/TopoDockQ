#!/usr/bin/env python3
"""
TopoDockQ Training Tutorial
Converted from 01_tutorial_train.ipynb

Notes:
- The csv files need to be downloaded first, and change their corresponding path.
- Please note that the hyperparameters presented in this example are for illustrative purposes only.
"""

# Standard libraries
import argparse

# Numerical and data handling
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt

# Import custom modules
from src.model import *
from src.train import *


def main():
    """Main training function"""
    
    # File paths
    df1_file = './data/processed_data/singlePPD_full_bins_features.csv'
    df2_file = './data/processed_data/singlePPD_DockQ.csv'
    
    # Hyperparameter for TopoDockQ
    input_dim = 2646
    neurons1 = 2048     
    neurons2 = 2048  
    neurons3 = 2048 
    neurons4 = 2048 
    
    lr = 0.0005
    batch_size = 512
    
    # num_epochs = 1000
    # patience = 1000
    
    num_epochs = 30
    patience = 30
    
    dropout = 0.0
    
    # Load and preprocess data
    print("Loading data...")
    df1 = pd.read_csv(df1_file)
    
    # Filter training data
    df_train = df1[df1['data_class'] != 'test']
    
    # Remove columns with all identical values
    identical_columns_count = sum(df_train.nunique() == 1)
    print(f"Number of columns with all identical values: {identical_columns_count}")
    
    identical_columns = df_train.columns[df_train.nunique() == 1]
    print("Columns with all identical values:", identical_columns.tolist())
    
    # Remove these columns from the DataFrame
    df1 = df1.drop(columns=identical_columns)
    
    # Load DockQ data
    df2 = pd.read_csv(df2_file)
    df2 = df2[['pdb_id', 'af_model_id', 'af_confidence', 'pdb2sql_DockQ', 'data_class']]
    
    # Merge datasets
    df1 = pd.merge(df1, df2, on=['pdb_id', 'af_model_id', 'pdb2sql_DockQ', 'data_class'], how='inner')
    
    df1_for_eval = df1
    print(f"Combined dataset shape: {df1.shape}")
    
    # Split data by class
    df_train = df1[df1['data_class'] == 'train']
    df_val = df1[df1['data_class'] == 'valid']
    df_test = df1[df1['data_class'] == 'test']
    
    # Check for data leakage between splits
    list_train = list(set(df_train['pdb_id'].to_list()))
    list_val = list(set(df_val['pdb_id'].to_list()))
    list_test = list(set(df_test['pdb_id'].to_list()))
    
    print(f"Train: {len(list_train)}, Val: {len(list_val)}, Test: {len(list_test)}")
    
    # Check for overlapping PDB IDs
    for i in list_train:
        if (i in list_val) or (i in list_test):
            print(f"Overlap in train: {i}")
    for i in list_test:
        if (i in list_val) or (i in list_train):
            print(f"Overlap in test: {i}")
    
    # Generate feature names
    filtration_values = [f"{x:.1f}" if x.is_integer() else f"{x:.2f}".rstrip("0") for x in np.arange(2.0, 10.25, 0.25)]
    
    # Generate persistent feature names for each filtration value
    persistent_features = []
    for filtration in filtration_values:
        persistent_features += [f"persistent_{filtration}_{str(i+1).zfill(2)}" for i in range(72)]
    
    # Generate static feature names
    static_features = [f"static_{str(i+1).zfill(2)}" for i in range(378)]
    
    # Combine both lists
    combined_feature_name_list = persistent_features + static_features
    
    feature_name_list = ['pdb2sql_DockQ'] + combined_feature_name_list
    
    # Filter valid columns
    valid_columns = [col for col in feature_name_list if col in df_train.columns]
    
    # Subset the DataFrame with the valid columns
    df_train = df_train[valid_columns]
    df_val = df_val[valid_columns]
    df_test = df_test[valid_columns]
    
    print(f"Final dataset shapes - Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}")
    
    def get_features_and_target(df):
        """Extract features and target from dataframe"""
        y = df['pdb2sql_DockQ'].values  # Select the 'pdb2sql_DockQ' column as the target
        X = df.drop(columns=['pdb2sql_DockQ']).values  # Drop the 'pdb2sql_DockQ' column to get the features
        return X, y
    
    # Extract features and targets
    print("Extracting features and targets...")
    X_train, y_train = get_features_and_target(df_train)
    X_val, y_val = get_features_and_target(df_val)
    X_test, y_test = get_features_and_target(df_test)
    
    # Standardize features
    print("Standardizing features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert arrays to PyTorch tensors and create datasets
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1000000)
    test_loader = DataLoader(test_data, batch_size=1000000)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
    model = TopoDockQ(input_dim, neurons1, neurons2, neurons3, neurons4, dropout).to(device)
    
    # Set up optimizer with Adam
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-08, amsgrad=False)
    
    # Set up learning rate scheduler
    r_adjust = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, last_epoch=-1)
    
    # Train and evaluate the model
    print("Starting training...")
    train_losses, val_losses, train_pccs, val_pccs = train_and_evaluate(
        model, train_loader, val_loader, optimizer, lr_scheduler=r_adjust,
        num_epochs=num_epochs, patience=patience, 
        model_save_path="./models/example_MLP_best_model.pth", 
        pcc_model_save_path="./models/example_MLP_best_pcc_model.pth"
    )
    
    # Save the last epoch model
    torch.save(model.state_dict(), f'./models/example_MLP_{num_epochs}epoch.pth')
    print("Training completed and model saved!")


if __name__ == "__main__":
    main()
