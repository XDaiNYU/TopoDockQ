#!/usr/bin/env python3
"""
TopoDockQ Inference Tutorial
Converted from 02_tutorial_inference.ipynb

Notes:
- The csv files need to be downloaded first, and change their corresponding path.
- The trained model used in this tutorial is a demonstration model, trained and saved in tutorial_train.ipynb. 
  To reproduce the optimal performance of the TopoDockQ model as reported in the paper, 
  please download and use the model file lr_0.0005_bs_512.zip.
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
    """Main inference function"""
    
    # File paths
    df1_file = './data/processed_data/singlePPD_full_bins_features.csv'
    df2_file = './data/processed_data/singlePPD_DockQ.csv'
    model_path = "./models/best_model.pth"
    
    # Hyperparameters (matching training configuration)
    lr = 0.0005
    batch_size = 512
    num_epochs = 20
    patience = 20
    dropout = 0.0
    
    # Model architecture parameters
    input_dim = 2646
    neurons1 = 2048     
    neurons2 = 2048  
    neurons3 = 2048 
    neurons4 = 2048 
    
    # Load and preprocess data (same as training)
    print("Loading data...")
    df1 = pd.read_csv(df1_file)
    
    # Filter training data for preprocessing
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
    
    # Generate feature names (same as training)
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
    
    # Standardize features (using training data statistics)
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
    
    # Load the pre-trained model
    print(f"Loading model from {model_path}...")
    best_model = TopoDockQ(input_dim, neurons1, neurons2, neurons3, neurons4, dropout).to(device)
    best_model.load_state_dict(torch.load(model_path, map_location=device))
    best_model.eval()
    
    # Perform inference on validation set
    print("Performing inference on validation set...")
    perform_inference(best_model, val_loader, device, "val_inference_results.csv")
    
    # Perform inference on test set
    print("Performing inference on test set...")
    perform_inference(best_model, test_loader, device, "test_inference_results.csv")
    
    print("Inference completed successfully!")


def perform_inference(model, data_loader, device, output_filename):
    """
    Perform inference on a dataset and save results to CSV
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing the dataset
        device: Device to run inference on
        output_filename: Name of the output CSV file
    """
    with torch.no_grad():
        predictions = []
        true_values = []
        
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions.append(outputs.cpu())
            true_values.append(targets.cpu())
    
    # Concatenate results
    predictions = torch.cat(predictions).numpy()
    true_values = torch.cat(true_values).numpy()
    
    # Save predictions to CSV
    results_df = pd.DataFrame({
        'True_DockQ': true_values.flatten(),
        'Predicted_DockQ(p-DockQ)': predictions.flatten()
    })
    results_df.to_csv(output_filename, index=False)
    
    # Calculate and print metrics
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error
    
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    pcc, _ = pearsonr(true_values.flatten(), predictions.flatten())
    
    print(f"Results saved to '{output_filename}'")
    print(f"RMSE: {rmse:.4f}")
    print(f"PCC: {pcc:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()
