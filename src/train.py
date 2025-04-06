import random
import numpy as np
import torch
def set_seed(seed):
    random.seed(seed)  # Set Python seed
    np.random.seed(seed)  # Set NumPy seed
    torch.manual_seed(seed)  # Set PyTorch CPU seed

    # If you are using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set PyTorch GPU seed
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU

        # To ensure deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate PCC
def pearson_corr(x, y):
    """
    Computes the Pearson Correlation Coefficient (PCC) between two tensors.
    """
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    covariance = torch.sum(x_diff * y_diff)
    x_var = torch.sum(x_diff ** 2)
    y_var = torch.sum(y_diff ** 2)
    pcc = covariance / torch.sqrt(x_var * y_var)
    return pcc

def extra_gpu_work(device, tensor_size=20000):
    """Dummy computation to keep the GPU busy."""
    dummy_data = torch.randn(tensor_size, tensor_size, device=device)
    result = torch.matmul(dummy_data, dummy_data)
    return result

def RMSELoss(y_pred, y_true):
    """Root Mean Squared Error (RMSE) loss."""
    return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def train_and_evaluate(model, train_loader, val_loader, optimizer, lr_scheduler=None, num_epochs=1000, patience=50, model_save_path="MLP_best_model.pth", pcc_model_save_path="MLP_best_pcc_model.pth"):
    set_seed(42)
    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_losses, val_losses = [], []
    train_pccs, val_pccs = [], []  # Store PCC values
    best_val_loss = float('inf')
    best_val_pcc = float('-inf')  # To track the highest PCC
    epochs_no_improve = 0  # Counter to keep track of the number of epochs with no improvement

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_preds, train_targets = [], []

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch data to the GPU
            X_batch, y_batch = X_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)

            predictions = model(X_batch)
            loss = RMSELoss(predictions.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * X_batch.size(0)

            # Store predictions and targets for PCC calculation
            train_preds.append(predictions.detach())
            train_targets.append(y_batch.detach())

        train_rmse = total_train_loss / len(train_loader.dataset)
        train_losses.append(train_rmse)

        # Compute PCC for training
        train_preds = torch.cat(train_preds).squeeze()
        train_targets = torch.cat(train_targets)
        train_pcc = pearson_corr(train_preds, train_targets).item()
        train_pccs.append(train_pcc)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # Ensure validation data is on GPU
                predictions = model(X_batch)
                loss = RMSELoss(predictions.squeeze(), y_batch)
                total_val_loss += loss.item() * X_batch.size(0)

                # Store predictions and targets for PCC calculation
                val_preds.append(predictions)
                val_targets.append(y_batch)

        val_rmse = total_val_loss / len(val_loader.dataset)
        val_losses.append(val_rmse)

        # Compute PCC for validation
        val_preds = torch.cat(val_preds).squeeze()
        val_targets = torch.cat(val_targets)
        val_pcc = pearson_corr(val_preds, val_targets).item()
        val_pccs.append(val_pcc)

        print(f'Epoch {epoch+1}: Train RMSE: {train_rmse}, Train PCC: {train_pcc}, Validation RMSE: {val_rmse}, Validation PCC: {val_pcc}')
        
        # Step the learning rate scheduler (if provided)
        if lr_scheduler:
            lr_scheduler.step()

        # Save the model with the best validation loss
        if val_rmse < best_val_loss:
            best_val_loss = val_rmse
            epochs_no_improve = 0
            # Save model state
            torch.save(model.state_dict(), model_save_path)
            print(f"Validation loss improved; model saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1

        # Save the model with the highest validation PCC
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            torch.save(model.state_dict(), pcc_model_save_path)
            print(f"Validation PCC improved; model saved at epoch {epoch+1}")

        # Save the model every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at epoch {epoch + 1}")

        # Early stopping logic based on RMSE
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs!')
            torch.save(model.state_dict(), 'final_epoch.pth')
            break

    # Plotting RMSE and PCC
    plt.figure(figsize=(12, 6))

    # Plot RMSE
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train RMSE')
    plt.plot(val_losses, label='Validation RMSE')
    plt.title('RMSE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)

    # Plot PCC
    plt.subplot(1, 2, 2)
    plt.plot(train_pccs, label='Train PCC')
    plt.plot(val_pccs, label='Validation PCC')
    plt.title('PCC Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PCC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return train_losses, val_losses, train_pccs, val_pccs


