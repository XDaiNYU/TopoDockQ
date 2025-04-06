import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoDockQ(nn.Module):
    def __init__(self, input_dim, neurons1, neurons2, neurons3, neurons4, dropout):
        super(TopoDockQ, self).__init__()
        self.input_dim = input_dim
        self.neurons1 = neurons1
        self.neurons2 = neurons2
        self.neurons3 = neurons3
        self.neurons4 = neurons4
        self.dropout = dropout

        # Define layers
        self.fc1 = nn.Linear(self.input_dim, self.neurons1)
        self.fc2 = nn.Linear(self.neurons1, self.neurons2)
        self.fc3 = nn.Linear(self.neurons2, self.neurons3)
        self.fc4 = nn.Linear(self.neurons3, self.neurons4)
        self.output_layer = nn.Linear(self.neurons4, 1)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(self.neurons1)
        self.bn2 = nn.BatchNorm1d(self.neurons2)
        self.bn3 = nn.BatchNorm1d(self.neurons3)
        self.bn4 = nn.BatchNorm1d(self.neurons4)

        # Dropout layer
        self.drop = nn.Dropout(self.dropout)

        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        # Forward pass with ReLU, batch normalization, and dropout
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.drop(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.drop(x)
        
        # Output layer (no activation function for regression)
        x = self.output_layer(x)
        return x




def RMSELoss(y_pred, y_true):
    """Root Mean Squared Error (RMSE) loss."""
    return torch.sqrt(nn.functional.mse_loss(y_pred, y_true))
