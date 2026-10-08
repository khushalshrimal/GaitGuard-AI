"""
GaitGuard AI - PyTorch BiLSTM Gait Classifier (Phase 7)
Implements a modest Bidirectional LSTM architecture for temporal cattle gait classification.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def set_seed(seed=42):
    """Sets random seeds for numpy, random, and torch for 100% reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class BiLSTMGaitClassifier(nn.Module):
    """
    Modest Bidirectional LSTM Neural Network for sequence classification (N, T, F).
    """

    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.3, dense_dim=16):
        super(BiLSTMGaitClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout
        
        # 1-Layer Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Dense classification head
        self.fc1 = nn.Linear(hidden_dim * 2, dense_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask=None):
        """
        Forward pass.
        Inputs:
            x: (batch_size, seq_len, input_dim)
            mask: (batch_size, seq_len) sequence mask (1 for valid, 0 for padded)
        Output:
            probs: (batch_size, 1) prediction probabilities in range [0, 1]
        """
        # lstm_out shape: (batch_size, seq_len, 2 * hidden_dim)
        lstm_out, _ = self.bilstm(x)
        
        if mask is not None:
            # Masked Mean Pooling over unpadded sequence frames
            # mask shape: (batch_size, seq_len, 1)
            mask_expanded = mask.unsqueeze(-1).float()
            masked_out = lstm_out * mask_expanded
            sum_out = torch.sum(masked_out, dim=1) # (batch_size, 2 * hidden_dim)
            valid_counts = torch.sum(mask_expanded, dim=1).clamp(min=1.0)
            pooled = sum_out / valid_counts
        else:
            pooled = torch.mean(lstm_out, dim=1)
            
        dropped = self.dropout(pooled)
        h1 = self.relu(self.fc1(dropped))
        logits = self.fc2(h1)
        probs = self.sigmoid(logits)
        
        return probs


class BiLSTMModelTrainer:
    """
    Helper class to train and evaluate BiLSTMGaitClassifier with Early Stopping.
    """

    def __init__(self, input_dim, hidden_dim=32, dropout=0.3, lr=1e-3, weight_decay=1e-4, seed=42):
        set_seed(seed)
        self.model = BiLSTMGaitClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss()

    def fit(self, X_train, y_train, mask_train, X_val, y_val, mask_val, max_epochs=100, batch_size=16, patience=15):
        """
        Trains model with early stopping on validation loss.
        """
        set_seed(42)
        X_tr = torch.tensor(X_train, dtype=torch.float32)
        y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
        m_tr = torch.tensor(mask_train, dtype=torch.float32)
        
        X_v = torch.tensor(X_val, dtype=torch.float32)
        y_v = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
        m_v = torch.tensor(mask_val, dtype=torch.float32)
        
        best_val_loss = float('inf')
        best_weights = None
        patience_counter = 0
        
        N_tr = X_tr.shape[0]
        
        for epoch in range(max_epochs):
            self.model.train()
            permutation = torch.randperm(N_tr)
            
            for i in range(0, N_tr, batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y, batch_m = X_tr[indices], y_tr[indices], m_tr[indices]
                
                self.optimizer.zero_grad()
                preds = self.model(batch_x, batch_m)
                loss = self.criterion(preds, batch_y)
                loss.backward()
                self.optimizer.step()
                
            # Validation check
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_v, m_v)
                val_loss = self.criterion(val_preds, y_v).item()
                
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
                    
        # Load best validation model weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        return best_val_loss

    def predict_proba(self, X, mask):
        """
        Predicts probabilities for sequence input X.
        """
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            m_t = torch.tensor(mask, dtype=torch.float32)
            probs = self.model(X_t, m_t).numpy().flatten()
        return probs
