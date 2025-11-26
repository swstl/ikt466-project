import torch
import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class GRU_Attention(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=5, num_classes=30, dropout=0.3):
        super(GRU_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU layers
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Single FC output layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru_layer(x)
        
        # Apply attention mechanism
        # gru_out shape: (batch, seq_len, hidden_size)
        attention_weights = self.attention(gru_out)  # (batch, seq_len, 1)
        
        # Apply attention weights to GRU outputs
        attended_output = torch.sum(attention_weights * gru_out, dim=1)  # (batch, hidden_size)
        
        # Classification
        out = self.classifier(attended_output)
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset
