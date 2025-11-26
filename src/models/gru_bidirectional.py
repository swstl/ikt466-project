import torch
import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class GRU_Bidirectional(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=5, num_classes=30, dropout=0.3):
        super(GRU_Bidirectional, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional GRU layers
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output size is doubled due to bidirectional
        gru_output_size = hidden_size * 2
        
        # Single FC output layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, num_classes)
        )
    
    def forward(self, x):
        # GRU forward pass
        gru_out, h_n = self.gru_layer(x)
        
        # Concatenate forward and backward hidden states from last layer
        # h_n shape: (num_layers * 2, batch, hidden_size)
        forward_hidden = h_n[-2]  # Last forward layer
        backward_hidden = h_n[-1]  # Last backward layer
        out = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        out = self.classifier(out)
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset
