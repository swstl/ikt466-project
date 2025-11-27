import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class GRU_Tanh(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=30, dropout=0.3):
        super(GRU_Tanh, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Two GRU layers with dropout
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier with Tanh activation
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # GRU forward pass
        gru_out, h_n = self.gru_layer(x)
        
        # Use the last hidden state
        out = h_n[-1]
        
        # Classification with Tanh
        out = self.classifier(out)
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset
