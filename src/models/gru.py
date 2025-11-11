import torch
import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class GRU(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3, bidirectional=True):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Bidirectional GRU for better context understanding
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate the actual output size from GRU
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Layer normalization for better training stability
        self.layer_norm = nn.LayerNorm(gru_output_size)
        
        # Attention mechanism to focus on important parts of the sequence
        self.attention = nn.Sequential(
            nn.Linear(gru_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Enhanced classifier with more layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_output_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights for better convergence
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[(n//4):(n//2)].fill_(1)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, _ = self.gru_layer(x)
        
        # Apply layer normalization
        gru_out = self.layer_norm(gru_out)
        
        # Apply attention mechanism
        attention_weights = self.attention(gru_out)
        attended_output = torch.sum(attention_weights * gru_out, dim=1)
        
        # Classification
        out = self.classifier(attended_output)
        
        return out


    @classmethod
    def supported_dataset(cls):
        return MFCCDataset 
