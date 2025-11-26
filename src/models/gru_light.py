import torch
import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class LightGRUCell(nn.Module):
    """
    Light GRU Cell - A simplified GRU with reduced parameters.
    Uses a single gate instead of separate update and reset gates.
    """
    def __init__(self, input_size, hidden_size):
        super(LightGRUCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Single gate for simplified gating mechanism
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate hidden state
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        
    def forward(self, x, hidden):
        combined = torch.cat([x, hidden], dim=1)
        
        # Single gate (simplified from update and reset gates)
        gate = torch.sigmoid(self.gate(combined))
        
        # Candidate hidden state
        candidate = torch.tanh(self.candidate(combined))
        
        # New hidden state with simplified gating
        new_hidden = gate * hidden + (1 - gate) * candidate
        
        return new_hidden

class GRU_Light(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=5, num_classes=30, dropout=0.3):
        super(GRU_Light, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        # Build Light GRU layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(LightGRUCell(cell_input_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout)
        
        # Single FC output layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states for all layers
        hidden = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                  for _ in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            input_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                hidden[layer] = self.cells[layer](input_t, hidden[layer])
                input_t = hidden[layer]
                
                # Apply dropout between layers (except last layer)
                if layer < self.num_layers - 1 and self.training:
                    input_t = self.dropout(input_t)
        
        # Use the last hidden state from the last layer
        out = self.classifier(hidden[-1])
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset
