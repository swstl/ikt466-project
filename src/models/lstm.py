import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class LSTM(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=30, dropout=0.3):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm_layer(x)

        out = h_n[-1]

        out = self.classifier(out)
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset 
