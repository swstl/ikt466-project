import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class GRU_Layers(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=1, num_classes=30, dropout=0.3, fc_layers=1):
        super(GRU_Layers, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc_layers = fc_layers

        # Configurable GRU layers
        self.gru_layer = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Build classifier with configurable FC layers
        classifier_layers = []
        
        if fc_layers == 1:
            # 1 FC layer: just output
            classifier_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            ])
        elif fc_layers == 2:
            # 2 FC layers: hidden -> output
            classifier_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes)
            ])
        elif fc_layers == 3:
            # 3 FC layers: hidden -> hidden -> output
            classifier_layers.extend([
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, num_classes)
            ])
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, x):
        # GRU forward pass
        gru_out, h_n = self.gru_layer(x)
        
        # Use the last hidden state
        out = h_n[-1]
        
        # Classification
        out = self.classifier(out)
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset
