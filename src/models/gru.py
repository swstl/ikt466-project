import torch.nn as nn

class GRU(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=10, dropout=0.3):
        super(GRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru_layer = nn.GRU(
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
        gru_out, h_n = self.gru_layer(x)

        out = h_n[-1]

        out = self.classifier(out)
        
        return out