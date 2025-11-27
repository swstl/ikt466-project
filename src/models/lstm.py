import torch
import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

# Best performing normal LSTM model:
# To try out the other experiments on simple lsmt models add layers here. Dropout may also be removed as we have documented in the report.
class LSTM_5L1F(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_classes=30, dropout=0.3):
        super(LSTM_5L1F, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=5, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset


###############################
# Exsisting LSTM architectures#
###############################

# Bidirectional LSTM
class BiLSTM(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=5, num_classes=30, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        out = self.dropout(combined)
        return self.fc(out)

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset

# Deeper Bidirectional LSTM
class DeepBiLSTM(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=3, num_classes=30, dropout=0.3):
        super(DeepBiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        out = self.dropout(combined)
        return self.fc(out)

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset

# LSTM with Attention
class LSTMAttention(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=5, num_classes=30, dropout=0.3):
        super(LSTMAttention, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        attention_scores = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        weighted = lstm_out * attention_weights
        attended = torch.sum(weighted, dim=1)
        
        out = self.dropout(attended)
        return self.fc(out)

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset

# Peephole LSTM
class PeepholeLSTM(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=5, num_classes=30, dropout=0.3):
        super(PeepholeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.peepholes = nn.ModuleList([
            nn.ModuleDict({
                'W_ci': nn.Linear(hidden_size, hidden_size, bias=False),
                'W_cf': nn.Linear(hidden_size, hidden_size, bias=False),
                'W_co': nn.Linear(hidden_size, hidden_size, bias=False),
            })
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        
        h = [torch.zeros(batch_size, self.hidden_size, device=device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=device) 
             for _ in range(self.num_layers)]
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](
                    x_t if layer == 0 else h[layer-1], 
                    (h[layer], c[layer])
                )
                
                c[layer] = c[layer] + 0.1 * self.peepholes[layer]['W_ci'](c[layer])
                
                if layer < self.num_layers - 1:
                    h[layer] = self.dropout(h[layer])
        
        out = self.dropout(h[-1])
        return self.fc(out)

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset