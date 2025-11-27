import torch
import torch.nn as nn
from data.dataset import MFCCDataset
from models.base import BASE

class GRU_Seq2Seq(BASE):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=30, dropout=0.3):
        super(GRU_Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Encoder: processes input sequence
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: generates output from encoded representation
        self.decoder = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection layer
        self.fc_out = nn.Linear(hidden_size, num_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Encode the input sequence
        encoder_outputs, hidden = self.encoder(x)
        
        # Use encoder's final hidden state as context
        # Decoder input: repeat the last encoder output for decoding steps
        decoder_input = encoder_outputs[:, -1:, :].repeat(1, seq_len, 1)
        
        # Decode using the context from encoder
        decoder_outputs, _ = self.decoder(decoder_input, hidden)
        
        # Use the last decoder output for classification
        last_output = decoder_outputs[:, -1, :]
        
        # Apply dropout and classify
        out = self.dropout(last_output)
        out = self.fc_out(out)
        
        return out

    @classmethod
    def supported_dataset(cls):
        return MFCCDataset
