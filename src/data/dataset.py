###############################################
######### This downloades the dataset #########
###############################################
from torch.utils.data import Dataset, DataLoader, random_split
from torch import Generator
from pathlib import Path

import numpy as np
import kagglehub
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Download the dataset from Kaggle
def download_dataset():
    """ Downloads the Google Speech Commands dataset from Kaggle """
    path = kagglehub.dataset_download("neehakurelli/google-speech-commands")
    print("😡😡😡😡😡 Dataset downloaded to: 😡😡😡😡", path)
    return path



class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('_')])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for cls in self.classes:
            cls_dir = self.data_dir / cls
            for file_path in cls_dir.glob('*.npy'):
                samples.append((file_path, self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        data = np.load(file_path)
        
        # Handle different data types:
        # - MFCC data is 2D (time_steps, features) - keep as is for RNNs
        # - Spectrogram data is 2D (freq_bins, time_steps) - add channel dim for CNNs
        if len(data.shape) == 2:
            # Check if this looks like MFCC data (typically has 13 features)
            # or spectrogram data (typically has more frequency bins)
            if data.shape[1] <= 20:  # Likely MFCC data (13 features)
                # Keep 2D for RNN models (will become 3D when batched)
                pass
            else:  # Likely spectrogram data
                # Add channel dimension for CNN models
                data = np.expand_dims(data, axis=0)
                
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        if self.transform:
            data_tensor = self.transform(data_tensor)
        return data_tensor, label_tensor

    @classmethod
    def create_loaders(cls, data_dir, batch_size=64, test_split=0.2, shuffle=True, random_seed=42, num_workers=1, pin_memory=False, persistent_workers=False):
        if device.type == 'cpu':
            num_workers = 0
            pin_memory = False
            persistent_workers = False

        dataset = cls(data_dir)
        total_size = len(dataset)
        test_size = int(total_size * test_split)
        train_size = total_size - test_size

        generator = Generator().manual_seed(random_seed)
        train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False
        )

        return train_loader, test_loader, dataset
