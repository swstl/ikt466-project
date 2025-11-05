import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from src.models.lstm.lstm import LSTM


class MFCCDataset(Dataset):
    
    def __init__(self, file_paths, labels, max_length=32):
        self.file_paths = file_paths
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        mfcc = np.load(self.file_paths[idx])
        label = self.labels[idx]
        
        if mfcc.shape[0] < self.max_length:
            pad = self.max_length - mfcc.shape[0]
            mfcc = np.pad(mfcc, ((0, pad), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:self.max_length]
        
        return torch.FloatTensor(mfcc), torch.LongTensor([label]).squeeze()


def load_dataset(data_dir, test_size=0.2, random_state=42):
    print(f"Loading data from: {data_dir}")
    
    class_folders = sorted([f for f in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, f))])
    
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    file_paths = []
    labels = []
    
    for class_name, label_idx in label_map.items():
        class_path = os.path.join(data_dir, class_name)
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        for npy_file in npy_files:
            file_paths.append(os.path.join(class_path, npy_file))
            labels.append(label_idx)
    
    print(f"Total samples: {len(file_paths)}")
    
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, random_state=random_state, 
        stratify=labels
    )
    
    print(f"Train samples: {len(train_paths)}")
    print(f"Test samples: {len(test_paths)}")
    
    return train_paths, test_paths, train_labels, test_labels, label_map


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in tqdm(loader, desc="Training", leave=False):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Validating", leave=False):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = 100 * accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels


def main():
    config = {
        'input_size': 13,
        'hidden_size': 128,
        'num_layers': 2,
        'num_classes': 31,
        'dropout': 0.3,
        'max_length': 32,
        'batch_size': 64,
        'lr': 0.001,
        'epochs': 50,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data_dir': os.path.join(project_root, 'data', 'processed', 'mfcc'),
        'checkpoint_dir': os.path.join(project_root, 'checkpoints'),
        'results_dir': os.path.join(project_root, 'results'),
    }
    
    print("="*60)
    print("Training LSTM on MFCC Temporal Sequences")
    print("="*60)
    print(f"Device: {config['device']}")
    print(f"Data directory: {config['data_dir']}")
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    train_paths, test_paths, train_labels, test_labels, label_map = load_dataset(
        config['data_dir'], test_size=0.2, random_state=42
    )
    
    train_dataset = MFCCDataset(train_paths, train_labels, config['max_length'])
    test_dataset = MFCCDataset(test_paths, test_labels, config['max_length'])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    model = LSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout']
    ).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_acc = 0
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, config['device']
        )
        
        val_loss, val_acc, val_preds, val_labels = validate(
            model, test_loader, criterion, config['device']
        )
        
        scheduler.step(val_acc)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_lstm.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config,
                'label_map': label_map
            }, checkpoint_path)
            print(f"✓ Saved best model with {val_acc:.2f}% accuracy")
    
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)
    
    checkpoint = torch.load(os.path.join(config['checkpoint_dir'], 'best_lstm.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, final_acc, final_preds, final_labels = validate(
        model, test_loader, criterion, config['device']
    )
    
    print(f"\nBest validation accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {final_acc:.2f}%")
    
    print("\nClassification Report:")
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]
    print(classification_report(
        final_labels, final_preds, 
        target_names=class_names, 
        zero_division=0
    ))
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', linewidth=2)
    plt.plot(val_accs, label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Training and Validation Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(config['results_dir'], 'lstm_training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining curves saved to: {plot_path}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved to: {config['checkpoint_dir']}/best_lstm.pth")
    print(f"Results saved to: {config['results_dir']}/")
    print("="*60)


if __name__ == "__main__":
    main()
