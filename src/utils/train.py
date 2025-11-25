from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.utils.evaluate import evaluate 
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
import torch.nn as nn
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the actual trianing:
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for _, (data, target) in enumerate(train_loader):
        # move data to device (GPU/CPU)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        output = model(data)

        # calculate loss
        loss = criterion(output, target)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

        # track statistics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# main training loop
def train(model, epochs=10, lr=0.001, delete_loaders=True):
    model = model.to(device, non_blocking=True)
    train_loader = model.train_loader
    test_loader = model.test_loader

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create TensorBoard writer with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    writer = SummaryWriter(f'runs/{model.get_name()}_{timestamp}')

    print(f"Training on {device}")
    print("-" * 60)

    best_acc_weights = None
    best_acc = 0
    best_acc_epoch = 0
    best_loss = float('inf')
    best_loss_weights = None
    best_loss_epoch = 0

    best_model_metrics = None

    for epoch in range(epochs):
        epoch_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - epoch_time

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
        print(f"  Training Time (epoch): {epoch_time:.2f} seconds")
        print("-" * 60)

        if test_acc > best_acc:
            best_acc = test_acc
            best_acc_weights = model.state_dict().copy()
            best_acc_epoch = epoch + 1

            # Calculate detailed metrics for best model
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    output = model(data)
                    _, predicted = torch.max(output.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())

            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)

            best_model_metrics = {
                'accuracy': best_acc,
                'f1': f1_score(all_targets, all_preds, average='weighted'),
                'precision': precision_score(all_targets, all_preds, average='weighted', zero_division=0), #type: ignore
                'recall': recall_score(all_targets, all_preds, average='weighted', zero_division=0), #type: ignore
                'confusion_matrix': confusion_matrix(all_targets, all_preds),
                'epoch': best_acc_epoch
            }

        if test_loss < best_loss:
            best_loss = test_loss
            best_loss_weights = model.state_dict().copy()
            best_loss_epoch = epoch + 1

    writer.close()


    os.makedirs('trained', exist_ok=True)
    model_path_acc = f'trained/{model.get_name()}_{timestamp}_acc:{best_acc:.2f}.pth'
    model_path_loss = f'trained/{model.get_name()}_{timestamp}_loss:{best_loss:.4f}.pth'
    torch.save(best_acc_weights, model_path_acc)
    torch.save(best_loss_weights, model_path_loss)

    if best_model_metrics is None:
        best_model_metrics = {
            'accuracy': best_acc,
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'confusion_matrix': np.array([]),
            'epoch': best_acc_epoch
        }

    # Print best model results
    print("\n" + "=" * 60)
    print("BEST MODEL RESULTS")
    print("=" * 60)
    print(f"Epoch: {best_model_metrics['epoch']}")
    print(f"Accuracy:  {best_model_metrics['accuracy']:.2f}%")
    print(f"F1 Score:  {best_model_metrics['f1']:.4f}")
    print(f"Precision: {best_model_metrics['precision']:.4f}")
    print(f"Recall:    {best_model_metrics['recall']:.4f}")
    print(f"\nBest test loss: {best_loss:.4f} (epoch {best_loss_epoch})")
    print("=" * 60)

    cm = best_model_metrics['confusion_matrix']
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix - Best Model (Epoch {best_model_metrics["epoch"]})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = f'trained/{model.get_name()}_{timestamp}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to: {cm_path}")
    print("=" * 60)

    if delete_loaders:
        del train_loader
        del test_loader
        del model.train_loader
        del model.test_loader

    return model
