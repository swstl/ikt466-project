from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.utils.evaluate import evaluate 

import torch
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
def train(model, train_loader, test_loader, epochs=10, lr=0.001):
    model = model.to(device, non_blocking=True)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # create TensorBoard writer with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/{model.name}_{timestamp}')

    print(f"Training on {device}")
    print("-" * 60)

    best_weights = None
    best_acc = 0

    for epoch in range(epochs):
        epoch_time = time.time()
        # train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_time = time.time() - epoch_time

        # evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # log to TensorBoard
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
            best_weights = model.state_dict()


    writer.close()

    os.makedirs('trained', exist_ok=True)
    model_path = f'trained/{model.name}_{timestamp}_{best_acc:.2f}.pth'
    torch.save(best_weights, model_path)
    print(f"Model saved to {model_path}")
    return model
