import torch

# evaluation function
def evaluate(model, test_loader, criterion, device):
    model.eval()  # set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)

            batch_size = target.size(0)
            test_loss += loss.item() * batch_size
            _, predicted = torch.max(output, 1)
            total += batch_size
            correct += (predicted == target).sum().item()

    avg_loss = test_loss / total 
    accuracy = 100 * correct / total
    return avg_loss, accuracy
