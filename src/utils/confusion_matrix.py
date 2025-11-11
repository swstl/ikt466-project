import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
from .evaluate import get_predictions


def generate_confusion_matrix(model, test_loader, device, model_name, timestamp, best_acc):
    """
    Generate and save a confusion matrix for the trained model.
    
    Args:
        model: The trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run inference on
        model_name: Name of the model for saving
        timestamp: Timestamp for unique file naming
        best_acc: Best accuracy achieved for display
    
    Returns:
        str: Path to the saved confusion matrix image
    """
    print("Generating confusion matrix...")
    predictions, targets = get_predictions(model, test_loader, device)
    
    # Create confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Get class names from dataset
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else [str(i) for i in range(len(np.unique(targets)))]
    
    # Create confusion matrix plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix\nTest Accuracy: {best_acc:.2f}%')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save confusion matrix
    os.makedirs('results', exist_ok=True)
    cm_path = f'results/{model_name}_{timestamp}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    return cm_path


def print_classification_report(model, test_loader, device):
    """
    Print a detailed classification report with precision, recall, and F1-score.
    
    Args:
        model: The trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to run inference on
    """
    from sklearn.metrics import classification_report
    
    predictions, targets = get_predictions(model, test_loader, device)
    
    # Get class names from dataset
    class_names = test_loader.dataset.classes if hasattr(test_loader.dataset, 'classes') else [str(i) for i in range(len(np.unique(targets)))]
    
    report = classification_report(targets, predictions, target_names=class_names)
    print("\nClassification Report:")
    print("=" * 60)
    print(report)