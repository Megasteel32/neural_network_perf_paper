import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from datetime import timedelta
from typing import Tuple, Type, Union
from pathlib import Path

# Model imports
from models.basenn.model import BaseNN
from models.partialcnn.model import PartialCNN
from models.fullcnn.model import FullCNN
from models.rnn.model import RNN

# Model registry
MODEL_REGISTRY = {
    'basenn': BaseNN,
    'fullcnn': FullCNN,
    'partialcnn': PartialCNN,
    'rnn': RNN
}


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    loss = F.nll_loss(logits, labels).item()
    pred = logits.argmax(dim=1)
    accuracy = (pred == labels).float().mean().item()
    return loss, accuracy


def train_epoch(model, optimizer, train_loader, device, epoch):
    model.train()
    total_loss = 0
    total_accuracy = 0
    start_time = time.time()

    steps_per_epoch = len(train_loader)
    update_interval = max(steps_per_epoch // 10, 1)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        batch_loss, batch_accuracy = compute_metrics(output, target)
        total_loss += batch_loss
        total_accuracy += batch_accuracy

        if (batch_idx + 1) % update_interval == 0:
            print(f"\rEpoch {epoch + 1}/{model.EPOCHS} - "
                  f"Progress: {((batch_idx + 1) / steps_per_epoch * 100):.1f}% - "
                  f"Loss: {total_loss / (batch_idx + 1):.3f} - "
                  f"Accuracy: {total_accuracy / (batch_idx + 1):.3f}",
                  end='', flush=True)

    print()
    return total_loss / len(train_loader), total_accuracy / len(train_loader)


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss, accuracy = compute_metrics(output, target)
        total_loss += loss
        total_accuracy += accuracy

    return total_loss / len(test_loader), total_accuracy / len(test_loader)


def train_model(model_type: str = 'basenn', use_augmentation: bool = False) -> None:
    """
    Train a neural network model with specified configuration.

    Args:
        model_type (str): Type of model to train ('basenn' or 'fullcnn')
        use_augmentation (bool): Whether to use augmented dataset
    """
    # Get model class from registry
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type must be one of {list(MODEL_REGISTRY.keys())}")

    ModelClass = MODEL_REGISTRY[model_type]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set paths based on configuration
    model_dir = Path('trained_models')
    if use_augmentation:
        model_dir = model_dir / f'{model_type}_augmented'
        data_dir = Path('datasets/augmented_emnist')
        train_images_file = 'augmented_images.pt'
        train_labels_file = 'augmented_labels.pt'
    else:
        model_dir = model_dir / model_type
        data_dir = Path('datasets/emnist')
        train_images_file = 'train_images.pt'
        train_labels_file = 'train_labels.pt'

    model_path = model_dir / 'model.pt'
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        print("Loading existing model...")
        model = torch.load(model_path, map_location=device)
        print("Model loaded successfully!")
        return model

    print("No existing model found. Training new model...")

    # Load dataset
    print("Loading dataset...")
    train_images = torch.load(data_dir / train_images_file)
    train_labels = torch.load(data_dir / train_labels_file)
    test_images = torch.load(data_dir / 'test_images.pt')
    test_labels = torch.load(data_dir / 'test_labels.pt')

    print(f"Dataset loaded: {len(train_images)} training samples, {len(test_images)} test samples")

    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=ModelClass.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=ModelClass.BATCH_SIZE, shuffle=False)

    # Initialize model and optimizer
    model = ModelClass().to(device)
    optimizer = optim.Adam(model.parameters(), lr=model.LEARNING_RATE)

    # Training loop
    print(f"\nStarting training for {model.EPOCHS} epochs...")
    best_test_accuracy = 0.0

    for epoch in range(model.EPOCHS):
        # Train and evaluate
        train_loss, train_accuracy = train_epoch(model, optimizer, train_loader, device, epoch)
        test_loss, test_accuracy = evaluate(model, test_loader, device)

        # Update best accuracy
        best_test_accuracy = max(best_test_accuracy, test_accuracy)

        print(f"\nEpoch {epoch + 1}/{model.EPOCHS} completed")
        print(f"Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}")
        print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}\n")

    # Save the model
    print("Saving model...")
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")

    print("\nModel is ready for use!")
    return model