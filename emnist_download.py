from matplotlib import pyplot as plt
import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader


def prepare_emnist_dataset(base_dir='datasets/emnist', display_samples=True):
    """
    Download, prepare and optionally display EMNIST dataset samples.

    Args:
        base_dir (str): Directory to store the dataset
        display_samples (bool): Whether to display sample images from the dataset

    Returns:
        tuple: (train_images, train_labels, test_images, test_labels) as torch tensors
    """
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)

    # Set up data transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download datasets
    train_dataset = EMNIST(
        root='datasets',
        split='digits',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = EMNIST(
        root='datasets',
        split='digits',
        train=False,
        download=True,
        transform=transform
    )

    # Create data loaders and get tensors
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    # Save tensors
    print("Saving dataset to", base_dir)
    torch.save(train_images, os.path.join(base_dir, 'train_images.pt'))
    torch.save(train_labels, os.path.join(base_dir, 'train_labels.pt'))
    torch.save(test_images, os.path.join(base_dir, 'test_images.pt'))
    torch.save(test_labels, os.path.join(base_dir, 'test_labels.pt'))

    print("\nDataset saved with shapes:")
    print(f"Training images: {train_images.shape}")
    print(f"Training labels: {train_labels.shape}")
    print(f"Test images: {test_images.shape}")
    print(f"Test labels: {test_labels.shape}")

    if display_samples:
        display_examples(train_images, train_labels, "EMNIST Training Dataset Examples")

    return train_images, train_labels, test_images, test_labels


def display_examples(images, labels, title, samples_per_class=10):
    """
    Display example digits from the dataset

    Args:
        images (torch.Tensor): Image tensor
        labels (torch.Tensor): Label tensor
        title (str): Title for the plot
        samples_per_class (int): Number of samples to display per class
    """
    plt.figure(figsize=(15, 8))
    plt.suptitle(title)

    for digit in range(10):
        digit_indices = (labels == digit).nonzero().squeeze()

        if len(digit_indices) == 0:
            continue

        indices = digit_indices[:samples_per_class]

        for idx, img_idx in enumerate(indices):
            ax = plt.subplot(10, samples_per_class, digit * samples_per_class + idx + 1)
            img = images[img_idx].squeeze()
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            if idx == 0:
                plt.title(f'Digit: {digit}')

    plt.tight_layout()
    plt.show()