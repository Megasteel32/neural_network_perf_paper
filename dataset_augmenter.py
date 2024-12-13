import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from pathlib import Path
import shutil
import os
import matplotlib.pyplot as plt


class EMNISTAugmenter(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

        # Pre-calculate augmentation parameters
        self.rotation_angles = torch.FloatTensor(len(images)).uniform_(-15, 15)
        self.shear_x = torch.FloatTensor(len(images)).uniform_(-0.1, 0.1)
        self.shear_y = torch.FloatTensor(len(images)).uniform_(-0.1, 0.1)
        self.translate_x = torch.FloatTensor(len(images)).uniform_(-3, 3)
        self.translate_y = torch.FloatTensor(len(images)).uniform_(-3, 3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Ensure 4D format [B, C, H, W]
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Apply rotation
        angle = self.rotation_angles[idx]
        image = TF.rotate(image, angle.item(), fill=0)

        # Apply shear
        shear_matrix = torch.tensor([
            [1.0, self.shear_x[idx], 0.0],
            [self.shear_y[idx], 1.0, 0.0]
        ])

        grid = F.affine_grid(
            shear_matrix.unsqueeze(0),
            image.size(),
            align_corners=False
        )
        image = F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros'
        )

        # Apply translation
        translate_matrix = torch.tensor([
            [1.0, 0.0, self.translate_x[idx] / image.size(-1)],
            [0.0, 1.0, self.translate_y[idx] / image.size(-2)]
        ])

        grid = F.affine_grid(
            translate_matrix.unsqueeze(0),
            image.size(),
            align_corners=False
        )
        image = F.grid_sample(
            image,
            grid,
            align_corners=False,
            mode='bilinear',
            padding_mode='zeros'
        )

        return image.squeeze(0), label


def create_augmented_dataset(images, labels, num_augmentations=2, batch_size=512):
    """Create augmented versions of the dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    original_size = len(images)
    total_size = original_size * (num_augmentations + 1)

    # Initialize output tensors
    augmented_images = torch.zeros((total_size, 1, 28, 28), dtype=torch.float32)
    augmented_labels = torch.zeros(total_size, dtype=torch.long)

    # Copy original images
    print("Copying original images...")
    if len(images.shape) == 3:
        images = images.unsqueeze(1)
    augmented_images[:original_size] = images
    augmented_labels[:original_size] = labels

    # Generate augmentations
    print("\nGenerating augmentations...")
    for aug_idx in range(num_augmentations):
        dataset = EMNISTAugmenter(images, labels)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )

        start_idx = original_size * (aug_idx + 1)
        current_idx = start_idx

        for batch_images, batch_labels in tqdm(dataloader,
                                               desc=f"Augmentation {aug_idx + 1}/{num_augmentations}"):
            batch_size = len(batch_images)
            end_idx = current_idx + batch_size

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            augmented_images[current_idx:end_idx] = batch_images.cpu()
            augmented_labels[current_idx:end_idx] = batch_labels.cpu()

            current_idx = end_idx

    return augmented_images, augmented_labels


def display_examples(images, labels, title, samples_per_class=10):
    """Display example digits from the dataset"""
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
            plt.imshow(img.cpu(), cmap='gray')
            plt.axis('off')
            if idx == 0:
                plt.title(f'Digit: {digit}')

    plt.tight_layout()
    plt.show()


def augment_emnist_dataset(
        source_dir='datasets/emnist',
        save_dir='datasets/augmented_emnist',
        num_augmentations=9,
        batch_size=512,
        force_regenerate=False
):
    """
    Main function to handle EMNIST dataset augmentation.

    Parameters:
    -----------
    source_dir : str
        Directory containing the original EMNIST dataset
    save_dir : str
        Directory to save the augmented dataset
    num_augmentations : int
        Number of augmented versions to create for each image
    batch_size : int
        Batch size for processing images
    force_regenerate : bool
        If True, regenerate augmentations even if they already exist

    Returns:
    --------
    tuple
        (augmented_images, augmented_labels)
    """
    save_dir = Path(save_dir)
    aug_images_path = save_dir / 'augmented_images.pt'
    aug_labels_path = save_dir / 'augmented_labels.pt'

    # Check if augmented datasets already exist
    if not force_regenerate and aug_images_path.exists() and aug_labels_path.exists():
        print("Loading existing augmented datasets...")
        aug_images = torch.load(aug_images_path)
        aug_labels = torch.load(aug_labels_path)
    else:
        print("Creating new augmented datasets...")
        # Load EMNIST data
        train_images = torch.load(os.path.join(source_dir, 'train_images.pt'))
        train_labels = torch.load(os.path.join(source_dir, 'train_labels.pt'))

        # Create augmentations
        aug_images, aug_labels = create_augmented_dataset(
            train_images,
            train_labels,
            num_augmentations=num_augmentations,
            batch_size=batch_size
        )

        # Save augmented dataset
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(aug_images, aug_images_path)
        torch.save(aug_labels, aug_labels_path)

        # Copy test set
        shutil.copy(os.path.join(source_dir, 'test_images.pt'), save_dir)
        shutil.copy(os.path.join(source_dir, 'test_labels.pt'), save_dir)

    print("\nDataset shape:")
    print(f"Training images: {aug_images.shape}")
    print(f"Training labels: {aug_labels.shape}")

    display_examples(aug_images, aug_labels, "Augmented Dataset Examples")

    return aug_images, aug_labels


