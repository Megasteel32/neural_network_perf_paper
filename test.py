import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report


def test_model(model_name):
    """
    Test a model on EMNIST and NovelDigits datasets
    Args:
        model_name: Name of the model folder (e.g., 'rnn', 'cnn', etc.)
    """
    model_path = f'trained_models/{model_name}/model.pt'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        model = torch.load(model_path)
        model = model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load datasets
    print("\nLoading datasets...")
    try:
        test_images = torch.load('datasets/emnist/test_images.pt')
        test_labels = torch.load('datasets/emnist/test_labels.pt')
        novel_images = torch.load('datasets/noveldigits/novel_digits_images.pt')
        novel_labels = torch.load('datasets/noveldigits/novel_digits_labels.pt')

        test_dataset = TensorDataset(test_images, test_labels)
        novel_dataset = TensorDataset(novel_images, novel_labels)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        novel_loader = DataLoader(novel_dataset, batch_size=64, shuffle=False)
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        return

    # Test function
    def evaluate(loader, name):
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Calculate accuracy
        accuracy = (all_preds == all_labels).mean() * 100

        # Print results
        print(f"\n{name} Results:")
        print("-" * 50)
        print(f"Accuracy: {accuracy:.2f}%")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

        return accuracy

    # Run tests
    try:
        emnist_acc = evaluate(test_loader, "EMNIST Test Set")
        novel_acc = evaluate(novel_loader, "NovelDigits Test Set")

        print("\nFinal Results:")
        print("-" * 50)
        print(f"EMNIST Accuracy: {emnist_acc:.2f}%")
        print(f"NovelDigits Accuracy: {novel_acc:.2f}%")

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")