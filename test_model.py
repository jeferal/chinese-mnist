import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch.utils.data import DataLoader

from utils.model import CNNNet
from utils.data import ChineseMnistDataset


def test_model(model, test_loader, criterion):
    """
    Test a pre-trained deep learning model on a test set.
    
    Args:
    - model: Pre-trained deep learning model
    - test_loader: DataLoader for the test set
    - criterion: Loss function for evaluation
    
    Returns:
    - test_loss: Average loss on the test set
    - test_accuracy: Accuracy on the test set
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()  # Assuming GPU is available
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / total
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%")
    
    return test_loss, test_accuracy

# Example usage:
# Assuming 'model' is your pre-trained model and 'test_loader' is DataLoader for the test set

# Define your loss function (e.g., CrossEntropyLoss for classification)
criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":

    model = CNNNet(num_classes=15)

    model_path = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/weights/model_1.pth"
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    annotations_file = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/data/chinese_mnist.csv"
    img_dir = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/data/data"
    dataset = ChineseMnistDataset(annotations_file, img_dir)

    generator1 = torch.Generator().manual_seed(42)
    SPLIT_SIZE = [10000, 2500, 2500]
    train_dataset, val_dataset, test_dataset = random_split(dataset, SPLIT_SIZE, generator=generator1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Call the testing function
    test_loss, test_accuracy = test_model(model, test_loader, criterion)
