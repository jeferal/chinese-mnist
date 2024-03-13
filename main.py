import torch

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from utils.model import CNNNet

import torch.nn as nn

from utils.data import ChineseMnistDataset
from utils.train import train_epoch, test_epoch, plot_results

"""
    Training of a Convolutional Neural Netowrk to classify
    a Chinese MNIST like dataset.

    Improvements TODO:
        1. Differentiate between train, test, validation datasets. Create validation procedure. With confusion matrix.
        2. Add propper loggers, tensorboard, etc.
        3. Create method to compare different trainings
        4. Learning rate schedule
        5. Hyperparameter tunning / search
        6. Post in medium with everything.
"""


HYPER_PARAMS = {
    "batch_size": 64,
    "num_epochs": 10,
    "learning_rate" : 1e-3,
    "weight_decay": 0.01,
    "criterion": nn.CrossEntropyLoss,
    "log_interval": 100
}

def main():
    # Configure device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # TODO: Understand this line
    torch.cuda.synchronize(device=device)

    # Create the custom Chinese Mnist Dataset
    annotations_file = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/data/chinese_mnist.csv"
    img_dir = "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/data/data"

    dataset = ChineseMnistDataset(annotations_file, img_dir)

    # Split the dataset
    SPLIT_SIZE = [10000, 2500, 2500]
    train_dataset, val_dataset, test_dataset = random_split(dataset, SPLIT_SIZE)

    # Create the train, validation and test datasets
    train_loader = DataLoader(train_dataset, batch_size=HYPER_PARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=HYPER_PARAMS["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=HYPER_PARAMS["batch_size"], shuffle=False)

    # Create the network model
    # The Chinese MNIST Dataset has 15 outputs
    network = CNNNet(num_classes=15)
    network.to(device)
    print(network)

    # Define the optimizer
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=HYPER_PARAMS['learning_rate'],
                                 weight_decay=HYPER_PARAMS['weight_decay'])
    
    criterion = HYPER_PARAMS['criterion']()

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # Loop for epochs
    for epoch in range(HYPER_PARAMS['num_epochs']):

        # Compute & save the average training loss for the current epoch
        train_loss, train_acc = train_epoch(train_loader, network, optimizer, criterion, HYPER_PARAMS["log_interval"], device, epoch)
        # Append the metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Test one epoch
        test_loss, test_accuracy = test_epoch(test_loader, network, criterion, device)

        # Append test metrics
        test_losses.append(test_loss)
        test_accs.append(test_accuracy)

    # Save the model
    torch.save(network.state_dict(), "/home/jesusferrandiz/Learning/pytorch/ml-ops-session-2/weights/model_1.pth")

    # Show the results
    # Plot the plots of the learning curves
    plot_results(train_losses, test_losses, train_accs, test_accs)

if __name__ == "__main__":
    main()