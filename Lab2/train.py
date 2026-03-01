import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random
import os
from model import MNISTConvNet

def set_seed(seed=42):
    """Ensures consistent results across different runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(batch_size=64):
    """Prepares Training and Testing DataLoaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)
    return train_loader, test_loader, train_set

def run_experiment(mode="full_train"):
    set_seed(42)
    train_loader, test_loader, full_train_set = get_loaders()
    model = MNISTConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    if mode == "sanity_check":
        subset = Subset(full_train_set, [0])
        train_loader = DataLoader(subset, batch_size=1)
        epochs = 50
    else:
        epochs = 5

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            if mode == "sanity_check": break

        avg_train_loss = total_train_loss / len(train_loader)
        history.append(avg_train_loss)

        if mode == "full_train":
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            accuracy = 100. * correct / len(test_loader.dataset)
            print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Test Accuracy: {accuracy:.2f}%")
        elif epoch % 10 == 0:
            print(f"Sanity Check Epoch {epoch}: Loss {avg_train_loss:.6f}")

    # Save the Gold Standard model
    if mode == "full_train":
        torch.save(model.state_dict(), "gold_standard_mnist.pth")

    # Final visualization
    plt.figure(figsize=(8, 4))
    plt.plot(history, label='Training Loss')
    plt.title(f"Optimization Path: {mode}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_experiment(mode="sanity_check")
    run_experiment(mode="full_train")