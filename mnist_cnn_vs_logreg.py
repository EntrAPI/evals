#!/usr/bin/env python3
"""
CNN vs Logistic Regression on CNN embeddings for MNIST 0 vs 1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mnist_full():
    """Load full MNIST (10 classes)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    return train_dataset, test_dataset


class CNN(nn.Module):
    """CNN for MNIST classification."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

    def get_embedding(self, x):
        """Extract embedding before final layer."""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_cnn(train_loader, test_loader, epochs=10):
    """Train CNN end-to-end."""
    model = CNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        acc = correct / total
        print(f"  Epoch {epoch+1}/{epochs}: Test Acc = {acc*100:.2f}%")

    return model, acc


def extract_embeddings(model, loader):
    """Extract embeddings from CNN."""
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            emb = model.get_embedding(data)
            embeddings.append(emb.cpu())
            labels.append(target)

    return torch.cat(embeddings), torch.cat(labels)


def train_logreg_on_embeddings(X_train, y_train, X_test, y_test, epochs=100):
    """Train logistic regression on embeddings."""
    X_train = X_train.to(DEVICE)
    y_train = y_train.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # Normalize
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = LogisticRegression(X_train.shape[1]).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    return test_acc, train_acc


def train_logreg_on_pixels(train_loader, test_loader, epochs=100):
    """Train logistic regression directly on pixel values."""
    # Flatten all data
    X_train, y_train = [], []
    X_test, y_test = [], []

    for data, target in train_loader:
        X_train.append(data.view(data.size(0), -1))
        y_train.append(target)
    for data, target in test_loader:
        X_test.append(data.view(data.size(0), -1))
        y_test.append(target)

    X_train = torch.cat(X_train).to(DEVICE)
    y_train = torch.cat(y_train).to(DEVICE)
    X_test = torch.cat(X_test).to(DEVICE)
    y_test = torch.cat(y_test).to(DEVICE)

    model = LogisticRegression(784).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(1) == y_test).float().mean().item()

    return test_acc


def run():
    print("=" * 70)
    print("Full MNIST: CNN vs LogReg on CNN Embeddings")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load data
    print("\nLoading full MNIST...")
    train_dataset, test_dataset = load_mnist_full()

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    results = {}

    # 1. Train CNN end-to-end
    print("\n" + "=" * 60)
    print("1. Training CNN (end-to-end)")
    print("=" * 60)
    cnn_model, cnn_acc = train_cnn(train_loader, test_loader, epochs=5)
    results['CNN (end-to-end)'] = cnn_acc

    # 2. Extract embeddings
    print("\n" + "=" * 60)
    print("2. Extracting CNN embeddings...")
    print("=" * 60)
    X_train_emb, y_train = extract_embeddings(cnn_model, train_loader)
    X_test_emb, y_test = extract_embeddings(cnn_model, test_loader)
    print(f"Embedding shape: {X_train_emb.shape}")

    # 3. Train LogReg on embeddings
    print("\n" + "=" * 60)
    print("3. Training LogReg on CNN embeddings")
    print("=" * 60)
    logreg_emb_acc, logreg_emb_train = train_logreg_on_embeddings(
        X_train_emb, y_train, X_test_emb, y_test
    )
    print(f"  Train: {logreg_emb_train*100:.2f}%, Test: {logreg_emb_acc*100:.2f}%")
    results['LogReg on CNN embeddings'] = logreg_emb_acc

    # 4. Train LogReg directly on pixels (baseline)
    print("\n" + "=" * 60)
    print("4. Training LogReg on raw pixels (baseline)")
    print("=" * 60)
    logreg_pixel_acc = train_logreg_on_pixels(train_loader, test_loader)
    print(f"  Test: {logreg_pixel_acc*100:.2f}%")
    results['LogReg on pixels'] = logreg_pixel_acc

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        bar = "█" * int(acc * 50)
        print(f"  {name:30s}: {acc*100:.2f}% {bar}")


if __name__ == "__main__":
    run()
