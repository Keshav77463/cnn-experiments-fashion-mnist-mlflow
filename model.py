from main import FashionCNN
import torchvision
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from main import (
    FashionCNN,
    train_loader,
    test_loader,
    device
)

import torch
model = FashionCNN()
print(model)

model = FashionCNN().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10   # start small
import mlflow
import mlflow.pytorch

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("FashionMNIST_CNN")
mlflow.autolog()
with mlflow.start_run():

    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("learning_rate", 0.001)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.4f}")
model.eval()
correct, total = 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
mlflow.log_metric("test_accuracy", test_accuracy)

print(f"Test Accuracy: {test_accuracy:.4f}")
