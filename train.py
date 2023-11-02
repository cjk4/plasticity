import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets

import matplotlib.pyplot as plt

import multiprocessing as mp

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_task(task_id, model, trainloader, criterion, optimizer, device):
    model.to(device)
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[labels != -1], labels[labels != -1])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print(f"Task {task_id}, Loss: {running_loss / len(trainloader)}")

# Function for evaluating a model on a specific task
def evaluate_task(task_id, model, testloader, device, result_queue):
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.max(1)[1]
            total += (labels != -1).sum().item()
            correct += (predicted[labels != -1] == labels[labels != -1]).sum().item()
    accuracy = 100 * correct / total
    print(f"Task {task_id}, Accuracy: {accuracy}%")
    result_queue.put(accuracy)