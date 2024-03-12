import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import wandb

import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

num_classes = 10

wandb.init(project='cnn_cifar10', entity='achun')
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 5,
  "batch_size": 4
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x, dim=1)
        return x
    
    transforms_aug = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
    ], p=0.3),
    transforms.RandomApply([
        transforms.RandomVerticalFlip(),
    ], p=0.3),
    transforms.RandomApply([
        transforms.RandomCrop(32, padding=4),
    ], p=0.3),
    transforms.RandomApply([
        transforms.RandomRotation(15),
    ], p=0.3),
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize with mean and std
])
    
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(json_file, 'r') as f:
            self.dataset_info = json.load(f)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset_info['labels'])

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataset_info['labels'][idx][0])
        image = Image.open(img_name)
        label = self.dataset_info['labels'][idx][1]

        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
json_file = "/home/adamchun/TARgan/dataset/cifar-10_imbalanced/dataset.json"
root_dir = "/home/adamchun/TARgan/dataset/cifar-10_imbalanced"
train_dataset = CustomCIFAR10Dataset(json_file=json_file, root_dir=root_dir, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./test_cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# Initialize the network, loss function, and optimizer
net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
wandb.watch(net, log_freq=100)

# Train the network
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            avg_loss = running_loss / 2000
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            wandb.log({"loss": avg_loss, "epoch": epoch})
            running_loss = 0.0

print('Finished Training')

# Test the network on the test data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Test the network on the test data
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.tolist())
        true_labels.extend(labels.tolist())

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
