import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools


import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader

num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



wandb.init(project='cnn_cifar10', entity='achun')
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 5,
  "batch_size": 4
}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional Layer Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)

        # Convolutional Layer Block 2
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)

        # Convolutional Layer Block 3
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)  
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)

        # Fully Connected Layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 128) 
        self.dropout4 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        # Removing softmax from here, assuming nn.CrossEntropyLoss will be used which includes softmax.
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
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# change json + root dir here
json_file = "/home/adamchun/TARgan/dataset/cifar-10_imbalanced/dataset.json"
root_dir = "/home/adamchun/TARgan/dataset/cifar-10_imbalanced"
train_dataset = CustomCIFAR10Dataset(json_file=json_file, root_dir=root_dir, transform=transform)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./test_cifar10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Initialize the network, loss function, and optimizer
net = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
wandb.watch(net, log_freq=100)

# Train the network
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

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
predicted_labels = []
true_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predicted_labels.extend(predicted.cpu().tolist())  # Move the data back to CPU for further processing
        true_labels.extend(labels.cpu().tolist())
        
# Compute confusion matrix
cnf_matrix = confusion_matrix(true_labels, predicted_labels)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[str(i) for i in range(num_classes)],
                    title='Confusion matrix, without normalization')

# Save the plot
plot_path = "confusion_matrix.png"
plt.savefig(plot_path)
plt.close()

# Classifcation Report
report = classification_report(true_labels, predicted_labels, target_names=[str(i) for i in range(num_classes)])
print(report)

# Log the plot to wandb
wandb.log({"confusion_matrix": wandb.Image(plot_path)})

print('Confusion matrix plot saved and logged to wandb.')

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
