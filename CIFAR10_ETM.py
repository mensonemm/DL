import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001


# Data augmentation and normalization for training
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Normalization for testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform_train)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False)

# Function to visualize images with class labels
def imshow(img, labels, class_names):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(8, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'Labels: {", ".join([class_names[l] for l in labels])}')
    plt.axis('off')
    plt.show()

# Get some random training images with labels
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Define class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Show images with class labels
imshow(torchvision.utils.make_grid(images), labels, class_names)

# CNN Model (ConvNet)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Adjust the input size of the fully connected layer
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# LSTM Model
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        # Input size: 32*32*3 for CIFAR-10 images
        self.lstm = nn.LSTM(32*32*3, 32, batch_first=True)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # Flatten the image to have features in sequence
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, -1, height * width * channels)  # Reshape to (batch, seq_len, features)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Function to calculate top-k accuracy
def topk_accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Function to train and evaluate a model and record the results
def train_and_evaluate_record(model_name, optimizer_name, loss_name, model, optimizer, criterion, train_loader, test_loader, epochs):
    model.to(device)
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




        # Evaluation phase
        model.eval()
        top1_acc, top5_acc, all_labels, all_predictions = [], [], [], []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                top1, top5 = topk_accuracy(outputs, labels, topk=(1, 5))
                top1_acc.append(top1.item())
                top5_acc.append(top5.item())
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        # Calculate F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=1)

        # Print and record the results
        print(f'Epoch [{epoch + 1}/{epochs}] - Top-1 Acc: {np.mean(top1_acc):.2f}%, '
              f'Top-5 Acc: {np.mean(top5_acc):.2f}%, F1 Score: {f1:.2f}')

        results.append([model_name, optimizer_name, loss_name, np.mean(top1_acc), np.mean(top5_acc), f1])

# Create models
cnn_model = ConvNet()
lstm_model = LSTMNet()

# Define loss functions
loss_functions = [('CrossEntropyLoss', nn.CrossEntropyLoss()), ('NLLLoss', nn.NLLLoss())]

# Define optimizers
optimizers = [('SGD', torch.optim.SGD(cnn_model.parameters(), lr=learning_rate)),
              ('Adam', torch.optim.Adam(cnn_model.parameters(), lr=learning_rate))]

# Create a list to store evaluation results
results = []

# Train and evaluate models
for model_name, model in [('CNN', cnn_model), ('LSTM', lstm_model)]:
    for optimizer_name, optimizer in optimizers:
        for loss_name, criterion in loss_functions:
            print(f"Training and evaluating {model_name} models with {loss_name} Loss and {optimizer_name} optimizer:")
            train_and_evaluate_record(model_name, optimizer_name, loss_name, model, optimizer, criterion, train_loader, test_loader, num_epochs)

# Create a DataFrame to display the results
results_df = pd.DataFrame(results, columns=['Model', 'Optimizer', 'Loss Function', 'Top-1 Accuracy', 'Top-5 Accuracy', 'F1 Score'])
print(results_df)
