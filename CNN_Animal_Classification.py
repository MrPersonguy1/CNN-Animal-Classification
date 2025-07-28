# -*- coding: utf-8 -*-
"""
##🟦Importing
"""

import torch
import torch.nn as nn

import torchvision  # Library with tools for computer vision
from torchvision import datasets  # Ready-to-use image datasets like CIFAR, MNIST
from torchvision.transforms import ToTensor, transforms  # Tools to process images
from torch.utils.data import DataLoader, Subset, Dataset  # Tools to handle data

from torchsummary import summary  # Prints a nice summary of your model
from tqdm import tqdm  # Adds progress bars to your loops

import matplotlib as plt  # Library for creating charts and visuals
import matplotlib.pyplot as plt  # The part of matplotlib for actually drawing plots
import numpy as np  # For working with arrays and math operations

print(torch.__version__)  # Shows which version of PyTorch you're using

"""
## 🟧Data Preprocessing
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_data = datasets.CIFAR10(  # Loads the CIFAR-10 training dataset
    root = 'data',
    train = True,
    download = True,
    transform = transform,
    target_transform = None,
)

test_data = datasets.CIFAR10(  # Loads the CIFAR-10 test dataset
    root = 'data',
    train = False,
    download = True,
    transform = transform,
    target_transform = None,
)

"""
### Data Exploration
"""

print(len(train_data))
print(train_data.data.shape)

print(len(test_data))
print(test_data.data.shape)

print(len(train_data.classes))
print(train_data.classes)

"""
### First image
"""

print(train_data[0][0].shape)
print(train_data[0][1])

"""
#### Visualize - Images will look weird because with normalized it
"""

image_num = 7 #@param {type:"raw"}
image = train_data[image_num][0].permute(1, 2, 0)
plt.imshow(image)
label = train_data[image_num][1]
plt.title(f'Class: {label}; {train_data.classes[label]}')
plt.show()

"""
## Building the Model
"""

class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding = 1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding = 1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding = 1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

    self.fully_connected = nn.Sequential(
        nn.Linear(64*4*4, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

    self.flatten = nn.Flatten()  # Flattens feature maps into a 1D vector



  def forward(self, x):
    # CODE HERE
    x = self.conv_layers(x)
    x = self.flatten(x)
    x = self.fully_connected(x)
    return x

model = ConvNet()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model.to(device)
summary(model, input_size=(3, 32, 32))

"""
## 🟥Training the Model

### Data Loader
"""

train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 32, shuffle = False)

images, labels = next(iter(train_dataloader))
print(f'Shape of the images: {images.shape}')
print(f'Shape of the labels: {labels.shape}')
print(f'Number of batches is: {len(iter(train_dataloader))}')

"""
### Loss Function & Optimizer
"""

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.001)

"""
### Train Loop
"""

def train_loop(train_dataloader, model, loss_fn, optimizer, epochs):
  model.train()
  train_loss = []

  for epoch in range(epochs):
    train_loss_epoch = 0
    for image, label in tqdm(train_dataloader, desc="Training Model"):
      optimizer.zero_grad()
      pred = model(image)
      loss = loss_fn(pred, label)
      loss.backward()
      optimizer.step()
      train_loss_epoch += loss.item()

    avg_loss = train_loss_epoch / len(train_dataloader)
    train_loss.append(avg_loss)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

  return train_loss

"""
### Train the Model
"""

num_epochs = 25
losses = train_loop(train_dataloader, model, loss_fn, optimizer, epochs=num_epochs)

print(losses)

epoch_list = list(range(1, 26))
plt.plot(epoch_list, losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

"""
## 🟪Testing
"""

def accuracy(correct, total):
  return correct/total * 100

def test_loop(test_dataloader, model):
  model.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for image, label in tqdm(test_dataloader, desc="Testing Model"):
      pred = model(image)
      correct += (pred.argmax(1) == label).type(torch.float).sum().item()
      total += len(label)

    print(f'Accuracy: {accuracy(correct, total)}')

test_loop(test_dataloader, model)

"""
###Visualize Testing
"""

rand_idx = 4 #@param {type:"raw"}
image, label = test_data[rand_idx]

print(image.shape)
with torch.no_grad():
  prediction = model(image.unsqueeze(0).to(device))

pred_idx = prediction[0].argmax().item()

plt.figure(figsize=(5,5))
plt.title(f'Prediction: {test_data.classes[pred_idx]} | Correct Label: {test_data.classes[label]}')
plt.imshow(image[0].squeeze(), cmap='gray')
plt.show()

"""
### Hyperparameter Tuning
"""

import torch
from pprint import pprint

print(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))

pretrained_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)

pretrained_model.to(device)

print(summary(pretrained_model, (3, 32, 32)))

test_loop(test_dataloader, pretrained_model)

rand_idx = torch.randint(0, len(test_data), (1,)).item()
image, label = test_data[rand_idx]

print(image.shape)

with torch.no_grad():
  prediction = pretrained_model(image.unsqueeze(0).to(device))
  print(prediction.shape)

pred_idx = prediction[0].argmax().item()

plt.figure(figsize=(5,5))
plt.title(f'Prediction: {test_data.classes[pred_idx]} | Correct Label: {test_data.classes[label]}')
plt.imshow(image.permute(1, 2, 0))
plt.show()
