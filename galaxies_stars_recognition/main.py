import lightning as L
import numpy as np
import pandas as pd
import torch
import os
import torchvision.transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

from Model import Model

# HyperParams
output_size_of_pooling = 3
learning_rate = 0.001
batch_size = 64
torch.set_num_threads(16)
num_classes = 2
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
epsilon = 0.001


filepath_train = "data/train"
filepath_test = "data/test"


def count_jpg_files(directory):
    jpg_files = get_all_jpg_files_in(directory)
    # Return the count of JPG files
    return len(jpg_files)


def get_all_jpg_files_in(directory):
    # List all files in the directory
    files = os.listdir(directory)
    # Filter files to include only JPG files
    jpg_files = [file for file in files if file.lower().endswith('.jpg')]
    return jpg_files


transform = transforms.Compose([
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Randomly crops and resizes images to 224x224
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flips images horizontally
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])
train_dataset = datasets.ImageFolder(root=filepath_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16)

# Check some dataset properties
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Class names: {train_dataset.classes}")
print(f"Number of images: {len(train_dataset)}")


def imshow(img, mean=mean, std=std):
    # Denormalize the image
    img = img.numpy().transpose((1, 2, 0))  # Change dimensions to (H,W,C)
    img = std * img + mean  # Denormalize
    img = np.clip(img, 0, 1)  # Clip to range [0, 1]
    plt.imshow(img)
    plt.axis('off')  # Turn off axis numbers and ticks


# Get a batch of images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Plot images
# fig = plt.figure(figsize=(8, 8))  # Figure size
# for idx in range(4):
#     ax = fig.add_subplot(2, 2, idx + 1, xticks=[], yticks=[])
#     imshow(images[idx])
#     ax.set_title(train_dataset.classes[labels[idx]])
#
# plt.show()

# transform = torchvision.transforms.Compose([ToTensor()])
# train_dataset = ImageDataset()
# test_dataset = CustomDataset(filepath_labels_csv, filepath_to_classes_csv, 'dataset/test', transform=transform)
#
model = Model.load_from_checkpoint(checkpoint_path="lightning_logs/version_2/checkpoints/88.7%epoch=32-step=8250.ckpt",
                                   batch_size=batch_size,
                                   learning_rate=learning_rate,
                                   num_classes=num_classes,
                                   output_size_of_pooling=output_size_of_pooling)
# Reset the optimizer state by reinitializing it

# Optionally, you can update the optimizer in the Lightning module if needed
# model = Model(batch_size, learning_rate, num_classes, epsilon)


#
# Set up your trainer with the profiler
import matplotlib.pyplot as plt
import numpy as np


def plot_large_weights_distribution(model, threshold=1.0):
    """
    Plot the distribution of weights for each layer in the given model, focusing on weights above a certain threshold.

    Parameters:
    model: torch.nn.Module
        The PyTorch model whose weights are to be plotted.
    threshold: float
        The threshold above which weights are considered 'large'. Default is 1.0.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Convert weights to numpy array and flatten
            weights = param.data.cpu().numpy().flatten()

            # Find large weights
            large_weights = weights[weights > threshold]

            # Plot the distribution of all weights
            plt.figure(figsize=(6, 4))
            plt.hist(weights, bins=50, color='blue', alpha=0.6, label='All weights')
            plt.hist(large_weights, bins=50, color='red', alpha=0.6, label=f'Weights > {threshold}')
            plt.title(f'Weight Distribution of {name}')
            plt.xlabel('Weight value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.legend()
            plt.show()


# Assuming `model` is a PyTorch model instance
plot_large_weights_distribution(model, threshold=1.0)

trainer = L.Trainer(max_epochs=200)

# trainer.fit(model)
trainer.test(model)


# Define a function to display images and their predicted labels
def display_predictions(model, dataloader, num_images=6):
    model.eval()  # Set the model to evaluation mode
    images_so_far = 0
    fig = plt.figure(figsize=(10, 10))

    with torch.no_grad():  # No need to compute gradients
        for i, (images, labels) in enumerate(dataloader):
            outputs = model(images)
            _, preds = torch.max(outputs, 1)  # Get the index of the max log-probability
            for j in range(images.size(0)):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(
                    f'Real:{dataloader.dataset.classes[labels[j]]} Predicted: {dataloader.dataset.classes[preds[j]]}')
                imshow(images[j])

                if images_so_far == num_images:
                    return


filepath_validate = "data/validate"
test_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 256x256
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
val_dataset = datasets.ImageFolder(root=filepath_validate, transform=test_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
#
# # `val_loader` is validation DataLoader
for i in range(8):
    display_predictions(model, val_loader, num_images=6)
plt.show()

filepath_hand_test="data/hand_test"
test_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 256x256
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
val_dataset = datasets.ImageFolder(root=filepath_hand_test, transform=test_transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
#
# # so `val_loader` is validation DataLoader
for i in range(1):
    display_predictions(model, val_loader, num_images=6)
plt.show()
