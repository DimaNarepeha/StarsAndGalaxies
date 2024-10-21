import lightning as L
import torch
import torchmetrics
import torchvision
from torch import nn
from torch.nn import functional as F, init
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor


class Model(L.LightningModule):
    def __init__(self, batch_size, learning_rate, num_classes, epsilon):
        super(Model, self).__init__()
        self.save_hyperparameters()
        # ADD BATCH NORMALIZATION
        # Convolutional layers
        # IN N,3,128,128
        self.conv3_32 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv32_32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        # OUT N,16,128,128
        # POOLED N,16,64,64
        self.conv32_64 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv64_64 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # OUT N,32,64,64
        # POOLED N,32,32,32
        self.conv64_128 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv128_128 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # OUT N,64,32,32
        # POOLED N,128,16,16

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer
        self.fc1 = nn.Linear(128 * 16 * 16, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, num_classes)
        # Define the convolutional layers

        self.dropout50 = nn.Dropout(0.50)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.weight_decay = 0.0001

        self.criterion = nn.CrossEntropyLoss()

        self.accuracy = torchmetrics.Accuracy(num_classes=2, average='macro', task='multiclass')

        self.test_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize images to 256x256
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),  # Randomly crops and resizes images to 224x224
            transforms.RandomHorizontalFlip(p=0.5),  # Randomly flips images horizontally
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),  # Convert images to PyTorch tensors
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.all_classes = ['galaxies', 'stars']
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)

    def forward(self, image):
        image = F.leaky_relu(self.conv3_32(image))
        image = F.leaky_relu(self.conv32_32(image))
        image = F.leaky_relu(self.conv32_32(image))
        image = self.pool(image)
        image = F.leaky_relu(self.conv32_64(image))
        image = F.leaky_relu(self.conv64_64(image))
        image = F.leaky_relu(self.conv64_64(image))
        image = self.pool(image)
        image = F.leaky_relu(self.conv64_128(image))
        image = F.leaky_relu(self.conv128_128(image))
        image = F.leaky_relu(self.conv128_128(image))
        image = self.pool(image)  # Output is now (128, 25, 25)
        image = torch.flatten(image, 1)  # Flatten the output
        image = F.leaky_relu(self.fc1(image))
        image = self.dropout50(image)
        image = F.leaky_relu(self.fc2(image))
        image = self.dropout50(image)
        image = self.fc3(image)
        return image

    def training_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)  # Forward pass
        loss = self.criterion(predictions, labels)  # Compute the loss
        predicted_classes = torch.argmax(F.softmax(predictions, dim=1), dim=1)
        predictions_softmax = F.softmax(predictions, dim=1)
        self.accuracy(predictions_softmax, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss  # Returning the loss for backpropagation

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        loss = self.criterion(predictions, labels)
        predicted_classes = torch.argmax(F.softmax(predictions, dim=1), dim=1)
        predictions_softmax = F.softmax(predictions, dim=1)
        self.conf_matrix(predictions, labels)
        self.accuracy(predictions_softmax, labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.accuracy, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        confmat = self.conf_matrix.compute().cpu().numpy()
        # Compute per-class accuracy from the confusion matrix
        per_class_accuracy = confmat.diagonal() / confmat.sum(axis=1)
        for idx, acc in enumerate(per_class_accuracy):
            self.log(f'class_{self.all_classes[idx]}_accuracy', acc, prog_bar=True)
        self.conf_matrix.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        predictions = self(images)
        loss = self.criterion(predictions, labels)
        predicted_classes = torch.argmax(F.softmax(predictions, dim=1), dim=1)
        predictions_softmax = F.softmax(predictions, dim=1)
        self.accuracy(predictions_softmax, labels)
        real_step_acc = (labels == predicted_classes).sum() / self.batch_size
        self.conf_matrix(predictions, labels)
        self.log('test_loss', loss, prog_bar=True)
        self.log('real_test_acc', real_step_acc, prog_bar=True)
        self.log('test_acc', self.accuracy, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        confmat = self.conf_matrix.compute().cpu().numpy()
        # Compute per-class accuracy from the confusion matrix
        per_class_accuracy = confmat.diagonal() / confmat.sum(axis=1)
        for idx, acc in enumerate(per_class_accuracy):
            self.log(f'class_{self.all_classes[idx]}_accuracy', acc)
        self.conf_matrix.reset()

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, eps=self.epsilon,
        #                              weight_decay=self.weight_decay)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,
                                    weight_decay=self.weight_decay)
        return optimizer

    def train_dataloader(self):
        # Set up and return the training DataLoader
        filepath_train = "data/train"

        train_dataset = datasets.ImageFolder(root=filepath_train, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

        return train_loader

    def test_dataloader(self):
        # Set up and return the training DataLoader
        filepath_train = "data/test"

        test_dataset = datasets.ImageFolder(root=filepath_train, transform=self.test_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return test_loader

    def val_dataloader(self):
        # Set up and return the validation DataLoader
        filepath_train = "data/validate"

        val_dataset = datasets.ImageFolder(root=filepath_train, transform=self.test_transform)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=16)

        return val_loader
