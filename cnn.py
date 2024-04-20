import os

from PIL.Image import Image
from PIL import Image

from LandmarkTester import LandmarkTester
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import mediapipe as mp
import time
import glob
import torchvision.models as models
import numpy as np

from SkeletonDetector import SkeletonDetector
from YoloDetector import YoloDetector
from YoloSkeleton import YoloSkeleton


class CNN():
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (0, 0))
        self.missed_abnormal=0
        self.misssed_normal=0
        self.missed=0

    def train_model_4_class(self,folder_path,model_name,augument,batch,epoch,lr):
        print('Training')
        BATCH_SIZE = batch
        EPOCHS = epoch
        LR = lr

        if augument:
            transform = transforms.Compose([
                transforms.Resize((80, 80)),  # Resize the images to 80x80
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomRotation(degrees=5),  # Random rotation within -10 to +10 degrees
                transforms.RandomAutocontrast(0.7),
                # transforms.RandomResizedCrop(size=(80, 80), scale=(0.8, 1.2)),  # Random resized crop
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            # Define data transformations for "normal" subfolder without augmentation
            transform= transforms.Compose([
                transforms.Resize((80, 80)),  # Resize the images to 80x80
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

        # Load the original dataset with augmentation only for "abnormal" subfolder
        # Load the original dataset with augmentation
        dataset = ImageFolder(root=folder_path, transform=transform)

        # Create a DataLoader for the dataset
        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

        # Define the neural network model, loss function, and optimizer
        model = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.Linear(84, 4)
        )

        loss_fun = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR)

        # Training loop
        for epoch in range(EPOCHS):
            print("epoch", epoch + 1)

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fun(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 20 == 19:
                    print(f'minibatch: {i + 1} loss: {running_loss / 20}')
                    running_loss = 0

        # Save the trained model
        filename = model_name
        torch.save(model.state_dict(), filename)
        print('Finished Training')

    def predict(self,image):
        image_pil = Image.fromarray(image)

        transform = transforms.Compose([
            transforms.Resize((80, 80)),  # Resize the images to 80x80
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Preprocess the image
        image = transform(image_pil).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output, 1)

        return predicted.item()

        # with torch.no_grad():
        #     output = self.model(image)
        #     probabilities = torch.softmax(output, dim=1)
        #
        # predicted_class_index = torch.argmax(probabilities)
        #
        # return probabilities.squeeze().tolist(),predicted_class_index

    def load_model(self,model_path):
        # Load the model
        model = nn.Sequential(
            nn.Conv2d(1, 8, 5),
            nn.ReLU(),
            nn.Conv2d(8, 16, 5),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.Linear(84, 4)  # Changed the number of output neurons to 4
        )

        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        self.model=model

        # Define the transformation to apply to the image



