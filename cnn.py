from PIL.Image import Image
from PIL import Image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim


class CNN():
    def __init__(self):
        self.missed_abnormal = 0
        self.misssed_normal = 0
        self.missed = 0

    def train_model_4_class(self, folder_path, model_name, augument, batch, epoch, lr):
        """
           Train a convolutional neural network for a 4-class classification task.

           Args:
               folder_path (str): Path to the folder containing the dataset.
               model_name (str): Name of the trained model file to be saved.
               augument (bool): Whether to apply data augmentation.
               batch (int): Batch size for training.
               epoch (int): Number of epochs for training.
               lr (float): Learning rate for the optimizer.
        """
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
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((80, 80)),  # Resize the images to 80x80
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

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

    def predict(self, image):
        """
            Make a prediction for a single image.

            Args:
                image (numpy.ndarray): Input image array.

            Returns:
                int: Predicted class index.
        """
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
    def predict_probs(self, image):
        """
        Make a prediction with probability outputs for a single image.

        Args:
            image (numpy.ndarray): Input image array.

        Returns:
            Tuple[int, numpy.ndarray]: Predicted class index and class probabilities.
        """
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
            probabilities = torch.softmax(output, dim=1)  # Calculate softmax probabilities
            _, predicted = torch.max(output, 1)  # Get the predicted class index

        return predicted.item(), probabilities.squeeze().numpy()  # Return predicted class label and probabilities

    def load_model(self, model_path):
        """
        Load a pre-trained convolutional neural network model.

        Args:
            model_path (str): Path to the saved model file.
        """
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

        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        self.model = model
