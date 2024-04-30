import torch
import torchvision
import netron
import torch.nn as nn

# Define the model
model = nn.Sequential(
    nn.Conv2d(1, 8, 5),
    nn.ReLU(),
    nn.Conv2d(8, 16, 5),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(16*4*4, 120),
    nn.ReLU(),
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 4)
)

# Define an example input
dummy_input = torch.randn(1, 1, 28, 28)

# Export the model to ONNX
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)

# Visualize the model using Netron
netron.start("model.onnx")
