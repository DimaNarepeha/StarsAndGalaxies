# Install torchviz if you haven't already
# !pip install torchviz

import torch
from torchviz import make_dot

from Model import Model

# Initialize the model with sample parameters
model = Model(batch_size=32, learning_rate=0.001, num_classes=2, epsilon=1e-8)

# Create a sample input tensor with the shape expected by the model (N, C, H, W)
sample_input = torch.randn(1, 3, 128, 128)  # Batch size 1, 3 color channels, 128x128 image

# Perform a forward pass to get the model output
output = model(sample_input)

# Generate and visualize the computational graph
dot = make_dot(output, params=dict(model.named_parameters()))
dot.format = 'png'
dot.render("model_graph")

# This will save the model graph as a PNG file named "model_graph.png"
# First, install torchinfo if not already installed
# !pip install torchinfo

from torchinfo import summary

# Initialize the model
model = Model(batch_size=32, learning_rate=0.001, num_classes=2, epsilon=1e-8)

# Display model summary for a 128x128 input image
summary(model, input_size=(32, 3, 128, 128))
