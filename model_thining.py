
import torch
import torch.nn as nn
from collections import OrderedDict
import cv2
import numpy as np

def linear_contrast_stretch(image, device='mps'):
    # Find the minimum and maximum pixel values
    # first convert the image to numpy array
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    # Move the tensor to the specified device (GPU/MPS)
    image = image.to(device)
    image = image.float()

# Use PyTorch operations for min, max, and linear stretch, which can run on GPU
    minVal = torch.min(image)
    maxVal = torch.max(image)

    # Apply the linear stretch formula directly with PyTorch operations
    stretched = 255 * (image - minVal) / (maxVal - minVal)

    return stretched

def gamma_stretch(image, gamma=1.0,device='mps'):
    # Ensure the input image is a PyTorch tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    # Move the tensor to the specified device (GPU/MPS)
    image = image.to(device)

    # Normalize the image to the range 0 to 1
    normalized = image / 255.0

    # Apply gamma correction
    corrected = torch.pow(normalized, gamma)

    # Convert back to an image scaled to 0-255 and ensure it's on the CPU for further processing/display
    gamma_corrected = corrected * 255.0

    return gamma_corrected

def midtone_stretch(image, steepness=10, midpoint=0.5, device='mps'):
    # Ensure the input image is a PyTorch tensor
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    # Move the tensor to the specified device (GPU/MPS)
    image = image.to(device)

    # Normalize the image to the range 0 to 1
    normalized = image / 255.0

    # Apply midtone stretch using a sigmoid function
    # The sigmoid function shifts and scales the input to adjust the contrast of midtones
    stretched = 1 / (1 + torch.exp(-steepness * (normalized - midpoint)))

    # Convert back to an image scaled to 0-255 and ensure it's on the CPU for further processing/display
    midtone_stretched = stretched * 255.0

    return midtone_stretched

def logarithmic_stretching(image, device='mps'):
    # Convert image to float32 to avoid overflow, and add a small value to avoid log(0)
    if not isinstance(image, torch.Tensor):
        image = torch.tensor(image, dtype=torch.float32)

    # Move the tensor to the GPU
    image = image.to(device)

    # Add a small value to avoid log(0) and convert image to float32 to avoid overflow
    img_float = image + 1.0

    # Apply logarithmic stretch using PyTorch operations
    log_stretch = torch.log(img_float) * (255 / torch.log(1 + torch.max(img_float)))
    return log_stretch

def histogram_equalization(image):
    # Convert to grayscale for simplicity
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    equalized = cv2.equalizeHist(gray)
    return equalized

def invert_image(image):
    # Invert the image
    inverted = 255 - image
    return inverted
""" 
It is a denoise model, remove the flake dots in the image.
"""

class ThinningNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('layer1', nn.Conv2d(1, 16, kernel_size=2, padding='same')),
            ('layer2', nn.BatchNorm2d(16)),
            ('layer3', nn.LeakyReLU()),
            ('layer4', nn.Conv2d(16, 16, kernel_size=3, padding='same')),
            ('layer5', nn.BatchNorm2d(16)),
            ('layer6', nn.LeakyReLU()),
            ('layer7', nn.Conv2d(16, 16, kernel_size=5, padding='same')),
            ('layer8', nn.BatchNorm2d(16)),
            ('layer9', nn.LeakyReLU()),
            ('layer10', nn.Conv2d(16, 1, kernel_size=3, padding='same')),
        ]))

    def forward(self, x):
        return self.net(x)
    
    def denoise(self, x):
        return self.net(x)
