
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

def Normalization(image):
    # Normalize the image to the range 0 to 1
    normalized = image / 255.0
    return normalized
""" 
It is a denoise model, remove the flake dots in the image.
"""
"""
class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(OrderedDict([
            ('layer1', nn.Conv2d(1, 32, kernel_size=5, padding='same')),
            ('layer2', nn.BatchNorm2d(32)),
            ('layer3', nn.LeakyReLU()),
            ('layer4', nn.Conv2d(32, 32, kernel_size=7, padding='same')),
            ('layer5', nn.BatchNorm2d(32)),
            ('layer6', nn.LeakyReLU()),
            ('layer7', nn.Conv2d(32, 1, kernel_size=5, padding='same')),
        ]))

    def forward(self, x):
        return self.net(x)
    
    def denoise(self, x):
        return self.net(x)
"""
# U-Net model definition
class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        logits = self.outc(y)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)