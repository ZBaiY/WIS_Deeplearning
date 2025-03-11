import torch
from torch.utils.data import Dataset
import torch.nn.functional as func
import torchvision.models as models
import torchvision.transforms as tfms
from PIL import Image
import numpy as np
import cv2
"""
crop the white background of each image, there are three channels
"""
def crop_background(img):
    mask = img.sum(axis=-1) != 255 
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    img = img[rows.min(): rows.max() + 1, cols.min(): cols.max() + 1]
    img = img.mean(axis=-1)
    return img

def crop_index(img):
    mask = img.sum(axis=-1) != 255 * 3
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return rows.min(), rows.max() + 1, cols.min(), cols.max() + 1

def crop_by_index(img, index):
    return img[index[0]: index[1], index[2]: index[3]]

""" 
change the image to the gray scale
"""

def to_gray(img):
    return img.mean(axis=-1)

"""
negative the image and make the image black and white
"""
def to_bw(img, threshold=20):
    img = 255 - img
    img[img < threshold] = 0
    img[img >= threshold] = 255
    return img
    
def thicken_lines(img, kernel_size=(4,4), iterations=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated_img = cv2.dilate(img, kernel, iterations=iterations)
    return dilated_img

def invert_image(image):
    # Invert the image
    inverted = 255 - image
    return inverted

class CustomDataset(Dataset):
    def __init__(self, image_paths):
        image_path_1 = image_paths[0]
        
        crop_ind = crop_index(np.array(Image.open(image_path_1)))
        
        self.noisy_images = [crop_by_index(np.array(Image.open(path).convert("L")),crop_ind) for path in image_paths]  # Convert noisy images to grayscale
        #self.noisy_images = [invert_image(img) for img in self.noisy_images]  # Convert noisy images to grayscale
        self.clean_images = [crop_by_index(np.array(Image.open(path.replace('01.png', '02.png')).convert("L")),crop_ind) for path in image_paths]  # Convert clean images to grayscale
        self.clean_images = [to_bw(img) for img in self.clean_images]  # Convert clean images to black and white
        self.clean_images = [thicken_lines(img, kernel_size=(1,1), iterations=5) for img in self.clean_images]  # Thicken lines of clean images

        return None
    
    def __len__(self):
       
        return len(self.noisy_images)

 
    ### converts image to suitable format for pytorch
    def transform_image(self, image):
        transform = tfms.Compose([
            tfms.ToTensor(),
            tfms.Resize((512,512),antialias=True)
        ])
        return transform(image)

    ### resize it back to the original size
    def resize_image(self, image, hight, width):
        transform = tfms.Compose([
            tfms.ToPILImage(),
            tfms.Resize((hight, width),antialias=True)
        ])
        return transform(image)

    def __getitem__(self, idx):

        input_img = self.noisy_images[idx]
        input_img = self.transform_image(input_img) # Convert to tensor, resize to 512x512.
    
        targ_img = self.clean_images[idx] # We want to predict the original image
        targ_img = self.transform_image(targ_img)
        
        return input_img.float(), targ_img.float()