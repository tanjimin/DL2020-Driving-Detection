import os
from PIL import Image

import numpy as np
import pandas as pd

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
unlabeled_scene_index = np.arange(106)
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]


class UnlabeledJigsawDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """
        self.image_folder = image_folder
        self.scene_index = scene_index
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])


    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

        image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
        # original shape 256*306
        original = Image.open(image_path)
        image = original
        sample = torchvision.transforms.RandomCrop((255,255))(original)
        
        crop_areas = [(i*85, j*85, (i+1)*85, (j+1)*85) for i in range(3) for j in range(3)]
        samples = [sample.crop(crop_area) for crop_area in crop_areas]
        samples = [torchvision.transforms.RandomCrop(64,64)(patch) for patch in samples]
        #augmentation collor jitter
        image = self.color_transform(image)
        samples = [self.color_transform(patch) for patch in samples]
        # augmentation - flips
        image = self.flips[0](image)
        image = self.flips[1](image)
        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        samples = [torchvision.transforms.functional.to_tensor(patch) for patch in samples]
        # normalize 
        image = self.normalize(image)
        samples = [self.normalize(patch) for patch in samples]
        random.shuffle(samples)

        return {'original': image,'patches': samples, 'index' : index}


class CameraPretextDataset(Dataset):
    
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        label_path = os.path.join(self.label_dir, self.data_names[idx])
        data = np.load(data_path)

        label = np.range(6)

    def __getitem__(self, index):
        scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
        sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
        image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

        image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
        # original shape 256*306
        original = Image.open(image_path)
        image = self.color_transform(original)
        return image, index % NUM_IMAGE_PER_SAMPLE