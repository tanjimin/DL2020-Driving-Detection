import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.models.resnet import resnet18

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = resnet18()
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])        
        self.projection_original_features = nn.Linear(512, 64)       
        self.connect_patches_feature = nn.Linear(576, 64)
        
    def forward_once(self, x):
        return self.network(x)
    
    def return_reduced_image_features(self, original):
        original_features = self.forward_once(original)
        original_features = original_features.view(-1,512)
        original_features = self.projection_original_features(original_features)
        return original_features
        
    def return_reduced_image_patches_features(self, original, patches):
        original_features = self.return_reduced_image_features(original)
        
        patches_features = []
        for i, patch in enumerate(patches):
            patch_features = self.return_reduced_image_features(patch)
            #print(patch_features.shape)
            patches_features.append(patch_features)
        
        patches_features = torch.cat(patches_features, axis = 1)
        #print(patches_features.shape)
        
        patches_features = self.connect_patches_feature(patches_features)
        return original_features, patches_features
         
    def forward(self, images = None, patches = None, mode = 0):
        '''
        mode 0: get 128 feature for image,
        mode 1: get 128 feature for image and patches       
        '''
        if mode == 0:
            return self.return_reduced_image_features(images)
        if mode == 1:
            return self.return_reduced_image_patches_features(images, patches)

