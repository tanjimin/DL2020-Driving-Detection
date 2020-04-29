import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class LaneSegmentationDataset(Dataset):
    
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        label_path = os.path.join(self.label_dir, self.data_names[idx])
        data = np.load(data_path)
        label = (np.load(label_path) * 1).astype(np.single)
        return data, label
        


class FrontLaneSegmentationDataset(Dataset):
    
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        label_path = os.path.join(self.label_dir, self.data_names[idx])
        data = np.load(data_path)
        label = (np.load(label_path) * 1).astype(np.single)
        
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
        # label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]
        label = torch.from_numpy(np.rot90(label[131:669,400:]).copy())
        return data[1,:], label

class FrontObjectSegmentationDataset(Dataset):
    
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        label_path = os.path.join(self.label_dir, self.data_names[idx])
        data = np.load(data_path)
        label = (np.load(label_path) * 1).astype(np.single)
    
        # data: (256, 16, 20)
        # label: (800, 800)
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

        # output label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]
        label = torch.from_numpy(np.rot90(label[131:669,400:]).copy())

        return data[1,:], label

if __name__ == "__main__":
    image_path = "/beegfs/cy1355/obj_binary_roadmap_train/image_tensor"
    label_path =  "/beegfs/cy1355/obj_binary_roadmap_train/road_map"

    train_loader = FrontObjectSegmentationDataset(image_path, label_path)
    for data, label in iter(train_loader):
        assert (label.shape[0] == 400) & (label.shape[1] == 538)
