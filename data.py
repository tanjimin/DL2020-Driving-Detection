import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from scipy import misc, ndimage
from torch.utils.data import Dataset
from utils import BboxGenerate

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

class CameraBasedLaneSegmentationDataset(Dataset):
    
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
        # label_processed: 6 * 400 * 538
        label_rot1 = ndimage.rotate(label, 30)
        label1 = label_rot1[547 - 400:547, 278: 278 + 538].copy()
        label4 = label_rot1[547:547+400, 278: 278 + 538][::-1,::-1].copy()

        label_rot2 = ndimage.rotate(label, -30)
        label3 = label_rot2[547-400:547, 278: 278 + 538 ].copy()
        label6 = label_rot2[547:547+400, 278: 278 + 538 ][::-1,::-1].copy()

        label2 = np.rot90(label[131:669,400:]).copy()
        label5 = np.rot90(label[131:669, :400][::-1,::-1]).copy()

        label_processed = torch.from_numpy(np.stack([label1, label2, label3, label4, label5, label6], axis = 0))
        return data, label_processed

class ObjectRegressionDataset(Dataset):
    
    def __init__(self, data_dir, label_dir, annotation_file, front_only = False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.front_only = front_only
        if front_only:
            self._filter_nonfront()
            self.sampler = BboxGenerate(400, 538, 20, 45)
        else:
            self.sampler = BboxGenerate(800, 800, 20, 45)

    def _filter_nonfront(self):
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['fl_x'] >= 0) | (self.annotation_dataframe['fr_x'] >= 0)]
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['bl_x'] >= 0) | (self.annotation_dataframe['br_x'] >= 0)]
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['fl_y'].abs() <= 269) | (self.annotation_dataframe['fr_y'].abs() <= 269)]
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['bl_y'].abs() <= 269) | (self.annotation_dataframe['br_y'].abs() <= 269)]

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        data = np.load(data_path)
        label_path = os.path.join(self.label_dir, self.data_names[idx])
        label = (np.load(label_path) * 1).astype(np.single)
        if self.front_only:
            label = label[131:669,400:]
    
        id_list = self.data_names[idx].split('_')
        scene_id, sample_id = int(id_list[1]), int(id_list[-1].split('.')[0])
        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['bl_x', 'bl_y', 'fl_x', 'fl_y', 'br_x', 'br_y', 'fr_x','fr_y']].to_numpy()
        pos_samples = torch.as_tensor(corners).view(-1, 8).float()

        neg_num= 100 - pos_samples.shape[0]
        neg_samples = torch.FloatTensor(self.sampler.sample(neg_num, label))
        if self.front_only:
            neg_samples[:,:,1] = neg_samples[:,:,1] - 269 
        else:
            neg_samples[:,:,1] = neg_samples[:,:,1] - 400
            neg_samples[:,:,0] = neg_samples[:,:,0] - 400
        neg_samples = neg_samples.view(-1,8)
        samples = torch.cat([pos_samples, neg_samples], 0)
        target = torch.cat([torch.ones(pos_samples.shape[0]), -1 * torch.ones(neg_num)]).float()
        # data: (256, 16, 20)
        # samples: (n = 100, 8)
        # target: 1000
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

        # output label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]
        #print(data.shape, samples.shape, target.shape)
        if self.front_only:
            return data[1,:], samples, target
        return data, samples, target

if __name__ == "__main__":
    image_path = "/beegfs/cy1355/obj_binary_roadmap_train/image_tensor"
    annotation_path = "/beegfs/cy1355/data/annotation.csv"
    #label_path =  "/beegfs/cy1355/obj_binary_roadmap_train/road_map"

    # train_loader = FrontObjectSegmentationDataset(image_path, label_path)
    # for data, label in iter(train_loader):
    #     assert (label.shape[0] == 400) & (label.shape[1] == 538)
    train_loader = ObjectRegressionDataset(image_path, annotation_path, True)
    for data, label in iter(train_loader):
        print(label.shape)

