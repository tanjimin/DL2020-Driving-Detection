import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from scipy import misc, ndimage
from torch.utils.data import Dataset

class CameraBasedObjectSegmentationDataset(Dataset):
    
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
        
        # label_processed: 6 * 400 * 538
        label_rot1 = ndimage.rotate(label, 30)
        label1 = label_rot1[547 - 400:547, 278: 278 + 538].copy()
        label4 = label_rot1[547:547+400, 278: 278 + 538][::-1,::-1].copy()

        label_rot2 = ndimage.rotate(label, -30)
        label6 = label_rot2[547-400:547, 278: 278 + 538 ].copy()
        label3 = label_rot2[547:547+400, 278: 278 + 538 ][::-1,::-1].copy()

        label2 = np.rot90(label[131:669,400:]).copy()
        label5 = np.rot90(label[131:669, :400][::-1,::-1]).copy()

        label_processed = torch.from_numpy(np.stack([label1, label2, label3, label4, label5, label6], axis = 0))
        return data, label_processed



if __name__ == "__main__":
    image_path = "/beegfs/jt3545/data/detection/train/image_tensor"
    annotation_path = "/beegfs/cy1355/data/annotation.csv"
    #label_path =  "/beegfs/cy1355/obj_binary_roadmap_train/road_map"

    # train_loader = FrontObjectSegmentationDataset(image_path, label_path)
    # for data, label in iter(train_loader):
    #     assert (label.shape[0] == 400) & (label.shape[1] == 538)
    train_loader = ObjectDetectionDataset(image_path,"/beegfs/cy1355/obj_binary_roadmap_train/road_map", annotation_path, True)
    data = train_loader[150]
