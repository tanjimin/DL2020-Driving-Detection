import os
import random
from PIL import Image
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt

image_folder = './beegfs/cy1355/data'
annotation_csv = './beegfs/cy1355/data/annotation.csv'

unlabeled_scene_index = np.arange(106)
labeled_scene_index = np.arange(106, 134)

NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
# re-define the camera order
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_LEFT.jpeg'
    ]

def convert_map_to_lane_map(ego_map, binary_lane):
    mask = (ego_map[0,:,:] == ego_map[1,:,:]) * (ego_map[1,:,:] == ego_map[2,:,:]) + (ego_map[0,:,:] == 250 / 255)

    if binary_lane:
        return (~ mask)
    return ego_map * (~ mask.view(1, ego_map.shape[1], ego_map.shape[2]))

def convert_map_to_road_map(ego_map):
    mask = (ego_map[0,:,:] == 1) * (ego_map[1,:,:] == 1) * (ego_map[2,:,:] == 1)

    return (~mask)

def collate_fn(batch):
    return tuple(zip(*batch))

class LabeledDataset(torch.utils.data.Dataset):    
    def __init__(self, image_folder, annotation_file, scene_index, transform, extra_info=True):
        """
        Args:
            image_folder (string): the location of the image folder
            annotation_file (string): the location of the annotations
            scene_index (list): a list of scene indices for the unlabeled data 
            transform (Transform): The function to process the image
            extra_info (Boolean): whether you want the extra information
        """
        
        self.image_folder = image_folder
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.scene_index = scene_index
        self.transform = transform
        self.extra_info = extra_info
    
    def __len__(self):
        return self.scene_index.size * NUM_SAMPLE_PER_SCENE

    def __getitem__(self, index):
        scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
        sample_id = index % NUM_SAMPLE_PER_SCENE
        sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

        images = []
        for image_name in image_names:
            image_path = os.path.join(sample_path, image_name)
            image = Image.open(image_path)
            images.append(self.transform(image))
        image_tensor = torch.stack(images)

        ######################################################
        # [3, 256, 1836]
        image_tensor_cat = torch.cat(images, dim = 2)
        ######################################################

        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        ego_path = os.path.join(sample_path, 'ego.png')
        ego_image = Image.open(ego_path)
        ego_image = torchvision.transforms.functional.to_tensor(ego_image)
        road_image = convert_map_to_road_map(ego_image)
        
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        if self.extra_info:
            actions = data_entries.action_id.to_numpy()
            # You can change the binary_lane to False to get a lane with 
            lane_image = convert_map_to_lane_map(ego_image, binary_lane=True)
            
            extra = {}
            extra['action'] = torch.as_tensor(actions)
            extra['ego_image'] = ego_image
            extra['lane_image'] = lane_image

            return image_tensor, target, road_image, extra
        else:
            return image_tensor, target, road_image, sample_path, image_tensor_cat

transform = torchvision.transforms.ToTensor()
labeled_trainset = LabeledDataset(image_folder=image_folder,
                                  annotation_file=annotation_csv,
                                  scene_index=labeled_scene_index,
                                  transform=transform,
                                  extra_info=False
                                 )
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

# load model
pretrained_model = torchvision.models.resnet18(pretrained=True)
modules = list(pretrained_model.children())[:-3]
res_model = nn.Sequential(*modules)
res_model.eval()


##########################################################
# save camera-level tensor
# output shape: [6, 256, 16, 20]
##########################################################
# cnt = 0
# for data, _, road_img, filename in trainloader:

#     output = res_model(data[0])
#     save_filename = filename[0].split('/')[-2] + '_' + filename[0].split('/')[-1]
    
#     if not os.path.exists('/Users/leo/Downloads/result/image_tensor'):
#         os.mkdir('/Users/leo/Downloads/result/image_tensor')
#     if not os.path.exists('/Users/leo/Downloads/result/road_map'):
#         os.mkdir('/Users/leo/Downloads/result/road_map')

#     assert road_img[0].shape == torch.Size([800, 800])
#     assert output.shape == torch.Size([6, 256, 16, 20])

#     np.save(os.path.join('/Users/leo/Downloads/result/road_map', save_filename), road_img[0].detach().numpy())    
#     np.save(os.path.join('/Users/leo/Downloads/result/image_tensor', save_filename), output.detach().numpy())
    
#     cnt += 1
#     if cnt % 100 == 0:
#         print(str(28 * 126 - cnt) + ' left')


##########################################################
# save polar image
##########################################################

def warpper(data):
    
    img = data[0].permute(1,2,0).numpy()
    img = (img * 255).astype(np.uint8)
    
    img = np.rot90(img)
    img = np.flip(img, 1)
    # (width, height * 1.5, 3)
    new_img = np.zeros((1836, 390, 3))
    # -height
    new_img[:, -256:, :] = img
    img = new_img
    img = cv2.resize(img, (3000, 3000))
    value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
    polar_image = cv2.linearPolar(src=img, center=(1500, 1500), maxRadius=1500, flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    polar_image = polar_image.astype(np.uint8)
    
    return polar_image

def check_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

cnt = 0
for _, _, road_img, filename, concat_width_img in trainloader:

    polar_image = warpper(concat_width_img)

    input_res = torch.Tensor(polar_image / 255).unsqueeze(0).permute(0,3,1,2) 
    output = res_model(input_res)
    save_filename = filename[0].split('/')[-2] + '_' + filename[0].split('/')[-1]
    
    # if not os.path.exists('/Users/leo/Downloads/polar/image_tensor'):
    #     os.mkdir('/Users/leo/Downloads/polar/image_tensor')
    # if not os.path.exists('/Users/leo/Downloads/polar/road_map'):
    #     os.mkdir('/Users/leo/Downloads/polar/road_map')
    # if not os.path.exists('/Users/leo/Downloads/polar/polar_image'):
    #     os.mkdir('/Users/leo/Downloads/polar/polar_image')
    check_path('/beegfs/cy1355/polar')
    check_path('/beegfs/cy1355/polar/image_tensor')
    check_path('/beegfs/cy1355/polar/road_map')
    check_path('/beegfs/cy1355/polar/polar_image')

    assert road_img[0].shape == torch.Size([800, 800])
    assert output.shape == torch.Size([1, 256, 188, 188])

    np.save(os.path.join('/beegfs/cy1355/polar/image_tensor', save_filename), output.detach().numpy())
    np.save(os.path.join('/beegfs/cy1355/polar/road_map', save_filename), road_img[0].detach().numpy())  
    cv2_filename = save_filename + '.png'
    cv2.imwrite(os.path.join('/beegfs/cy1355/polar/polar_image', cv2_filename), polar_image[...,::-1])
    
    if cnt % 10 == 0:
        print(str(28 * 126 - cnt) + ' left')
    cnt += 1