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

from preprocess_module import retrieve_grid_poly_pair, check_contained, \
                           covert_bbox_roadmap_to_binary, main_binary_roadmap_objdetection

cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")

image_folder = '/beegfs/cy1355/data'
annotation_csv = '/beegfs/cy1355/data/annotation.csv'

# image_folder = '../Project/z_own/data'
# annotation_csv = '../Project/z_own/data/annotation.csv'

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
trainloader = torch.utils.data.DataLoader(labeled_trainset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)


# load model
pretrained_model = torchvision.models.resnet18(pretrained=True)
modules = list(pretrained_model.children())[:-3]
res_model = nn.Sequential(*modules)
res_model.eval()
res_model.to(device)

def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)

##########################################################
# save camera-level tensor
# input shape: [6, 3, 256, 306]
# output shape: [6, 256, 16, 20]
##########################################################
# cnt = 0
# for data, _, road_img, filename, _ in trainloader:
#     import pdb;pdb.set_trace()

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
# output shape: [256, 188, 188]
##########################################################

# def warpper(data):
    
#     img = data[0].permute(1,2,0).numpy()
#     img = (img * 255).astype(np.uint8)
    
#     img = np.rot90(img)
#     img = np.flip(img, 1)
#     # (width, height * 1.5, 3)
#     new_img = np.zeros((1836, 390, 3))
#     # -height
#     new_img[:, -256:, :] = img
#     img = new_img
#     img = cv2.resize(img, (3000, 3000))

#     polar_image = cv2.linearPolar(src=img, center=(1500, 1500), maxRadius=1500, flags=cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
#     polar_image = polar_image.astype(np.uint8)
    
#     return polar_image

# cnt = 0
# for _, _, road_img, filename, concat_width_img in trainloader:

#     polar_image = warpper(concat_width_img)
#     polar_image = np.rot90(np.rot90(np.rot90(np.fliplr(polar_image))))

#     input_res = torch.Tensor(polar_image / 255).unsqueeze(0).permute(0,3,1,2) 
#     # input_res ([1, 3, 3000, 3000])
#     input_res = input_res.to(device)
#     output = res_model(input_res)
    
#     save_filename = filename[0].split('/')[-2] + '_' + filename[0].split('/')[-1]
#     output = output.squeeze(0)
    
#     check_path('/beegfs/cy1355/polar_tensor')
#     check_path('/beegfs/cy1355/polar_tensor/image_tensor')
#     check_path('/beegfs/cy1355/polar_tensor/road_map')
#     check_path('/beegfs/cy1355/polar_tensor/polar_image')

#     assert road_img[0].shape == torch.Size([800, 800])
#     assert output.shape == torch.Size([256, 188, 188])

#     np.save(os.path.join('/beegfs/cy1355/polar_tensor/image_tensor', save_filename), output.detach().cpu().numpy())
#     np.save(os.path.join('/beegfs/cy1355/polar_tensor/road_map', save_filename), road_img[0].detach().numpy())  
#     cv2_filename = save_filename + '.png'
#     cv2.imwrite(os.path.join('/beegfs/cy1355/polar_tensor/polar_image', cv2_filename), polar_image[...,::-1])
    
#     if cnt % 10 == 0:
#         print(str(28 * 126 - cnt) + ' left')
#     cnt += 1


##########################################################
# save binary roadmap with bbox
##########################################################

cnt = 0
for _, target, _, filename, _ in trainloader:
    target_bbox = target[0]['bounding_box']
    binary_roadmap_objdetect = main_binary_roadmap_objdetection(target_bbox)

    save_filename = filename[0].split('/')[-2] + '_' + filename[0].split('/')[-1]
    check_path('/beegfs/cy1355/obj_binary_roadmap/')
    check_path('/beegfs/cy1355/obj_binary_roadmap/road_map')
    check_path('/beegfs/cy1355/obj_binary_roadmap/roadmap_image')
    assert binary_roadmap_objdetect.shape == (800, 800)

    np.save(os.path.join('/beegfs/cy1355/obj_binary_roadmap/road_map', save_filename), binary_roadmap_objdetect)

    cv2_filename = save_filename + '.png'
    cv2.imwrite(os.path.join('/beegfs/cy1355/obj_binary_roadmap/roadmap_image', cv2_filename), binary_roadmap_objdetect[...,::-1]*255)
    
    if cnt % 10 == 0:
        print(str(28 * 126 - cnt) + ' left')
    cnt += 1


    
