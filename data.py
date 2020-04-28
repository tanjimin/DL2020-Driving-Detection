import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from matplotlib.patches import Rectangle, Polygon
import itertools

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
    
    def __init__(self, data_dir, label_dir, annotation_file):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))
        self.annotation_dataframe = pd.read_csv(annotation_file)

    def __len__(self):
        return len(self.data_names)


    ################################################################################
    def retrieve_grid_poly_pair(self, target_bbox):
        '''
        retrieve list of grids to be searched and list of polygon objects
        only return grids in the outer polygon of each bbox
        '''

        polygon_list = []
        search_grid_list = []
        for i, bb in enumerate(target_bbox):
            # You can check the implementation of the draw box to understand how it works 
            point_squence = torch.stack([bb[:, 0], bb[:, 1], bb[:, 3], bb[:, 2], bb[:, 0]])
            x = point_squence.T[0] * 10 + 400
            y = -point_squence.T[1] * 10 + 400
            xy_pair = torch.stack((x,y)).T

            # get outer polygon
            x_min = int(min(xy_pair[:,0]))
            x_max = int(max(xy_pair[:,0])) + 1
            y_min = int(min(xy_pair[:,1]))
            y_max = int(max(xy_pair[:,1])) + 1
            # get the search grid
            search_grid = [*itertools.product(range(x_min-1, x_max+1), range(y_min-1, y_max+1))]
            search_grid_list.extend(search_grid)
            
            polygon_list.append(Polygon(xy_pair, fill = True))

        return search_grid_list, polygon_list

    def check_contained(self, pixel, init_matrix, polygon_list):
        '''
        check if pixel is contained in any polygon object
        assign 1 if contained
        '''

        for poly in polygon_list:
            if poly.contains_point(pixel):
                init_matrix[pixel[1],pixel[0]] = 1
                return

    def covert_bbox_roadmap_to_binary(self, search_grid_list, polygon_list):
        '''
        Input grids to be searched and list of polygon objects defined by bbox coordinates
        Output 800*800 binary grid
        Convert roadmap with bbox to binary roadmap with bbox (lanes are not included)
        '''

        init_matrix = np.zeros((800,800))
        
        for pixel in search_grid_list:
            self.check_contained(pixel, init_matrix, polygon_list)
        
        return init_matrix
    ################################################################################

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        data = np.load(data_path)
        
        scene_id = int(self.data_names[idx].strip('.npy').split('_')[1])
        sample_id = int(self.data_names[idx].strip('.npy').split('_')[-1])

        data_entries = self.annotation_dataframe[(self.annotation_dataframe['scene'] == scene_id) & (self.annotation_dataframe['sample'] == sample_id)]
        corners = data_entries[['fl_x', 'fr_x', 'bl_x', 'br_x', 'fl_y', 'fr_y','bl_y', 'br_y']].to_numpy()
        categories = data_entries.category_id.to_numpy()
        
        # target: len = 2, dict with 2 pairs of key-value 
        target = {}
        target['bounding_box'] = torch.as_tensor(corners).view(-1, 2, 4)
        target['category'] = torch.as_tensor(categories)

        search_grid_list, polygon_list = self.retrieve_grid_poly_pair(target['bounding_box'])
        out_binary_bbox_map = self.covert_bbox_roadmap_to_binary(search_grid_list, polygon_list)

        # data: (256, 16, 20)
        # out_binary_bbox_map: (800, 800)
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
        # label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]

        label = torch.from_numpy(np.rot90(out_binary_bbox_map[131:669,400:]).copy())

        return data[1,:], label


