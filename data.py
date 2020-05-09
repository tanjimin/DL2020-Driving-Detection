import os
from math import sqrt
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from scipy import misc, ndimage
from torch.utils.data import Dataset
from utils import BboxGenerate
from preprocess_module import main_binary_roadmap_objdetection

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
        # if directly loading numpy, the camera orders are CHANGED
        # orig data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT
        # now: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_RIGHT, CAM_BACK, CAM_BACK_LEFT

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

class ObjectRegressionDataset(Dataset):
    
    def __init__(self, data_dir, label_dir, annotation_file, front_only = False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.front_only = front_only
        if front_only:
            self._filter_nonfront()
            self.box_sampler= BboxGenerate(538, 400, 20, 45)
        else:
            self.box_sampler = BboxGenerate(800, 800, 20, 45)

    def _filter_nonfront(self, camera_idx = None):
        """
        camera_idx: 0-fl 1-f 2-fr 3-br 4-b 5-bl
        """
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['fl_x'] >= 0) | (self.annotation_dataframe['fr_x'] >= 0)]
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['bl_x'] >= 0) | (self.annotation_dataframe['br_x'] >= 0)]
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['fl_y'].abs() <= 26.9) | (self.annotation_dataframe['fr_y'].abs() <= 26.9)]
        self.annotation_dataframe = self.annotation_dataframe[(self.annotation_dataframe['bl_y'].abs() <= 26.9) | (self.annotation_dataframe['br_y'].abs() <= 26.9)]


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
        pos_samples = torch.as_tensor(corners).view(-1, 4, 2).float()
        pos_mean = pos_samples.mean(axis = 1, keepdim = True)
        diff = torch.as_tensor([-4.5/2, -1, 4.5/2, -1, -4.5/2, 1, 4.5/2, 1]).view(-1, 8)
        pos_means = pos_mean.repeat(1, 4, 1).view(-1, 8)
        pos_samples = pos_means + diff 

        neg_num= 400 - pos_samples.shape[0]
        neg_samples = torch.FloatTensor(self.box_sampler.sample(neg_num, label))
        if self.front_only:
            neg_samples[:,:,1] = neg_samples[:,:,1] - 269 
        else:
            neg_samples[:,:,1] = neg_samples[:,:,1] - 400
            neg_samples[:,:,0] = neg_samples[:,:,0] - 400
        neg_samples = neg_samples.view(-1,8) / 10
        samples = torch.cat([pos_samples, neg_samples], 0)
        target = torch.cat([torch.ones(pos_samples.shape[0]), torch.zeros(neg_num)]).float()
        # data: (256, 16, 20)
        # samples: (n = 100, 8)
        # target: 1000
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

        # output label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]
        #print(data.shape, samples.shape, target.shape)
        if self.front_only:
            return data[1,:], samples, target, label
        else:
            return data, samples, target, label


class CameraBasedObjectRegressionDataset(Dataset):
    
    def __init__(self, data_dir, label_dir, annotation_file):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))
        self.annotation_dataframe = pd.read_csv(annotation_file)
        self.camera_dataframes = [self._filter_nonfront(i) for i in range(6)]
        self.box_sampler= BboxGenerate(538, 400, 20, 45) ## to be confirmed
        self.rotate_degrees = [-np.pi/3., 0, np.pi/3., (2*np.pi)/3., np.pi, -(2*np.pi)/3.]
        self.rotate_matrix = [self._generate_rotate_matrix(degree) for degree in self.rotate_degrees]

    def _generate_rotate_matrix(self, degree):
        return torch.FloatTensor([[np.cos(degree), -np.sin(degree)],[np.sin(degree), np.cos(degree)]])

    def _filter_nonfront(self, camera_idx):
        """
        camera_idx: 0-fl 1-f 2-fr 3-br 4-b 5-bl
        """
        # to be confirmed: no need for rotation?
        camera_dataframe = self.annotation_dataframe.copy()
        positions = ['fl_', 'fr_','br_','bl_']
        if camera_idx == 0:
            k = sqrt(3)
            # y - sqrt(3)*x + 53.8 > 0
            idxsets = [((camera_dataframe[pos+'y'] - k * camera_dataframe[pos + 'x'] + 53.8) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # y - sqrt(3)*x - 53.8 < 0
            idxsets = [((camera_dataframe[pos+'y'] - k * camera_dataframe[pos+'x'] - 53.8) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # sqrt(3) * y +  x > 0
            idxsets = [((k * camera_dataframe[pos+'y'] + camera_dataframe[pos+'x'] ) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # sqrt(3) * y +  x - 80 < 0
            idxsets = [((k * camera_dataframe[pos+'y'] + camera_dataframe[pos+'x'] -80 ) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
        elif camera_idx == 1:
            # x > 0
            idxsets = [(camera_dataframe[pos + 'x'] > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # abs(y) < 26.9
            idxsets = [(camera_dataframe[pos + 'y'].abs() < 26.9) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
        elif camera_idx == 2:
            k = sqrt(3)
            # sqrt(3) * y -  x < 0
            idxsets = [((k * camera_dataframe[pos+'y'] - camera_dataframe[pos+'x'] ) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # sqrt(3) * y - x + 80 > 0
            idxsets = [((k * camera_dataframe[pos+'y'] - camera_dataframe[pos+'x'] + 80 ) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # y + sqrt(3)*x - 53.8 < 0
            idxsets = [((camera_dataframe[pos+'y'] + k * camera_dataframe[pos + 'x'] - 53.8) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # y + sqrt(3)*x + 53.8 > 0
            idxsets = [((camera_dataframe[pos+'y'] + k * camera_dataframe[pos + 'x'] + 53.8) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
        elif camera_idx == 3:
            k = sqrt(3)
            # y - sqrt(3)*x + 53.8 > 0
            idxsets = [((camera_dataframe[pos+'y'] - k * camera_dataframe[pos + 'x'] + 53.8) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # y - sqrt(3)*x - 53.8 < 0
            idxsets = [((camera_dataframe[pos+'y'] - k * camera_dataframe[pos+'x'] - 53.8) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # sqrt(3) * y +  x < 0
            idxsets = [((k * camera_dataframe[pos+'y'] + camera_dataframe[pos+'x'] ) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # sqrt(3) * y +  x + 80 > 0
            idxsets = [((k * camera_dataframe[pos+'y'] + camera_dataframe[pos+'x'] + 80 ) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
        elif camera_idx == 4:
            # x < 0
            idxsets = [(camera_dataframe[pos + 'x'] < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # abs(y) < 26.9
            idxsets = [(camera_dataframe[pos + 'y'].abs() < 26.9) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
        elif camera_idx == 5:
            k = sqrt(3)
            # sqrt(3) * y -  x > 0
            idxsets = [((k * camera_dataframe[pos+'y'] - camera_dataframe[pos+'x'] ) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # sqrt(3) * y - x - 80 < 0
            idxsets = [((k * camera_dataframe[pos+'y'] - camera_dataframe[pos+'x'] - 80 ) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # y + sqrt(3)*x - 53.8 < 0
            idxsets = [((camera_dataframe[pos+'y'] + k * camera_dataframe[pos + 'x'] - 53.8) < 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
            # y + sqrt(3)*x + 53.8 > 0
            idxsets = [((camera_dataframe[pos+'y'] + k * camera_dataframe[pos + 'x'] + 53.8) > 0) for pos in positions]
            idxset = idxsets[0] | idxsets[1]| idxsets[2] | idxsets[3]
            camera_dataframe = camera_dataframe[idxset]
        return camera_dataframe

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_names[idx])
        data = np.load(data_path)
        label_path = os.path.join(self.label_dir, self.data_names[idx])
        label = (np.load(label_path) * 1).astype(np.single)
        id_list = self.data_names[idx].split('_')
        scene_id, sample_id = int(id_list[1]), int(id_list[-1].split('.')[0])
    

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
        
        # corners
        samples_cameras = []
        targets_cameras = []
        for camera_idx in range(6):
            data_entries = self.camera_dataframes[camera_idx][(self.camera_dataframes[camera_idx]['scene'] == scene_id) & (self.camera_dataframes[camera_idx]['sample'] == sample_id)]
            corners = data_entries[['bl_x', 'bl_y', 'fl_x', 'fl_y', 'br_x', 'br_y', 'fr_x','fr_y']].to_numpy()
            pos_samples = torch.as_tensor(corners).view(-1, 4, 2).float()
            pos_mean = pos_samples.mean(axis = 1, keepdim = True)
            diff = torch.as_tensor([-4.5/2, -1, 4.5/2, -1, -4.5/2, 1, 4.5/2, 1]).view(-1, 8)
            pos_means = pos_mean.repeat(1, 4, 1).view(-1, 4, 2).unsqueeze(2)
            pos_means_processed = torch.matmul(pos_means, self.rotate_matrix[camera_idx].T) # n*4*1*2
            pos_samples_processed = pos_means_processed.squeeze().view(-1,8) + diff 

            neg_num= 400 - pos_samples.shape[0]
            neg_samples = torch.FloatTensor(self.box_sampler.sample(neg_num, None))
            neg_samples[:,:,1] = neg_samples[:,:,1] - 269 
            neg_samples = neg_samples.view(-1,8) / 10
            samples = torch.cat([pos_samples_processed, neg_samples], 0)
            target = torch.cat([torch.ones(pos_samples_processed.shape[0]), torch.zeros(neg_num)]).float()
            samples_cameras.append(samples)
            targets_cameras.append(target)

        # data: (256, 16, 20)
        # samples: (n = 100, 8)
        # target: 1000
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

        # output label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]
        #print(data.shape, samples.shape, target.shape)
        targets_cameras = torch.from_numpy(np.stack(targets_cameras, axis = 0))
        samples_cameras = torch.from_numpy(np.stack(samples_cameras, axis = 0))

        return data, samples_cameras, targets_cameras, label_processed            

class ObjectDetectionDataset(Dataset):
    
    def __init__(self, data_dir, label_dir, annotation_file, front_only = False):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.data_names = sorted(os.listdir(data_dir))
        self.annotation_dataframe = pd.read_csv(annotation_file, encoding='utf-8')
        self.front_only = front_only
        if front_only:
            self._filter_nonfront()
            self.box_sampler= BboxGenerate(400, 538, 20, 45)
        else:
            self.box_sampler = BboxGenerate(800, 800, 20, 45)

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
        corners = data_entries[['bl_x', 'fl_x', 'br_x', 'fr_x', 'bl_y', 'fl_y', 'br_y','fr_y']].to_numpy()
        pos_samples = torch.as_tensor(corners).view(-1, 8).float()
        neg_num= 100 - pos_samples.shape[0]
        neg_samples = torch.FloatTensor(self.box_sampler.sample(neg_num, label))
        if self.front_only:
            neg_samples[:,:,1] = neg_samples[:,:,1] - 269 
        else:
            neg_samples[:,:,1] = neg_samples[:,:,1] - 400
            neg_samples[:,:,0] = neg_samples[:,:,0] - 400
        neg_samples = neg_samples.view(-1,8) / 10
        samples = torch.cat([pos_samples, neg_samples], 0)
        #import pdb; pdb.set_trace()
        sample_pixels = [] 
        for sample_idx in range(samples.shape[0]):
            sample = samples[sample_idx].view(1, 2, 4)
            print(sample)
            image = main_binary_roadmap_objdetection(sample.view(1,2,4)) 
            image = np.rot90(image[131:669,400:])
            sample_pixels.append(image)
        samples =  np.append(None, sample_pixels, axis = 0)

        import cv2
        cv2.imwrite('sample.png', image * 255)
        import pdb; pdb.set_trace()
        target = torch.cat([torch.ones(pos_samples.shape[0]), torch.zeros(neg_num)]).float()
        # data: (256, 16, 20)
        # samples: (n = 100, 8)
        # target: 1000
        # data: CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT, CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

        # output label: [height = 538, width = 400] ---> rotate counterclockwise [h = 400, width = 538]
        #print(data.shape, samples.shape, target.shape)
        if self.front_only:
            return data[1,:], samples, target, label
        else:
            return data, samples, target, label

if __name__ == "__main__":
    #image_path = "/beegfs/jt3545/data/detection/train/image_tensor"
    image_path = "/beegfs/cy1355/camera_tensor_train/image_tensor"
    annotation_path = "/beegfs/cy1355/data/annotation.csv"
    #label_path =  "/beegfs/cy1355/obj_binary_roadmap_train/road_map"

    # train_loader = FrontObjectSegmentationDataset(image_path, label_path)
    # for data, label in iter(train_loader):
    #     assert (label.shape[0] == 400) & (label.shape[1] == 538)
    train_loader = CameraBasedObjectRegressionDataset(image_path,"/beegfs/cy1355/obj_binary_roadmap_train/road_map", annotation_path)
    data = train_loader[1]
    print(data[0].shape, data[1].shape, data[2].shape, data[3].shape)
