import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torchvision
import itertools
import torchvision.transforms as transforms
from utils import concat_cameras, non_max_suppression
from train import epoch_loop 
from data import CameraBasedObjectRegressionDataset 
from model import FrontDynamicModel, BoundingBoxEncoder, BoundingBoxClassifier
from loss import focal_loss
import pandas as pd

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 1000
    param['run_name'] = 'bbox_reg_full'
    param['model_dir'] = 'saves_bbox_reg_full/static_bbox_reg_full_10'

def main():
    param = {}
    set_flags(param)
    init_loggers(param)
    init_data(param)
    init_model(param)
    init_optimizers(param)
    param['ts'] = 0
    bbox_list = validation(param)
    print(bbox_list)

def init_loggers(param):
    pass

def validation(param):
    # assume batch size = 1
    batch = None
    for batch_i in param['validation_loader']:
        batch = batch_i
        break
    threshould = 0.2
    inputs, samples, labels, graph = batch
    inputs = inputs.to(param['device'])
    labels = labels.to(param['device'])
    samples = samples.to(param['device'])
    
    camera_model = param['model'][0].train()
    bbox_model = param['model'][1].train()
    classifier = param['model'][2].train()
    # inputs of shape batch * 6 * 256 * 16 * 20
    camera_inputs = inputs.view(-1, 256, 16, 20)
    camera_feature = camera_model(camera_inputs).unsqueeze(1)
    
    all_bbox = param['validation_loader'].dataset.box_sampler.get_bbox().copy()
    all_bbox[:,:,1] = all_bbox[:,:,1] - 269 
    all_bbox_coord = all_bbox.mean(axis = 1).astype(int)
    all_bbox_size = all_bbox.shape[0]
    bbox_model = param['model'][1].eval()
    classifier = param['model'][2].eval()
    model_input = torch.LongTensor(all_bbox).view(-1, 8).to(param['device']) / 10
    bbox_feature = bbox_model(model_input.float())
    canvases = []
    for camera_idx in range(camera_feature.shape[0]):
        camera_batches = camera_feature[camera_idx].repeat(all_bbox_size, 1)
        pred = classifier(camera_batches, bbox_feature).squeeze(1)
        max_value = pred.max().item()
        min_value = pred.min().item()
        pred = pred > threshould
        canvas = np.zeros((538, 400))
        for coord_idx, coord in enumerate(all_bbox_coord):
            canvas[coord[0], coord[1]] = pred[coord_idx]
        canvases.append(canvas)
    # fusing function to generate 800 * 800
    canvases = np.asarray(canvases)
    canvases = np.rot90(canvases, axes = (1, 2)).reshape((1, 6, 400, 538))
    road_map = concat_cameras(canvases).reshape(800, 800)
    
    bbox_list = []
    for x_coord in range(800):
        for y_coord in range(800):
            if road_map[x_coord, y_coord] == 1:
                proc_x = (x_coord - 400) / 10
                proc_y = (y_coord - 400) / 10
                bbox = [proc_x - 4.5/2, proc_y - 1, proc_x + 4.5/2, proc_y - 1, proc_x - 4.5/2, proc_y + 1, proc_x + 4.5/2, proc_y + 1]
                bbox_list.append(bbox)
    bbox_list = np.asarray(bbox_list) * 10
    bbox_list = bbox_list.reshape(1, -1, 2, 4) 
    return bbox_list    

def init_data(param):
    validationset = CameraBasedObjectRegressionDataset("/beegfs/cy1355/obj_binary_roadmap_val/image_tensor", "/beegfs/cy1355/obj_binary_roadmap_val/road_map", "/beegfs/cy1355/data/annotation.csv")
    validationloader = torch.utils.data.DataLoader(validationset, 
                                              batch_size=1, 
                                              shuffle=True, 
                                              num_workers=2,
                                              pin_memory=True)
    param['validation_loader'] = validationloader
    
def init_model(param):
    model = torch.load(param['model_dir'])
    param['model'] = model

def init_optimizers(param):
    pass

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
