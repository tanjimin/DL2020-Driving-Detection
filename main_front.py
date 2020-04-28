import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from train import epoch_loop 
from data import FrontLaneSegmentationDataset
from model import FrontStaticModel 

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 500
    param['run_name'] = 'front'

def main():
    param = {}
    set_flags(param)
    init_loggers(param)
    init_data(param)
    init_model(param)
    init_optimizers(param)
    epoch_loop(param)

def init_loggers(param):
    pass

def init_data(param):
    trainset = FrontLaneSegmentationDataset("/beegfs/cy1355/camera_tensor_train/image_tensor", "/beegfs/cy1355/camera_tensor_train/road_map")
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=16, 
                                              shuffle=True, 
                                              num_workers=0)
    param['train_loader'] = trainloader

    validationset = FrontLaneSegmentationDataset("/beegfs/cy1355/camera_tensor_val/image_tensor", "/beegfs/cy1355/camera_tensor_val/road_map")
    validationloader = torch.utils.data.DataLoader(validationset, 
                                              batch_size=16, 
                                              shuffle=True, 
                                              num_workers=0)
    param['validation_loader'] = validationloader
def init_model(param):
    model = FrontStaticModel().to(param['device'])
    param['model'] = model

def init_optimizers(param):
    criterion = nn.BCELoss()
    parameters = param['model'].parameters()
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)
    param['criterion'] = criterion
    param['optimizer'] = optimizer

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
