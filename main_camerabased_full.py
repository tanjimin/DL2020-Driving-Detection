import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from train import epoch_loop 
from data import LaneSegmentationDataset
from model import FrontStaticModel, FusionSixCameras

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 500
    param['run_name'] = 'camerabased_full'
    

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
    
    batch_size_n = 64

    trainset = LaneSegmentationDataset("/beegfs/cy1355/camera_tensor_train/image_tensor", "/beegfs/cy1355/camera_tensor_train/road_map")
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size = batch_size_n, 
                                              shuffle=True, 
                                              num_workers=0)
    param['train_loader'] = trainloader

    validationset = LaneSegmentationDataset("/beegfs/cy1355/camera_tensor_val/image_tensor", "/beegfs/cy1355/camera_tensor_val/road_map")
    validationloader = torch.utils.data.DataLoader(validationset, 
                                              batch_size = batch_size_n, 
                                              shuffle=True, 
                                              num_workers=0)
    param['validation_loader'] = validationloader
def init_model(param):
    fusion = FusionSixCameras().to(param['device'])
    model = FrontStaticModel().to(param['device'])
    model = torch.load('./static_camerabased_100')
    print('*** Model loads successfully ***')
    
    for param_ in model.parameters():
        param_.requires_grad = False
    
    param['model'] = (model, fusion)

def init_optimizers(param):
    criterion = nn.BCELoss()

    # parameters = param['model'][0].parameters()
    parameters_fusion = param['model'][1].parameters()

    # optimizer = optim.SGD(parameters, lr=0.00005, momentum=0.9)
    optimizer_fusion = optim.SGD(parameters_fusion, lr=0.001, momentum=0.9)

    param['criterion'] = criterion
    # param['optimizer'] = (optimizer, optimizer_fusion)
    param['optimizer'] = optimizer_fusion

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
