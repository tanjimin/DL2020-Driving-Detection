import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from train import epoch_loop 
from data import LaneSegmentationDataset
from model import FusionLayer, StaticModel

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 500

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
    trainset = LaneSegmentationDataset("/beegfs/cy1355/camera_tensor/image_tensor", "/beegfs/cy1355/camera_tensor/road_map")
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=16, 
                                              shuffle=True, 
                                              num_workers=2)
    param['train_loader'] = trainloader

def init_model(param):
    fusion = FusionLayer().to(param['device'])
    model = StaticModel().to(param['device'])
    param['model'] = (fusion, model) 

def init_optimizers(param):
    criterion = nn.BCELoss()
    parameters = list(param['model'][0].parameters()) + list(param['model'][1].parameters())
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)
    param['criterion'] = criterion
    param['optimizer'] = optimizer

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
