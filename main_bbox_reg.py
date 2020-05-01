import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from train import epoch_loop 
from data import ObjectSegmentationDataset
from model import FrontDynamicModel, BoundingBoxEncoder 
#from loss import focal_loss
import pandas as pd

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 500
    param['run_name'] = 'bbox_reg'

def main():
    param = {}
    set_flags(param)
    init_loggers(param)
    init_data(param)
    init_model(param)
    init_optimizers(param)
    param['ts'] = 0
    epoch_loop(param)

def init_loggers(param):
    pass


def init_data(param):
    trainset = FrontObjectSegmentationDataset("/beegfs/cy1355/obj_binary_roadmap_train/image_tensor", "/beegfs/cy1355/data/annotation.csv", True)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=32, 
                                              shuffle=True, 
                                              num_workers=0 )

    param['train_loader'] = trainloader

    validationset = FrontObjectSegmentationDataset("/beegfs/cy1355/obj_binary_roadmap_val/image_tensor", "/beegfs/cy1355/data/annotation.csv", True)
    validationloader = torch.utils.data.DataLoader(validationset, 
                                              batch_size=32, 
                                              shuffle=True, 
                                              num_workers=0)
    param['validation_loader'] = validationloader
    
def init_model(param):
    models = [FrontDynamicModel().to(param['device']) ]
    models.append(BoundingBoxEncoder().to(param['device']))
    param['model'] = models

def init_optimizers(param):
    criterion = nn.CosineEmbeddingLoss(0.1)
    parameters = param['model'].parameters()
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)
    param['criterion'] = criterion
    param['optimizer'] = optimizer

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
