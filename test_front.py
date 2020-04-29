import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from train import validation_loop 
from data import FrontLaneSegmentationDataset
from model import FrontStaticModel 

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 500
    param['run_name'] = 'front'
    param['model_dir'] = 'saves_front/static_polar_490'

def main():
    param = {}
    set_flags(param)
    init_loggers(param)
    init_data(param)
    init_model(param)
    init_optimizers(param)
    param['ts'] = 0
    validation_loop(490, param)
    print(param['ts'] / len(param['validation_loader'].dataset))

def init_loggers(param):
    pass

def init_data(param):
    validationset = FrontLaneSegmentationDataset("/beegfs/cy1355/camera_tensor_val/image_tensor", "/beegfs/cy1355/camera_tensor_val/road_map")
    validationloader = torch.utils.data.DataLoader(validationset, 
                                              batch_size=16, 
                                              shuffle=True, 
                                              num_workers=0)
    param['validation_loader'] = validationloader
def init_model(param):
    model = torch.load(param['model_dir']).eval() 
    param['model'] = model

def init_optimizers(param):
    criterion = nn.BCELoss()
    param['criterion'] = criterion

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
