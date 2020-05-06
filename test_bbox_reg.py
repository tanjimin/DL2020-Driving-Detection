import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from validation import validation_loop 
from data import ObjectRegressionDataset
from model import FrontDynamicModel, BoundingBoxEncoder, BoundingBoxClassifier
from loss import focal_loss

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 500
    param['run_name'] = 'bbox_reg'
    param['model_dir'] = 'saves_bbox_reg/static_bbox_reg_10'

def main():
    param = {}
    set_flags(param)
    init_loggers(param)
    init_data(param)
    init_model(param)
    init_optimizers(param)
    param['ts'] = 0
    validation_loop(10, param)
    print(param['ts'] / len(param['validation_loader'].dataset))

def init_loggers(param):
    pass

def init_data(param):
    validationset = ObjectRegressionDataset("/beegfs/cy1355/obj_binary_roadmap_val/image_tensor", 
                                            "/beegfs/cy1355/obj_binary_roadmap_val/road_map", "/beegfs/cy1355/data/annotation.csv", True)
    validationloader = torch.utils.data.DataLoader(validationset, 
                                              batch_size=32, 
                                              shuffle=True, 
                                              num_workers=2,
                                              pin_memory=True)
    param['validation_loader'] = validationloader

def init_model(param):
    model = torch.load(param['model_dir'], map_location = 'cpu')
    param['model'] = model

def init_optimizers(param):
    criterion = focal_loss
    param['criterion'] = criterion

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
