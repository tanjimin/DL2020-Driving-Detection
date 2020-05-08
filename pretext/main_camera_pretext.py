import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from trainpretext import epoch_loop 
from datapretext import UnlabeledJigsawDataset
from modelpretext import Network
from utilspretext import NoiseContrastiveEstimator

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 5000
    param['run_name'] = 'camera_pretext'
    

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

    unlabeled_scene_index = np.arange(106)

    trainset = UnlabeledJigsawDataset("/beegfs/cy1355/data", unlabeled_scene_index)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size = batch_size_n, 
                                              shuffle=True, 
                                              num_workers=0)
    param['train_loader'] = trainloader
    #no validation loader here


def init_model(param):
    model = Network().to(param['device'])
    #model = torch.load('./pretrained_rotation_7')
    print('*** Train from Scratch ***')
    
    param['model'] = model

def init_optimizers(param):
    criterion = NoiseContrastiveEstimator(param['device'])

    parameters = param['model'].parameters()
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)

    param['criterion'] = criterion
    param['optimizer'] = optimizer

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
