import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from train import epoch_loop 

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 50

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
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), 
                                                         (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True, 
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size=4, 
                                              shuffle=True, 
                                              num_workers=2)
    param['train_loader'] = trainloader

def init_model(param):
    from model_test import Net
    net = Net()
    param['model'] = net

def init_optimizers(param):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(param['model'].parameters(), 
                          lr=0.001, 
                          momentum=0.9)
    param['criterion'] = criterion
    param['optimizer'] = optimizer

def init_distributed(param):
    pass


if __name__ == "__main__":
    main()
