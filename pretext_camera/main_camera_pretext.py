import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
from trainpretext import epoch_loop 
from datapretext import CameraPretextDataset
from modelpretext import Network
from utilspretext import NoiseContrastiveEstimator

# Setting initial parameters for training
def set_flags(param):
    param['device'] = torch.device("cuda:0" if torch.cuda.is_available() 
                                            else "cpu")
    param['epochs'] = 5000
    param['run_name'] = 'camera_pretext_2'
    

def main():
    param = {}
    set_flags(param)
    init_loggers(param)
    init_data(param)
    init_model(param)
    init_optimizers(param)
    for epoch in range(param['epochs']):
        param['running_loss'] = 0.0 
        for batch_i, (inputs, labels) in enumerate(tqdm(param['train_loader'])):
            param['optimizer'].zero_grad()
            inputs = inputs.to(param['device'])
            labels = labels.to(param['device'])
            pirl_features = param['model'][0]
            classifier = param['model'][1].train()
            outputs = pirl_features(inputs).view(-1, 512)
            outputs_c = classifier(outputs)
            loss = param['criterion'](outputs_c, labels.float())
            loss.backward()
            param['optimizer'].step()
            param['running_loss'] += loss.item()
        print("Epoch {}, Train Loss: {}".format(epoch, param['running_loss'] / len(param['train_loader'])))


def init_loggers(param):
    pass

def init_data(param):
    
    batch_size_n = 64

    unlabeled_scene_index = np.arange(106)

    trainset = CameraPretextDataset("/beegfs/cy1355/data", unlabeled_scene_index)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                              batch_size = batch_size_n, 
                                              shuffle=True, 
                                              num_workers=0)
    param['train_loader'] = trainloader
    #no validation loader here


def init_model(param):
    model = Network().to(param['device'])

    model_dict = model.state_dict()
    pretrained_dict = torch.load('./epoch_15')
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model = model.network()
    print('*** Load model successfully ***')
    for param_ in model.parameters():
        param_.requires_grad = False
    model_ll = nn.Sequential(nn.Linear(512, 6), nn.Softmax())
    param['model'] = (model, model_ll)

def init_optimizers(param):
    criterion = nn.BCELoss()

    parameters = param['model'][1].parameters()
    optimizer = optim.SGD(parameters, lr=0.001, momentum=0.9)

    param['criterion'] = criterion
    param['optimizer'] = optimizer

def init_distributed(param):
    pass

if __name__ == "__main__":
    main()
