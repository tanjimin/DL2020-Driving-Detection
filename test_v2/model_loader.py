"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# import your model class
# import ...
from model import *
from utils import *

# Put your transform function here, we will use it for our dataloader
# For bounding boxes task
def get_transform_task1(): 
    return torchvision.transforms.ToTensor()
# For road map task
def get_transform_task2(): 
    return torchvision.transforms.ToTensor()

class ModelLoader():
    # Fill the information for your team
    team_name = 'MalaProject'
    team_number = 21
    round_number = 3
    team_member = ['Jimin Tan', 'Yakun Wang', 'Chenqin Yang']
    contact_email = ['jt3545@nyu.edu', 'yw3918@nyu.edu', 'cy1355@nyu.edu']

    def __init__(self, model_file='model_all.pt'):
        # You should 
        #       1. create the model object
        #       2. load your state_dict
        #       3. call cuda()
        # self.model = ...

        self.model_file = model_file

        resnet = resnet18().cuda()
        static = FrontStaticModel().cuda()
        unet = UNet(n_channels=1, n_classes=1, bilinear=True).cuda()

        model_all_state_dict = torch.load(self.model_file)

        resnet.load_state_dict(model_all_state_dict['resnet'])
        static.load_state_dict(model_all_state_dict['static'])
        unet.load_state_dict(model_all_state_dict['unet'])

        resnet = resnet.eval()
        static = static.eval()
        unet = unet.eval()

        self.resnet = resnet
        self.static = static
        self.unet = unet
        

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 

        batch_size, rotations, channels, height, width = samples.size()
        samples_input = samples.view([batch_size*rotations, channels, height, width])

        # drop last two layers
        modules = list(self.resnet.children())[:-3]
        res_model = nn.Sequential(*modules)
        res_model = res_model.eval()

        # resnet model output size [batch*6, 256, 16, 20]
        resnet_output = res_model(samples_input)

        # reorder camera orders for the subsequent models
        static_input = reorder_tensor(resnet_output).cuda()

        # static model output size [batch*6, 400, 538]
        cameras_outputs = self.static(static_input).squeeze(1)

        # unet model output size [batch, 800, 800]
        cameras_outputs = cameras_outputs.view(-1, 6, 400, 538)
        unet_input = concat_cameras(cameras_outputs.cpu().numpy()).unsqueeze(1).cuda()
        outputs = self.unet(unet_input).squeeze(1)

        return outputs > 0.5
