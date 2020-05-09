"""
You need to implement all four functions in this file and also put your team info as a variable
Then you should submit the python file with your model class, the state_dict, and this file
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
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
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

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
        model_all_state_dict = torch.load(self.model_file)

        ########################
        # Static
        ########################
        network = Network()
        network.load_state_dict(model_all_state_dict['resnet'])
        modules = list(network.children())[0][:-2]

        resnet = nn.Sequential(*modules).cuda()
        static = FrontStaticModel().cuda()
        unet = UNet(n_channels=1, n_classes=1, bilinear=True).cuda()

        static.load_state_dict(model_all_state_dict['static'])
        unet.load_state_dict(model_all_state_dict['unet'])

        ########################
        # Dynamic 
        ########################
        dynamic = FrontDynamicModel().cuda()
        bbox_encoder = BoundingBoxEncoder().cuda()
        bbox_classifier = BoundingBoxClassifier().cuda()

        dynamic.load_state_dict(model_all_state_dict['dynamic'])
        bbox_encoder.load_state_dict(model_all_state_dict['bbox_encoder'])
        bbox_classifier.load_state_dict(model_all_state_dict['bbox_classifier'])

        resnet = resnet.eval()
        static = static.eval()
        unet = unet.eval()
        dynamic = dynamic.eval()
        bbox_encoder = bbox_encoder.eval()
        bbox_classifier = bbox_classifier.eval()

        self.resnet = resnet
        self.static = static
        self.unet = unet
        self.dynamic = dynamic
        self.bbox_encoder = bbox_encoder
        self.bbox_classifier = bbox_classifier
        

    def get_bounding_boxes(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a tuple with size 'batch_size' and each element is a cuda tensor [N, 2, 4]
        # where N is the number of object
        batch_size, rotations, channels, height, width = samples.size()
        samples_input = samples.view([batch_size*rotations, channels, height, width])

        # resnet model output size [batch*6, 256, 16, 20]
        resnet_output = self.resnet(samples_input)

        # Front Dynamic
        # camera_feature: torch.Size([6, 1, 1024])
        camera_feature = self.dynamic(resnet_output).unsqueeze(1)
        import pdb;pdb.set_trace()

        all_bbox = param['validation_loader'].dataset.box_sampler.get_bbox().copy()
        all_bbox[:,:,1] = all_bbox[:,:,1] - 269 
        all_bbox_coord = all_bbox.mean(axis = 1)
        all_bbox_size = all_bbox.shape[0]

        model_input = torch.LongTensor(all_bbox).view(-1, 8).cuda() / 10
        bbox_feature = self.bbox_encoder(model_input.float())

        canvases = []
        for camera_idx in range(camera_feature.shape[0]):
            camera_batches = camera_feature[camera_idx].unsqueeze(0).repeat(all_bbox_size, 1)
            pred = self.bbox_classifier(camera_batches, bbox_feature).squeeze(1)
            max_value = pred.max().item()
            min_value = pred.min().item()
            pred = pred > threshould
            canvas = np.zeros(538, 400)
            for coord_idx, coord in enumerate(all_bbox_coord):
                canvas[coord[0], coord[1]] = pred[coord_idx]
            canvases.append(canvas)

        road_map = concat_cameras(canvases)

        bbox_list = []
        for x_coord in range(800):
            for y_coord in range(800):
                if road_map[x_coord, y_coord] == 1:
                    proc_x = (x_coord - 400) / 10
                    proc_y = (y_coord - 400) / 10
                    bbox = [proc_x - 4.5/2, proc_y - 1, proc_x + 4.5/2, proc_y - 1, proc_x - 4.5/2, proc_y + 1, proc_x + 4.5/2, proc_y + 1]
                    bbox_list.append(bbox)
        bbox_list = bbox_list.reshape(1, -1, 4, 2) 

        return torch.rand(1, 15, 2, 4) * 10

    def get_binary_road_map(self, samples):
        # samples is a cuda tensor with size [batch_size, 6, 3, 256, 306]
        # You need to return a cuda tensor with size [batch_size, 800, 800] 

        batch_size, rotations, channels, height, width = samples.size()
        samples_input = samples.view([batch_size*rotations, channels, height, width])

        # resnet model output size [batch*6, 256, 16, 20]
        resnet_output = self.resnet(samples_input)

        # reorder camera orders for the subsequent models
        static_input = reorder_tensor(resnet_output).cuda()

        # static model output size [batch*6, 400, 538]
        cameras_outputs = self.static(static_input).squeeze(1)

        # unet model output size [batch, 800, 800]
        cameras_outputs = cameras_outputs.view(-1, 6, 400, 538)
        unet_input = concat_cameras(cameras_outputs.cpu().numpy()).unsqueeze(1).cuda()
        outputs = self.unet(unet_input).squeeze(1)

        return outputs > 0.5
