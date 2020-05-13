import os
import random

import numpy as np
import pandas as pd
import math

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import transforms as TF
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

from PIL import Image


image_folder = '/beegfs/cy1355/data'
unlabeled_scene_index = np.arange(106)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
gau_noise = AddGaussianNoise()


transform = transforms.Compose([
    transforms.ColorJitter(brightness = 0.5, contrast = 1),
    transforms.RandomRotation(degrees = 45),
    transforms.RandomChoice([transforms.RandomHorizontalFlip(p=0.2), \
                             transforms.RandomVerticalFlip(p=0.2)]),
    transforms.ToTensor(),
    transforms.Lambda(gau_noise)
])

### All four rotated images are returned
NUM_SAMPLE_PER_SCENE = 126
NUM_IMAGE_PER_SAMPLE = 6
image_names = [
    'CAM_FRONT_LEFT.jpeg',
    'CAM_FRONT.jpeg',
    'CAM_FRONT_RIGHT.jpeg',
    'CAM_BACK_LEFT.jpeg',
    'CAM_BACK.jpeg',
    'CAM_BACK_RIGHT.jpeg',
    ]

# The dataset class for unlabeled data.
class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder, scene_index, first_dim, transform):
        """
        Args:
            image_folder (string): the location of the image folder
            scene_index (list): a list of scene indices for the unlabeled data 
            first_dim ({'sample', 'image'}):
                'sample' will return [batch_size, NUM_IMAGE_PER_SAMPLE, 3, H, W]
                'image' will return [batch_size, 3, H, W] and the index of the camera [0 - 5]
                    CAM_FRONT_LEFT: 0
                    CAM_FRONT: 1
                    CAM_FRONT_RIGHT: 2
                    CAM_BACK_LEFT: 3
                    CAM_BACK.jpeg: 4
                    CAM_BACK_RIGHT: 5
            transform (Transform): The function to process the image
        """

        self.image_folder = image_folder
        self.scene_index = scene_index
        self.transform = transform

        assert first_dim in ['sample', 'image']
        self.first_dim = first_dim

    def __len__(self):
        if self.first_dim == 'sample':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE
        elif self.first_dim == 'image':
            return self.scene_index.size * NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE
    
    def __getitem__(self, index):
        if self.first_dim == 'sample':
            scene_id = self.scene_index[index // NUM_SAMPLE_PER_SCENE]
            sample_id = index % NUM_SAMPLE_PER_SCENE
            sample_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}') 

            images = []
            for image_name in image_names:
                image_path = os.path.join(sample_path, image_name)
                image = Image.open(image_path)
                images.append(self.transform(image))
            image_tensor = torch.stack(images)
            
            return image_tensor

        elif self.first_dim == 'image':
            scene_id = self.scene_index[index // (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)]
            sample_id = (index % (NUM_SAMPLE_PER_SCENE * NUM_IMAGE_PER_SAMPLE)) // NUM_IMAGE_PER_SAMPLE
            image_name = image_names[index % NUM_IMAGE_PER_SAMPLE]

            image_path = os.path.join(self.image_folder, f'scene_{scene_id}', f'sample_{sample_id}', image_name) 
            
            image = Image.open(image_path)
            
            ################################################
            # counter-clockwise rotate
            rotated_imgs = [self.transform(TF.functional.rotate(image, 0)),
                            self.transform(TF.functional.rotate(image, 90)),
                            self.transform(TF.functional.rotate(image, 180)),
                            self.transform(TF.functional.rotate(image, 270))]
            rotation_labels = torch.LongTensor([0, 1, 2, 3])
            ################################################
            rand_mask = torch.randperm(4)
            return torch.stack(rotated_imgs, dim=0)[rand_mask], rotation_labels[rand_mask]
            # return torch.stack(rotated_imgs, dim=0), rotation_labels
            
def _collate_fun(batch):
    batch = default_collate(batch)
    assert(len(batch)==2)
    batch_size, rotations, channels, height, width = batch[0].size()
    batch[0] = batch[0].view([batch_size*rotations, channels, height, width])
    batch[1] = batch[1].view([batch_size*rotations])
    return batch

# All four rotated images
unlabeled_trainset = UnlabeledDataset(image_folder=image_folder, scene_index=unlabeled_scene_index, first_dim='image', transform=transform)
trainloader = torch.utils.data.DataLoader(unlabeled_trainset, batch_size=4, shuffle=True, num_workers=0, collate_fn=_collate_fun)

######################################################################
# ResNet 18
######################################################################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet18()
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.5)
criterion = nn.CrossEntropyLoss()
num_epochs = 1000

if not os.path.exists('./saved_model/'):
	os.mkdir('./saved_model/')

def train(model, device, train_loader, optimizer, epoch, log_interval = 100):
    
    model.train()
    
    for batch_idx, (rot_image, rot_target) in enumerate(train_loader):
    
        data, rot_target = rot_image.to(device), rot_target.to(device)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        _, preds = torch.max(output, 1)
        loss = criterion(output, rot_target)
        accuracy = (rot_target == preds).sum().float() / len(rot_target)
        
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(
                epoch, batch_idx * len(data)/4, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), accuracy))

    torch.save(model, './saved_model/pretrained_rotation_{}'.format(epoch))


for epoch in range(1, num_epochs + 1):
    train(model, device, trainloader, optimizer, epoch, log_interval = 100)

######################################################################
# ResNet Old version
######################################################################                   
# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size):
#         super(BasicBlock, self).__init__()
#         padding = (kernel_size-1)//2
#         self.layers = nn.Sequential()
#         self.layers.add_module('Conv', nn.Conv2d(in_planes, out_planes, \
#             kernel_size=kernel_size, stride=1, padding=padding, bias=False))
#         self.layers.add_module('BatchNorm', nn.BatchNorm2d(out_planes))
#         self.layers.add_module('ReLU',      nn.ReLU(inplace=True))

#     def forward(self, x):
#         return self.layers(x)

#         feat = F.avg_pool2d(feat, feat.size(3)).view(-1, self.nChannels)

# class GlobalAveragePooling(nn.Module):
#     def __init__(self):
#         super(GlobalAveragePooling, self).__init__()

#     def forward(self, feat):
#         num_channels = feat.size(1)
#         return F.avg_pool2d(feat, (feat.size(2), feat.size(3))).view(-1, num_channels)

# class NetworkInNetwork(nn.Module):
#     def __init__(self, opt):
#         super(NetworkInNetwork, self).__init__()

#         num_classes = opt['num_classes']
#         num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
#         num_stages = opt['num_stages'] if ('num_stages' in opt) else 3
#         use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True


#         assert(num_stages >= 3)
#         nChannels  = 192
#         nChannels2 = 160
#         nChannels3 = 96

#         blocks = [nn.Sequential() for i in range(num_stages)]
#         # 1st block
#         blocks[0].add_module('Block1_ConvB1', BasicBlock(num_inchannels, nChannels, 5))
#         blocks[0].add_module('Block1_ConvB2', BasicBlock(nChannels,  nChannels2, 1))
#         blocks[0].add_module('Block1_ConvB3', BasicBlock(nChannels2, nChannels3, 1))
#         blocks[0].add_module('Block1_MaxPool', nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

#         # 2nd block
#         blocks[1].add_module('Block2_ConvB1',  BasicBlock(nChannels3, nChannels, 5))
#         blocks[1].add_module('Block2_ConvB2',  BasicBlock(nChannels,  nChannels, 1))
#         blocks[1].add_module('Block2_ConvB3',  BasicBlock(nChannels,  nChannels, 1))
#         blocks[1].add_module('Block2_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1))

#         # 3rd block
#         blocks[2].add_module('Block3_ConvB1',  BasicBlock(nChannels, nChannels, 3))
#         blocks[2].add_module('Block3_ConvB2',  BasicBlock(nChannels, nChannels, 1))
#         blocks[2].add_module('Block3_ConvB3',  BasicBlock(nChannels, nChannels, 1))

#         if num_stages > 3 and use_avg_on_conv3:
#             blocks[2].add_module('Block3_AvgPool', nn.AvgPool2d(kernel_size=3,stride=2,padding=1))
#         for s in range(3, num_stages):
#             blocks[s].add_module('Block'+str(s+1)+'_ConvB1',  BasicBlock(nChannels, nChannels, 3))
#             blocks[s].add_module('Block'+str(s+1)+'_ConvB2',  BasicBlock(nChannels, nChannels, 1))
#             blocks[s].add_module('Block'+str(s+1)+'_ConvB3',  BasicBlock(nChannels, nChannels, 1))

#         # global average pooling and classifier
#         blocks.append(nn.Sequential())
#         blocks[-1].add_module('GlobalAveragePooling',  GlobalAveragePooling())
#         blocks[-1].add_module('Classifier', nn.Linear(nChannels, num_classes))

#         self._feature_blocks = nn.ModuleList(blocks)
#         self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier',]
#         assert(len(self.all_feat_names) == len(self._feature_blocks))

#     def _parse_out_keys_arg(self, out_feat_keys):

#         # By default return the features of the last layer / module.
#         out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

#         if len(out_feat_keys) == 0:
#             raise ValueError('Empty list of output feature keys.')
#         for f, key in enumerate(out_feat_keys):
#             if key not in self.all_feat_names:
#                 raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
#             elif key in out_feat_keys[:f]:
#                 raise ValueError('Duplicate output feature key: {0}.'.format(key))

#         # Find the highest output feature in `out_feat_keys
#         max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

#         return out_feat_keys, max_out_feat

#     def forward(self, x, out_feat_keys=None):
#         """Forward an image `x` through the network and return the asked output features.

#         Args:
#           x: input image.
#           out_feat_keys: a list/tuple with the feature names of the features
#                 that the function should return. By default the last feature of
#                 the network is returned.

#         Return:
#             out_feats: If multiple output features were asked then `out_feats`
#                 is a list with the asked output features placed in the same
#                 order as in `out_feat_keys`. If a single output feature was
#                 asked then `out_feats` is that output feature (and not a list).
#         """
#         out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
#         out_feats = [None] * len(out_feat_keys)

#         feat = x
#         for f in range(max_out_feat+1):
#             feat = self._feature_blocks[f](feat)
#             key = self.all_feat_names[f]
#             if key in out_feat_keys:
#                 out_feats[out_feat_keys.index(key)] = feat

#         out_feats = out_feats[0] if len(out_feats)==1 else out_feats
#         return out_feats


#     def weight_initialization(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m.weight.requires_grad:
#                     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                     m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 if m.weight.requires_grad:
#                     m.weight.data.fill_(1)
#                 if m.bias.requires_grad:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 if m.bias.requires_grad:
#                     m.bias.data.zero_()

# def create_model(opt):
#     return NetworkInNetwork(opt)

# opt = {'num_classes':4, 'num_stages': 5}