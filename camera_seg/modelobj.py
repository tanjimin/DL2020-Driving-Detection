import torch
import torch.nn as nn
import torch.functional as F

class FusionLayer(nn.Module):
    """
    To combine context from 6 cameras and return a single context matrix
    """
    def __init__(self, method = 'concat'):
        super(FusionLayer, self).__init__()
        self.method = method

    def forward(self, inputs):
        """
        Inputs:
            inputs: batch * 6 * C * H * W
        Outputs:
            out: batch * C * 3H * 2W
        """
        C, H, W = inputs.shape[2], inputs.shape[3], inputs.shape[4]
        chunk_list = torch.chunk(inputs, 6, 1)
        chunk1 = torch.cat(chunk_list[:3], 3)
        chunk2 = torch.cat(chunk_list[::-1][:3],3)
        out = torch.cat([chunk1, chunk2], 4)
        return out.squeeze()



class StaticModel(nn.Module):
    def __init__(self):
        super(StaticModel, self).__init__()

        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 4, 2, 0),
                        nn.BatchNorm2d(64),
                        nn.ReLU())

        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(64, 16, 6, (2,3), (1,5)),
                        nn.BatchNorm2d(16),
                        nn.ReLU())

        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(16, 4, 6, 2, (1,5)),
                        nn.BatchNorm2d(4),
                        nn.ReLU())
        
        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(4, 1, 6, 2, (0,5)),
                        nn.Sigmoid()
                        )


    def forward(self, inputs):
        """
        Inputs:
            inputs: batch * C * 3H * 2W
        Outputs:
            out: batch * 1 * 800 * 800
        """
        
        out = self.deconv1(inputs)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out[:,:,:,69:-69]

class StaticPolarModel(nn.Module):
    def __init__(self):
        super(StaticPolarModel, self).__init__()

        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 6, 1, 0),
                        nn.BatchNorm2d(64),
                        nn.ReLU())

        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(64, 16, 6, 1, 0),
                        nn.BatchNorm2d(16),
                        nn.ReLU())

        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(16, 4, 6, 2, 1),
                        nn.BatchNorm2d(4),
                        nn.ReLU())
        
        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(4, 1, 6, 2, 0),
                        nn.Sigmoid()
                        )


    def forward(self, inputs):
        """
        Inputs:
            inputs: batch * C * H * W
        Outputs:
            out: batch * 1 * 800 * 800
        """
        
        out = self.deconv1(inputs)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out


class FrontStaticModel(nn.Module):
    def __init__(self):
        super(FrontStaticModel, self).__init__()

        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(256, 64, 4, 2, 0),
                        nn.BatchNorm2d(64),
                        nn.ReLU())

        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(64, 16, 6, 2, (1,0)),
                        nn.BatchNorm2d(16),
                        nn.ReLU())

        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(16, 4, 6, 3, (5,0)),
                        nn.BatchNorm2d(4),
                        nn.ReLU())
        
        self.deconv4 = nn.Sequential(
                        nn.ConvTranspose2d(4, 1, 6, 2, (5,0)),
                        nn.Sigmoid()
                        )


    def forward(self, inputs):
        """
        Inputs:
            inputs: batch * C * H * W. currently 6 * 256 * 16 * 20
        Outputs:
            out: batch * 1 * 400 * 538
        """
        
        out = self.deconv1(inputs)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out

class FrontDynamicModel(nn.Module):
    """
    Inputs:
        inputs: Image representations   batch * C * H * W. currently 6 * 256 * 16 * 20 
    Ouputs:
        out: Flattened representations   batch * (H*W) * C'
    """
    def __init__(self):
        super(FrontDynamicModel, self).__init__()

        self.conv1 = nn.Sequential(
                        nn.Conv2d(256, 512, 5, 2, (2, 0)),
                        nn.BatchNorm2d(512),
                        nn.ReLU())

        self.conv2 = nn.Sequential(
                        nn.Conv2d(512, 1024, 3, 2, 0),
                        nn.BatchNorm2d(1024),
                        nn.ReLU())

        self.conv3 = nn.Sequential(
                        nn.Conv2d(1024, 2048, 3, 1, 0),
                        nn.BatchNorm2d(2048),
                        nn.ReLU())
        
        self.fc = nn.Linear(2048, 1024)

    def forward(self, inputs):
        """
        Inputs:
            inputs: batch * C * H * W. currently 6 * 256 * 16 * 20
        Outputs:
            out: batch * 1 * 400 * 538
        """
        
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out.view(-1, 2048))
        return out

    """    
    def __init__(self):
        super(FrontDynamicModel, self).__init__() 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_original_features = nn.Linear(256, 32)

    def forward(self, inputs):
        out = self.avgpool(inputs)
        #out = inputs.permute(0, 2, 3, 1).view(-1, 16 * 20, 256)
        out = torch.flatten(out, 1)
        out = self.projection_original_features(out)
        return out
    """
class BoundingBoxClassifier(nn.Module):
    """
    Inputs:
        inputs: Image representations   batch * C * H * W. currently 6 * 256 * 16 * 20 
    Ouputs:
        out: Flattened representations   batch * (H*W) * C'
    """
    def __init__(self):
        super(BoundingBoxClassifier, self).__init__()

        self.encoder = nn.Sequential(
                        nn.Linear(1024 + 32, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Linear(64, 128),
                        nn.BatchNorm1d(128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.BatchNorm1d(64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                        nn.Sigmoid())
                       

    def forward(self, camera, bbox):
        """
        Inputs:
            camera: batch * H (currently n * 1024) 
            bbox: batch * H (currently n * 32)
        Outputs:
            out: batch * 1 * 400 * 538
        """
        inputs = torch.cat((camera, bbox), dim = 1)
        out = self.encoder(inputs)
        return out

class BoundingBoxEncoder(nn.Module):
    """
    Inputs:
        samples: Bounding box coordinates   (batch * samples) * 8  
    Ouputs:
        out: Bounding box representations   (batch * samples) * 8 * C'
    """
    def __init__(self):
        super(BoundingBoxEncoder, self).__init__() 
        self.fc1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.fc2 = nn.Linear(16, 32)

    def forward(self, samples):
        out = self.fc1(samples.unsqueeze(1))
        out = self.fc2(out)
        return out.squeeze()


class FusionSixCameras(nn.Module):
    def __init__(self):
        super(FusionSixCameras, self).__init__()

        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels = 6, out_channels = 2, \
                                           kernel_size = 6, stride = 2, padding = 2),
                        nn.BatchNorm2d(2),
                        nn.ReLU())

        # self.conv2 = nn.Sequential(
        #                 nn.Conv2d(in_channels = 2, out_channels = 2, \
        #                           kernel_size = 5, stride = (1,4), padding = 2),
        #                 nn.ReLU())

        self.deconv3 = nn.Sequential(
                        nn.ConvTranspose2d(in_channels = 2, out_channels = 1, \
                                           kernel_size = (5,4), stride = (1,3), padding = (2,4)),
                        nn.Sigmoid())

    def forward(self, inputs):
        """
        Inputs:
            inputs: batch * 6 * 400 * 538
        Outputs:
            out: batch * 1 * 800 * 800
        """
        
        out = self.deconv1(inputs)
        out = nn.functional.max_pool2d(out, (1,4))
        out = self.deconv3(out)
        return out


# class BoundingBoxDecoder(nn.Module):
#     """
#     Inputs:
#         Concatenated vector from FrontDynamicModel and BoundingBoxEncoder (batch * samples) * (32+32)
#     Ouputs:
#         Bounding boc probabilities   (batch * samples) * 1
#     """
#     def __init__(self):
#         super(BoundingBoxDecoder, self).__init__() 
#         self.fc1 = nn.Sequential(nn.Linear(32+32, 16), nn.ReLU())
#         self.fc2 = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

#     def forward(slef, inputs):
#         out = self.fc1(inputs) 
#         out = self.fc2(out)
#         return out



####################################################################
# Unet Block
####################################################################
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,\
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        out = self.sig(logits)
        return out
####################################################################

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


if __name__ == "__main__":
    # inputs = torch.rand((8, 6, 256, 16, 20))

    # fusion = FusionLayer().eval()
    # static = StaticModel().eval()
    # with torch.no_grad():
    #     out1 = fusion(inputs)
    #     print(out1.shape) # 8, 256, 48, 40
    #     out2 = static(out1)
    #     print(out2.shape)

    # inputs2 = torch.rand((8, 256, 188, 188))
    # staticpolar = StaticPolarModel().eval()
    # with torch.no_grad():
    #     out3 = staticpolar(inputs2)
    #     print(out3.shape) 

    # inputs3 = torch.rand((8, 256, 16, 20))
    # frontStatic = FrontStaticModel().eval()
    # with torch.no_grad():
    #     out4 = frontStatic(inputs3)
    #     print(out4.shape) 

    inputs4 = torch.rand((8, 256, 16, 20))
    inputs5 = torch.rand((8, 8))
    m1 = FrontDynamicModel().eval()
    m2 = BoundingBoxEncoder().eval()
    #m3 = BoundingBoxDecoder().eval()
    with torch.no_grad():
        out5_1 = m1(inputs4)
        out5_2 = m2(inputs5)
        print(out5_1.shape)
        print(out5_2.shape)


