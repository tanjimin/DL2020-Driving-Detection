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


