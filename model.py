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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_original_features = nn.Linear(256, 32)

    def forward(self, inputs):
        out = self.avgpool(inputs)
        #out = inputs.permute(0, 2, 3, 1).view(-1, 16 * 20, 256)
        out = torch.flatten(out, 1)
        out = self.projection_original_features(out)
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


