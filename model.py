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

if __name__ == "__main__":
    inputs = torch.rand((8, 6, 256, 16, 20))

    fusion = FusionLayer().eval()
    static = StaticModel().eval()
    with torch.no_grad():
        out1 = fusion(inputs)
        print(out1.shape) # 8, 256, 48, 40
        out2 = static(out1)
        print(out2.shape)
