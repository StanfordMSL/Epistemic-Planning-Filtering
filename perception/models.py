import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Baseline Perception Model
class SmallEncoder3DBaseline(nn.Module):
    def __init__(self):
        super(SmallEncoder3DBaseline, self).__init__()

        image_channels = 1
        state_channels = 6
        ngf = 16

        self.main = nn.Sequential(

            nn.Conv2d(image_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Dropout2d(p=0.2),

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8)

        )

        self.pose_layer =  nn.Conv2d(ngf * 8, state_channels, 4, 1, 0, bias=False)
        self.cov_layer = nn.Conv2d(ngf * 8, state_channels, 4, 1, 0, bias=False)

    def forward(self, input):
        x_int = self.main(input)
        x_int = F.relu(x_int)
        pose = self.pose_layer(x_int)
        cov_int = self.cov_layer(x_int)
        cov = F.relu(cov_int)
        return pose, cov

# Perception Model
class SmallEncoder3D(nn.Module):
    def __init__(self):
        super(SmallEncoder3D, self).__init__()

        image_channels = 1
        state_channels = 6
        ngf = 16

        self.main = nn.Sequential(

            nn.Conv2d(image_channels, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Dropout2d(p=0.2),

            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.Conv2d(ngf * 8, state_channels, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)

# Generator Model
class FourierDecoder(nn.Module):
    def __init__(self):
        super(FourierDecoder, self).__init__()
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_channels = 1
        state_channels = 512
        ngf = 32 
        m = 256

        B = np.random.normal(0.0, 5., size=(m, 6))
        if device == torch.device("cpu"):
            self.B_tensor = torch.from_numpy(B)
        else:
            self.B_tensor = torch.from_numpy(B).cuda()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(state_channels, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # adding dropout layer
            nn.Dropout2d(p=0.2),

            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, image_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def fourier_encoding(self, pixel_coord):
        batch_size = pixel_coord.size()[0]
        pixel_coord = torch.squeeze(pixel_coord, 3)
        
        B_tensor = self.B_tensor.tile((batch_size, 1, 1))
        B_tensor = B_tensor.float()

        z = 2.*np.pi*pixel_coord
        z = z.float()

        z = torch.bmm(B_tensor, z)
        z = torch.cat((torch.sin(z), torch.cos(z)), axis=1)
        z = torch.unsqueeze(z,3)
        return z

    def forward(self, pixel_coord):
        pixel_coord = self.fourier_encoding(pixel_coord)
        image_out = self.main(pixel_coord)
        return image_out
