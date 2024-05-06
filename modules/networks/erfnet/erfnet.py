# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def add_coords(x: torch.Tensor) -> torch.Tensor:
  batch_size, _, height, width = x.shape
  xm = torch.linspace(-1, 1, width, device=x.device).view(1, 1, -1).expand(1, width, width)  # x-coords in rows
  ym = torch.linspace(-1, 1, height, device=x.device).view(1, -1, 1).expand(1, height, height) # y-coords in cols
  xym = torch.cat((xm, ym), 0).unsqueeze(0) # [1 x 2 x H x W]
  xym_batched = torch.cat(batch_size * [xym])

  out = torch.cat([x, xym_batched], dim=1) # [b x (c + 2) x h x w]

  return out

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput,
                              (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(
            1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(
            0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        DROPOUT_01 = 0.03
        for _ in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, DROPOUT_01, 1)) 

        self.layers.append(DownsamplerBlock(64, 128))

        DROPOUT_02 = 0.3 
        for _ in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, DROPOUT_02, 2))
            self.layers.append(non_bottleneck_1d(128, DROPOUT_02, 4))
            self.layers.append(non_bottleneck_1d(128, DROPOUT_02, 8))
            self.layers.append(non_bottleneck_1d(128, DROPOUT_02, 16))

        if pretrained:
            self.load_encoder_pretrained()

    def load_encoder_pretrained(self):
        path = "/home/janw/projects/instance_segmentation/pretrained/erfnet/erfnet_encoder_pretrained.pth.tar"
        ckpt_dict = torch.load(path)['state_dict']
        
        ckpt_dict_new = {}
        ckpt_dict_new['state_dict'] = OrderedDict()

        for key in ckpt_dict.keys():
            if "classifier" not in key:
                key_new = key.replace("module.features.encoder.", "")
                ckpt_dict_new['state_dict'][key_new] = ckpt_dict[key].clone()
        del ckpt_dict

        self.load_state_dict(ckpt_dict_new['state_dict'], strict=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            # if layer.__class__ .__name__ == 'DownsamplerBlock':
            #     output = add_coords(output) 
            output = layer(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)

class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            # if layer.__class__ .__name__ == 'UpsamplerBlock':
            #     output = add_coords(output)
            output = layer(output)

        # output = add_coords(output)
        output = self.output_conv(output)

        return output

# ERFNet
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(num_classes)

    def forward(self, input):
        output = self.encoder(input)

        return self.decoder.forward(output)