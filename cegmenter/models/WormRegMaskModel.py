# This is the one used in 2020-04-01 Regression Testing
# STD lib imports
from collections import OrderedDict
import os
# 3rd party libs
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
# local stuff
from models.layers import *





class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=(0, 1, 2, 3),
            num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1): #what the fuck are the values of i for this? range(start, stop, step)
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            

            self.convs[("depthconvM", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("depthconvX", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            self.convs[("depthconvY", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:       
                self.outputs[("Mask", i)] = self.sigmoid(self.convs[("depthconvM", i)](x))
                self.outputs[("X_Coord", i)] = self.convs[("depthconvX", i)](x)
                self.outputs[("Y_Coord", i)] = self.convs[("depthconvY", i)](x)

        return self.outputs


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = 'reglib'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152}
        resnets_pretrained_path = {18: 'resnet18-5c106cde.pth', 34: 'resnet34-333f7ec4.pth', 50: 'resnet50.pth', 101: 'resnet101.pth', 152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()

        if pretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
        
        
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []

        x = self.encoder.conv1(input_image)
        #print(x.shape)
        x = self.encoder.bn1(x)
        #print(x.shape)
        self.features.append(self.encoder.relu(x))
        #print(self.features[0].shape)
        #print(len(self.features[0]))
        #print(self.features[0])
        self.features.append(
            self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        #what is self.features here? are they the perceptrons with weights? or are they the result of putting the previous layer through the weights?
        return self.features

class WormRegModel(nn.Module):
    def __init__(self, num_layers, pretrained=False):
        super(WormRegModel, self).__init__()
        self.scale = (0,1,2,3) 
        self.encoder = ResnetEncoder(num_layers, pretrained)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, self.scale)
        
        
    def forward(self, inputs1):  
        inputs = (inputs1 )
        features = self.encoder(inputs)
        outputs = self.decoder(features)
        return outputs
