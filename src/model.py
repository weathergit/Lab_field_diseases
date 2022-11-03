# -*- coding: utf-8 -*-
# @Time    : 2022/10/24 下午9:05
# @Author  : weather
# @Software: PyCharm

from torchvision import models
from torch import nn as nn


"""
class Model(nn.Module):
    def __int__(self):
        pass
    
    def __len__(self):
        pass
    
    def __getitem__(self, item):
        pass
         
    def forward():
        pass
"""


def set_model(num_cls, name:str):
    if name == 'effi0':
        efficientnet = models.efficientnet_b0(pretrained=True)
        efficientnet.classifier[1] = nn.Linear(in_features=1280, out_features=num_cls)
        return efficientnet
    elif name == 'mobilev3':
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        mobilenet.classifier[3] = nn.Linear(in_features=1024, out_features=num_cls)
        return mobilenet
    elif name == 'res34':
        resnet = models.resnet34(pretrained=True)
        resnet_infeature = resnet.fc.in_features
        resnet.fc = nn.Linear(resnet_infeature, num_cls)
        return resnet
    else:
        raise ValueError("only Efficient Net B0, MobileNet V3 and ResNet34 are used.")


if __name__ == "__main__":

    model = set_model(5, 'effi0')
    print(model)