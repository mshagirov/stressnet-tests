from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18 as pt_resnet18

from models import TORCH_DEVICE


def resnet18_classifier(
    num_classes:int = 2,
    pretrained:bool = True,
    weights_path:str|Path|None = None,
    device:torch.device = TORCH_DEVICE,
    dropout:float = 0.0,
) -> nn.Module:
    '''
    Vanilla ResNet18 classifier head on top of the ImageNet-pretrained backbone.

    num_classes  : number of output classes (e.g., "num_classes=2" for age young/adult)
    pretrained   : load pretrained weights from torchvision.models.ResNet18_Weights
    weights_path : optional path to fine-tuned weights, loaded after construction
    device       : device used to map loaded weights onto
    dropout      : if > 0, insert nn.Dropout(dropout) before the final Linear layer
                   (head becomes Sequential(Dropout, Linear); pass the same value
                   when reloading weights saved from a dropout head)
    '''
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model_ft = pt_resnet18(weights=weights)

    num_ftrs = model_ft.fc.in_features
    if dropout > 0:
        model_ft.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, num_classes))
    else:
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    if weights_path is not None:
        model_ft.load_state_dict(
            torch.load(weights_path, weights_only=True, map_location=torch.device(device))
        )
    return model_ft
