import math

import torch
from torchvision import transforms

im_dim = 256
im_dim_v2 = 96
# v5.2-gridaveraged images are 1376x1376; largest square fully valid after
# 360-degree rotation: floor(1376/sqrt(2)) = 972
im_dim_v4 = 972

# STIFMaps examples:
# brightness_range = (.9,1.1), 
# contrast_range = (.5,1.5), 
# sharpness_range = (1.5,.67),
#
#   transforms.ColorJitter(brightness=brightness_range, contrast=contrast_range),
#   transforms.RandomAdjustSharpness(sharpness_range[0], p=.5),
#   transforms.RandomAdjustSharpness(sharpness_range[1], p=.5),

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=360),
        transforms.CenterCrop(im_dim),
        transforms.Resize(224),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(im_dim),
        transforms.Resize(224),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

data_transforms_v2 = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=360),
        transforms.CenterCrop(im_dim_v2),
        transforms.Resize(224),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(im_dim_v2),
        transforms.Resize(224),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

data_transforms_v3 = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=360),
        transforms.CenterCrop(im_dim_v2),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(im_dim_v2),
    ]),
}

class AddGaussianNoise:
    '''Add Gaussian noise to a float image tensor with probability `p`.

    Usage:
        noise_TF = AddGaussianNoise(std=0.02, p=0.5)
        x_noisy = noise_TF(x)
    '''
    def __init__(self, std:float=0.02, p:float=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(()) > self.p:
            return x
        return (x + torch.randn_like(x)*self.std).clamp(0, 1)

    def __repr__(self):
        return f'{self.__class__.__name__}(std={self.std}, p={self.p})'

# Xv4: stronger augmentation for small training sets (age classifier v2 runs)
data_transforms_v4 = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=360),
        transforms.RandomResizedCrop(im_dim_v4, scale=(0.6, 1.0)),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        AddGaussianNoise(std=0.02, p=0.5),
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop(im_dim_v4),
        transforms.Resize(224),
    ]),
}

data_transforms_inference = data_transforms['val']
data_transforms_inference_v2 = data_transforms_v2['val']
data_transforms_inference_v3 = data_transforms_v3['val']
data_transforms_inference_v4 = data_transforms_v4['val']

# transforms for y_tgt
class TargetNormalise:
    '''Transform input `x`: normalise(x) = (x - offset)/scale

    Usage:
        stiffness_TF = TargetNormalise(4.5, 15.0)
        y_norm = stiffness_TF(y)
    '''
    def __init__(self, offset:float=0, scale:float=1):
        self.offset = offset
        self.scale = scale
    
    def __call__(self, x):
        return (x - self.offset)/self.scale

    def __repr__(self):
        offset, scale = self.offset, self.scale
        return f'{self.__class__.__name__}({offset=}, {scale=})'
    
class TargetLog:
    '''Transform input `x` : normalise(x) = math.log(x)

    Usage:
        stiffness_TF = TargetLog()
        y_norm = stiffness_TF(y)
    ''' 
    def __call__(self, x):
        return math.log(x)
    def __repr__(self):
        return f'{self.__class__.__name__}()'

stiffness_transform = transforms.Compose([
    TargetLog(),
    TargetNormalise(offset=1.85, scale=3.0)
])

stiffness_norm_v1 = transforms.Compose([
    TargetLog(),
    TargetNormalise(offset=1.85, scale=3.0)
])

stiffness_norm_v2 = transforms.Compose([
    TargetNormalise(offset=0.0, scale=25.0)
])

stiffness_norm_v3 = transforms.Compose([
    TargetLog(),
    TargetNormalise(offset=1.5, scale=1.0)
])
