from torchvision import transforms
import math

im_dim = 256

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

data_transforms_inference = data_transforms['val']

# transforms for y_tgt
class TargetNormalise:
    '''Transform input `x`: normalise(x) = (x - offset)/scale

    Usage:
        stiffness_TF = TargetNormalise(4.5, 15.0)
        y_norm = stiffness_TF(y)
    '''
    def __init__(self, offset=0, scale=1):
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
