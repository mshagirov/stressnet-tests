from pathlib import Path

import torch
from torchvision import models
from torch import nn


if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
else:
    TORCH_DEVICE = torch.device("cpu")

def get_predict_func(m:nn.Module, device:torch.device=TORCH_DEVICE):
    @torch.inference_mode()
    def predict(X_in:torch.Tensor):
        assert torch.is_inference_mode_enabled()
        return m(X_in.to(device))

    return predict


def fc_layers(in_features:int, out_features=1, hidden_layers=[], p=0) -> nn.Module:
    '''
    Generates and returns a torch.nn.Sequential Module (Neural Net) with fully-connected layers
    
    in_features  : number of input features to the fully-connected layers
    hidden_layers: list of int that represent number of neurons in hidden
                   layers

    For a single fully-connected layer use Linear(in_features, out_features) from torch.nn, e.g.:

        fc = nn.Linear(in_features, out_features)
    '''
    fc_net = []
    max_h_idx = len(hidden_layers) - 1 

    for k, (num_in, num_out) in enumerate( zip( [in_features] + hidden_layers[:-1], hidden_layers)):
        fc_net.append(nn.Linear(num_in, num_out))
        fc_net.append(nn.ReLU(inplace=True))
        if (p > 0) and (k < max_h_idx):
            # no dropout before and after hidden layers
            fc_net.append(nn.Dropout(p))
    
    if hidden_layers:
        fc_net.append( nn.Linear( hidden_layers[-1], out_features))
    else:
        fc_net.append( nn.Linear( in_features, out_features))
    return nn.Sequential(*fc_net)

def resnet18(weights:str|Path, device:torch.device = TORCH_DEVICE) -> nn.Module:
    '''
    The last FC-layers consist of torch.nn.nn.Linear Module 
    '''
    
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    
    model_ft.load_state_dict(
        torch.load(weights, weights_only=True, map_location=torch.device(device))
    )
    return model_ft

def resnet18_seq(weights:str|Path, device:torch.device = TORCH_DEVICE) -> nn.Module:
    '''
    The last FC-layers consist of torch.nn.Sequential Module from fc_layers() function above
    '''
    
    model_ft = models.resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = fc_layers(num_ftrs, hidden_layers=[])

    model_ft.load_state_dict(
        torch.load(weights, weights_only=True, map_location=torch.device(device))
    )
    return model_ft

