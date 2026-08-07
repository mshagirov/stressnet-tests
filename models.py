from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models import resnet18 as pt_resnet18


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

def resnet18(weights_path:str|Path, device:torch.device = TORCH_DEVICE) -> nn.Module:
    '''
    The last FC-layers consist of torch.nn.nn.Linear Module 
    '''
    
    model_ft = pt_resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)
    
    model_ft.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=torch.device(device))
    )
    return model_ft

def resnet18_seq(weights_path:str|Path, device:torch.device = TORCH_DEVICE) -> nn.Module:
    '''
    The last FC-layers consist of torch.nn.Sequential Module from fc_layers() function above
    '''
    
    model_ft = pt_resnet18()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = fc_layers(num_ftrs, hidden_layers=[])

    model_ft.load_state_dict(
        torch.load(weights_path, weights_only=True, map_location=torch.device(device))
    )
    return model_ft


class ResNet18WithAgeLoc(nn.Module):
    '''
    Usage:

    ```
    batch_size = 4
    x_images = torch.randn(batch_size, 3, 224, 224)  # (B, 3, H, W)
    x_age_loc = torch.randn(batch_size, 2)            # (B, 2)
    
    model = ResNet18WithAgeLoc(num_classes=1, pretrained=True)
    output = model(x_images, x_age_loc)
    ```
    '''

    def __init__(
        self,
        num_classes: int = 1,
        input_dim_mlp:int = 2,
        hidden_dim_mlp: int = 32,
        embed_dim_mlp: int = 8,
        pretrained: bool = True
    ):
        '''
        num_classes : num dim-s of the model's output (e.g., "num_classes=1" for model(x) -> Stiffness (scalar) )
        input_dim_mlp: num dim-s MLP input layer,
        hidden_dim_mlp : num dim-s hidden MLP hidden layer
        embed_dim_mlp : num dim-s of MLP output
        pretrained: load pretrained wights from torchvision.models.ResNet18_Weights
        '''

        super(ResNet18WithAgeLoc, self).__init__()
        
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.cnn = pt_resnet18(weights=weights)
        
        # original FC input
        cnn_feature_dim = self.cnn.fc.in_features
        # Remove original FC layer
        self.cnn.fc = nn.Identity()
        
        # Age_Location MLP Branch (2-Layer)
        self.mlp = nn.Sequential(
            # Layer 1
            nn.Linear(in_features=input_dim_mlp, out_features=hidden_dim_mlp),
            nn.BatchNorm1d(hidden_dim_mlp),
            nn.ReLU(inplace=True),
            
            # Layer 2 (Output size = 8 neurons)
            nn.Linear(in_features=hidden_dim_mlp, out_features=embed_dim_mlp),
            nn.BatchNorm1d(embed_dim_mlp),
            nn.ReLU(inplace=True)
        )
        
        # output layer
        combined_dim = cnn_feature_dim + embed_dim_mlp  # CNN_features + MLP_output 
        self.fc_head = nn.Linear(in_features=combined_dim, out_features=num_classes)

    def forward(self, x_img: torch.Tensor, x_ageloc: torch.Tensor) -> torch.Tensor:
        """
        x_img:    Image tensor of shape (Batch_Size, 3, H, W)
        x_ageloc: Scalar tensor of shape (Batch_Size, 2)
        """
        img_features = self.cnn(x_img)
        
        ageloc_features = self.mlp(x_ageloc)
        
        combined_features = torch.cat((img_features, ageloc_features), dim=1)

        out = self.fc_head(combined_features)
        return out
