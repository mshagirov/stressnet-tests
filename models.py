from torch import nn

def fc_layers(in_features, out_features=1, hidden_layers=[], p=0):
    '''
    in_features : number of input features to the fully-connected layers
    hidden_layers: list of int that represent number of neurons in hiddenr
    layers
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


