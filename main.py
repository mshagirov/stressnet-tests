import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import ValidationDatasetAgeLoc
from models import get_predict_func, resnet18
from plots import plot_scatter
from transforms import data_transforms_inference, stiffness_transform


def parse_args(args):
    parser = argparse.ArgumentParser( prog='StressNet Predict', description='Plots StressNet Predictions')

    parser.add_argument('-o', '--output', help='directory path for output files', required=True)
    parser.add_argument('-w', '--weights', help='path to the model weights',required=True)
    parser.add_argument('-i', '--input', help='root directory for the input dataset', required=True)
    parser.add_argument('-l', '--labels', help='name of the spreadsheet file with the labels inside the root directory',required=True)
    parser.add_argument('-c', '--chsuffix', help='Channel directory suffix (for dir-s inside the root dir-y)',required=True)
    parser.add_argument('-p', '--phase', help='Training or validation phase', choices=['train', 'val'],required=True)
    
    return parser.parse_args(args)

def main():
    args = parse_args(sys.argv[1:])

    save_dir = Path(args.output)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")

    # Weights
    WEIGHTS_PATH = Path(args.weights) #MODEL_ROOT / (model_name + "_30epochWarmup.pt")
    model_name = WEIGHTS_PATH.name

    if not WEIGHTS_PATH.exists():
        raise ValueError(f"Can't find:\n  {WEIGHTS_PATH}")

    model_ft = resnet18(WEIGHTS_PATH).to(device)
    model_ft.eval();

    if not model_ft.training:
        print(f'Model ({model_name}) is in evaluation mode')

    batch_size = 2

    DATA_ROOT = Path(args.input)#"../faris_cnn/Faris_Data_for_ML_v3/Training_Data")
    DATA_LABELS = args.labels
    DATA_CH_SUFFIX = args.chsuffix
    loader = DataLoader(
        ValidationDatasetAgeLoc(DATA_LABELS, DATA_ROOT, ch_dir_suffix=DATA_CH_SUFFIX,
                          transform=data_transforms_inference,
                          target_transform=stiffness_transform),
        batch_size=batch_size, shuffle=False, num_workers=2)

    predict = get_predict_func(model_ft, device=torch.device(device))

    y_tgt = []
    y_pred = []
    x_names = []

    for x, y, fnames in loader:
        x_names.extend(fnames)
        y_tgt.append(y.cpu().numpy())
        y_pred.append(predict(x).cpu().numpy())

    y_tgt = np.concatenate(y_tgt).ravel()
    y_pred = np.concatenate(y_pred).ravel()

    plot_scatter(y_tgt, y_pred, x_names, args.phase, save_dir/model_name)

    df = pd.DataFrame({"Image": x_names, "Ytgt": y_tgt, "Ypred": y_pred })
    df.to_csv(save_dir/model_name/f'{args.phase}_preds.csv', index=False)

if __name__ == "__main__":
    main()
