import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models import resnet18, get_predict_func
from dataset import ValidationDataset
from transforms import data_transforms_inference, stiffness_transform
from plots import plot_scatter


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <save_dir>")
        sys.exit(1)
    save_dir = Path(sys.argv[1])

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")

    # Weights
    model_name = 'resnet18_2_v3_fc_'
    MODEL_ROOT = Path('../faris_cnn/saved_models') / model_name
    WEIGHTS_PATH = MODEL_ROOT / (model_name + '_1000epochFinetune_Adam.pt')
    if not WEIGHTS_PATH.exists():
        raise ValueError(f"Can't find:\n  {WEIGHTS_PATH}")

    model_ft = resnet18(WEIGHTS_PATH)
    model_ft.eval();

    if not model_ft.training:
        print(f'Model ({model_name}) is in evaluation mode')

    batch_size = 2
    dataloaders = {}

    TRAIN_ROOT = Path("../faris_cnn/Faris_Data_for_ML_v3/Training_Data")
    TRAIN_LABELS = "Stiffness_Data_Training.xlsx"
    dataloaders["train"] = DataLoader(
            ValidationDataset(TRAIN_LABELS, TRAIN_ROOT, ch_dir_suffix="_Train",
                              transform=data_transforms_inference,
                              target_transform=stiffness_transform),
            batch_size=batch_size, shuffle=False, num_workers=2)
    
    VAL_ROOT   = Path("../faris_cnn/Faris_Data_for_ML_v3/Prediction_Data")
    VAL_LABELS = "Stiffness_Data_Prediction.xlsx"
    dataloaders["val"] = DataLoader(
            ValidationDataset(VAL_LABELS, VAL_ROOT, ch_dir_suffix="_Prediction",
                              transform=data_transforms_inference,
                              target_transform=stiffness_transform), 
            batch_size=batch_size, shuffle=False, num_workers=1) 
    
    predict = get_predict_func(model_ft, device=torch.device(device))

    for phase in ("train", "val"):
        print(f"{phase=}")
        loader = dataloaders[phase]

        y_tgt = []
        y_pred = []
        x_names = []
        
        for x, y, fnames in loader:
            x_names.extend(fnames)
            y_tgt.append(y.cpu().numpy())
            y_pred.append(predict(x).cpu().numpy())

        plot_scatter(y_tgt, y_pred, x_names, phase, save_dir/model_name)

if __name__ == "__main__":
    main()
