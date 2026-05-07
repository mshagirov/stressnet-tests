from pathlib import Path
import torch
from torch.utils.data import DataLoader

from models import resnet18
from dataset import ValidationDataset
from transforms import data_transforms_inference

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using {device} device")

    # Weights
    model_name = "resnet18_2_v3_fc_"
    MODEL_ROOT = Path('../faris_cnn/saved_models') / model_name
    WEIGHTS_PATH = MODEL_ROOT / (model_name + "_30epochWarmup.pt")
    if not WEIGHTS_PATH.exists():
        raise ValueError(f"Can't find:\n  {WEIGHTS_PATH}")

    model_ft = resnet18(WEIGHTS_PATH)
    model_ft.eval();

    # input shape: [N_EXAMPLES, 3, 224, 224]

    if not model_ft.training:
        print(f'Model ({model_name}) is in evaluation mode')

    batch_size = 2
    dataloaders = {}

    TRAIN_ROOT = Path("../faris_cnn/Faris_Data_for_ML_v3/Training_Data")
    TRAIN_LABELS = "Stiffness_Data_Training.xlsx"
    dataloaders["train"] = DataLoader(
            ValidationDataset(TRAIN_LABELS, TRAIN_ROOT, ch_dir_suffix="_Train",
                              transform=data_transforms_inference,
                              target_transform=None),
            batch_size=batch_size, shuffle=False, num_workers=2)
    
    VAL_ROOT   = Path("../faris_cnn/Faris_Data_for_ML_v3/Prediction_Data")
    VAL_LABELS = "Stiffness_Data_Prediction.xlsx"
    dataloaders["val"] = DataLoader(
            ValidationDataset(VAL_LABELS, VAL_ROOT, ch_dir_suffix="_Prediction",
                              transform=data_transforms_inference,
                              target_transform=None), 
            batch_size=batch_size, shuffle=False, num_workers=1)


    for phase in ("train", "val"):
        loader = dataloaders[phase]
        for x, y, fname in loader:
            print(x.size())
            print(y)
            print(fname)
            break


if __name__ == "__main__":
    main()
