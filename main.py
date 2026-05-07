from pathlib import Path
import torch

from models import resnet18

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




if __name__ == "__main__":
    main()
