import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from models import TORCH_DEVICE

AGE_TO_IDX = {'young': 0, 'adult': 1}


def train_classifier(
    model: torch.nn.Module,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    return_best_val=False,
    device=TORCH_DEVICE,
):
    '''Classifier version of trainer.train_model.

    Expects dataloaders built over AgeDataset, whose batches collate to
    (images, age_strings, filenames); age strings are mapped through
    AGE_TO_IDX ('young'=0, 'adult'=1) into class-index labels.

    Returns (model, losses, accs) where losses/accs map phase -> per-epoch history.
    '''
    model = model.to(device)
    since = time.time()

    phases = list(dataset_sizes.keys())
    losses = {k: [] for k in phases}
    accs = {k: [] for k in phases}

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = Path(tempdir)/'best_model_params.pt'

        torch.save(model.state_dict(), best_model_params_path)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'Epoch {epoch:0>3}/{num_epochs - 1:0>3}; ', end='')

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for images, ages, _fnames in dataloaders[phase]:
                    batch_size = images.size(0)
                    labels = torch.tensor([AGE_TO_IDX[a] for a in ages], dtype=torch.long)
                    images = images.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * batch_size
                    running_corrects += (preds == labels).sum().item()
                if (phase == 'train') and scheduler is not None:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}; ', end='')
                losses[phase].append(epoch_loss)
                accs[phase].append(epoch_acc)

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if return_best_val:
                        torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print('-' * 10)
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val loss: {best_loss:4f}')

        if return_best_val:
            # load best model weights
            model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model, losses, accs
