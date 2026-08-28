'''Categorical-stiffness classifier trainer.

Adaptation of classifier_trainer_v2.train_classifier_v2 for the 4-way
stiffness categories of AgeStiffnessDataset ('adult_low', 'adult_high',
'young_low', 'young_high'). Keeps the same features:
  - MixUp augmentation (train phase only),
  - early stopping on val loss (patience),
  - best-val weights written to a persistent checkpoint_path (survives across
    calls, enabling two-stage progressive-unfreeze training),
  - optional skipping of the final best-weights reload (load_best_at_end=False)
    so a second training stage can continue from the last weights.

Label smoothing / weight decay are handled by the criterion / optimizer the
caller passes in, e.g. nn.CrossEntropyLoss(label_smoothing=0.1) and
torch.optim.AdamW(..., weight_decay=1e-4).
'''

import time
from pathlib import Path

import numpy as np
import torch

from models import TORCH_DEVICE

STIFFNESS_TO_IDX = {
    'adult_low': 0,
    'adult_high': 1,
    'young_low': 2,
    'young_high': 3,
}


def train_classifier_stiffness(
    model: torch.nn.Module,
    criterion,
    optimizer,
    scheduler,
    dataloaders,
    dataset_sizes,
    num_epochs=25,
    device=TORCH_DEVICE,
    checkpoint_path=None,
    initial_best_loss=float('inf'),
    early_stopping_patience=None,
    mixup_alpha=0.0,
    load_best_at_end=True,
):
    '''Train a 4-way stiffness-category classifier.

    Expects dataloaders built over AgeStiffnessDataset, whose batches collate
    to (images, category_strings, filenames); category strings are mapped
    through STIFFNESS_TO_IDX ('adult_low'=0, 'adult_high'=1, 'young_low'=2,
    'young_high'=3) into class-index labels.

    checkpoint_path         : file where best-val state_dict is saved (persistent;
                              pass the same path across training stages)
    initial_best_loss       : best val loss from a previous stage (default inf)
    early_stopping_patience : stop after this many epochs without val improvement
                              (None disables early stopping)
    mixup_alpha             : Beta(alpha, alpha) mixing coefficient for train-phase
                              MixUp; 0 disables
    load_best_at_end        : reload best-val weights into the model before returning
                              (set False for intermediate stages)

    Returns (model, losses, accs, best_loss) where losses/accs map
    phase -> per-epoch history.
    '''
    model = model.to(device)
    since = time.time()

    phases = list(dataset_sizes.keys())
    losses = {k: [] for k in phases}
    accs = {k: [] for k in phases}

    best_loss = initial_best_loss
    epochs_without_improvement = 0
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

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
            for images, categories, _fnames in dataloaders[phase]:
                batch_size = images.size(0)
                labels = torch.tensor([STIFFNESS_TO_IDX[c] for c in categories],
                                      dtype=torch.long)
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train' and mixup_alpha > 0:
                        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                        perm = torch.randperm(batch_size, device=device)
                        images = lam*images + (1.0 - lam)*images[perm]
                        outputs = model(images)
                        loss = lam*criterion(outputs, labels) \
                            + (1.0 - lam)*criterion(outputs, labels[perm])
                    else:
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

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

            # save best model weights to the persistent checkpoint
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    epochs_without_improvement = 0
                    if checkpoint_path is not None:
                        torch.save(model.state_dict(), checkpoint_path)
                else:
                    epochs_without_improvement += 1

        print()

        if (early_stopping_patience is not None
                and epochs_without_improvement > early_stopping_patience):
            print(f'Early stopping: no val improvement for '
                  f'{epochs_without_improvement} epochs (patience {early_stopping_patience})')
            break

    time_elapsed = time.time() - since
    print('-' * 10)
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')

    if load_best_at_end and checkpoint_path is not None and Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    return model, losses, accs, best_loss