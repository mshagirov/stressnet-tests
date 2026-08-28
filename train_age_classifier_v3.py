'''Train age classifiers (young vs adult) — v3: staged gradual-unfreeze ladder.

Explores simplification/mixing around the best v2 strategy
(gradual_unfreeze: 10 frozen epochs, differential LR, AdamW), keeping every
other hyperparameter fixed. The experiment ladder:

    stage 1 : gradual_unfreeze with vs without label smoothing
              (10 frozen epochs, 30 total)
    stage 2 : stage-1 winner + MixUp alpha in {0, 0.1, 0.2}
              (10 frozen epochs, 40 total)
    stage 3 : stage-2 winner + dropout p in {0, 0.2, 0.5}

Each stage is a separate CLI invocation — no code changes between stages:

    uv run python train_age_classifier_v3.py                  # stage 1
    uv run python train_age_classifier_v3.py --ls <best> \\
        --mixup 0 0.1 0.2 --epochs 40                         # stage 2
    uv run python train_age_classifier_v3.py --ls <best> \\
        --mixup <best> --dropouts 0 0.2 0.5                   # stage 3

Run dirs follow the usual convention
`<arch>_datav<N>_Xv<1|4>_<strategy>`; strategy names carry explicit tags for
every swept knob (`_mx00`, `_do02`, ...) so stages never collide.

Differences from v2: only training-set and held-out (Prediction_Data)
predictions are saved as CSVs/plots (internal-val predictions are computed
for training/early stopping but not written out); trainer, datasets and
transforms are reused unchanged from the v2 modules.
'''

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from classifier_trainer_v2 import train_classifier_v2
from classifiers import resnet18_classifier
from dataset import AgeDataset, AgeDatasetNoNucleus
from models import TORCH_DEVICE
from train_age_classifier_v2 import (
    DATASET_CONFIGS,
    REPORT_COLUMNS,
    SAVE_ROOT,
    TRAIN_TRANSFORMS,
    append_report,
    build_datasets,
    build_loaders,
    build_optimizer,
    plot_history,
    plot_predictions,
    predict_phase,
    set_backbone_trainable,
)

FREEZE_EPOCHS = 10


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Train age classifiers, v3 staged gradual-unfreeze ladder')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS),
                        default=None)
    parser.add_argument('--ls', nargs='+', type=float, default=None,
                        help='label-smoothing values to sweep '
                             '(default: 0.1 0.0 = stage 1)')
    parser.add_argument('--mixup', nargs='+', type=float, default=None,
                        help='MixUp alpha values to sweep (default: [0.0])')
    parser.add_argument('--dropouts', nargs='+', type=float, default=None,
                        help='dropout p values to sweep (default: [0.5])')
    parser.add_argument('--epochs', type=int, default=30,
                        help='total epochs (stage 2 uses 40)')
    parser.add_argument('--freeze-epochs', type=int, default=None,
                        help='frozen-backbone epochs (default: 10); '
                             'run name gets a _fz<NN> tag')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='fraction of Training_Data held out for internal validation')
    parser.add_argument('--report', type=Path, default=Path('training_report_v3.csv'))
    parser.add_argument('--no-nucleus', action='store_true',
                        help='zero out the Nucleus channel (AgeDatasetNoNucleus); '
                             'run name gets a _nonuc tag')
    parser.add_argument('--debug', action='store_true',
                        help='smoke test: 1 dataset, 2 epochs, small subset')
    return parser.parse_args(args)


def build_strategies(args) -> dict:
    '''Compose the strategy grid from CLI sweeps; names encode swept knobs.'''
    ls_values = [0.1, 0.0] if args.ls is None else args.ls
    mixup_values = [0.0] if args.mixup is None else args.mixup
    dropout_values = [0.5] if args.dropouts is None else args.dropouts
    strategies = {}
    for ls in ls_values:
        for mixup_alpha in mixup_values:
            for dropout in dropout_values:
                lbl = 'wlblsmooth' if ls > 0 else 'nolblsmooth'
                name = f'gradual_unfreeze_{lbl}'
                if args.mixup is not None:
                    name += f'_mx{round(mixup_alpha*100):02d}'
                if args.dropouts is not None:
                    name += f'_do{round(dropout*100):02d}'
                freeze_epochs = FREEZE_EPOCHS if args.freeze_epochs is None \
                    else args.freeze_epochs
                if args.freeze_epochs is not None:
                    name += f'_fz{freeze_epochs:02d}'
                strategies[name] = {
                    'transform': 'Xv1', 'dropout': dropout,
                    'label_smoothing': ls, 'weight_decay': 1e-4,
                    'lr_head': 1e-3, 'lr_backbone': 1e-5,
                    'freeze_epochs': freeze_epochs, 'epochs': args.epochs,
                    'patience': 10, 'mixup_alpha': mixup_alpha,
                }
    return strategies


def run_training(data_key: str, strategy_name: str, s: dict, args) -> dict:
    model_name = f'resnet18_classifier_{data_key}_{s["transform"]}_{strategy_name}'
    if args.no_nucleus:
        model_name += '_nonuc'
    if args.debug:
        model_name += '_debug'
    model_dir = SAVE_ROOT/model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {model_name} ===')
    dataset_cls = AgeDatasetNoNucleus if args.no_nucleus else AgeDataset
    datasets = build_datasets(DATASET_CONFIGS[data_key],
                              TRAIN_TRANSFORMS[s['transform']], args.val_frac,
                              dataset_cls=dataset_cls)
    if args.debug:
        datasets['train'] = Subset(datasets['train'].dataset,
                                   datasets['train'].indices[:64])
        datasets['val'] = Subset(datasets['val'].dataset,
                                 datasets['val'].indices[:32])
        datasets['train_infer'] = datasets['val']
        s['epochs'] = 2
        s['freeze_epochs'] = 1
    dataloaders = build_loaders(datasets, args.batch_size)
    dataset_sizes = {phase: len(datasets[phase]) for phase in ('train', 'val')}

    model = resnet18_classifier(num_classes=2, pretrained=True, dropout=s['dropout'])
    criterion = nn.CrossEntropyLoss(label_smoothing=s['label_smoothing'])
    weights_file = f'{model_name}_{s["epochs"]}epoch_AdamW.pt'
    checkpoint_path = model_dir/weights_file

    since = time.time()
    losses, accs = {'train': [], 'val': []}, {'train': [], 'val': []}
    stage_boundary = None

    # Stage 1: frozen backbone, head-only training.
    freeze_epochs = min(s['freeze_epochs'], s['epochs'] - 1)
    print(f'-- Stage 1: frozen backbone, {freeze_epochs} epochs, '
          f'head lr={s["lr_head"]} --')
    set_backbone_trainable(model, False)
    optimizer = build_optimizer(model, s, backbone_trainable=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=freeze_epochs)
    model, l1, a1, best_loss = train_classifier_v2(
        model, criterion, optimizer, scheduler,
        dataloaders=dataloaders, dataset_sizes=dataset_sizes,
        num_epochs=freeze_epochs, device=TORCH_DEVICE,
        checkpoint_path=checkpoint_path,
        early_stopping_patience=None,
        mixup_alpha=s['mixup_alpha'],
        load_best_at_end=False,
    )
    for phase in losses:
        losses[phase] += l1[phase]
        accs[phase] += a1[phase]

    # Stage 2: unfreeze, differential LR (head lr/10).
    stage2_epochs = s['epochs'] - freeze_epochs
    stage_boundary = freeze_epochs
    lr_head = s['lr_head']/10
    print(f'-- Stage 2: unfrozen, {stage2_epochs} epochs, '
          f'head lr={lr_head}, backbone lr={s["lr_backbone"]} --')
    set_backbone_trainable(model, True)
    optimizer = build_optimizer(model, s, backbone_trainable=True,
                                lr_head=lr_head)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, stage2_epochs))
    model, l2, a2, best_loss = train_classifier_v2(
        model, criterion, optimizer, scheduler,
        dataloaders=dataloaders, dataset_sizes=dataset_sizes,
        num_epochs=stage2_epochs, device=TORCH_DEVICE,
        checkpoint_path=checkpoint_path,
        initial_best_loss=best_loss,
        early_stopping_patience=s['patience'],
        mixup_alpha=s['mixup_alpha'],
        load_best_at_end=True,
    )
    for phase in losses:
        losses[phase] += l2[phase]
        accs[phase] += a2[phase]

    elapsed = time.time() - since
    stopped_epoch = len(losses['train'])
    print(f'Best-val weights: {checkpoint_path}')

    plot_history(losses, accs, model_dir/f'{model_name}_loss.png',
                 stage_boundary=stage_boundary)

    # Final predictions with inference transforms; Prediction_Data (heldout)
    # is scored only here, after training is fully done. Only train + heldout
    # artifacts are written (per v3 spec).
    model = model.to(TORCH_DEVICE).eval()
    preds_by_phase = {}
    for phase, key in (('train', 'train_infer'), ('heldout', 'heldout')):
        pred_loader = DataLoader(datasets[key], batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        preds_df = predict_phase(model, pred_loader, TORCH_DEVICE, desc=phase)
        preds_df.to_csv(model_dir/f'{model_name}_{phase}_preds.csv', index=False)
        preds_by_phase[phase] = preds_df
    plot_predictions(preds_by_phase, model_dir/f'{model_name}_preds.png')

    row = {
        'model_name': model_name,
        'dataset': data_key,
        'strategy': strategy_name,
        'transform': s['transform'],
        'epochs': s['epochs'],
        'batch_size': args.batch_size,
        'lr_head': s['lr_head'],
        'lr_backbone': s['lr_backbone'],
        'weight_decay': s['weight_decay'],
        'label_smoothing': s['label_smoothing'],
        'dropout': s['dropout'],
        'mixup_alpha': s['mixup_alpha'],
        'best_val_loss': min(losses['val']),
        'final_train_loss': losses['train'][-1],
        'final_val_loss': losses['val'][-1],
        'final_train_acc': accs['train'][-1],
        'final_val_acc': accs['val'][-1],
        'best_train_acc': max(accs['train']),
        'best_val_acc': max(accs['val']),
        'heldout_acc': preds_by_phase['heldout']['correct'].mean(),
        'stopped_epoch': stopped_epoch,
        'train_time_s': round(elapsed, 1),
        'weights_file': weights_file,
    }
    append_report(row, args.report)
    return row


def main():
    args = parse_args(sys.argv[1:])
    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Using device: {TORCH_DEVICE}')

    strategies = build_strategies(args)
    print('Strategies:')
    for name, s in strategies.items():
        print(f'  {name}: {s}')

    if args.debug:
        datasets = args.datasets or list(DATASET_CONFIGS)[:1]
    else:
        datasets = args.datasets or list(DATASET_CONFIGS)

    results = []
    for data_key in datasets:
        for strategy_name, s in strategies.items():
            results.append(run_training(data_key, strategy_name, s, args))

    print('\n=== Summary ===')
    print(pd.DataFrame(results)[REPORT_COLUMNS].to_string(index=False))


if __name__ == '__main__':
    main()
