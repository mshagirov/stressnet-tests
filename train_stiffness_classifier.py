'''Train a 4-way stiffness-category classifier (adult_low/adult_high/
young_low/young_high) using AgeStiffnessDataset.

Hyperparameters are the best ones found for the age classifiers
(gradual_unfreeze_nolblsmooth_mx00_do00): Xv1 transforms, 10 frozen
backbone epochs then differential-LR fine-tune (head lr_head/10, backbone
lr_backbone), AdamW wd 1e-4, cosine annealing, patience 10, and NO label
smoothing / MixUp / dropout. The internal train/val split is stratified by
the 4 stiffness categories (stratify_col='Category') instead of age Group.

Reuses datasets, loaders, optimizer and plotting from the age-classifier v2
modules; only the prediction/plotting helpers and the report schema are
specialized for the 4 stiffness categories.

Usage (from inside stressnet-tests/):
    uv run python train_stiffness_classifier.py            # datav5, best config
    uv run python train_stiffness_classifier.py --debug    # fast smoke test
'''

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from classifier_trainer_stiffness import STIFFNESS_TO_IDX, train_classifier_stiffness
from classifiers import resnet18_classifier
from dataset import AgeStiffnessDataset
from models import TORCH_DEVICE
from train_age_classifier_v2 import (
    DATASET_CONFIGS,
    REPORT_COLUMNS,
    SAVE_ROOT,
    TRAIN_TRANSFORMS,
    build_datasets,
    build_loaders,
    build_optimizer,
    plot_history,
    set_backbone_trainable,
)

CATEGORY_ORDER = ['adult_low', 'adult_high', 'young_low', 'young_high']
STRATEGY_NAME = 'gradual_unfreeze_nolblsmooth_mx00_do00_catstrat'
STRATIFY_COL = 'Category'

# Best age-classifier config; no label smoothing / MixUp / dropout.
STRATEGY = {
    'transform': 'Xv1', 'dropout': 0.0, 'label_smoothing': 0.0, 'weight_decay': 1e-4,
    'lr_head': 1e-3, 'lr_backbone': 1e-5, 'freeze_epochs': 10, 'epochs': 40,
    'patience': 10, 'mixup_alpha': 0.0,
}

STIFFNESS_REPORT_COLUMNS = REPORT_COLUMNS + ['heldout_macro_acc']


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Train 4-way stiffness-category classifiers '
                    '(gradual_unfreeze, no ls/mixup/dropout)')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS),
                        default=None)
    parser.add_argument('--epochs', type=int, default=STRATEGY['epochs'])
    parser.add_argument('--freeze-epochs', type=int, default=STRATEGY['freeze_epochs'],
                        help='frozen-backbone epochs')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='fraction of Training_Data held out for internal validation')
    parser.add_argument('--report', type=Path, default=Path('training_report_stiffness.csv'))
    parser.add_argument('--debug', action='store_true',
                        help='smoke test: 1 dataset, 2 epochs, small subset')
    return parser.parse_args(args)


def predict_phase_stiffness(model: nn.Module, loader: DataLoader, device: torch.device,
                            desc: str = '') -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, categories, fnames in tqdm(loader, desc=desc):
            probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
            for fname, cat, p in zip(fnames, categories, probs):
                y_true = STIFFNESS_TO_IDX[cat]
                pred = int(p.argmax())
                rows.append({
                    'Image': fname, 'category': cat, 'y_true': y_true,
                    'p_adult_low': p[0], 'p_adult_high': p[1],
                    'p_young_low': p[2], 'p_young_high': p[3],
                    'p_pred': p[pred], 'pred': pred,
                    'correct': int(y_true == pred),
                })
    return pd.DataFrame(rows)


def plot_predictions_stiffness(preds_by_phase: dict, save_path: Path):
    fig, axs = plt.subplots(1, len(preds_by_phase),
                            figsize=[7*len(preds_by_phase), 6], squeeze=False)
    rng = np.random.default_rng(42)
    for ax, (phase, df) in zip(axs.ravel(), preds_by_phase.items()):
        jitter = rng.uniform(-0.04, 0.04, size=len(df))
        colors = np.where(df['correct'].astype(bool), 'tab:green', 'tab:red')
        ax.scatter(df['y_true'] + jitter, df['p_pred'], c=colors, alpha=.5, s=18)
        ax.axhline(0.25, ls='--', c='r', alpha=.4)
        acc = df['correct'].mean()
        ax.set_title(f'{phase} (acc={acc:.3f})')
        ax.set_xlim(-0.25, 3.25)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(4))
        ax.set_xticklabels(CATEGORY_ORDER, rotation=30)
        ax.set_ylabel('P(predicted class)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


def append_report_stiffness(row: dict, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = report_path.exists()
    with report_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=STIFFNESS_REPORT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Report row appended: {report_path}')


def run_training(data_key: str, s: dict, args) -> dict:
    model_name = f'resnet18_stiffness_classifier_{data_key}_{s["transform"]}_{STRATEGY_NAME}'
    if args.debug:
        model_name += '_debug'
    model_dir = SAVE_ROOT/model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {model_name} ===')
    datasets = build_datasets(DATASET_CONFIGS[data_key],
                              TRAIN_TRANSFORMS[s['transform']], args.val_frac,
                              dataset_cls=AgeStiffnessDataset,
                              stratify_col=STRATIFY_COL)
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

    model = resnet18_classifier(num_classes=4, pretrained=True, dropout=s['dropout'])
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
    model, l1, a1, best_loss = train_classifier_stiffness(
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
    model, l2, a2, best_loss = train_classifier_stiffness(
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
    # is scored only here, after training is fully done.
    model = model.to(TORCH_DEVICE).eval()
    preds_by_phase = {}
    for phase, key in (('train', 'train_infer'), ('heldout', 'heldout')):
        pred_loader = DataLoader(datasets[key], batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)
        preds_df = predict_phase_stiffness(model, pred_loader, TORCH_DEVICE, desc=phase)
        preds_df.to_csv(model_dir/f'{model_name}_{phase}_preds.csv', index=False)
        preds_by_phase[phase] = preds_df
    plot_predictions_stiffness(preds_by_phase, model_dir/f'{model_name}_preds.png')

    row = {
        'model_name': model_name,
        'dataset': data_key,
        'strategy': STRATEGY_NAME,
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
    heldout = preds_by_phase['heldout']
    row['heldout_macro_acc'] = heldout.groupby('category')['correct'].mean().mean()
    append_report_stiffness(row, args.report)
    return row


def main():
    args = parse_args(sys.argv[1:])
    torch.manual_seed(42)
    np.random.seed(42)
    print(f'Using device: {TORCH_DEVICE}')

    s = dict(STRATEGY)
    s['epochs'] = args.epochs
    s['freeze_epochs'] = args.freeze_epochs

    if args.debug:
        datasets = args.datasets or list(DATASET_CONFIGS)[:1]
    else:
        datasets = args.datasets or ['datav5']

    print(f'Strategy: {STRATEGY_NAME}: {s}')
    results = []
    for data_key in datasets:
        results.append(run_training(data_key, dict(s), args))

    print('\n=== Summary ===')
    print(pd.DataFrame(results)[STIFFNESS_REPORT_COLUMNS].to_string(index=False))


if __name__ == '__main__':
    main()