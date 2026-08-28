'''Train regularized age classifiers (young vs adult) — v2.

Targets the overfitting seen in the train_age_classifier.py runs
(~0.99 train acc vs ~0.50 val acc). Five causes, five fixes:

1. No regularization       -> AdamW weight decay, dropout before the FC head,
                              label smoothing (criterion-level).
2. No LR scheduling        -> cosine annealing; differential LRs for the
                              pretrained backbone vs the new head.
3. Train/val domain shift  -> the val phase now uses an internal stratified
                              80/20 split of Training_Data; Prediction_Data is
                              NEVER trained on and is scored only once, after
                              training, as a held-out test set.
4. Weak augmentation       -> Xv4 transforms (random-resized crop, elastic,
                              brightness/contrast jitter, Gaussian noise) + MixUp.
5. Pretrained features destroyed early -> frozen-backbone and gradual-unfreeze
                              strategies.

Strategies (all ImageNet-pretrained, dropout head, label smoothing, AdamW):
    frozen_pretrained : backbone frozen for the whole run; only the head trains.
    gradual_unfreeze  : head-only for freeze_epochs, then full model with
                        differential LR (head lr_head/10, backbone lr_backbone).
    regularized       : full fine-tune with differential LR from epoch 0.
    mixup_augmented   : regularized + Xv4 strong transforms + MixUp.

Usage (from inside stressnet-tests/):
    uv run python train_age_classifier_v2.py                    # all 8 runs
    uv run python train_age_classifier_v2.py --debug            # fast smoke test
    uv run python train_age_classifier_v2.py --datasets datav5 \
        --strategies regularized mixup_augmented
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

from classifier_trainer_v2 import train_classifier_v2
from classifiers import resnet18_classifier
from dataset import AgeDataset
from models import TORCH_DEVICE
from transforms import data_transforms, data_transforms_inference, data_transforms_v4

SAVE_ROOT = Path('/mnt/data/faris-data/saved_models')
DATA_ROOT = Path('/mnt/data/faris-data/datasets')

DATASET_CONFIGS = {
    'datav5': {
        'root': DATA_ROOT/'Faris_Data_for_ML_v5',
        'ch_names': ('Nucleus', 'Collagen'),
        'ch_name_prefix': ('C1-', 'C2-'),
    },
    'datav6': {
        'root': DATA_ROOT/'Faris_Data_for_ML_v6_Col+Nuc+Actin',
        'ch_names': ('Nucleus', 'Collagen', 'Actin'),
        'ch_name_prefix': ('C1-', 'C2-', 'C3-'),
    },
}
TRAIN_DIR = ('Training_Data', '_Train', 'Stiffness_Data_Training.xlsx')
HELDOUT_DIR = ('Prediction_Data', '_Prediction', 'Stiffness_Data_Prediction.xlsx')

TRAIN_TRANSFORMS = {'Xv1': data_transforms['train'], 'Xv4': data_transforms_v4['train']}

# Strategy hyperparameters. lr_backbone=0.0 means "keep backbone frozen".
STRATEGIES = {
    'frozen_pretrained': {
        'transform': 'Xv1', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-3, 'lr_backbone': 0.0, 'freeze_epochs': None, 'epochs': 30,
        'patience': 8, 'mixup_alpha': 0.0,
    },
    'gradual_unfreeze': {
        'transform': 'Xv1', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-3, 'lr_backbone': 1e-5, 'freeze_epochs': 10, 'epochs': 30,
        'patience': 10, 'mixup_alpha': 0.0,
    },
    'regularized': {
        'transform': 'Xv1', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-4, 'lr_backbone': 1e-5, 'freeze_epochs': 0, 'epochs': 30,
        'patience': 10, 'mixup_alpha': 0.0,
    },
    'mixup_augmented': {
        'transform': 'Xv4', 'dropout': 0.5, 'label_smoothing': 0.1, 'weight_decay': 1e-4,
        'lr_head': 1e-4, 'lr_backbone': 1e-5, 'freeze_epochs': 0, 'epochs': 40,
        'patience': 10, 'mixup_alpha': 0.2,
    },
}

REPORT_COLUMNS = [
    'model_name', 'dataset', 'strategy', 'transform', 'epochs', 'batch_size',
    'lr_head', 'lr_backbone', 'weight_decay', 'label_smoothing', 'dropout',
    'mixup_alpha', 'best_val_loss', 'final_train_loss', 'final_val_loss',
    'final_train_acc', 'final_val_acc', 'best_train_acc', 'best_val_acc',
    'heldout_acc', 'stopped_epoch', 'train_time_s', 'weights_file',
]


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='Train regularized age classifiers (young/adult), v2')
    parser.add_argument('--datasets', nargs='+', choices=list(DATASET_CONFIGS),
                        default=None)
    parser.add_argument('--strategies', nargs='+', choices=list(STRATEGIES),
                        default=None)
    parser.add_argument('--epochs', type=int, default=None,
                        help='override per-strategy epoch count')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='fraction of Training_Data held out for internal validation')
    parser.add_argument('--report', type=Path, default=Path('training_report_v2.csv'))
    parser.add_argument('--debug', action='store_true',
                        help='smoke test: 1 dataset x 1 strategy, 2 epochs, small subset')
    return parser.parse_args(args)


def stratified_split(groups: pd.Series, val_frac: float, seed: int = 42):
    '''Stratified split of row indices by the Group column (A/Y).'''
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for g in pd.unique(groups):
        idx = np.flatnonzero((groups == g).to_numpy())
        rng.shuffle(idx)
        n_val = round(len(idx)*val_frac)
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return sorted(train_idx), sorted(val_idx)


def build_datasets(cfg: dict, train_transform, val_frac: float,
                   dataset_cls=AgeDataset):
    '''Internal stratified train/val split of Training_Data (+ held-out dataset).

    The val phase uses the internal val split (inference transforms);
    Prediction_Data is returned separately as 'heldout' and never trained on.
    dataset_cls allows swapping in an AgeDataset subclass (e.g.
    AgeDatasetNoNucleus) for all three splits.
    '''
    sub_dir, suffix, labels = TRAIN_DIR
    common = {
        'ch_names': cfg['ch_names'],
        'ch_name_prefix': cfg['ch_name_prefix'],
        'ch_dir_suffix': suffix,
    }
    ds_train_tf = dataset_cls(labels, cfg['root']/sub_dir, transform=train_transform, **common)
    ds_infer = dataset_cls(labels, cfg['root']/sub_dir,
                           transform=data_transforms_inference, **common)

    train_idx, val_idx = stratified_split(ds_infer.img_labels['Group'], val_frac)
    print(f'Internal split: {len(train_idx)} train / {len(val_idx)} val '
          f'(of {len(ds_infer)} Training_Data samples)')
    print(repr(ds_train_tf))

    h_sub, h_suffix, h_labels = HELDOUT_DIR
    ds_heldout = dataset_cls(h_labels, cfg['root']/h_sub,
                             transform=data_transforms_inference,
                             ch_names=cfg['ch_names'],
                             ch_name_prefix=cfg['ch_name_prefix'],
                             ch_dir_suffix=h_suffix)
    print(repr(ds_heldout))

    return {
        'train': Subset(ds_train_tf, train_idx),
        'val': Subset(ds_infer, val_idx),
        'heldout': ds_heldout,
        # inference-transform versions of both splits, for final predictions
        'train_infer': Subset(ds_infer, train_idx),
        'val_infer': Subset(ds_infer, val_idx),
    }


def build_loaders(datasets: dict, batch_size: int) -> dict:
    return {
        phase: DataLoader(datasets[phase], batch_size=batch_size,
                          shuffle=(phase == 'train'), num_workers=4)
        for phase in ('train', 'val')
    }


def set_backbone_trainable(model: nn.Module, trainable: bool):
    for name, p in model.named_parameters():
        if not name.startswith('fc.'):
            p.requires_grad = trainable


def build_optimizer(model: nn.Module, s: dict, backbone_trainable: bool,
                    lr_head: float | None = None):
    groups = [{'params': list(model.fc.parameters()),
               'lr': lr_head if lr_head is not None else s['lr_head']}]
    if backbone_trainable:
        backbone = [p for n, p in model.named_parameters() if not n.startswith('fc.')]
        groups.append({'params': backbone, 'lr': s['lr_backbone']})
    return torch.optim.AdamW(groups, weight_decay=s['weight_decay'])


def predict_phase(model: nn.Module, loader: DataLoader, device: torch.device,
                  desc: str = '') -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, ages, fnames in tqdm(loader, desc=desc):
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            for fname, age, p_adult in zip(fnames, ages, probs):
                y_true = int(age == 'adult')
                pred = int(p_adult >= 0.5)
                rows.append({
                    'Image': fname, 'age': age, 'y_true': y_true,
                    'p_adult': p_adult, 'pred': pred, 'correct': int(y_true == pred),
                })
    return pd.DataFrame(rows)


def plot_history(losses: dict, accs: dict, save_path: Path, stage_boundary: int | None = None):
    fig, axs = plt.subplots(1, 2, figsize=[12, 5])
    for ax, history, title in zip(axs, (losses, accs), ('Loss', 'Accuracy')):
        for phase in history:
            ax.plot(history[phase], label=phase)
        if stage_boundary:
            ax.axvline(stage_boundary - 0.5, ls=':', c='k', alpha=.5,
                       label='unfreeze')
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


def plot_predictions(preds_by_phase: dict, save_path: Path):
    fig, axs = plt.subplots(1, len(preds_by_phase), figsize=[7*len(preds_by_phase), 6], squeeze=False)
    rng = np.random.default_rng(42)
    for ax, (phase, df) in zip(axs.ravel(), preds_by_phase.items()):
        jitter = rng.uniform(-0.04, 0.04, size=len(df))
        colors = np.where(df['correct'].astype(bool), 'tab:green', 'tab:red')
        ax.scatter(df['y_true'] + jitter, df['p_adult'], c=colors, alpha=.5, s=18)
        ax.axhline(0.5, ls='--', c='r', alpha=.4)
        acc = df['correct'].mean()
        ax.set_title(f'{phase} (acc={acc:.3f})')
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['young', 'adult'])
        ax.set_ylabel('P(adult)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved: {save_path}')


def append_report(row: dict, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = report_path.exists()
    with report_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'Report row appended: {report_path}')


def run_training(data_key: str, strategy_name: str, args) -> dict:
    s = dict(STRATEGIES[strategy_name])
    if args.epochs is not None:
        s['epochs'] = args.epochs
        if s['freeze_epochs']:
            s['freeze_epochs'] = min(s['freeze_epochs'], max(1, args.epochs//3))

    model_name = f'resnet18_classifier_{data_key}_{s["transform"]}_{strategy_name}'
    if args.debug:
        model_name += '_debug'
    model_dir = SAVE_ROOT/model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {model_name} ===')
    train_transform = TRAIN_TRANSFORMS[s['transform']]
    datasets = build_datasets(DATASET_CONFIGS[data_key], train_transform, args.val_frac)
    if args.debug:
        datasets['train'] = Subset(datasets['train'].dataset,
                                   datasets['train'].indices[:64])
        datasets['val'] = Subset(datasets['val'].dataset,
                                 datasets['val'].indices[:32])
        datasets['train_infer'] = datasets['val']
        datasets['val_infer'] = datasets['val']
        s['epochs'] = 2
        if s['freeze_epochs']:
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

    if s['freeze_epochs'] is None or s['freeze_epochs'] > 0:
        # Stage 1: backbone frozen, head-only training.
        freeze_epochs = s['freeze_epochs'] if s['freeze_epochs'] else s['epochs']
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
            early_stopping_patience=(s['patience'] if s['freeze_epochs'] is None else None),
            mixup_alpha=s['mixup_alpha'],
            load_best_at_end=(s['freeze_epochs'] is None),
        )
        for phase in losses:
            losses[phase] += l1[phase]
            accs[phase] += a1[phase]

    if s['freeze_epochs'] is not None:
        # Stage 2 (gradual_unfreeze) or single-stage full fine-tune.
        stage2_epochs = s['epochs'] - (s['freeze_epochs'] or 0)
        if s['freeze_epochs']:
            stage_boundary = s['freeze_epochs']
            # smaller head LR once the backbone is unfrozen
            lr_head = s['lr_head']/10
            print(f'-- Stage 2: unfrozen, {stage2_epochs} epochs, '
                  f'head lr={lr_head}, backbone lr={s["lr_backbone"]} --')
        else:
            lr_head = s['lr_head']
            best_loss = float('inf')
            print(f'-- Full fine-tune: {stage2_epochs} epochs, '
                  f'head lr={lr_head}, backbone lr={s["lr_backbone"]} --')
        set_backbone_trainable(model, True)
        optimizer = build_optimizer(model, s, backbone_trainable=True, lr_head=lr_head)
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
    # is scored only here, after training is fully done.
    model = model.to(TORCH_DEVICE).eval()
    preds_by_phase = {}
    for phase, key in (('train', 'train_infer'), ('val', 'val_infer'), ('heldout', 'heldout')):
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

    if args.debug:
        datasets = args.datasets or list(DATASET_CONFIGS)[:1]
        strategies = args.strategies or ['regularized']
    else:
        datasets = args.datasets or list(DATASET_CONFIGS)
        strategies = args.strategies or list(STRATEGIES)

    results = []
    for data_key in datasets:
        for strategy_name in strategies:
            results.append(run_training(data_key, strategy_name, args))

    print('\n=== Summary ===')
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
