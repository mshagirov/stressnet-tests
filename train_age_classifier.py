'''Train age classifiers (young vs adult) on all dataset versions.

For every dataset version (v5 2-channel, v6 3-channel) and both weight
initialisations (ImageNet-pretrained, scratch) this script:
  - trains resnet18_classifier for num_epochs on Training_Data only
    (Prediction_Data is used for the val phase / prediction plots, never
    for gradient updates),
  - saves weights, loss/accuracy plot, per-phase prediction plots and CSVs
    into /mnt/data/faris-data/saved_models/<model_name>/,
  - appends one summary row per run to <report_csv> in the current directory.

Usage (from inside stressnet-tests/):
    uv run python train_age_classifier.py
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier_trainer import train_classifier
from classifiers import resnet18_classifier
from dataset import AgeDataset
from models import TORCH_DEVICE
from transforms import data_transforms, data_transforms_inference

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
PHASE_DIRS = {
    'train': ('Training_Data', '_Train', 'Stiffness_Data_Training.xlsx'),
    'val': ('Prediction_Data', '_Prediction', 'Stiffness_Data_Prediction.xlsx'),
}

REPORT_COLUMNS = [
    'model_name', 'dataset', 'init', 'epochs', 'batch_size', 'lr',
    'best_val_loss', 'final_train_loss', 'final_val_loss',
    'final_train_acc', 'final_val_acc', 'train_time_s', 'weights_file',
]


def parse_args(args):
    parser = argparse.ArgumentParser(description='Train age classifiers (young/adult)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--report', type=Path, default=Path('training_report.csv'))
    return parser.parse_args(args)


def build_datasets(cfg: dict):
    datasets = {}
    for phase, (sub_dir, suffix, labels) in PHASE_DIRS.items():
        transform = data_transforms[phase] if phase == 'train' else data_transforms_inference
        ds = AgeDataset(
            labels,
            cfg['root']/sub_dir,
            ch_names=cfg['ch_names'],
            ch_name_prefix=cfg['ch_name_prefix'],
            ch_dir_suffix=suffix,
            transform=transform,
        )
        print(repr(ds))
        datasets[phase] = ds
    return datasets


def build_loaders(datasets: dict, batch_size: int) -> dict:
    return {
        phase: DataLoader(ds, batch_size=batch_size, shuffle=(phase == 'train'), num_workers=4)
        for phase, ds in datasets.items()
    }


def predict_phase(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, ages, fnames in tqdm(loader):
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            for fname, age, p_adult in zip(fnames, ages, probs):
                y_true = int(age == 'adult')
                pred = int(p_adult >= 0.5)
                rows.append({
                    'Image': fname, 'age': age, 'y_true': y_true,
                    'p_adult': p_adult, 'pred': pred, 'correct': int(y_true == pred),
                })
    return pd.DataFrame(rows)


def plot_history(losses: dict, accs: dict, save_path: Path):
    fig, axs = plt.subplots(1, 2, figsize=[12, 5])
    for ax, history, title in zip(axs, (losses, accs), ('Loss', 'Accuracy')):
        for phase in history:
            ax.plot(history[phase], label=phase)
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


def run_training(data_key: str, init: str, args) -> dict:
    model_name = f'resnet18_classifier_{data_key}_Xv1_{init}'
    model_dir = SAVE_ROOT/model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n=== {model_name} ===')
    datasets = build_datasets(DATASET_CONFIGS[data_key])
    dataloaders = build_loaders(datasets, args.batch_size)
    dataset_sizes = {phase: len(ds) for phase, ds in datasets.items()}

    model = resnet18_classifier(num_classes=2, pretrained=(init == 'pretrained'))
    since = time.time()
    model, losses, accs = train_classifier(
        model,
        nn.CrossEntropyLoss(),
        torch.optim.Adam(model.parameters(), lr=args.lr),
        scheduler=None,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        num_epochs=args.epochs,
        return_best_val=True,
        device=TORCH_DEVICE,
    )
    elapsed = time.time() - since

    weights_file = f'{model_name}_{args.epochs}epoch_Adam.pt'
    torch.save(model.state_dict(), model_dir/weights_file)
    print(f'Saved: {model_dir/weights_file}')

    plot_history(losses, accs, model_dir/f'{model_name}_loss.png')

    # Prediction runs use inference transforms for BOTH phases.
    model = model.to(TORCH_DEVICE).eval()
    preds_by_phase = {}
    for phase in ('train', 'val'):
        pred_loader = DataLoader(
            AgeDataset(
                PHASE_DIRS[phase][2],
                DATASET_CONFIGS[data_key]['root']/PHASE_DIRS[phase][0],
                ch_names=DATASET_CONFIGS[data_key]['ch_names'],
                ch_name_prefix=DATASET_CONFIGS[data_key]['ch_name_prefix'],
                ch_dir_suffix=PHASE_DIRS[phase][1],
                transform=data_transforms_inference,
            ),
            batch_size=args.batch_size, shuffle=False, num_workers=4,
        )
        preds_df = predict_phase(model, pred_loader, TORCH_DEVICE)
        preds_df.to_csv(model_dir/f'{model_name}_{phase}_preds.csv', index=False)
        preds_by_phase[phase] = preds_df
    plot_predictions(preds_by_phase, model_dir/f'{model_name}_preds.png')

    row = {
        'model_name': model_name,
        'dataset': data_key,
        'init': init,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'best_val_loss': min(losses['val']),
        'final_train_loss': losses['train'][-1],
        'final_val_loss': losses['val'][-1],
        'final_train_acc': accs['train'][-1],
        'final_val_acc': accs['val'][-1],
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

    results = []
    for data_key in DATASET_CONFIGS:
        for init in ('pretrained', 'scratch'):
            results.append(run_training(data_key, init, args))

    print('\n=== Summary ===')
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == '__main__':
    main()
