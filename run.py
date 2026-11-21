#!/usr/bin/env python
"""Entry point for PreDiff experiments.

Usage:
    python run.py --dataset beauty --gpu 0
"""

import argparse
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import yaml

import torch

from datasets import load_data, split_data, RecDataset, LazyRecTrainDataset
from models.prediff import PreDiff
from trainer import train, set_seed
from torch.utils.data import DataLoader


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = ('beauty', 'toys', 'ml1m', 'ml25m')


class _Tee:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()


def _worker_init_fn(worker_id):
    """Ensure each DataLoader worker has a unique random seed."""
    seed = torch.initial_seed() % (2**32) + worker_id
    import random as _random
    import numpy as _np
    _random.seed(seed)
    _np.random.seed(seed)


def make_parser():
    parser = argparse.ArgumentParser(description='PreDiff - Sequential Recommendation')
    parser.add_argument('--dataset', choices=DATASETS, default='beauty')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML. Default: configs/')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data path (default: data/)')
    parser.add_argument('--save_dir', type=str, default=os.path.join(ROOT_DIR, 'results'),
                        help='Parent directory for runs (default: results/)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Override config: key=value pairs, e.g. --override lr=0.0005')
    return parser


def apply_overrides(cfg, overrides):
    for override in overrides:
        key, value = override.split('=', 1)
        if value.lower() in ('true', 'false'):
            cfg[key] = value.lower() == 'true'
        else:
            try:
                cfg[key] = int(value)
            except ValueError:
                try:
                    cfg[key] = float(value)
                except ValueError:
                    cfg[key] = value


def load_config(args):
    """Load dataset YAML, then apply runtime arguments and explicit overrides."""
    config_path = args.config or os.path.join(ROOT_DIR, 'configs', f'{args.dataset}.yaml')
    with open(config_path) as f:
        all_cfg = yaml.safe_load(f)
    cfg = {'num_workers': 4}
    cfg.update(all_cfg['_defaults'])
    cfg.update(all_cfg['prediff'])
    print(f"[config] Loaded {config_path}")

    cfg['dataset'] = args.dataset
    cfg['gpu'] = args.gpu
    cfg['seed'] = args.seed
    cfg['data_dir'] = args.data_dir
    cfg['save_dir'] = args.save_dir
    apply_overrides(cfg, args.override)
    return cfg


def build_model(cfg, item_size, encoder_seq_len):
    """Build PreDiff model from config dict."""
    return PreDiff(
        item_size=item_size,
        hidden_dim=cfg['hidden_dim'],
        encoder_seq_len=encoder_seq_len,
        num_encoder_layers=cfg['num_encoder_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        dropout_prob=cfg['dropout_prob'],
        diffusion_steps=cfg['diffusion_steps'],
        noise_schedule=cfg['noise_schedule'],
        decoder_layers=cfg['decoder_layers'],
        corr_alpha=cfg['corr_alpha'],
        corr_tau=cfg['corr_tau'],
        dist_dropout=float(cfg['dist_dropout']),
    )


def _run_training(args, run_dir):
    cfg = load_config(args)
    cfg['save_dir'] = run_dir
    set_seed(cfg['seed'])
    device = torch.device(f'cuda:{cfg["gpu"]}' if torch.cuda.is_available() else 'cpu')

    # Load data
    if cfg.get('data_dir') is not None:
        data_path = cfg['data_dir']
    else:
        data_path = os.path.join(ROOT_DIR, 'data', f'{cfg["dataset"]}.pickle')
    item_size, encoder_seq_len, user_seq = load_data(data_path)

    # Split
    valid_seq, test_seq = split_data(user_seq)

    # Dataloaders
    train_dataset = LazyRecTrainDataset(user_seq, encoder_seq_len)
    print(f"[data] Training: {len(train_dataset)} samples")
    val_dataset = RecDataset(valid_seq, item_size, encoder_seq_len)
    test_dataset = RecDataset(test_seq, item_size, encoder_seq_len)

    num_workers = cfg['num_workers']
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Build model
    model = build_model(cfg, item_size, encoder_seq_len)

    # Compute eval_interval
    steps_per_epoch = len(train_loader)
    eval_epochs = cfg['eval_epochs']
    cfg['eval_interval'] = steps_per_epoch * eval_epochs
    print(f"[data] steps_per_epoch={steps_per_epoch}, eval_interval={cfg['eval_interval']}")

    print(f"Dataset: {cfg['dataset']}, Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Step-based training: {cfg['learning_steps']} steps, eval every {cfg['eval_interval']} steps, patience={cfg['patience']}")
    train(model, train_loader, val_loader, test_loader, cfg, device)


if __name__ == '__main__':
    args = make_parser().parse_args()
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run_dir = os.path.join(os.path.abspath(args.save_dir), f'{args.dataset}-{timestamp}')
    os.makedirs(run_dir)
    log_path = os.path.join(run_dir, 'train.log')
    with open(log_path, 'w', encoding='utf-8') as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(_Tee(sys.stderr, log_file)):
            print(f"[run] Output: {run_dir}")
            print(f"[run] Log: {log_path}")
            _run_training(args, run_dir)
