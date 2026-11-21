#!/usr/bin/env python
"""Evaluate PreDiff checkpoint on test set."""
import argparse
import os
import torch

from run import DATASETS, ROOT_DIR, apply_overrides, build_model
from datasets.base import load_data, split_data, RecDataset
from torch.utils.data import DataLoader
from trainer import evaluate, set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='beauty')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--data_dir', type=str, default=None, help='Override interaction pickle path')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Checkpoint file or directory containing prediff_<dataset>_best.pt')
    parser.add_argument('--override', type=str, nargs='*', default=[])
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    set_seed(2026)

    if os.path.isdir(args.ckpt):
        ckpt_path = os.path.join(args.ckpt, f'prediff_{args.dataset}_best.pt')
    else:
        ckpt_path = args.ckpt
    print(f"Loading {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    cfg = ckpt['config'].copy()
    apply_overrides(cfg, args.override)

    data_path = args.data_dir or cfg.get('data_dir') or os.path.join(ROOT_DIR, 'data', f'{args.dataset}.pickle')
    item_size, encoder_seq_len, user_seq = load_data(data_path)
    test_seq = split_data(user_seq, only_test=True)
    test_loader = DataLoader(
        RecDataset(test_seq, item_size, encoder_seq_len),
        batch_size=args.batch_size, shuffle=False)

    model = build_model(cfg, item_size, encoder_seq_len)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)

    test_metrics = evaluate(model, test_loader, device, ks=[10, 20])
    print(f"\nTest Results — {args.dataset}")
    print(f"{'='*50}")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
