"""Training loop, evaluation, and logging for PreDiff."""

import os
import time
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_best_checkpoint(best_state, config, best_step, best_val_metrics):
    """Save best checkpoint immediately on new best."""
    save_dir = config['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"prediff_{config['dataset']}_best.pt")
    torch.save({
        'model_state_dict': best_state,
        'config': config,
        'best_step': best_step,
        'best_val_metrics': best_val_metrics,
    }, ckpt_path)


def evaluate(model, dataloader, device, ks=[10, 20]):
    """Full-ranking Hit@K/NDCG@K with timing."""
    model.eval()
    max_k = max(ks)
    total_hit = {k: 0.0 for k in ks}
    total_ndcg = {k: 0.0 for k in ks}
    total_samples = 0

    use_cuda_events = device.type == 'cuda'
    inference_time_ms = 0.0

    if use_cuda_events:
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            if use_cuda_events:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(torch.cuda.current_stream(device))
            else:
                start_time = time.perf_counter()

            scores = model.compute_scores(batch)

            if use_cuda_events:
                end_event.record(torch.cuda.current_stream(device))
                torch.cuda.synchronize(device)
                inference_time_ms += start_event.elapsed_time(end_event)
            else:
                inference_time_ms += (time.perf_counter() - start_time) * 1000

            # mask padding token
            scores[:, 0] = -torch.inf
            labels = batch['target_id'].squeeze(-1)
            total_samples += labels.size(0)

            _, topk_idx = torch.topk(scores, k=max_k, dim=-1)
            labels_exp = labels.unsqueeze(-1)
            hit_matrix = (topk_idx == labels_exp).float()

            for k in ks:
                total_hit[k] += hit_matrix[:, :k].sum(dim=-1).sum().item()
                positions = torch.arange(1, k + 1, dtype=torch.float32, device=device)
                dcg = (hit_matrix[:, :k] / torch.log2(positions + 1)).sum(dim=-1)
                total_ndcg[k] += dcg.sum().item()

    if use_cuda_events:
        torch.cuda.synchronize(device)

    results = {}
    for k in ks:
        results[f'HR@{k}'] = round(total_hit[k] / total_samples * 100, 4)
        results[f'NDCG@{k}'] = round(total_ndcg[k] / total_samples * 100, 4)

    avg_time_ms = inference_time_ms / total_samples if total_samples > 0 else 0
    results['latency_ms'] = round(avg_time_ms, 4)

    return results


def infinite_loader(data_loader):
    """Yield batches infinitely by cycling through the dataloader."""
    while True:
        yield from data_loader



def train(model, train_loader, val_loader, test_loader, config, device):
    """Step-based training loop with early stopping."""
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.get('lr', 0.001),
                                   weight_decay=config.get('weight_decay', 0.0))

    # Training config
    learning_steps = config.get('learning_steps', 150000)
    eval_interval = config.get('eval_interval', 3000)
    log_interval = config.get('log_interval', 100)
    patience = config.get('patience', 10)
    ks = [10, 20]

    best_hr20 = -float('inf')
    best_step = 0
    bad_count = 0
    best_state = None
    best_val_metrics = None

    data_iter = infinite_loader(train_loader)
    running_loss = 0
    log_count = 0

    set_seed(config.get('seed', 0))

    model.train()
    for step in range(1, learning_steps + 1):
        batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        loss = model.compute_loss(batch)['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss = loss.item()

        running_loss += total_loss
        log_count += 1

        if step % log_interval == 0:
            avg_loss = running_loss / log_count
            print(f"Step {step}/{learning_steps}: loss={avg_loss:.4f}")

            running_loss = 0
            log_count = 0

        if step % eval_interval == 0:
            val_metrics = evaluate(model, val_loader, device, ks)

            print(f"  Val: {val_metrics}")

            if val_metrics['HR@20'] > best_hr20:
                best_hr20 = val_metrics['HR@20']
                best_step = step
                bad_count = 0
                best_val_metrics = val_metrics
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  ★ New best! HR@20={best_hr20}")
                _save_best_checkpoint(best_state, config, best_step, best_val_metrics)
            else:
                bad_count += 1
                if patience > 0 and bad_count >= patience:
                    print(f"Early stopping at step {step} (best was step {best_step})")
                    break
            model.train()

    # Test with best model
    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device, ks)

    print(f"\n{'='*50}")
    print(f"Best val (step {best_step}): {best_val_metrics}")
    print(f"Test: {test_metrics}")
    print(f"{'='*50}")

    return test_metrics
