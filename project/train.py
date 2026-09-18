"""Accelerate training with MLflow tracking and a single champion bundle."""
import argparse
import hashlib
import json
import math
from contextlib import ExitStack
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, set_seed
from filelock import FileLock
from transformers import get_linear_schedule_with_warmup
from .checkpoint import promote_champion, save_bundle
from .config import Config, load_labels
from .dataset import PersonQueryDataset, QueryCollator, make_tokenizer
from .model import PersonQueryParser, compute_loss
from .metrics import calculate_metrics, decode_indices
from .tracking import tracking_run, log_epoch, log_champion


def main_process_call(accelerator, operation):
    """Propagate tracking/checkpoint errors so other ranks do not wait forever."""
    result = [None, None]
    if accelerator.is_main_process:
        try:
            result[0] = operation()
        except Exception as exc:
            result[1] = f'{type(exc).__name__}: {exc}'
    broadcast_object_list(result)
    if result[1]:
        raise RuntimeError(result[1])
    return result[0]


@torch.no_grad()
def validate(model, loader, accelerator, labels, config):
    model.eval()
    predictions, targets, losses = {}, {}, {}
    count = 0
    for batch in loader:
        truth = batch.pop('targets')
        with accelerator.autocast():
            outputs = model(**batch)
        gathered_outputs, gathered_truth = accelerator.gather_for_metrics((outputs, truth))
        loss, components = compute_loss(gathered_outputs, gathered_truth, config.loss_weights)
        n = len(gathered_truth['gender'])
        count += n
        for key, value in {'total': loss, **components}.items():
            losses[key] = losses.get(key, 0.0) + value.item() * n
        predicted = decode_indices(gathered_outputs, labels, config)
        for key in predicted:
            predictions.setdefault(key, []).append(predicted[key].cpu())
            targets.setdefault(key, []).append(gathered_truth[key].cpu())
    result = calculate_metrics({k: torch.cat(v) for k, v in predictions.items()},
                               {k: torch.cat(v) for k, v in targets.items()}, labels)
    result.update({f'validation_{key}_loss': value / count for key, value in losses.items()})
    result['validation_loss'] = result.pop('validation_total_loss')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--validation', required=True)
    parser.add_argument('--output', required=True, help='New output directory; only champion/ will be retained')
    parser.add_argument('--config')
    parser.add_argument('--tracking-uri', help='MLflow server/database URI; defaults to project/mlflow.db')
    parser.add_argument('--experiment', default='person-query-parser')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    accelerator = Accelerator(mixed_precision=config.mixed_precision,
                              gradient_accumulation_steps=config.gradient_accumulation_steps)
    if accelerator.device.type != 'cuda' and config.mixed_precision != 'no':
        raise ValueError('This pipeline enables mixed precision only on CUDA; choose no on CPU/MPS')
    set_seed(config.seed)
    labels = load_labels()
    output = Path(args.output).resolve()
    signature = {'train_sha256': hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                 'validation_sha256': hashlib.sha256(Path(args.validation).read_bytes()).hexdigest(),
                 'world_size': accelerator.num_processes, 'labels': labels}
    with ExitStack() as stack:
        def start():
            output.parent.mkdir(parents=True, exist_ok=True)
            stack.enter_context(FileLock(str(output.parent / f'.{output.name}.training.lock'), timeout=0))
            if output.exists() and any(output.iterdir()):
                raise ValueError('Output is not empty. Choose a new output; existing champions are preserved.')
            output.mkdir(parents=True, exist_ok=True)
            return stack.enter_context(tracking_run(config, signature, output,
                args.tracking_uri, args.experiment, args.run_name or output.name))
        run_id = main_process_call(accelerator, start)
        tokenizer = make_tokenizer(config)
        train = PersonQueryDataset(args.train, tokenizer, config, labels)
        val = PersonQueryDataset(args.validation, tokenizer, config, labels)
        if set(train.texts) & set(val.texts):
            raise ValueError('Train and validation contain duplicate normalized texts')
        collator = QueryCollator(tokenizer)
        train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val, batch_size=config.batch_size, collate_fn=collator)
        model = PersonQueryParser(config, labels)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
        updates = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(updates * config.warmup_ratio), updates)
        step = 0
        for epoch in range(1, config.epochs + 1):
            model.train()
            if hasattr(train_loader, 'set_epoch'):
                train_loader.set_epoch(epoch - 1)
            totals, count = {}, 0
            for batch in train_loader:
                targets = batch.pop('targets')
                n = len(targets['gender'])
                with accelerator.accumulate(model):
                    with accelerator.autocast():
                        loss, components = compute_loss(model(**batch), targets, config.loss_weights)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                        step += 1
                    optimizer.zero_grad()
                count += n
                for key, value in {'total': loss, **components}.items():
                    totals[key] = totals.get(key, 0.0) + value.detach().item() * n
            metrics = validate(model, val_loader, accelerator, labels, config)
            names = list(totals)
            stats = torch.tensor([totals[key] for key in names] + [count], device=accelerator.device)
            stats = accelerator.reduce(stats, reduction='sum').cpu().tolist()
            metrics.update({f'train_{key}_loss': stats[i] / stats[-1] for i, key in enumerate(names)})
            def finish_epoch():
                promoted = promote_champion(output, accelerator.unwrap_model(model), tokenizer,
                    config, labels, metrics, {'epoch': epoch, 'global_step': step,
                                             'mlflow_run_id': run_id, 'dataset_signature': signature})
                log_epoch(metrics, epoch, step, scheduler.get_last_lr()[0], promoted)
                if promoted:
                    log_champion(output / 'champion')
                print(json.dumps({'epoch': epoch, 'global_step': step, 'mlflow_run_id': run_id,
                                  'champion_promoted': promoted, **metrics}), flush=True)
            main_process_call(accelerator, finish_epoch)


if __name__ == '__main__':
    main()
