"""Train InkCloser (CODE BLOCKS 13 / 13b / SAVE CHECKPOINT of the notebook).

Defaults reproduce the reported model: AdamW (wd 1e-4), lr 1e-3 for convolutions and
3e-4 for the two attention modules, cosine schedule configured for 30 epochs
(T_max = 30, eta_min = 1e-6) of which the first 5 are run (11,250 steps at batch 4),
fp16 autocast with gradient scaling, gradient clipping at 1.0, LineLoss(3, 1, 1, 0.5).

A checkpoint (same format as the notebook, so block 13b can restore it and vice
versa) is written to --out-dir after every epoch as last.pt; --resume continues from
one. The test set is evaluated at the end of every epoch for monitoring, as in the
notebook (--eval-batches limits it, 0 disables it).

Examples
  # the paper run (5 of 30 scheduled epochs), ~several hours on a T4
  python code/train.py --out-dir /content/checkpoints/inkcloser

  # smoke test: 20 steps, short evaluation, previews
  python code/train.py --out-dir /tmp/smoke --max-steps 20 --eval-batches 2 --preview-every 10

  # resume after a disconnect
  python code/train.py --out-dir /content/checkpoints/inkcloser --resume /content/checkpoints/inkcloser/last.pt
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inkcloser.data import (DEFAULT_SPLIT_ROOT, build_datasets, build_loaders,  # noqa: E402
                            load_split, preview_tensors)
from inkcloser.loss import LineLoss  # noqa: E402
from inkcloser.metrics import evaluate  # noqa: E402
from inkcloser.model import LineInpainter  # noqa: E402
from inkcloser.viz import LOSS_KEYS, save_curves, save_predictions  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--out-dir", required=True, help="checkpoints, logs and previews")
    # schedule
    ap.add_argument("--epochs", type=int, default=30, help="length of the cosine schedule (T_max)")
    ap.add_argument("--stop-after", type=int, default=5,
                    help="stop after this many epochs (paper: 5 of 30); 0 = run all --epochs")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many optimizer steps in total (check runs)")
    ap.add_argument("--train-subset", type=int, default=None,
                    help="train on the first N training images only (check runs)")
    # optimisation
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-attn", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--eta-min", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    # loss
    ap.add_argument("--hole-weight", type=float, default=3.0)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--gap-dice-weight", type=float, default=0.5)
    ap.add_argument("--gap-radius", type=int, default=7)
    # misc
    ap.add_argument("--seed", type=int, default=0, help="-1 = unseeded, like the notebook")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=None,
                    help="test batches evaluated per epoch (default: all; 0 = no evaluation)")
    ap.add_argument("--preview-every", type=int, default=500, help="steps; 0 = no step previews")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--keep-epochs", action="store_true",
                    help="also keep inkrepair_epXXX.pt for every epoch (~520 MB each)")
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    return ap.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optim(model, args):
    attn_params = model.attention_parameters()
    attn_ids = {id(p) for p in attn_params}
    conv_params = [p for p in model.parameters() if id(p) not in attn_ids]
    groups = [{'params': conv_params, 'lr': args.lr},
              {'params': attn_params, 'lr': args.lr_attn}]
    groups = [g for g in groups if g['params']]          # ablations may have no attention
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    n_opt = sum(p.numel() for g in optimizer.param_groups for p in g['params'])
    assert n_opt == sum(p.numel() for p in model.parameters()), "params missing from optimizer"
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                           eta_min=args.eta_min)
    return optimizer, scheduler


def save_checkpoint(path, model, optimizer, scheduler, scaler, history, loss_log,
                    global_step, epochs_done, args):
    tmp = path + ".tmp"
    torch.save(dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(),
        scaler=scaler.state_dict(),
        history=history,
        loss_log=loss_log,
        global_step=global_step,
        epochs_done=epochs_done,
        args=vars(args),
    ), tmp)
    os.replace(tmp, path)            # never leave a half-written last.pt


def main(argv=None, model_fn=None):
    """model_fn: zero-argument factory of the network (default LineInpainter); used by ablations."""
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    preview_dir = os.path.join(args.out_dir, "previews")
    with open(os.path.join(args.out_dir, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    generator = None
    if args.seed >= 0:
        seed_everything(args.seed)
        generator = torch.Generator().manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")

    # data (CODE BLOCK 4)
    split = load_split(args.split_root)
    train_dataset, test_dataset, sample_dataset = build_datasets(split)
    if args.train_subset:   # check runs only: first N training images
        train_dataset.items = train_dataset.items[:args.train_subset]
        train_dataset.files = train_dataset.files[:args.train_subset]
        print(f"[check run] training on the first {len(train_dataset)} images only")
    train_loader, test_loader = build_loaders(train_dataset, test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers, generator=generator)
    ex_corrupted, ex_clean, example_files = preview_tensors(sample_dataset)
    print("train_loader:", len(train_loader), "batches/epoch")

    # model / optimiser (CODE BLOCK 13)
    model = (model_fn or LineInpainter)().to(device)
    optimizer, scheduler = build_optim(model, args)
    scaler = GradScaler('cuda', enabled=use_amp)
    criterion = LineLoss(hole_weight=args.hole_weight, bce_weight=args.bce_weight,
                         dice_weight=args.dice_weight, gap_dice_weight=args.gap_dice_weight,
                         gap_radius=args.gap_radius).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M params")

    history, loss_log = [], {k: [] for k in LOSS_KEYS}
    global_step, start_epoch = 0, 0
    if args.resume:   # CODE BLOCK 13b
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)  # trusted
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        history, loss_log = ckpt["history"], ckpt["loss_log"]
        global_step, start_epoch = ckpt["global_step"], ckpt["epochs_done"]
        print(f"resumed {args.resume} at epoch {start_epoch}/{args.epochs}, step {global_step}, "
              f"lr {[round(g['lr'], 6) for g in optimizer.param_groups]}")

    end_epoch = min(args.epochs, args.stop_after) if args.stop_after else args.epochs
    last_path = os.path.join(args.out_dir, "last.pt")
    hist_path = os.path.join(args.out_dir, "history.json")
    stop = False
    t_start = time.perf_counter()

    for epoch in range(start_epoch, end_epoch):
        model.train()
        t_epoch = time.perf_counter()
        for i, (corrupted, clean) in enumerate(train_loader):
            corrupted = corrupted.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            # mixed-precision forward pass, loss in float32
            optimizer.zero_grad()
            with autocast(device.type, dtype=torch.float16, enabled=use_amp):
                output = model(corrupted)
            loss = criterion(output.float(), clean, corrupted)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            for k in LOSS_KEYS:
                loss_log[k].append(criterion.last_components[k])

            if i % args.log_every == 0:
                c = criterion.last_components
                rate = (i + 1) / (time.perf_counter() - t_epoch)
                print(f"Epoch [{epoch + 1}/{args.epochs}] Step [{i}/{len(train_loader)}] "
                      f"total={c['total']:.4f} bce={c['bce']:.4f} dice={c['dice']:.4f} "
                      f"hole_only={c['hole_only_bce']:.4f} hole_frac={c['hole_frac']:.4f} "
                      f"({rate:.2f} it/s)", flush=True)

            if args.preview_every and global_step % args.preview_every == 0:
                save_predictions(model, device, ex_corrupted, ex_clean, example_files,
                                 f"step_{global_step:06d}", preview_dir)
                save_curves(loss_log, history, f"step_{global_step:06d}", preview_dir)

            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        if stop:   # partial epoch: save, but do not count it as done or step the schedule
            save_checkpoint(last_path, model, optimizer, scheduler, scaler, history, loss_log,
                            global_step, len(history), args)
            print(f"stopped at --max-steps {args.max_steps} (mid-epoch {epoch + 1}); saved {last_path}")
            if args.eval_batches != 0:
                m = evaluate(model, test_loader, device, max_batches=args.eval_batches)
                print("  TEST " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
            break

        print(f"Epoch [{epoch + 1}/{args.epochs}] done in {time.perf_counter() - t_epoch:.0f} s")
        if args.eval_batches != 0:
            m = evaluate(model, test_loader, device, max_batches=args.eval_batches)
        else:
            m = {}
        history.append(m)
        if m:
            print("  TEST " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
        save_predictions(model, device, ex_corrupted, ex_clean, example_files,
                         f"epoch_{epoch + 1:03d}", preview_dir)
        save_curves(loss_log, history, f"epoch_{epoch + 1:03d}", preview_dir)

        scheduler.step()
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, history, loss_log,
                        global_step, len(history), args)
        if args.keep_epochs:
            ep_path = os.path.join(args.out_dir, f"inkrepair_ep{len(history):03d}.pt")
            save_checkpoint(ep_path, model, optimizer, scheduler, scaler, history, loss_log,
                            global_step, len(history), args)
        with open(hist_path, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"  saved {last_path} (epoch {len(history)}, step {global_step})")

    print(f"Done! {global_step} steps, {len(history)} epochs, "
          f"{(time.perf_counter() - t_start) / 60:.1f} min this run")


if __name__ == "__main__":
    main()
