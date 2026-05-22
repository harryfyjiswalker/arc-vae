"""
ARC-VAE: Main Training Script
==============================
Run this on a GPU machine after cloning both repos:

  git clone https://github.com/MarcYin/ARC.git
  git clone https://github.com/princeemensah/transformer-prosailvae.git
  pip install pynndescent jax jaxlib scipy tqdm torch

Then:
  python train_arc_vae.py

Outputs are saved to ./outputs/
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "ARC"))

from synthetic_dataloader import GermanyMaizeDataset
from training import ARCVAE, train


def get_args():
    p = argparse.ArgumentParser(description="Train ARC-VAE")

    # Data
    p.add_argument("--n_train",      type=int,   default=50_000)
    p.add_argument("--n_val",        type=int,   default=5_000)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--crop_type",    type=str,   default="maize",
                   choices=["maize", "wheat"])

    # Architecture
    p.add_argument("--d_model",      type=int,   default=128)
    p.add_argument("--n_layers",     type=int,   default=4)
    p.add_argument("--n_heads",      type=int,   default=4)
    p.add_argument("--n_queries",    type=int,   default=4)
    p.add_argument("--d_ff",         type=int,   default=256)
    p.add_argument("--dropout",      type=float, default=0.1)

    # Training
    p.add_argument("--n_epochs_s1",  type=int,   default=20,
                   help="Stage 1 epochs (reconstruction only)")
    p.add_argument("--n_epochs_s2",  type=int,   default=30,
                   help="Stage 2 epochs (KL annealing)")
    p.add_argument("--beta_target",  type=float, default=0.3,
                   help="Final KL weight")
    p.add_argument("--lambda_sup",   type=float, default=0.0,
                   help="Weight on supervised auxiliary loss "
                        "(0 = pure VAE; 0.1-1.0 = hybrid VAE+supervised). "
                        "Range-normalised MSE on encoder posterior mean. "
                        "Only used during training; no effect at inference.")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42)

    # I/O
    p.add_argument("--out_dir",      type=str,   default="./outputs")
    p.add_argument("--log_every",    type=int,   default=100)
    p.add_argument("--device",       type=str,   default="auto")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from latest checkpoint in out_dir")

    return p.parse_args()


def main():
    args = get_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    # Output directory
    run_name = (f"{args.crop_type}_d{args.d_model}_L{args.n_layers}"
                f"_e{args.n_epochs_s1}+{args.n_epochs_s2}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M')}")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    print(f"\nGenerating {args.n_train:,} training samples...")
    train_ds = GermanyMaizeDataset(
        n_samples=args.n_train, seed=args.seed,
        crop_type=args.crop_type
    )
    print(f"Generating {args.n_val:,} validation samples...")
    val_ds = GermanyMaizeDataset(
        n_samples=args.n_val, seed=args.seed + 99999,
        crop_type=args.crop_type
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=0, pin_memory=(device == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0, pin_memory=(device == "cuda")
    )

    # Model
    model = ARCVAE(
        crop_type=args.crop_type,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_queries=args.n_queries,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    n_enc = sum(p.numel() for p in model.encoder.parameters()
                if p.requires_grad)
    print(f"\nEncoder parameters (trainable): {n_enc:,}")
    print(f"Decoder parameters (frozen):    "
          f"{sum(p.numel() for p in model.decoder.parameters()):,}")

    # Train
    ckpt_path = out_dir / "checkpoint_latest.pt"
    resume_from = str(ckpt_path) if args.resume and ckpt_path.exists() else None
    if resume_from:
        print(f"Will resume from {resume_from}")

    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs_s1=args.n_epochs_s1,
        n_epochs_s2=args.n_epochs_s2,
        beta_target=args.beta_target,
        lambda_sup=args.lambda_sup,
        lr=args.lr,
        device=device,
        log_every=args.log_every,
        checkpoint_dir=str(out_dir),
        resume_from=resume_from,
    )

    # Save model and history
    torch.save({
        "model_state": model.state_dict(),
        "config":      vars(args),
        "history":     history,
    }, out_dir / "model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nSaved to {out_dir}")
    print(f"Final val_rec: {history['val_rec'][-1]:.4f}")
    print(f"Final val_kl:  {history['val_kl'][-1]:.4f}")


if __name__ == "__main__":
    main()
