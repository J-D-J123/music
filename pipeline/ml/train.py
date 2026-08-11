"""Stage 3: training loop. Adam + CrossEntropyLoss (PAD ignored), run across
the three curriculum stages, checkpointing between each so a later stage
always resumes from the previous stage's weights rather than from scratch.

NOTE on stage 3 ("style fine-tuning"): the three datasets described so far
(MAESTRO, PDMX, Zenodo Violin) cover macro pretraining and domain adaptation
cleanly, but nothing distinct has been defined yet for a *style* fine-tuning
set (e.g. a specific performer/recording corpus). Stage 3 below reuses the
`violin` source at a much lower learning rate as a placeholder so the
pipeline runs end-to-end — swap in a real style-specific --sources value
once that dataset exists.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam

from pipeline.ml.data_prep import PAD
from pipeline.ml.dataset import load_pieces, split_pieces, filter_by_source, get_dataloader
from pipeline.ml.model import NoteTransformer

DEFAULT_STAGES = [
    {"name": "macro_pretrain", "sources": {"maestro"}, "epochs": 10, "lr": 3e-4},
    {"name": "domain_adapt", "sources": {"pdmx", "violin"}, "epochs": 5, "lr": 1e-4},
    {"name": "style_finetune", "sources": {"violin"}, "epochs": 3, "lr": 3e-5},
]


def train_one_epoch(model, dataloader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, total_tokens = 0.0, 0

    for input_ids, target_ids, loss_mask in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)  # [batch, seq_len, vocab_size]

        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        n_valid = loss_mask.sum().item()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for input_ids, target_ids, loss_mask in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))

        n_valid = loss_mask.sum().item()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid

    return total_loss / max(total_tokens, 1)


def run_stage(model, stage: dict, all_train_pieces, all_val_pieces, device,
              seq_len: int, stride: int, batch_size: int, checkpoint_dir: Path, stage_idx: int, total_stages: int):
    name, sources, epochs, lr = stage["name"], stage["sources"], stage["epochs"], stage["lr"]
    
    # Status update: Stage overview
    print(f"\n╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║ Stage {stage_idx}/{total_stages}: {name.upper():<54} ║")
    print(f"║ Sources: {str(sources):<57} ║")
    print(f"║ Epochs: {epochs:<58} ║")
    print(f"║ Learning Rate: {lr:<51} ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")

    train_pieces = filter_by_source(all_train_pieces, sources)
    val_pieces = filter_by_source(all_val_pieces, sources)
    if not train_pieces:
        print(f"[train] stage '{name}': no training pieces for sources={sources} — skipping")
        return model

    # DataLoaders now use multiple workers for parallel data processing
    train_loader = get_dataloader(train_pieces, seq_len=seq_len, stride=stride,
                                   batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = get_dataloader(val_pieces, seq_len=seq_len, stride=stride,
                                 batch_size=batch_size, shuffle=False, num_workers=4) if val_pieces else None

    # Safely get original model parameters if wrapped in DataParallel
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = Adam(raw_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_val = float("inf")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\n--- [Status] Stage '{name}' | Epoch {epoch}/{epochs} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device)
            print(f"✔ Status: Epoch {epoch}/{epochs} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val:
                best_val = val_loss
                save_model = raw_model if isinstance(model, nn.DataParallel) else model
                torch.save(save_model.state_dict(), checkpoint_dir / f"{name}_best.pt")
                print(f"💾 New best validation model saved for stage '{name}'!")
        else:
            print(f"✔ Status: Epoch {epoch}/{epochs} Complete | Train Loss: {train_loss:.4f} (No validation data)")

    final_path = checkpoint_dir / f"{name}_final.pt"
    save_model = raw_model if isinstance(model, nn.DataParallel) else model
    torch.save(save_model.state_dict(), final_path)
    print(f"🎯 Stage '{name}' completed successfully — checkpoint saved to {final_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed_sequences.pkl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--resume-from", type=Path, default=None,
                         help="optional checkpoint to load before stage 1 (e.g. resuming a run)")
    parser.add_argument("--no-parallel", action="store_true", 
                         help="Disable multi-GPU DataParallel even if multiple GPUs are available")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpus = torch.cuda.device_count()
        print(f"[train] Found {n_gpus} CUDA device(s) available.")
    else:
        device = torch.device("cpu")
        n_gpus = 0
        print(f"[train] Using device: CPU")

    pieces = load_pieces(args.input)
    train_pieces, val_pieces = split_pieces(pieces)

    model = NoteTransformer(
        d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        max_len=max(2048, args.seq_len + 1),
    ).to(device)

    # Wrap model for multi-GPU training
    if n_gpus > 1 and not args.no_parallel:
        print(f"[train] Wrapping model with nn.DataParallel across {n_gpus} GPUs.")
        model = nn.DataParallel(model)
    elif n_gpus > 1:
        print(f"[train] Multi-GPU detected, but --no-parallel flag was passed. Running on a single device.")

    if args.resume_from is not None:
        state_dict = torch.load(args.resume_from, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        print(f"[train] Resumed weights from {args.resume_from}")

    total_stages = len(DEFAULT_STAGES)
    for idx, stage in enumerate(DEFAULT_STAGES, start=1):
        model = run_stage(
            model, stage, train_pieces, val_pieces, device,
            seq_len=args.seq_len, stride=args.stride, batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir, stage_idx=idx, total_stages=total_stages
        )

    print("\n🎉 All curriculum stages complete!")


if __name__ == "__main__":
    main()