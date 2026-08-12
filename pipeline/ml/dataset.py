"""Stage 2: PyTorch Dataset + DataLoader for next-note prediction.

Takes the pickle written by data_prep.py (a list of {"source", "file",
"tokens"} dicts) and produces fixed-length (X, Y) windows: Y is X shifted by
one timestep, per the rolling-window formatting already decided on. Splitting
happens per PIECE, not per window, so a validation window can never share a
piece with a training window.
"""
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from pipeline.ml.data_prep import PAD, VOCAB_SIZE  # keep one source of truth for the vocab

DEFAULT_SEQ_LEN = 64
DEFAULT_STRIDE = 32  # 50% overlap between consecutive windows from the same piece


@dataclass
class Piece:
    source: str
    file: str
    tokens: list[int]


def load_pieces(pickle_path: Path) -> list[Piece]:
    with open(pickle_path, "rb") as f:
        raw = pickle.load(f)
    pieces = [Piece(**p) for p in raw]
    print(f"[dataset] loaded {len(pieces)} pieces from {pickle_path}")
    return pieces


def split_pieces(pieces: list[Piece], val_fraction: float = 0.1,
                  seed: int = 0) -> tuple[list[Piece], list[Piece]]:
    """Piece-level split so no window in val was ever adjacent to a training
    window from the same piece."""
    shuffled = list(pieces)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if shuffled else 0
    val, train = shuffled[:n_val], shuffled[n_val:]
    print(f"[dataset] split: {len(train)} train pieces, {len(val)} val pieces")
    return train, val


def filter_by_source(pieces: list[Piece], sources: set[str]) -> list[Piece]:
    """Curriculum-stage helper — e.g. filter_by_source(pieces, {'maestro'})
    for the pretraining stage, {'violin'} for domain adaptation, etc."""
    kept = [p for p in pieces if p.source in sources]
    print(f"[dataset] filtered to sources={sources}: {len(kept)} of {len(pieces)} pieces")
    return kept


class NextNoteDataset(Dataset):
    """Sliding-window dataset over a list of Pieces.

    Each __getitem__ returns:
        input_ids:  LongTensor[seq_len]      — tokens[i : i+seq_len]
        target_ids: LongTensor[seq_len]      — tokens[i+1 : i+seq_len+1]
        loss_mask:  BoolTensor[seq_len]      — True where target_ids != PAD

    Pieces shorter than seq_len+1 are right-padded with PAD to exactly that
    length and yield a single window, rather than being dropped — short
    pieces (e.g. brief etudes) are still useful signal, they're just fully
    masked past their real content.
    """

    def __init__(self, pieces: list[Piece], seq_len: int = DEFAULT_SEQ_LEN,
                 stride: int = DEFAULT_STRIDE):
        self.pieces = pieces
        self.seq_len = seq_len
        self.stride = stride
        self.index: list[tuple[int, int]] = []  # (piece_idx, window_start)

        for pi, piece in enumerate(pieces):
            n = len(piece.tokens)
            if n < 2:
                continue  # can't form even one (input, target) pair
            if n <= seq_len:
                self.index.append((pi, 0))
                continue
            # need window_start .. window_start + seq_len + 1 <= n
            last_start = n - (seq_len + 1)
            for start in range(0, last_start + 1, stride):
                self.index.append((pi, start))
            # make sure the tail of the piece is always covered even if
            # `stride` overshoots it
            if (last_start % stride) != 0:
                self.index.append((pi, last_start))

        print(f"[dataset] built {len(self.index)} windows "
              f"(seq_len={seq_len}, stride={stride}) from {len(pieces)} pieces")

    def __len__(self) -> int:
        return len(self.index)

    def _padded_window(self, tokens: list[int], start: int):
        """Grab seq_len+1 tokens starting at `start`, padding on the right
        with PAD if the piece runs out."""
        chunk = tokens[start:start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [PAD] * (self.seq_len + 1 - len(chunk))
        return chunk

    def __getitem__(self, idx: int):
        piece_idx, start = self.index[idx]
        tokens = self.pieces[piece_idx].tokens
        chunk = self._padded_window(tokens, start)

        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        loss_mask = target_ids != PAD

        return input_ids, target_ids, loss_mask


def get_dataloader(pieces: list[Piece], seq_len: int = DEFAULT_SEQ_LEN,
                    stride: int = DEFAULT_STRIDE, batch_size: int = 64,
                    shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    ds = NextNoteDataset(pieces, seq_len=seq_len, stride=stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, drop_last=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed_sequences.pkl"))
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pieces = load_pieces(args.input)
    train_pieces, val_pieces = split_pieces(pieces)

    train_loader = get_dataloader(train_pieces, seq_len=args.seq_len,
                                   stride=args.stride, batch_size=args.batch_size)
    val_loader = get_dataloader(val_pieces, seq_len=args.seq_len,
                                 stride=args.stride, batch_size=args.batch_size,
                                 shuffle=False)

    x, y, mask = next(iter(train_loader))
    print(f"[dataset] sample batch: input_ids={tuple(x.shape)} "
          f"target_ids={tuple(y.shape)} loss_mask={tuple(mask.shape)} "
          f"vocab_size={VOCAB_SIZE}")
