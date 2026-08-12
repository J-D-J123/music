"""Post-hoc pitch correction for the audio-transcriber pipeline, using the
trained NoteTransformer from train.py as a learned "does this note make
musical sense here" check — same idea as intonation.py's key-based
correction, but driven by the model instead of hand-written key theory, and
narrow in the same way: only notes the model is both confident *and*
disagreeing about get touched, everything else is left alone.

How it works: convert the note sequence to the exact same pitch-token stream
used at training time (reusing data_prep.py's tokenizer logic so training and
inference can't silently drift apart), run it through the model with teacher
forcing, and read off — for every note — how likely the model thought THAT
pitch was, given everything before it. A note gets flagged only when the
model was both (a) genuinely surprised by the actual pitch (low log-prob) and
(b) confident about one specific alternative (much higher log-prob) —
mirroring intonation.py's "only act when the evidence breaks the tie
cleanly" rule.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from pipeline.ml.data_prep import PAD, BOS, EOS, REST, PITCH_OFFSET, MIDI_MIN, MIDI_MAX
from pipeline.ml.model import NoteTransformer

# How surprised the model has to be at the note it actually saw before it's
# even a candidate — natural-log probability, so -4.0 ~= "the model gave this
# pitch under ~2% probability."
SURPRISE_LOGPROB_THRESHOLD = -4.0

# How much better the model's favourite alternative has to be, in nats,
# before it's trusted over the measured pitch. This is the "breaks the tie
# cleanly" guard — a note that's merely somewhat unlikely, with no strong
# alternative, is left alone exactly like intonation.py leaves ambiguous-
# but-tied notes.
MIN_CONFIDENCE_MARGIN = 2.0

# Must match whatever --d-model/--nhead/--num-layers/--seq-len (via max_len)
# were actually passed to train.py — these aren't stored in the checkpoint,
# only the weights are. Defaults here match train.py's own CLI defaults.
DEFAULT_MODEL_KWARGS = dict(d_model=256, nhead=8, num_layers=6, max_len=2048)

_MODEL_CACHE: dict[str, tuple[NoteTransformer, torch.device]] = {}


def load_model(checkpoint_path: Path, device: torch.device | None = None,
                use_cache: bool = True, **model_kwargs) -> tuple[NoteTransformer, torch.device]:
    """Loads once and reuses across calls (keyed by checkpoint path) since
    run.py processes tracks in a loop — avoids re-reading weights from disk
    and rebuilding the model every single track."""
    key = str(checkpoint_path)
    if use_cache and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {**DEFAULT_MODEL_KWARGS, **model_kwargs}
    model = NoteTransformer(**kwargs).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            f"checkpoint at {checkpoint_path} doesn't match this architecture "
            f"({kwargs}) — if you trained with non-default --d-model/--nhead/"
            f"--num-layers/--seq-len, pass the same values to load_model()."
        ) from e

    model.eval()
    print(f"[correct_notes] loaded {checkpoint_path} onto {device}")

    if use_cache:
        _MODEL_CACHE[key] = (model, device)
    return model, device


def _tokens_with_note_indices(notes: list[dict]) -> tuple[list[int], list[int | None]]:
    """Mirrors data_prep.py's _tokens_from_note_events EXACTLY (same BOS/
    REST/EOS logic), but also returns, per token, which index into `notes`
    it came from (None for BOS/REST/EOS tokens) — needed to write
    corrections back onto the original note list. If that tokenizer's logic
    ever changes, this needs to change with it or train/inference will
    silently drift apart.

    Note on timing units: this only checks whether a gap is > 0, never its
    size — so it works the same whether `start`/`end` are in seconds
    (pre-quantize, real pipeline usage) or quarter-lengths (training data).
    """
    if not notes:
        return [], []

    tokens = [BOS]
    indices: list[int | None] = [None]
    prev_end = notes[0]["start"]

    for i, n in enumerate(notes):
        gap = n["start"] - prev_end
        if gap > 0:
            tokens.append(REST)
            indices.append(None)
        pitch = max(MIDI_MIN, min(MIDI_MAX, n["pitch"]))
        tokens.append(pitch + PITCH_OFFSET)
        indices.append(i)
        prev_end = n["end"]

    tokens.append(EOS)
    indices.append(None)
    return tokens, indices


def score_notes(notes: list[dict], model: NoteTransformer, device: torch.device,
                 max_len: int = 2048) -> list[dict]:
    """Returns one dict per scoreable note (skips the very first note — there's
    no context to judge it against):
        {'index': idx into `notes`, 'pitch': measured pitch,
         'actual_logprob': how likely the model thought the real pitch was,
         'suggested_pitch': the model's favourite pitch at that position,
         'suggested_logprob': that pitch's log-prob}
    """
    tokens, note_idx_by_token = _tokens_with_note_indices(notes)
    if len(tokens) < 2:
        return []

    if len(tokens) > max_len:
        print(f"[correct_notes] piece has {len(tokens)} tokens > max_len={max_len} "
              f"— truncating to the first {max_len} (positional encoding can't "
              f"cover the rest)")
        tokens = tokens[:max_len]
        note_idx_by_token = note_idx_by_token[:max_len]

    input_ids = torch.tensor(tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
    targets = tokens[1:]
    target_note_idx = note_idx_by_token[1:]  # aligned with `targets`

    with torch.no_grad():
        logits = model(input_ids)  # [1, T-1, vocab]
        log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # [T-1, vocab]

    # Restrict "alternative suggestion" search to real pitch tokens only —
    # suggesting BOS/EOS/PAD/REST as a "corrected note" would be nonsense.
    pitch_lo, pitch_hi = PITCH_OFFSET, PITCH_OFFSET + (MIDI_MAX - MIDI_MIN)

    results = []
    for t, note_i in enumerate(target_note_idx):
        if note_i is None:
            continue  # this position's target was REST/EOS, not a note

        actual_token = targets[t]
        actual_lp = log_probs[t, actual_token].item()

        pitch_logp = log_probs[t, pitch_lo:pitch_hi + 1]
        best_offset = int(torch.argmax(pitch_logp).item())
        suggested_token = pitch_lo + best_offset
        suggested_lp = pitch_logp[best_offset].item()

        results.append({
            "index": note_i,
            "pitch": notes[note_i]["pitch"],
            "actual_logprob": actual_lp,
            "suggested_pitch": suggested_token - PITCH_OFFSET,
            "suggested_logprob": suggested_lp,
        })

    return results


def apply_corrections(notes: list[dict], scored: list[dict],
                       surprise_threshold: float = SURPRISE_LOGPROB_THRESHOLD,
                       min_margin: float = MIN_CONFIDENCE_MARGIN) -> list[dict]:
    """Mutates and returns `notes`: overwrites 'pitch' in place on notes
    where the model was both surprised by the measured pitch AND clearly
    prefers a specific alternative. Everything else is left untouched."""
    changed = 0
    for s in scored:
        if s["pitch"] == s["suggested_pitch"]:
            continue  # model agrees with what was measured — nothing to do
        margin = s["suggested_logprob"] - s["actual_logprob"]
        if s["actual_logprob"] < surprise_threshold and margin >= min_margin:
            notes[s["index"]]["pitch"] = s["suggested_pitch"]
            changed += 1

    print(f"[correct_notes] {len(scored)} note(s) scored, {changed} corrected "
          f"(surprise_threshold={surprise_threshold}, min_margin={min_margin})")
    return notes


def correct_pitches(notes: list[dict], checkpoint_path: Path,
                     device: torch.device | None = None, **model_kwargs) -> list[dict]:
    """One-call convenience for run.py: load (cached) model, score, apply."""
    model, device = load_model(checkpoint_path, device=device, **model_kwargs)
    scored = score_notes(notes, model, device)
    return apply_corrections(notes, scored)