"""Smooths a note sequence's pitches using Viterbi decoding over a simple
melodic transition model: small intervals are common, large leaps are rare.
This is a real HMM-style correction, not key-snapping — it stays valid on
chromatic pieces (like Hungarian Dance) where forcing notes into a scale
would break intentionally chromatic passages."""
import numpy as np

CANDIDATE_RANGE = 3  # consider pitches within +/- this many semitones of the original guess
LEAP_PENALTY_SCALE = 0.15  # higher = more strongly prefers small, stepwise motion


def _transition_log_prob(pitch_a: int, pitch_b: int) -> float:
    interval = abs(pitch_a - pitch_b)
    # Smooth, monotonically decreasing preference for small intervals —
    # not a hard cutoff, so genuine leaps are still allowed when confidence supports them.
    return -LEAP_PENALTY_SCALE * interval


def smooth_pitches(notes: list[dict]) -> list[dict]:
    """notes: list of dicts with 'pitch' (int) and 'velocity' (used as a
    confidence proxy, 0-127). Returns notes with 'pitch' possibly adjusted."""
    if len(notes) < 2:
        return notes

    # Build candidate pitches + observation log-probabilities per note
    candidates = []
    for n in notes:
        base = n["pitch"]
        conf = max(n.get("velocity", 64) / 127.0, 0.01)
        obs = []
        for offset in range(-CANDIDATE_RANGE, CANDIDATE_RANGE + 1):
            cand_pitch = base + offset
            # Confidence peaks at the original guess, falls off for nearby alternatives
            cand_conf = conf * np.exp(-0.5 * (offset / 1.5) ** 2)
            obs.append((cand_pitch, np.log(max(cand_conf, 1e-6))))
        candidates.append(obs)

    n_notes = len(candidates)
    n_cands = len(candidates[0])

    dp = np.full((n_notes, n_cands), -np.inf)
    backptr = np.zeros((n_notes, n_cands), dtype=int)

    for c in range(n_cands):
        dp[0][c] = candidates[0][c][1]

    for t in range(1, n_notes):
        for c in range(n_cands):
            pitch_b = candidates[t][c][0]
            best_score = -np.inf
            best_prev = 0
            for c_prev in range(n_cands):
                pitch_a = candidates[t - 1][c_prev][0]
                score = dp[t - 1][c_prev] + _transition_log_prob(pitch_a, pitch_b)
                if score > best_score:
                    best_score = score
                    best_prev = c_prev
            dp[t][c] = best_score + candidates[t][c][1]
            backptr[t][c] = best_prev

    path = [int(np.argmax(dp[-1]))]
    for t in range(n_notes - 1, 0, -1):
        path.append(backptr[t][path[-1]])
    path.reverse()

    changed = 0
    for i, c in enumerate(path):
        new_pitch = candidates[i][c][0]
        if new_pitch != notes[i]["pitch"]:
            changed += 1
        notes[i]["pitch"] = new_pitch

    print(f"[pitch_smoothing] adjusted {changed} of {n_notes} notes via Viterbi smoothing")
    return notes