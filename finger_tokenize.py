"""Convert finger flow sequences to discrete tokens for transformer pretraining."""

import numpy as np

# ── Vocabulary ────────────────────────────────────────────────────────────────

FINGER_SCRATCH = 5          # scratch treated as a 6th "finger"
LANE_SCRATCH   = 0          # scratch lane token

NUM_FINGERS    = 6           # 0-4 = hand fingers, 5 = scratch
NUM_LANES      = 8           # 0 = scratch, 1-7 = key lanes
DURATION_BINS  = 32

# Duration is measured in real seconds, not beats: physical execution pacing
# and finger-recovery time are governed by wall-clock time, not musical time
# (consistent with dp_fingering.py's transition cost, which is also time- not
# beat-normalized — a fast passage is harder regardless of its beat label).
# MAX_DUR_SEC doubles as both the quantization curve's scale and the
# physiological rest cap: a finger is fully recovered well within this many
# seconds of idleness, so longer waits are indistinguishable from it.
MAX_DUR_SEC = 2.0

# Special tokens appended after normal vocab
MASK_FINGER  = NUM_FINGERS      # 6
MASK_LANE    = NUM_LANES        # 8
PAD_FINGER   = NUM_FINGERS + 1  # 7
PAD_LANE     = NUM_LANES + 1    # 9

FINGER_VOCAB = NUM_FINGERS + 2  # 8  (0-5 normal, 6 mask, 7 pad)
LANE_VOCAB   = NUM_LANES   + 2  # 10 (0-7 normal, 8 mask, 9 pad)


# ── Duration quantization ─────────────────────────────────────────────────────

def quantize_duration(duration_sec: float) -> int:
    """Log-quantize a real-time duration (seconds) into [0, DURATION_BINS-1]."""
    if duration_sec <= 0.0:
        return 0
    log_val = np.log1p(min(duration_sec, MAX_DUR_SEC)) / np.log1p(MAX_DUR_SEC)
    return min(int(log_val * DURATION_BINS), DURATION_BINS - 1)


def dequantize_duration(bin_idx: int) -> float:
    """Inverse of quantize_duration — returns approximate seconds."""
    t = bin_idx / DURATION_BINS
    return float(np.expm1(t * np.log1p(MAX_DUR_SEC)))


# ── Flow → token array ────────────────────────────────────────────────────────

def flow_to_tokens(flow: list) -> np.ndarray:
    """
    Convert a list[TimestepFlow] to a (N, 3) int16 token array.

    Each row: [finger_id, lane, dur_bin]
    - finger_id 0-4: hand fingers (PINKY..THUMB); 5: scratch
    - lane 1-7: key lanes; 0: scratch
    - dur_bin: log-quantized real seconds since this finger last fired

    Simultaneous tokens within a timestep are ordered by lane (low to high).
    Scratch token is emitted first within its timestep when present.
    """
    tokens: list[tuple[int, int, int]] = []

    for ts in flow:
        step: list[tuple[int, int, int]] = []

        if ts.scratch:
            gap_sec = ts.time_since_last_scratch or 0.0
            step.append((FINGER_SCRATCH, LANE_SCRATCH, quantize_duration(gap_sec)))

        active = sorted(
            (fs.lane, fs.finger, fs.duration_since_last)
            for fs in ts.fingers
            if fs.is_active
        )
        for lane, finger, dur_sec in active:
            step.append((finger, lane, quantize_duration(dur_sec)))

        tokens.extend(step)

    if not tokens:
        return np.zeros((0, 3), dtype=np.int16)
    return np.array(tokens, dtype=np.int16)
