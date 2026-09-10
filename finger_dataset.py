"""PyTorch Dataset for masked finger token pretraining."""

import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from chart_io import load_chart, split_hands
from dp_fingering import infer_fingering
from simulate import finger_flow
from finger_tokenize import (
    flow_to_tokens,
    MASK_FINGER, MASK_LANE,
    DURATION_BINS,
)


class FingerTokenDataset(Dataset):
    """
    Sliding-window dataset over finger token sequences extracted from charts.

    Each item is a masked window: finger+lane are masked at random positions,
    duration bins remain visible. The model predicts the masked finger and lane.

    Parameters
    ----------
    chart_dir   : directory containing *.npy + *.json chart pairs
    window      : number of tokens per training window
    stride      : sliding window stride
    mask_rate   : fraction of tokens to mask
    min_seq_len : discard sequences shorter than this
    """

    def __init__(
        self,
        chart_dir: str,
        window: int = 64,
        stride: int = 32,
        mask_rate: float = 0.15,
        min_seq_len: int = 64,
    ):
        self.window    = window
        self.stride    = stride
        self.mask_rate = mask_rate

        self.sequences: list[np.ndarray] = _build_sequences(
            chart_dir, min_seq_len=min_seq_len
        )

        # Flat index of (sequence_idx, start_token)
        self._index: list[tuple[int, int]] = []
        for si, seq in enumerate(self.sequences):
            for start in range(0, len(seq) - window + 1, stride):
                self._index.append((si, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        si, start = self._index[idx]
        window = self.sequences[si][start : start + self.window].copy()  # (W, 3)

        n_mask = max(1, int(self.mask_rate * self.window))
        mask_pos = np.random.choice(self.window, n_mask, replace=False)

        label_finger = np.full(self.window, -100, dtype=np.int64)
        label_lane   = np.full(self.window, -100, dtype=np.int64)
        label_finger[mask_pos] = window[mask_pos, 0]
        label_lane[mask_pos]   = window[mask_pos, 1]

        window[mask_pos, 0] = MASK_FINGER
        window[mask_pos, 1] = MASK_LANE

        return {
            "input_ids":    torch.from_numpy(window.astype(np.int64)),
            "label_finger": torch.from_numpy(label_finger),
            "label_lane":   torch.from_numpy(label_lane),
        }


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _build_sequences(
    chart_dir: str,
    min_seq_len: int = 64,
) -> list[np.ndarray]:
    """
    Load all charts in chart_dir, run Viterbi DP + finger_flow, tokenize.
    Returns one token array per hand per chart that meets min_seq_len.
    """
    sequences = []
    paths = sorted(Path(chart_dir).expanduser().glob("*.npy"))
    skipped = 0

    for npy_path in tqdm(paths, desc="preprocessing charts"):
        try:
            arr, bpm_arr, meta = load_chart(npy_path)
        except Exception:
            skipped += 1
            continue

        p1_events, p2_events = split_hands(arr, bpm_arr)

        for events in (p1_events, p2_events):
            if len(events) < 10:
                continue
            try:
                _, path = infer_fingering(events, k_best=1)[0]
            except Exception:
                skipped += 1
                continue

            flow = finger_flow(path)
            tokens = flow_to_tokens(flow)

            if len(tokens) >= min_seq_len:
                sequences.append(tokens)

    print(f"  Built {len(sequences)} sequences ({skipped} charts skipped)")
    return sequences
