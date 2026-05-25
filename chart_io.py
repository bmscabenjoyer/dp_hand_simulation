"""Load IIDX DP chart arrays and extract per-hand note event sequences."""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

ROWS_PER_BAR  = 192
ROWS_PER_BEAT = 48
BPM_COL       = 16   # column 16 stores BPM × 100

P1_LANES  = list(range(0, 8))   # scratch + keys 1–7
P2_LANES  = list(range(8, 16))  # keys 8–14 + scratch

# Note values that represent an active head event (requires motor action)
HEAD_VALUES = frozenset({1, 2, 4, 6, 8})


@dataclass
class NoteEvent:
    row:   int          # absolute row in the chart array
    lane:  int          # global lane (0–15)
    value: int          # raw note value (1–9)
    bpm:   float        # BPM at this row
    time_sec: float     # absolute time in seconds from chart start


def load_chart(npy_path: str | Path):
    """
    Returns
    -------
    arr      : (total_rows, 17) uint32 numpy array
    bpm      : (total_rows,) float32 — fill-forward BPM per row
    meta     : dict from JSON sidecar
    """
    npy_path = Path(npy_path)
    arr = np.load(npy_path)

    # Fill-forward BPM from column 16 (stored as BPM × 100)
    bpm_raw = arr[:, BPM_COL].astype(np.float32)
    bpm = np.zeros(len(arr), dtype=np.float32)
    current = 0.0
    for i, v in enumerate(bpm_raw):
        if v > 0:
            current = v / 100.0
        bpm[i] = current

    json_path = npy_path.with_suffix('.json')
    meta = {}
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)

    return arr, bpm, meta


def row_to_seconds(row: int, bpm: np.ndarray) -> float:
    """Integrate BPM to convert a row index to absolute time in seconds."""
    if row == 0:
        return 0.0
    # Each row is 1/48 of a beat; dt per row = 60 / (bpm * 48)
    dt = 60.0 / (bpm[:row] * 48.0)
    return float(dt.sum())


def extract_events(arr: np.ndarray, bpm: np.ndarray,
                   lanes: list[int]) -> list[NoteEvent]:
    """
    Extract head note events from the given lanes, sorted by row.
    Scratch lanes (0, 15) are included if present in lanes.
    """
    events = []
    for row in range(arr.shape[0]):
        for lane in lanes:
            v = int(arr[row, lane])
            if v in HEAD_VALUES:
                events.append(NoteEvent(
                    row=row,
                    lane=lane,
                    value=v,
                    bpm=float(bpm[row]),
                    time_sec=row_to_seconds(row, bpm),
                ))
    return events


def split_hands(arr: np.ndarray, bpm: np.ndarray):
    """Return (p1_events, p2_events) sorted by row."""
    p1 = extract_events(arr, bpm, P1_LANES)
    p2 = extract_events(arr, bpm, P2_LANES)
    return p1, p2
