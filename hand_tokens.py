"""
Tokenise a DP chart into a sequence of two-hand positions.

Design rationale (measured, not assumed)
----------------------------------------
Over 217k timesteps of lv12 charts:

  * only 121 distinct hand positions occur; 19 cover 95% of timesteps
  * given the set of lanes pressed, the inferred fingering is the *default*
    (isolated cost-minimal) choice 95.5% of the time

So the finger assignment is very nearly a deterministic function of the notes,
and spending a token field on it wastes capacity. What is worth keeping is the
4.5% residual: the timesteps where surrounding context — tempo, neighbouring
notes, a chord pinning other fingers — forces a non-default fingering. That is
the 運指変更 signal, and one bit captures all of it.

Hence the token layout below: a discrete position id, a one-bit deviation flag,
and timing. The finger identities themselves are recoverable from the position
id whenever the flag is clear.

Both hands share one timeline. P1 and P2 positions sit in the same row rather
than being encoded separately and concatenated downstream, so simultaneity
across hands — the DP-specific skill — is visible to a model reading the rows.

Token layout (one row per unified timestep)
-------------------------------------------
    p1_pos    uint8   position id, 0 = this hand presses nothing here
    p2_pos    uint8   position id, 0 = this hand presses nothing here
    p1_dev    uint8   1 = fingering deviates from the isolated default
    p2_dev    uint8   1 = fingering deviates from the isolated default
    dur_bin   uint8   log-quantised seconds since the previous timestep
    times_sec float32 absolute time, kept for windowing and diagnostics

Position id encoding
--------------------
    id = scratch_bit << 7 | lane_bitmask        (bit i-1 set <=> local lane i)
    id 0 is reserved for "no press", so a bare scratch is id 128.
Local lanes are hand-local (1 = pinky side, 7 = thumb side on both hands), so a
pattern and its P1/P2 mirror produce identical ids.
"""

import numpy as np
from functools import lru_cache
from pathlib import Path

from chart_io import load_chart, split_hands, NoteEvent
from finger_model import is_scratch, to_local
from dp_fingering import _enumerate_hand_states, _state_cost, infer_fingering

# ── vocabulary ────────────────────────────────────────────────────────────────

POSITION_VOCAB = 256          # 7 lane bits + 1 scratch bit
NO_PRESS       = 0

DURATION_BINS = 32
MAX_DUR_SEC   = 2.0           # beyond this a hand is fully rested; bins saturate


def position_id(local_lanes, scratch: bool) -> int:
    """Pack a hand position into a single id. See module docstring."""
    mask = 0
    for lane in local_lanes:
        mask |= 1 << (lane - 1)
    return (int(scratch) << 7) | mask


def decode_position(pid: int) -> tuple[list[int], bool]:
    """Inverse of position_id — returns (local lanes, scratch)."""
    scratch = bool(pid >> 7)
    mask = pid & 0x7F
    return [i + 1 for i in range(7) if mask & (1 << i)], scratch


def quantize_duration(seconds: float) -> int:
    """Log-quantise a real-time gap into [0, DURATION_BINS-1]."""
    if seconds <= 0.0:
        return 0
    v = np.log1p(min(seconds, MAX_DUR_SEC)) / np.log1p(MAX_DUR_SEC)
    return min(int(v * DURATION_BINS), DURATION_BINS - 1)


# ── default (context-free) fingering, for the deviation bit ───────────────────

@lru_cache(maxsize=None)
def isolated_fingering(local_lanes: tuple[int, ...]) -> tuple:
    """
    The fingering this lane set would get with no surrounding context — the
    static cost minimum. Comparing the contextual Viterbi choice against this is
    what the deviation bit reports.
    """
    states = _enumerate_hand_states(list(local_lanes)) if local_lanes else [()]
    best, best_cost = (), float('inf')
    for s in states:
        c = _state_cost(s, has_scratch=False)
        if c < best_cost:
            best, best_cost = s, c
    return tuple(sorted((a.finger, a.lane) for a in best))


# ── per-hand pass ─────────────────────────────────────────────────────────────

def _hand_timeline(events: list[NoteEvent]) -> dict[int, tuple[int, int]]:
    """
    Run the fingering selector over one hand and return {row: (pos_id, dev)}.

    The cost model is used here and only here — to choose a plausible fingering.
    Its numeric output is deliberately discarded; measured against difficulty it
    carries little (rho ~= 0.25) and it is not what this representation encodes.
    """
    if not events:
        return {}
    _, path = infer_fingering(events, k_best=1)[0]

    out: dict[int, tuple[int, int]] = {}
    for state, group in path:
        scratch = any(is_scratch(e.lane) for e in group)
        lanes   = sorted({to_local(e.lane) for e in group if not is_scratch(e.lane)})
        chosen  = tuple(sorted((a.finger, a.lane) for a in state))
        dev     = int(chosen != isolated_fingering(tuple(lanes)))
        out[group[0].row] = (position_id(lanes, scratch), dev)
    return out


# ── chart pass ────────────────────────────────────────────────────────────────

def tokenize_chart(npy_path: str | Path) -> dict:
    """
    Tokenise one chart into a two-hand position sequence.

    Returns arrays of equal length T, one row per unified timestep (a row where
    either hand presses something).
    """
    arr, bpm, _ = load_chart(npy_path)
    p1_events, p2_events = split_hands(arr, bpm)

    p1 = _hand_timeline(p1_events)
    p2 = _hand_timeline(p2_events)

    row_time = {e.row: e.time_sec for e in p1_events}
    row_time.update({e.row: e.time_sec for e in p2_events})
    rows = sorted(set(p1) | set(p2))

    if not rows:
        return {k: np.empty(0, dtype=d) for k, d in
                (('p1_pos', np.uint8), ('p2_pos', np.uint8),
                 ('p1_dev', np.uint8), ('p2_dev', np.uint8),
                 ('dur_bin', np.uint8), ('times_sec', np.float32))}

    times = np.array([row_time[r] for r in rows], dtype=np.float64)
    gaps  = np.zeros(len(rows), dtype=np.float64)
    gaps[1:] = np.diff(times)

    return {
        'p1_pos':    np.array([p1.get(r, (NO_PRESS, 0))[0] for r in rows], dtype=np.uint8),
        'p2_pos':    np.array([p2.get(r, (NO_PRESS, 0))[0] for r in rows], dtype=np.uint8),
        'p1_dev':    np.array([p1.get(r, (NO_PRESS, 0))[1] for r in rows], dtype=np.uint8),
        'p2_dev':    np.array([p2.get(r, (NO_PRESS, 0))[1] for r in rows], dtype=np.uint8),
        'dur_bin':   np.array([quantize_duration(g) for g in gaps], dtype=np.uint8),
        'times_sec': times.astype(np.float32),
    }


# ── cache builder ─────────────────────────────────────────────────────────────

def build_cache(chart_dir: str, output_dir: str, overwrite: bool = False) -> None:
    """Tokenise every chart in chart_dir, writing one .npz per chart."""
    src, dst = Path(chart_dir).expanduser(), Path(output_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob('*.npy'))
    done = skipped = failed = 0
    print(f'Tokenising {len(paths)} charts  ->  {dst}')
    for p in paths:
        out = dst / (p.stem + '.npz')
        if out.exists() and not overwrite:
            skipped += 1
            continue
        try:
            np.savez_compressed(out, **tokenize_chart(p))
            done += 1
        except Exception as e:
            print(f'  ERR {p.name}: {e}')
            failed += 1
    print(f'  done={done}  skipped={skipped}  failed={failed}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chart-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    build_cache(a.chart_dir, a.output_dir, a.overwrite)
