"""
Run the finger simulation on a chart and print per-bar difficulty features.

Usage:
    python simulate.py path/to/chart.npy [--bars START END] [--k 3]
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

from chart_io import load_chart, split_hands, ROWS_PER_BAR
from dp_fingering import infer_fingering
from finger_model import FINGER_NAMES, HOME_A, is_scratch


# ── Status flow ───────────────────────────────────────────────────────────────

@dataclass
class FingerState:
    finger: int
    lane: int                    # active lane, or home lane (1/2/4/6/7) if idle
    is_active: bool
    duration_since_last: float   # seconds since this finger last played; 0.0 at start


@dataclass
class TimestepFlow:
    time_sec: float
    row: int
    bpm: float                           # local BPM at this timestep (for tempo-aware tokenization)
    scratch: bool
    scratch_key_lanes: list[int]         # local lanes active simultaneously with scratch
    time_since_last_scratch: float | None  # None until first scratch occurs
    fingers: list[FingerState]           # all 5 fingers, active or idle


def finger_flow(path: list[tuple]) -> list[TimestepFlow]:
    """
    Convert a Viterbi path into a per-timestep finger status flow.

    Idle fingers rest at Home A (pinky=1, ring=2, middle=4, index=6, thumb=7).
    Scratch timesteps record the local key lanes active at the same time.
    Every timestep after a scratch records time elapsed since that scratch.
    """
    last_active: dict[int, float | None] = {f: None for f in range(5)}
    last_scratch_time: float | None = None
    records: list[TimestepFlow] = []

    for state, events in path:
        t = events[0].time_sec
        row = events[0].row
        has_scratch = any(is_scratch(ev.lane) for ev in events)
        active_by_finger = {a.finger: a for a in state}

        finger_states: list[FingerState] = []
        for f in range(5):
            if f in active_by_finger:
                a = active_by_finger[f]
                lane = a.lane
                is_active = True
            else:
                lane = HOME_A[f]
                is_active = False
            dur = (t - last_active[f]) if last_active[f] is not None else 0.0
            if is_active:
                last_active[f] = t
            finger_states.append(FingerState(
                finger=f, lane=lane, is_active=is_active, duration_since_last=dur,
            ))

        tsls = (t - last_scratch_time) if last_scratch_time is not None else None

        if has_scratch:
            scratch_key_lanes = sorted(lane for a in state for lane in a.covered_lanes)
            last_scratch_time = t
        else:
            scratch_key_lanes = []

        records.append(TimestepFlow(
            time_sec=t,
            row=row,
            bpm=events[0].bpm,
            scratch=has_scratch,
            scratch_key_lanes=scratch_key_lanes,
            time_since_last_scratch=tsls,
            fingers=finger_states,
        ))

    return records


def bar_of(row: int) -> int:
    return row // ROWS_PER_BAR


def classify_transition(prev_state, next_state, time_delta_sec: float) -> str:
    """Classify a hand state transition into a motor vocabulary label."""
    if prev_state is None or not next_state:
        return 'start'

    prev_lanes = set()
    for a in prev_state:
        prev_lanes.update(a.covered_lanes)
    next_lanes = set()
    for a in next_state:
        next_lanes.update(a.covered_lanes)

    same_lanes = (prev_lanes == next_lanes)
    is_jack    = same_lanes and (
        {a.finger for a in prev_state} == {a.finger for a in next_state}
    )

    if is_jack:
        return 'jack'

    if len(next_lanes) > 1 and len(prev_lanes) > 1 and next_lanes == prev_lanes:
        return 'chord_repeat'

    lane_list = sorted(next_lanes)
    if len(lane_list) >= 2:
        diffs = [lane_list[i+1] - lane_list[i] for i in range(len(lane_list)-1)]
        if all(d == 1 for d in diffs):
            return 'chord_adjacent'
        return 'chord'

    if len(next_lanes) == 1 and len(prev_lanes) == 1:
        dl = list(next_lanes)[0] - list(prev_lanes)[0]
        if dl == 1:
            return 'roll_up'
        if dl == -1:
            return 'roll_down'
        if dl == 0:
            return 'jack'
        return 'jump'

    return 'mixed'


def print_bar_summary(bar_features: dict, total_bars: int):
    print(f"\n{'bar':>4}  {'n_ev':>5}  {'jacks':>6}  {'rolls':>6}  dominant_transition")
    for bar in range(total_bars):
        f = bar_features.get(bar)
        if f is None or f['n_events'] == 0:
            continue
        n       = f['n_events']
        jacks   = f['transitions'].get('jack', 0)
        rolls   = f['transitions'].get('roll_up', 0) + f['transitions'].get('roll_down', 0)
        dom_t   = max(f['transitions'], key=f['transitions'].get) if f['transitions'] else '—'
        print(f"{bar:>4}  {n:>5}  {jacks:>6}  {rolls:>6}  {dom_t}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('chart', help='path to .npy chart file')
    p.add_argument('--bars', nargs=2, type=int, default=None,
                   metavar=('START', 'END'), help='bar range to analyse')
    p.add_argument('--k', type=int, default=1, help='number of best fingerings')
    p.add_argument('--hand', choices=['p1', 'p2', 'both'], default='both')
    args = p.parse_args()

    arr, bpm, meta = load_chart(args.chart)
    print(f"Chart: {meta.get('title', Path(args.chart).stem)}")
    print(f"  BPM: {meta.get('bpm')}  bars: {meta.get('total_bars')}  "
          f"level: {meta.get('level')}")

    p1_events, p2_events = split_hands(arr, bpm)

    if args.bars:
        lo_row = args.bars[0] * 192
        hi_row = args.bars[1] * 192
        p1_events = [e for e in p1_events if lo_row <= e.row < hi_row]
        p2_events = [e for e in p2_events if lo_row <= e.row < hi_row]

    hands = []
    if args.hand in ('p1', 'both'):
        hands.append(('P1', p1_events))
    if args.hand in ('p2', 'both'):
        hands.append(('P2', p2_events))

    total_bars = meta.get('total_bars', arr.shape[0] // 192)

    for hand_name, events in hands:
        print(f"\n── {hand_name} ({len(events)} events) ──")
        if not events:
            print('  (no events)')
            continue

        results = infer_fingering(events, k_best=args.k)
        cost, path = results[0]
        print(f"  search score (Viterbi objective, not a difficulty measure): "
              f"{cost:.3f}  ({len(path)} timesteps)")

        # Aggregate per-bar technique features (structural only — no cost/velocity
        # numbers here; cost is an internal search objective for dp_fingering's
        # Viterbi DP, not a difficulty measure, so it stops at the pathfinding step
        # and never becomes a reported feature).
        bar_features = defaultdict(lambda: {
            'n_events': 0, 'transitions': defaultdict(int),
        })

        prev_state = None
        for state, evs in path:
            bar   = bar_of(evs[0].row)
            label = classify_transition(prev_state, state, evs[0].time_sec)

            bar_features[bar]['n_events'] += len(evs)
            bar_features[bar]['transitions'][label] += 1

            prev_state = state

        print_bar_summary(bar_features, total_bars)


if __name__ == '__main__':
    main()
