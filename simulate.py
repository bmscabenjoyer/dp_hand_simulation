"""
Run the finger simulation on a chart and print per-bar difficulty features.

Usage:
    python simulate.py path/to/chart.npy [--bars START END] [--k 3]
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from chart_io import load_chart, split_hands, ROWS_PER_BAR
from dp_fingering import infer_fingering
from finger_model import FINGER_NAMES, is_scratch


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
    print(f"\n{'bar':>4}  {'n_ev':>5}  {'cost/ev':>8}  {'peak_vel':>9}  "
          f"{'jacks':>6}  {'rolls':>6}  dominant_transition")
    for bar in range(total_bars):
        f = bar_features.get(bar)
        if f is None or f['n_events'] == 0:
            continue
        n       = f['n_events']
        avg_c   = f['total_cost'] / n
        peak    = f['peak_velocity']
        jacks   = f['transitions'].get('jack', 0)
        rolls   = f['transitions'].get('roll_up', 0) + f['transitions'].get('roll_down', 0)
        dom_t   = max(f['transitions'], key=f['transitions'].get) if f['transitions'] else '—'
        print(f"{bar:>4}  {n:>5}  {avg_c:>8.3f}  {peak:>9.3f}  "
              f"{jacks:>6}  {rolls:>6}  {dom_t}")


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
        print(f"  optimal path cost: {cost:.3f}  ({len(path)} timesteps)")

        # Aggregate per-bar features
        bar_features = defaultdict(lambda: {
            'n_events': 0, 'total_cost': 0.0,
            'peak_velocity': 0.0, 'transitions': defaultdict(int),
        })

        prev_state = None
        for state, evs in path:
            bar   = bar_of(evs[0].row)
            n     = len(evs)
            sc    = sum(__import__('finger_model').assignment_cost(a) for a in state)
            label = classify_transition(prev_state, state, evs[0].time_sec)

            bar_features[bar]['n_events']     += n
            bar_features[bar]['total_cost']   += sc
            bar_features[bar]['transitions'][label] += 1
            if prev_state and evs:
                from finger_model import transition_velocity
                for a in state:
                    prev_a = next((x for x in prev_state if x.finger == a.finger), None)
                    vel = transition_velocity(prev_a, a, max(evs[0].time_sec - 1e-3, 1e-6))
                    bar_features[bar]['peak_velocity'] = max(
                        bar_features[bar]['peak_velocity'], vel)

            prev_state = state

        print_bar_summary(bar_features, total_bars)


if __name__ == '__main__':
    main()
