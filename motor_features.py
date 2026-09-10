"""
Constraint-based motor features derived from the fingering search.

Why not path cost
-----------------
The Viterbi objective answers "what does the easiest available fingering cost".
Measured against ereter ratings that is a weak difficulty signal (rho ~= 0.25),
and it gets weaker as the cost model improves, because a better model routes
*around* hard transitions — the optimiser optimises away the thing being counted.

Difficulty is closer to how hard the good fingering is to find and hold onto, so
these features read the *shape of the search* rather than its optimum:

  ambiguity   how much better the best option is than the alternatives, and how
              many alternatives are near-optimal at all
  instability whether the chart forces the same lane to be played by different
              fingers at different times (運指変更)
  lookahead   how far the correct choice depends on notes not yet reachable by a
              local decision — greedy vs full Viterbi disagreement

Ambiguity is measured locally (best vs second-best next state, given the chosen
predecessor) rather than by k-best backtracking: infer_fingering's k-best paths
are drawn from the top entries of the final DP frontier, so they tend to differ
only near the end of the chart and understate real diversity.
"""

import math
import numpy as np
from collections import defaultdict

from chart_io import NoteEvent
from finger_model import is_scratch, to_local
from dp_fingering import (
    HandState, _enumerate_hand_states, _state_cost, _transition_cost,
)

# A state counts as a live alternative within this much of the best option.
# Measured at 0.10/0.10 the count sat at 1.02 +/- 0.02 across 684 charts — too
# tight to separate "one forced fingering" from "several equally good ones", so
# the band is wider than a first guess would suggest.
VIABLE_ABS_TOL = 0.50
VIABLE_REL_TOL = 0.25

# Window used for the peak-instability feature.
PEAK_WINDOW_SEC = 4.0


# ── timestep grouping (mirrors infer_fingering) ───────────────────────────────

def group_timesteps(events: list[NoteEvent]) -> list[list[NoteEvent]]:
    """Group simultaneous events (same row) into timesteps."""
    if not events:
        return []
    out, cur_row, cur = [], events[0].row, [events[0]]
    for ev in events[1:]:
        if ev.row == cur_row:
            cur.append(ev)
        else:
            out.append(cur)
            cur_row, cur = ev.row, [ev]
    out.append(cur)
    return out


def states_for(group: list[NoteEvent]) -> tuple[list[HandState], bool]:
    """Valid hand states for one timestep, plus whether scratch is engaged."""
    active = [ev.lane for ev in group]
    has_scratch = any(is_scratch(l) for l in active)
    local = sorted({to_local(l) for l in active if not is_scratch(l)})
    return (_enumerate_hand_states(local) if local else [()]), has_scratch


# ── lookahead: greedy decode for comparison against Viterbi ───────────────────

def greedy_fingering(timesteps: list[list[NoteEvent]]):
    """
    Decode left-to-right taking the locally cheapest state at every timestep.

    Returns (total_cost, path) in the same shape as infer_fingering's path so the
    two can be compared directly. Where greedy and Viterbi disagree, the correct
    choice depended on notes the local decision could not see.
    """
    prev_state, path, total = None, [], 0.0
    for t, group in enumerate(timesteps):
        states, has_scratch = states_for(group)
        dt = (group[0].time_sec - timesteps[t - 1][0].time_sec) if t else 0.0
        best_cost, best_state = math.inf, ()
        for s in states:
            c = _state_cost(s, has_scratch)
            if t:
                c += _transition_cost(prev_state, s, dt)
            if c < best_cost:
                best_cost, best_state = c, s
        total += best_cost if math.isfinite(best_cost) else 0.0
        path.append((best_state, group))
        prev_state = best_state
    return total, path


# ── ambiguity: how forced is each choice along the chosen path ────────────────

def ambiguity_profile(path, timesteps):
    """
    For each timestep after the first, cost every valid state against the state
    actually chosen at t-1.

    Returns (margins, n_viable, times):
      margins[t]  second-best minus best — how much the winner wins by
      n_viable[t] how many states sit within tolerance of the best
    """
    margins, n_viable, times = [], [], []
    for t in range(1, len(path)):
        group = timesteps[t]
        dt = group[0].time_sec - timesteps[t - 1][0].time_sec
        if dt <= 0:
            continue
        states, has_scratch = states_for(group)
        prev_state = path[t - 1][0]
        costs = []
        for s in states:
            c = _state_cost(s, has_scratch) + _transition_cost(prev_state, s, dt)
            if math.isfinite(c):
                costs.append(c)
        if not costs:
            continue
        costs.sort()
        best = costs[0]
        margins.append(costs[1] - best if len(costs) > 1 else 0.0)
        tol = best + VIABLE_ABS_TOL + VIABLE_REL_TOL * abs(best)
        n_viable.append(sum(1 for c in costs if c <= tol))
        times.append(group[0].time_sec)
    return np.array(margins), np.array(n_viable, dtype=float), np.array(times)


# ── instability: does the chart force a lane to change fingers ────────────────

def instability_profile(path):
    """
    Track, per hand-local lane, which finger plays it each time it fires.

    Returns (change_times, entropy, n_uses):
      change_times  times at which a lane was played by a different finger than
                    the previous time that same lane fired (運指変更 events)
      entropy       usage-weighted mean entropy of the per-lane finger
                    distribution — 0 when every lane has one settled fingering
      n_uses        total lane activations, for normalisation
    """
    last_finger: dict[int, int] = {}
    counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    change_times: list[float] = []
    n_uses = 0

    for state, group in path:
        t = group[0].time_sec
        for a in state:
            for lane in a.covered_lanes:
                n_uses += 1
                counts[lane][a.finger] += 1
                if lane in last_finger and last_finger[lane] != a.finger:
                    change_times.append(t)
                last_finger[lane] = a.finger

    total_h, total_w = 0.0, 0
    for lane, dist in counts.items():
        n = sum(dist.values())
        h = -sum((c / n) * math.log(c / n) for c in dist.values() if c)
        total_h += h * n
        total_w += n
    entropy = (total_h / total_w) if total_w else 0.0
    return np.array(change_times), entropy, n_uses


# ── aggregation ───────────────────────────────────────────────────────────────

def _p95(a): return float(np.percentile(a, 95)) if len(a) else 0.0
def _mean(a): return float(np.mean(a)) if len(a) else 0.0


def path_cost(path, timesteps) -> float:
    """Replay a path and total its static + transition cost.

    Used so the greedy and Viterbi paths are scored by the identical accounting;
    infer_fingering's returned cost cannot be compared directly against a greedy
    total because it accumulates through the DP's own beam.
    """
    total, prev_state = 0.0, None
    for t, (state, group) in enumerate(path):
        has_scratch = any(is_scratch(e.lane) for e in group)
        c = _state_cost(state, has_scratch)
        if t:
            dt = group[0].time_sec - timesteps[t - 1][0].time_sec
            if dt > 0:
                c += _transition_cost(prev_state, state, dt)
        if math.isfinite(c):
            total += c
        prev_state = state
    return total


def _peak_rate(times: np.ndarray, window: float = PEAK_WINDOW_SEC) -> float:
    """Highest count of `times` falling inside any `window`-second span."""
    if len(times) == 0:
        return 0.0
    best, j = 0, 0
    for i in range(len(times)):
        while times[i] - times[j] > window:
            j += 1
        best = max(best, i - j + 1)
    return float(best)


def hand_features(events: list[NoteEvent], viterbi_path, tag: str) -> dict:
    """Constraint features for one hand. `viterbi_path` comes from infer_fingering."""
    # nviable_min is deliberately absent: the chosen state is always viable, so a
    # per-chart minimum is 1 on every chart of any length (std 0.0 over 684) and
    # carries nothing.
    keys = ['margin_mean', 'margin_p95', 'nviable_mean',
            'greedy_agree', 'greedy_regret',
            'instab_entropy', 'instab_rate', 'instab_peak']
    out = {f'{tag}_{k}': 0.0 for k in keys}
    if not events or not viterbi_path:
        return out

    timesteps = group_timesteps(events)

    margins, n_viable, _ = ambiguity_profile(viterbi_path, timesteps)
    out[f'{tag}_margin_mean']  = _mean(margins)
    out[f'{tag}_margin_p95']   = _p95(margins)
    out[f'{tag}_nviable_mean'] = _mean(n_viable)

    _, g_path = greedy_fingering(timesteps)
    n = min(len(g_path), len(viterbi_path))
    agree = sum(1 for i in range(n) if g_path[i][0] == viterbi_path[i][0])
    out[f'{tag}_greedy_agree'] = agree / max(n, 1)
    g_cost = path_cost(g_path, timesteps)
    v_cost = path_cost(viterbi_path, timesteps)
    out[f'{tag}_greedy_regret'] = (g_cost - v_cost) / max(abs(v_cost), 1e-6)

    change_times, entropy, n_uses = instability_profile(viterbi_path)
    out[f'{tag}_instab_entropy'] = entropy
    out[f'{tag}_instab_rate']    = len(change_times) / max(n_uses, 1)
    out[f'{tag}_instab_peak']    = _peak_rate(change_times)
    return out
