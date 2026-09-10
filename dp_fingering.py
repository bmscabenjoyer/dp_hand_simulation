"""
Viterbi DP to infer minimum-cost finger assignment sequence for one hand.

For each timestep (set of simultaneously active lanes), enumerates valid
finger assignments and finds the globally optimal sequence via DP.

A 'hand state' is a tuple of Assignments, one per simultaneously active lane.
"""

from itertools import product
from typing import Optional
import math

from finger_model import (
    Assignment, assignment_cost, transition_cost, alternation_cost,
    valid_single_assignments, valid_two_key_assignments,
    is_scratch, to_local, SCRATCH_COST,
    P1_SCRATCH, P2_SCRATCH,
)
from chart_io import NoteEvent


# ── Hand state ────────────────────────────────────────────────────────────────

HandState = tuple[Assignment, ...]   # one Assignment per active lane


def _enumerate_hand_states(active_lanes: list[int]) -> list[HandState]:
    """
    Enumerate all valid finger assignments for a set of simultaneously active lanes.
    Scratch lanes are handled separately (no finger model, fixed cost).
    Key lanes may be covered by a two-key press (one assignment for two lanes).
    """
    key_lanes   = [l for l in active_lanes if not is_scratch(l)]
    local_lanes = [to_local(l) for l in key_lanes]

    def _cover(remaining: list[int]) -> list[list[Assignment]]:
        """Recursively enumerate ways to cover remaining local lanes."""
        if not remaining:
            return [[]]
        results = []
        lane = remaining[0]
        rest = remaining[1:]

        # single-key options for this lane
        for a in valid_single_assignments(lane):
            for tail in _cover(rest):
                # check no finger conflict
                fingers_used = {x.finger for x in tail}
                if a.finger not in fingers_used:
                    results.append([a] + tail)

        # two-key option: can this lane pair with the next?
        if rest and rest[0] == lane + 1:
            for a in valid_two_key_assignments(lane, rest[0]):
                for tail in _cover(rest[1:]):
                    fingers_used = {x.finger for x in tail}
                    if a.finger not in fingers_used:
                        results.append([a] + tail)

        return results

    key_states = _cover(local_lanes)
    return [tuple(sorted(s, key=lambda a: a.lane)) for s in key_states]


def _state_cost(state: HandState, has_scratch: bool) -> float:
    """Static cost of a hand state (sum of assignment costs + scratch if active)."""
    cost = sum(assignment_cost(a) for a in state)
    if has_scratch:
        cost += SCRATCH_COST
    return cost


def _transition_cost(
    prev_state: Optional[HandState],
    next_state: HandState,
    time_delta_sec: float,
) -> float:
    """
    Total rate-based transition cost between two hand states.

    Per-finger channels (lane switch, jack) are charged for each finger active in
    next_state against its own previous assignment; cross-finger handoff is
    charged once for the whole state against the binding coupled pair.

    Note this is first-order: a finger that skipped a timestep reads as idle
    rather than as firing at the two-step rate. That is an approximation in the
    *search*; exact per-finger rates are recovered from the chosen path during
    feature extraction, where the full timeline is available.

    Scratch cost is intentionally NOT added here — it's already counted once,
    as a flat static cost, in _state_cost (consistent with how the deviation
    term is treated: once per timestep, not time-normalized).
    """
    cost = 0.0

    prev_by_finger = {}
    if prev_state:
        for a in prev_state:
            prev_by_finger[a.finger] = a

    for a in next_state:
        prev_a = prev_by_finger.get(a.finger)
        cost += transition_cost(prev_a, a, time_delta_sec)

    cost += alternation_cost(prev_state, next_state, time_delta_sec)

    return cost


# ── Viterbi DP ────────────────────────────────────────────────────────────────

def infer_fingering(
    events: list[NoteEvent],
    k_best: int = 1,
) -> list[tuple[float, list[tuple[HandState, list[NoteEvent]]]]]:
    """
    Run Viterbi DP over a sequence of note events for one hand.

    Groups simultaneous events (same row) into timesteps.
    Returns k_best solutions as (total_cost, path) pairs, where each path
    element is (HandState, events_at_that_timestep).

    Parameters
    ----------
    events  : sorted list of NoteEvent for one hand
    k_best  : number of near-optimal paths to return

    Returns
    -------
    List of (cost, path) sorted ascending by cost.
    path[t] = (HandState, [NoteEvent, ...])
    """
    if not events:
        return [(0.0, [])]

    # Group events by row (simultaneous notes)
    timesteps: list[list[NoteEvent]] = []
    cur_row, cur_group = events[0].row, [events[0]]
    for ev in events[1:]:
        if ev.row == cur_row:
            cur_group.append(ev)
        else:
            timesteps.append(cur_group)
            cur_row, cur_group = ev.row, [ev]
    timesteps.append(cur_group)

    # DP tables: list of {state: (cost, parent_state_index)}
    # Using k-best paths per state (beam)
    INF = float('inf')

    # dp[t] = list of (cost, state, parent_idx_in_dp[t-1])
    dp: list[list[tuple[float, HandState, int]]] = []

    for t, group in enumerate(timesteps):
        active_global = [ev.lane for ev in group]
        has_scratch   = any(is_scratch(l) for l in active_global)
        key_lanes     = [l for l in active_global if not is_scratch(l)]
        local_lanes   = sorted(set(to_local(l) for l in key_lanes))

        states = _enumerate_hand_states(local_lanes) if local_lanes else [()]

        if t == 0:
            # First timestep: initialise with static costs
            entries = []
            for state in states:
                c = _state_cost(state, has_scratch)
                entries.append((c, state, -1))
            entries.sort(key=lambda x: x[0])
            dp.append(entries[:max(k_best * 10, 50)])
        else:
            prev_group = timesteps[t - 1]
            time_delta = group[0].time_sec - prev_group[0].time_sec

            entries = []
            for state in states:
                static = _state_cost(state, has_scratch)
                best_k = []
                for pi, (prev_cost, prev_state, _) in enumerate(dp[t - 1]):
                    trans = _transition_cost(prev_state, state, time_delta)
                    total = prev_cost + static + trans
                    best_k.append((total, pi))
                    if len(best_k) >= k_best * 20:
                        break
                best_k.sort()
                for total, pi in best_k[:k_best]:
                    entries.append((total, state, pi))

            entries.sort(key=lambda x: x[0])
            dp.append(entries[:max(k_best * 10, 50)])

    # Backtrack k_best paths
    results = []
    for rank in range(min(k_best, len(dp[-1]))):
        cost, state, _ = dp[-1][rank]
        path_states = [state]
        t = len(dp) - 1
        idx = rank
        while t > 0:
            _, _, parent_idx = dp[t][idx]
            t -= 1
            idx = parent_idx
            path_states.append(dp[t][idx][1])
        path_states.reverse()
        path = list(zip(path_states, timesteps))
        results.append((cost, path))

    return results
