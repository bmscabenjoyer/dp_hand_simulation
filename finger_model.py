"""
Biomechanical finger model for IIDX DP.

Lanes 1–7 = P1 keys, lane 0 = P1 scratch.
Lanes 8–14 = P2 keys (mirrored), lane 15 = P2 scratch.

Two home positions per hand:
  Home A: pinky=1, ring=2, middle=4, index=6, thumb=7
  Home B: pinky=1, ring=3, middle=4, index=5, thumb=7
P2 is the mirror: thumb on the left (lane 8), pinky on the right (lane 14).
"""

from dataclasses import dataclass, field
from typing import Optional

# ── Finger identifiers ────────────────────────────────────────────────────────

PINKY  = 0
RING   = 1
MIDDLE = 2
INDEX  = 3
THUMB  = 4
FINGER_NAMES = ['pinky', 'ring', 'middle', 'index', 'thumb']

# ── Home positions ─────────────────────────────────────────────────────────────
# Lane numbers within a hand-local coordinate (1–7 for both P1 and P2).
# P2 lanes 8–14 map to local lanes 1–7 respectively (thumb side = 1).

# {finger: lane} for each home position
HOME_A = {PINKY: 1, RING: 2, MIDDLE: 4, INDEX: 6, THUMB: 7}
HOME_B = {PINKY: 1, RING: 3, MIDDLE: 4, INDEX: 5, THUMB: 7}

# ── Valid finger positions ─────────────────────────────────────────────────────
# Explicit list of local lanes (1–7) each finger can be assigned to.
# Edit these lists directly — non-contiguous positions are fine.

FINGER_POSITIONS: dict[int, list[int]] = {
    PINKY:  [1],
    RING:   [2, 3],
    MIDDLE: [3, 4, 5],
    INDEX:  [5, 6],
    THUMB:  [5,7],
}

# Fingers that can perform a two-key press (flat of finger covers two adjacent lanes).
# Ring and middle have sufficient width; others are too narrow or rigid.
TWO_KEY_CAPABLE = {RING, MIDDLE}

# ── Cost constants ─────────────────────────────────────────────────────────────

# Base deviation cost per lane away from nearest home position. This is the only
# surviving static term: it is positional (the hand is displaced from rest), not
# a judgement about which finger is doing the work. It also anchors the Viterbi's
# choice for isolated notes, where no transition cost applies.
DEVIATION_COST_PER_LANE = 1.0

# Base cost for a two-key press (on top of deviation cost).
TWO_KEY_BASE_COST = {RING: 0.8, MIDDLE: 1.2}

# Scratch lane engagement cost (requires wrist rotation / arm movement).
SCRATCH_COST = 4.0

# ── Rate ceilings ──────────────────────────────────────────────────────────────
# Motor difficulty is convex in demanded rate, not linear in 1/Δt: going from 4
# to 6 notes/sec costs almost nothing, going from 10 to 12 is a wall. Every
# channel therefore prices as (demanded_rate / ceiling) ** RATE_EXPONENT.
#
# Finger identity enters here — as a ceiling, not a flat surcharge. The pinky is
# not expensive to use, it is expensive to use *fast*, so it only costs anything
# when the chart actually demands speed from it.

RATE_EXPONENT = 2.5

# Channel 1 — same finger, moving between lanes (ring rocking 2 <-> 3).
SWITCH_RATE = {PINKY: 5.0, RING: 6.5, MIDDLE: 8.0, INDEX: 8.5, THUMB: 7.0}

# Channel 2 — same finger, same lane, re-struck (jack / 縦連). Hardest channel:
# no lateral movement assists the release, so ceilings sit below SWITCH_RATE.
JACK_RATE = {PINKY: 4.0, RING: 5.0, MIDDLE: 6.5, INDEX: 7.0, THUMB: 5.5}

# Channel 3 — two different fingers alternating. Easiest channel, but anatomically
# coupled pairs cap lower: pinky/ring and ring/middle share extensor tendons and
# cannot be driven independently at speed.
ALT_RATE_DEFAULT = 14.0
ALT_RATE_PAIRS = {
    frozenset({PINKY,  RING}):    9.0,
    frozenset({RING,   MIDDLE}): 10.0,
    frozenset({MIDDLE, INDEX}):  13.0,
    frozenset({INDEX,  THUMB}):  13.0,
}

# Each lane of travel beyond the first lowers the rate a switch can sustain.
SWITCH_DIST_FALLOFF = 0.45

# ── Hand coordinate helpers ───────────────────────────────────────────────────

P1_KEY_LANES   = list(range(1, 8))   # lanes 1–7
P2_KEY_LANES   = list(range(8, 15))  # lanes 8–14
P1_SCRATCH     = 0
P2_SCRATCH     = 15

def to_local(lane: int) -> int:
    """Convert global lane number to hand-local lane (1–7).

    P2 is the physical mirror of P1: the right hand's thumb sits on the inside
    of the cabinet (global lane 8) and its pinky on the outside (global lane 14),
    so P2 keys are reflected rather than shifted. Reflecting here lets both hands
    share one FINGER_POSITIONS / HOME table, and makes local lane numbers mean
    the same finger on either side.
    """
    if lane in P1_KEY_LANES:
        return lane
    if lane in P2_KEY_LANES:
        return 15 - lane   # lane 8 → 7 (thumb side), lane 14 → 1 (pinky side)
    raise ValueError(f'scratch lane {lane} has no local key lane')

def is_scratch(lane: int) -> bool:
    return lane in (P1_SCRATCH, P2_SCRATCH)

def is_p1(lane: int) -> bool:
    return lane <= 7

# ── Assignment dataclass ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Assignment:
    """A finger pressing one or two adjacent local lanes."""
    finger: int
    lane:   int             # primary (or lower) local lane
    lane2:  Optional[int] = None  # set for two-key press; must equal lane + 1

    @property
    def is_two_key(self) -> bool:
        return self.lane2 is not None

    @property
    def center(self) -> float:
        return self.lane if self.lane2 is None else (self.lane + self.lane2) / 2

    @property
    def covered_lanes(self) -> frozenset:
        if self.lane2 is None:
            return frozenset([self.lane])
        return frozenset([self.lane, self.lane2])


# ── Cost functions ─────────────────────────────────────────────────────────────

def nearest_home_distance(finger: int, local_lane: float) -> float:
    """Minimum distance from local_lane to either home position for this finger."""
    d_a = abs(local_lane - HOME_A[finger])
    d_b = abs(local_lane - HOME_B[finger])
    return min(d_a, d_b)


def assignment_cost(a: Assignment) -> float:
    """Static cost of holding a single assignment (no temporal component)."""
    if a.is_two_key:
        if a.finger not in TWO_KEY_CAPABLE:
            return float('inf')
        positions = FINGER_POSITIONS[a.finger]
        if a.lane not in positions or a.lane2 not in positions:
            return float('inf')
        base = TWO_KEY_BASE_COST[a.finger]
        dev  = nearest_home_distance(a.finger, a.center)
    else:
        if a.lane not in FINGER_POSITIONS[a.finger]:
            return float('inf')
        base = 0.0
        dev  = nearest_home_distance(a.finger, a.lane)
    return base + dev * DEVIATION_COST_PER_LANE


def _rate_cost(rate: float, ceiling: float) -> float:
    """Convex penalty for demanding `rate` Hz from a channel capped at `ceiling`."""
    if ceiling <= 0.0:
        return float('inf')
    return (rate / ceiling) ** RATE_EXPONENT


def transition_cost(
    prev_assignment: Optional[Assignment],
    next_assignment: Assignment,
    time_delta_sec: float,
) -> float:
    """
    Cost of driving one finger from prev_assignment to next_assignment in the
    time available. Covers channels 1 (lane switch) and 2 (jack); cross-finger
    alternation is channel 3, priced by alternation_cost over whole hand states.

    A finger that was idle costs nothing to engage — the static deviation term
    in assignment_cost already accounts for where it has to reach.
    """
    if time_delta_sec <= 0:
        return float('inf')
    if prev_assignment is None:
        return 0.0

    rate = 1.0 / time_delta_sec

    if prev_assignment.covered_lanes == next_assignment.covered_lanes:
        return _rate_cost(rate, JACK_RATE[next_assignment.finger])

    distance = abs(prev_assignment.center - next_assignment.center)
    if prev_assignment.is_two_key != next_assignment.is_two_key:
        distance += 0.5
    ceiling = (SWITCH_RATE[next_assignment.finger]
               / (1.0 + SWITCH_DIST_FALLOFF * max(distance - 1.0, 0.0)))
    return _rate_cost(rate, ceiling)


def alternation_cost(prev_state, next_state, time_delta_sec: float) -> float:
    """
    Channel 3: cost of handing off between two different fingers of one hand.

    Charged once per timestep against the *binding* pair — the coupled pair with
    the lowest ceiling — rather than once per finger, so wide chords are not
    penalised simply for being wide (density already measures that).
    """
    if time_delta_sec <= 0 or not prev_state or not next_state:
        return 0.0

    prev_fingers = {a.finger for a in prev_state}
    next_fingers = {a.finger for a in next_state}
    engaged  = next_fingers - prev_fingers
    released = prev_fingers - next_fingers
    if not engaged or not released:
        return 0.0

    ceiling = min(
        ALT_RATE_PAIRS.get(frozenset({f_out, f_in}), ALT_RATE_DEFAULT)
        for f_in in engaged for f_out in released
    )
    return _rate_cost(1.0 / time_delta_sec, ceiling)


# ── Valid assignment enumeration ───────────────────────────────────────────────

def valid_single_assignments(local_lane: int) -> list[Assignment]:
    """All finger assignments that can cover local_lane with finite cost."""
    out = []
    for finger in range(5):
        if local_lane in FINGER_POSITIONS[finger]:
            a = Assignment(finger=finger, lane=local_lane)
            if assignment_cost(a) < float('inf'):
                out.append(a)
    return out


def valid_two_key_assignments(local_lane_a: int, local_lane_b: int) -> list[Assignment]:
    """All two-key assignments that cover both lanes (must be adjacent)."""
    if local_lane_b != local_lane_a + 1:
        return []
    out = []
    for finger in TWO_KEY_CAPABLE:
        a = Assignment(finger=finger, lane=local_lane_a, lane2=local_lane_b)
        if assignment_cost(a) < float('inf'):
            out.append(a)
    return out
