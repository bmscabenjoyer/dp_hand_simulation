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

# ── Reach envelopes ────────────────────────────────────────────────────────────
# (min_lane, max_lane) each finger can reach from its home position.
# Based on anatomy: middle has most reach, pinky the least.

REACH = {
    PINKY:  (1, 2),
    RING:   (1, 4),   # sits naturally between lanes 2–3; can stretch to 1 or 4
    MIDDLE: (2, 6),
    INDEX:  (4, 7),
    THUMB:  (5, 7),
}

# Fingers that can perform a two-key press (flat of finger covers two adjacent lanes).
# Ring and middle have sufficient width; others are too narrow or rigid.
TWO_KEY_CAPABLE = {RING, MIDDLE}

# ── Cost constants ─────────────────────────────────────────────────────────────

# Base deviation cost per lane away from nearest home position.
DEVIATION_COST_PER_LANE = 1.0

# Extra cost for using a weaker finger on a demanding note.
WEAK_FINGER_COST = {PINKY: 1.5, RING: 0.5, MIDDLE: 0.0, INDEX: 0.0, THUMB: 0.5}

# Base cost for a two-key press (on top of deviation cost).
TWO_KEY_BASE_COST = {RING: 0.8, MIDDLE: 1.2}

# Jack penalty: effective distance added when the same lane fires twice.
# Divided by time_available in the transition velocity formula.
JACK_PENALTY = 3.0

# Scratch lane engagement cost (requires wrist rotation / arm movement).
SCRATCH_COST = 4.0

# ── Hand coordinate helpers ───────────────────────────────────────────────────

P1_KEY_LANES   = list(range(1, 8))   # lanes 1–7
P2_KEY_LANES   = list(range(8, 15))  # lanes 8–14
P1_SCRATCH     = 0
P2_SCRATCH     = 15

def to_local(lane: int) -> int:
    """Convert global lane number to hand-local lane (1–7)."""
    if lane in P1_KEY_LANES:
        return lane
    if lane in P2_KEY_LANES:
        return lane - 7   # lane 8 → 1, lane 14 → 7
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
        base = TWO_KEY_BASE_COST[a.finger]
        dev  = nearest_home_distance(a.finger, a.center)
    else:
        lo, hi = REACH[a.finger]
        if not (lo <= a.lane <= hi):
            return float('inf')
        base = 0.0
        dev  = nearest_home_distance(a.finger, a.lane)
    return base + dev * DEVIATION_COST_PER_LANE + WEAK_FINGER_COST[a.finger]


def transition_velocity(
    prev_assignment: Optional[Assignment],
    next_assignment: Assignment,
    time_delta_sec: float,
) -> float:
    """
    Cost of moving from prev_assignment to next_assignment given the time available.

    Returns effective_distance / time_delta — higher = harder.
    """
    if time_delta_sec <= 0:
        return float('inf')

    if prev_assignment is None:
        # First note: cost is deviation from home only, normalised by time
        return assignment_cost(next_assignment) / time_delta_sec

    same_finger = (prev_assignment.finger == next_assignment.finger)
    same_lanes  = (prev_assignment.covered_lanes == next_assignment.covered_lanes)

    if same_finger and same_lanes:
        # Jack: same finger, same lane(s) fired again
        effective_distance = JACK_PENALTY
    else:
        # Spatial distance between the two assignments' centres
        effective_distance = abs(prev_assignment.center - next_assignment.center)
        # Additional cost if changing from single to two-key or vice versa
        if prev_assignment.is_two_key != next_assignment.is_two_key:
            effective_distance += 0.5

    return effective_distance / time_delta_sec


# ── Valid assignment enumeration ───────────────────────────────────────────────

def valid_single_assignments(local_lane: int) -> list[Assignment]:
    """All finger assignments that can cover local_lane with finite cost."""
    out = []
    for finger in range(5):
        lo, hi = REACH[finger]
        if lo <= local_lane <= hi:
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
