"""
Named physical quantities computed per time window from a hand-position token
stream.

Everything here is deterministic and label-free — these are measurements of what
a chart asks the hands to do, not predictions of how hard players find it. Each
quantity has a name and a unit so a result can be stated in words rather than
read off a coefficient vector.

Difficulty labels enter only afterwards, to test which of these measurements
actually track player outcomes. That test is the point: a quantity earns its
place by having a stable, signed relationship with what players do, not by
improving an aggregate score.

Windows are time-based (not token-based) because every quantity here is a rate,
and rates need a wall-clock denominator.
"""

import numpy as np
from collections import Counter

from hand_tokens import decode_position, isolated_fingering, NO_PRESS

# Measured, per-window split-half reliability:
#     4s window  0.45   adjacent-window correlation 0.32
#     8s window  0.36   adjacent-window correlation 0.15
#
# Lengthening the window does NOT buy reliability here. The obvious argument —
# more events per estimate, so less noise — fails because the real variation and
# the noise occupy the same timescale: averaging over 8s smooths away genuine
# 4-8s structure just as fast as it smooths away error.
#
# So a single window is roughly half signal at any length, and per-window values
# should not be trusted individually (this is why change-point segmentation on
# this profile finds mostly noise). Chart-level summaries are unaffected, since
# they average over dozens of windows: the ranked effect sizes move by <0.03
# between 4s and 8s, well inside their confidence intervals.
#
# 8s is kept because it is marginally cheaper and no worse, not because it is
# more reliable.
WINDOW_SEC = 8.0
STRIDE_SEC = 4.0

# One entry per quantity: name -> one-line meaning, used when reporting results.
QUANTITIES = {
    'nps':             'presses per second (either hand)',
    'finger_peak_hz':  'highest rate demanded of any single finger',
    'fingers_busy':    'distinct fingers used in the window',
    'pos_change_hz':   'hand-shape changes per second',
    'pos_repeat_frac': 'consecutive presses reusing the same hand shape (jack-like)',
    'both_hands_frac': 'moments where both hands press together',
    'scratch_key_frac':'moments where a hand must scratch and press keys at once',
    'hand_imbalance':  'how lopsided the load is between the two hands',
    'pos_entropy':     'variety of distinct hand shapes used',
    'deviation_rate':  'presses needing a non-default fingering',
    'chord_mean':      'average keys held down per press, per hand',
    'gap_regularity':  'how evenly spaced the presses are (1 = metronomic)',
}


def _fingers_for(pos_id: int) -> frozenset:
    """Canonical finger set for a hand position (correct ~95.5% of the time;
    the deviation bit flags the remainder)."""
    lanes, _ = decode_position(pos_id)
    return frozenset(f for f, _ in isolated_fingering(tuple(sorted(lanes))))


def window_physics(tok: dict, lo: int, hi: int) -> dict:
    """Compute every quantity over token rows [lo, hi)."""
    p1, p2 = tok['p1_pos'][lo:hi], tok['p2_pos'][lo:hi]
    d1, d2 = tok['p1_dev'][lo:hi], tok['p2_dev'][lo:hi]
    t = tok['times_sec'][lo:hi]
    n = len(p1)
    out = {k: 0.0 for k in QUANTITIES}
    if n < 2:
        return out

    span = max(float(t[-1] - t[0]), 1e-3)
    a1, a2 = p1 > NO_PRESS, p2 > NO_PRESS

    out['nps'] = n / span
    out['both_hands_frac'] = float((a1 & a2).mean())

    presses = int(a1.sum() + a2.sum())
    out['hand_imbalance'] = abs(int(a1.sum()) - int(a2.sum())) / max(presses, 1)

    # per-finger activation counts; hands kept separate (10 fingers total)
    fcount = Counter()
    scratch_key = 0
    chord_total = 0
    for arr, active, hand in ((p1, a1, 0), (p2, a2, 1)):
        for pid in arr[active]:
            lanes, scratch = decode_position(int(pid))
            chord_total += len(lanes)
            if scratch and lanes:
                scratch_key += 1
            for f in _fingers_for(int(pid)):
                fcount[(hand, f)] += 1
            if scratch:
                fcount[(hand, 'scr')] += 1
    out['finger_peak_hz'] = (max(fcount.values()) / span) if fcount else 0.0
    out['fingers_busy']   = float(len(fcount))
    out['scratch_key_frac'] = scratch_key / max(presses, 1)
    out['chord_mean'] = chord_total / max(presses, 1)

    # hand-shape change vs repeat, measured within each hand's own press sequence
    changes = repeats = pairs = 0
    for arr, active in ((p1, a1), (p2, a2)):
        seq = arr[active]
        for i in range(1, len(seq)):
            pairs += 1
            if seq[i] == seq[i - 1]:
                repeats += 1
            else:
                changes += 1
    out['pos_change_hz']   = changes / span
    out['pos_repeat_frac'] = repeats / max(pairs, 1)

    counts = Counter(int(x) for x in np.concatenate([p1[a1], p2[a2]]))
    total = sum(counts.values())
    out['pos_entropy'] = -sum((c / total) * np.log(c / total)
                              for c in counts.values()) if total else 0.0

    out['deviation_rate'] = ((d1[a1].sum() + d2[a2].sum()) / max(presses, 1))

    gaps = np.diff(t.astype(np.float64))
    gaps = gaps[gaps > 0]
    if len(gaps) > 1 and gaps.mean() > 0:
        out['gap_regularity'] = float(1.0 / (1.0 + gaps.std() / gaps.mean()))
    return out


def chart_physics(tok: dict) -> dict:
    """
    Slide a window over a chart and summarise each quantity two ways:
      _mean  what the chart sustains
      _peak  what its hardest stretch demands (95th percentile over windows)
    """
    t = tok['times_sec']
    keys = list(QUANTITIES)
    if len(t) < 4:
        return {f'{k}_{s}': 0.0 for k in keys for s in ('mean', 'peak')}

    rows, start = [], float(t[0])
    while start < float(t[-1]):
        lo = int(np.searchsorted(t, start, 'left'))
        hi = int(np.searchsorted(t, start + WINDOW_SEC, 'right'))
        if hi - lo >= 2:
            rows.append(window_physics(tok, lo, hi))
        start += STRIDE_SEC

    if not rows:
        return {f'{k}_{s}': 0.0 for k in keys for s in ('mean', 'peak')}

    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        out[f'{k}_mean'] = float(v.mean())
        out[f'{k}_peak'] = float(np.percentile(v, 95))
    return out
