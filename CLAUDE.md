# CLAUDE.md — dp_hand_simulation

Biomechanical finger simulation for IIDX Double Play (DP) charts.
Goal: infer optimal finger assignments per note event, compute time-normalized
transition costs, and produce per-window motor feature vectors for difficulty
estimation and pattern similarity retrieval.

---

## References

### Foundational
- **Parncutt et al. (1997)** — "An ergonomic model of keyboard fingering for
  melodic fragments." *Music Perception* 14(4), 341–382.
  - Defined biomechanical cost rules (stretch, position change, weak finger,
    same-finger repeat) and used DP to find minimum-cost fingering.
  - Weights fitted empirically against human difficulty ratings.
  - Time-independent (no velocity normalisation) — key limitation for IIDX.

### Modern learned approach
- **Nakamura, Saito, Yoshii (2019)** — "Statistical Learning and Estimation of
  Piano Fingering." arXiv:1904.10237.
  - HMM where hidden states = finger assignments, observations = notes.
  - Transition probabilities learned from annotated fingering data (PIG dataset).
  - High-order HMMs (conditioning on 2–3 previous states) outperform first-order.
  - Key insight: multiple valid fingerings exist; evaluation uses k-best paths.
  - Limitation flagged: two-hand interdependence not modelled.

### Relevance to IIDX
- Piano models are time-independent; IIDX requires time-normalised transition
  velocity (`effective_distance / time_available`) because BPM multiplies every
  cost.
- Annotations in piano = finger-to-note assignment per event. IIDX equivalent =
  finger-to-lane assignment per chord (simultaneous note group).
- HMM vs hand-crafted costs: both use Viterbi DP for inference; HMM learns
  weights from data, hand-crafted costs use intuition + ereter calibration.
  For our purposes hand-crafted costs calibrated against ereter ratings are
  sufficient without annotation overhead.

---

## IIDX DP physical dimensions

### Lane layout
```
P1 side                          P2 side
SCR  1  2  3  4  5  6  7   |   1  2  3  4  5  6  7  SCR
 0   1  2  3  4  5  6  7   |   8  9 10 11 12 13 14  15
```
- Lanes 0–7: P1 (scratch=0, keys 1–7)
- Lanes 8–15: P2 (keys 8–14, scratch=15)
- P2 is a mirror of P1: thumb is on the inside (lane 8), pinky on the outside (lane 14)

### Note values in the .npy array (column 0–15)
| Value | Meaning |
|-------|---------|
| 0 | empty |
| 1 | tap note head |
| 2 | CN (charge note) head |
| 3 | CN body |
| 4 | HCN (hell charge note) head |
| 5 | HCN body |
| 6 | BSS head |
| 7 | BSS body |
| 8 | MSS head |
| 9 | MSS body |

Head values requiring motor action: {1, 2, 4, 6, 8}.
Column 16: BPM × 100 (uint32, 0 = no change; fill-forward to get dense BPM).

### Timing grid
- `ROWS_PER_BAR = 192` (4 beats × 48 subdivisions per beat)
- 1 row = 1/48 of a beat
- At 150 BPM: 1 row ≈ 8.33 ms

---

## Finger home positions

Two standard home positions. Players switch between them for most charts
and deviate to extended positions for unusual note formations.

### P1 (left hand) — local lanes 1–7

| Lane | Home A  | Home B  |
|------|---------|---------|
| 1    | pinky   | pinky   |
| 2    | ring    | —       |
| 3    | —       | ring    |
| 4    | middle  | middle  |
| 5    | —       | index   |
| 6    | index   | —       |
| 7    | thumb   | thumb   |

### P2 (right hand) — local lanes 1–7 (global lanes 8–14), mirrored

| Local | Global | Home A  | Home B  |
|-------|--------|---------|---------|
| 1     | 8      | thumb   | thumb   |
| 2     | 9      | —       | index   |
| 3     | 10     | index   | —       |
| 4     | 11     | middle  | middle  |
| 5     | 12     | —       | ring    |
| 6     | 13     | ring    | —       |
| 7     | 14     | pinky   | pinky   |

### Extended positions (advanced charts)
- Middle on lane 3 (shift from home lane 4, cost = 1 deviation unit)
- Thumb on lane 5 (large inward reach from home 7, high cost)
- Ring finger pressing lanes 2+3 simultaneously (two-key press; ring sits
  naturally between home A lane 2 and home B lane 3, lowest two-key cost)
- Only ring and middle are wide enough for two-key presses

### Finger reach envelopes (local lanes)
| Finger | Min | Max | Notes |
|--------|-----|-----|-------|
| pinky  | 1   | 2   | very limited |
| ring   | 1   | 4   | spans both home positions |
| middle | 2   | 6   | longest finger, most reach |
| index  | 4   | 7   | |
| thumb  | 5   | 7   | inward reach costly |

---

## Transition cost model

### Core formula
```
transition_velocity = effective_distance / time_available_seconds
```
Time-available is the gap between consecutive note events derived from the
BPM-integrated timing grid. This makes every cost BPM-aware automatically.

### Effective distance
- **Normal transition**: absolute distance between assignment centres in lane space
- **Jack** (same lane, same finger fires again): `JACK_PENALTY` constant
  (non-zero despite zero spatial movement — finger must release and re-engage)
- **Two-key ↔ single key**: +0.5 extra distance for mode change

### Static assignment cost (position cost)
```
cost = deviation_from_nearest_home × DEVIATION_COST_PER_LANE
     + WEAK_FINGER_COST[finger]
     + TWO_KEY_BASE_COST[finger]  (if two-key press)
```

### Current cost constants (uncalibrated — intuition only)
```python
DEVIATION_COST_PER_LANE = 1.0
WEAK_FINGER_COST = {pinky: 1.5, ring: 0.5, middle: 0.0, index: 0.0, thumb: 0.5}
TWO_KEY_BASE_COST = {ring: 0.8, middle: 1.2}
JACK_PENALTY = 3.0
SCRATCH_COST = 4.0
```

These need calibration against ereter ratings (see workflow below).

### Transition type vocabulary
| Label | Description |
|-------|-------------|
| `roll_up` | single note, lane +1 from previous |
| `roll_down` | single note, lane −1 from previous |
| `jack` | same lane fires again with same finger |
| `chord` | multiple simultaneous lanes, non-adjacent |
| `chord_adjacent` | multiple simultaneous lanes, all adjacent |
| `chord_repeat` | same chord as previous timestep |
| `jump` | single note, non-adjacent lane change (currently too broad — needs refinement) |
| `mixed` | combination |

---

## Current workflow

### Data flow
```
.npy + .json
     │
     ▼
chart_io.py          — load array, fill-forward BPM, integrate to seconds,
                        extract NoteEvent list per hand
     │
     ▼
finger_model.py      — define valid assignments, compute static and transition costs
     │
     ▼
dp_fingering.py      — Viterbi DP over hand states → k-best (cost, path) pairs
                        path[t] = (HandState, [NoteEvent, ...])
     │
     ▼
simulate.py          — aggregate per bar: n_events, avg_cost, peak_velocity,
                        transition type histogram → printed table
```

### Running
```bash
python simulate.py path/to/chart.npy [--bars START END] [--hand p1|p2|both] [--k 3]
```

### Known gaps
1. **Transition classifier too coarse** — `jump` conflates small reaches
   (within envelope, cheap) with full repositions (hand shift, expensive).
   Needs a `reach` vs `reposition` split based on whether the target lane
   falls within the current finger's envelope.
2. **No per-window feature vector** — need to aggregate bar features into a
   fixed-size vector per chart for ereter calibration and NN retrieval.
3. **P1+P2 combined features missing** — simultaneous scratch+key (the hardest
   DP-specific skill) requires cross-hand awareness not yet implemented.
4. **Cost weights uncalibrated** — current values are intuition-only. Planned
   calibration: extract feature vectors for all 684 labeled lv12 charts, regress
   against ereter EC/HC/EXH ratings, iterate on primitive costs until regression
   coefficients align with game knowledge.

### Calibration plan
1. Run simulation on all lv12 charts → per-chart feature vectors
   (mean velocity, peak velocity, jack fraction, reposition fraction,
   scratch density, chord density, etc.)
2. RidgeCV regression against `rating_hc` / `rating_ec` / `rating_exh`
3. Check which features get high coefficients — if `jack_fraction` is
   underweighted vs game knowledge, raise `JACK_PENALTY` and repeat
4. Target: Spearman ρ > 0.5 on held-out charts from cost features alone
   (XGBoost on encoder embeddings currently gets ρ ≈ 0.73; cost features
   should add orthogonal signal)

---

## Related repo
`~/projects/iidx_dp_learning` — MAE encoder pretraining and fine-tuning on
ereter difficulty ratings. The encoder (v12, 4-row patches, 33ms resolution)
produces segment embeddings used for pattern similarity retrieval. The motor
feature vectors from this repo are intended as complementary features alongside
the encoder embeddings.
