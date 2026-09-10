"""
Generate per-window finger embeddings (P1+P2 concatenated) and search by similarity.

Each index entry is a 12-second window: P1 and P2 are embedded separately then
concatenated into a 256-dim vector. Flip search swaps P1↔P2 to find charts where
the pattern appears on the opposite side.

Build index:
    python embed_charts.py build \\
        --chart-dir ~/projects/textagextractor/dp12_active/charts \\
        --checkpoint checkpoints/finger_pretrain/encoder_best.pt \\
        --output embeddings/dp12_finger.npz

Query (best-matching window in a chart):
    python embed_charts.py query --index embeddings/dp12_finger.npz --title "AA" --k 12

Query (specific time in a chart):
    python embed_charts.py query --index embeddings/dp12_finger.npz --title "AA" --time 60 --k 12
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

WIN_SEC    = 12.0
STRIDE_SEC = 6.0
MIN_TOKENS = 5       # minimum tokens for a hand side to be embedded (else zeros)
D_HALF     = 128     # embedding dim per hand
D_FULL     = D_HALF * 2  # P1+P2 concat


# ── Helpers ───────────────────────────────────────────────────────────────────

def _embed_flow_segment(
    flow: list,
    t_start: float,
    t_end: float,
    model,
    device,
) -> np.ndarray:
    """
    Embed a time slice of a finger flow. Returns (D_HALF,) float32.
    Returns zeros if the segment has fewer than MIN_TOKENS.
    """
    import torch
    from finger_tokenize import flow_to_tokens

    sub  = [ts for ts in flow if t_start <= ts.time_sec < t_end]
    toks = flow_to_tokens(sub)

    if len(toks) < MIN_TOKENS:
        return np.zeros(D_HALF, dtype=np.float32)

    # Truncate to MAX_SEQ_LEN then mean-pool sub-windows if still too long
    from finger_transformer import MAX_SEQ_LEN
    if len(toks) <= MAX_SEQ_LEN:
        t = torch.from_numpy(toks.astype(np.int64)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode(t).squeeze(0).cpu().numpy()
    else:
        # split and mean-pool
        parts = []
        for s in range(0, len(toks) - MAX_SEQ_LEN + 1, MAX_SEQ_LEN // 2):
            chunk = toks[s : s + MAX_SEQ_LEN]
            t = torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(device)
            with torch.no_grad():
                parts.append(model.encode(t).squeeze(0).cpu().numpy())
        emb = np.mean(parts, axis=0)

    return emb.astype(np.float32)


def _load_flows(npy_path: Path):
    """Load chart and run Viterbi DP for both hands. Returns (flow1, flow2, meta)."""
    from chart_io import load_chart, split_hands
    from dp_fingering import infer_fingering
    from simulate import finger_flow

    arr, bpm_arr, meta = load_chart(npy_path)
    p1_events, p2_events = split_hands(arr, bpm_arr)

    def _infer(events):
        if len(events) < 5:
            return []
        try:
            _, path = infer_fingering(events, k_best=1)[0]
            return finger_flow(path)
        except Exception:
            return []

    return _infer(p1_events), _infer(p2_events), meta


# ── Build index ───────────────────────────────────────────────────────────────

def build_index(
    chart_dir: str,
    checkpoint: str,
    output: str,
    win_sec: float  = WIN_SEC,
    stride_sec: float = STRIDE_SEC,
    device_str: str = "cpu",
):
    import torch
    from finger_transformer import FingerTransformer

    device = torch.device(device_str)
    model  = FingerTransformer()
    ckpt   = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state  = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint}")

    chart_dir_p = Path(chart_dir).expanduser()
    paths = sorted(chart_dir_p.glob("*.npy"))

    all_embs: list[np.ndarray] = []
    all_meta: list[dict]       = []

    for npy_path in tqdm(paths, desc="embedding"):
        try:
            flow1, flow2, meta = _load_flows(npy_path)
        except Exception:
            continue

        # Chart duration from whichever flow is non-empty
        dur = max(
            (flow1[-1].time_sec if flow1 else 0.0),
            (flow2[-1].time_sec if flow2 else 0.0),
        )
        if dur < win_sec:
            continue

        title = str(meta.get("title") or npy_path.stem)

        t = 0.0
        while t + win_sec <= dur + stride_sec:
            t_end = t + win_sec

            e1 = _embed_flow_segment(flow1, t, t_end, model, device)
            e2 = _embed_flow_segment(flow2, t, t_end, model, device)

            # Skip windows where both sides are empty
            if np.all(e1 == 0) and np.all(e2 == 0):
                t += stride_sec
                continue

            emb = np.concatenate([e1, e2])           # (256,)
            emb /= np.linalg.norm(emb) + 1e-8        # L2 normalise

            all_embs.append(emb)
            all_meta.append({
                "title":    title,
                "path":     str(npy_path),
                "t_start":  round(t, 2),
                "t_end":    round(t_end, 2),
                "level":    meta.get("level"),
                "bpm":      meta.get("bpm"),
                "diftype":  meta.get("diftype", ""),
            })
            t += stride_sec

    embeddings = np.stack(all_embs).astype(np.float32)
    meta_json  = np.array([json.dumps(m) for m in all_meta])

    out = Path(output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=embeddings, meta=meta_json)

    print(f"\nSaved {len(all_embs)} windows → {out}")
    print(f"  shape: {embeddings.shape}  "
          f"win={win_sec}s  stride={stride_sec}s")


# ── Query ─────────────────────────────────────────────────────────────────────

def query_index(
    index_path: str,
    title: str,
    time_sec: float | None = None,
    k: int = 12,
    show_flip: bool = True,
):
    data  = np.load(Path(index_path).expanduser(), allow_pickle=True)
    embs  = data["embeddings"]                    # (N, 256)
    metas = [json.loads(s) for s in data["meta"]]

    # Find query windows for this chart
    q_indices = [
        i for i, m in enumerate(metas)
        if title.lower() in m["title"].lower()
    ]
    if not q_indices:
        print(f"No windows found for '{title}'.")
        _print_sample_titles(metas)
        return

    # If time specified, pick closest window; else use all and pick best match
    if time_sec is not None:
        q_indices = sorted(
            q_indices,
            key=lambda i: abs(metas[i]["t_start"] - time_sec),
        )[:1]

    # Pick the query window that has the highest mean similarity to others
    # (most "representative" if time not specified)
    if len(q_indices) > 1:
        candidate_embs = embs[q_indices]           # (Q, 256)
        mean_sim = (candidate_embs @ embs.T).mean(axis=1)
        best_local = int(np.argmax(mean_sim))
        qi = q_indices[best_local]
    else:
        qi = q_indices[0]

    q_meta = metas[qi]
    q_emb  = embs[qi]                             # (256,)
    q_flip = np.concatenate([q_emb[D_HALF:], q_emb[:D_HALF]])  # swap P1↔P2

    sims      = embs @ q_emb                      # (N,)
    sims_flip = embs @ q_flip                     # (N,)

    print(f"\nQuery: {q_meta['title']}  "
          f"t={q_meta['t_start']:.0f}–{q_meta['t_end']:.0f}s  "
          f"lv{q_meta['level']}  {q_meta['bpm']} BPM\n")

    # Collect top-k entries, considering both normal and flip similarity
    seen_windows: set[int] = {qi}

    def _header(label: str):
        print(f"{'─'*4} {label} {'─'*(54 - len(label))}")
        print(f"{'#':>3}  {'sim':>6}  {'t':>8}  {'lv':>3}  title")

    def _rows(sims_arr: np.ndarray, label: str):
        _header(label)
        order  = np.argsort(-sims_arr)
        shown  = 0
        for idx in order:
            if idx in seen_windows:
                continue
            m = metas[idx]
            print(f"{shown+1:>3}  {sims_arr[idx]:>6.3f}  "
                  f"{m['t_start']:>4.0f}–{m['t_end']:>3.0f}s  "
                  f"{str(m['level']):>3}  {m['title']}")
            seen_windows.add(idx)
            shown += 1
            if shown >= k:
                break

    _rows(sims, "NORMAL")
    if show_flip:
        print()
        _rows(sims_flip, "FLIP  (P1↔P2 swapped)")


def _print_sample_titles(metas: list[dict], n: int = 20):
    print("Sample titles:")
    seen: set[str] = set()
    for m in metas:
        t = m["title"]
        if t not in seen:
            print(f"  {t}")
            seen.add(t)
        if len(seen) >= n:
            break


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--chart-dir",  required=True)
    b.add_argument("--checkpoint", required=True)
    b.add_argument("--output",     default="embeddings/dp12_finger.npz")
    b.add_argument("--win",        type=float, default=WIN_SEC,    dest="win_sec")
    b.add_argument("--stride",     type=float, default=STRIDE_SEC, dest="stride_sec")
    b.add_argument("--device",     default="cpu")

    q = sub.add_parser("query")
    q.add_argument("--index",    required=True)
    q.add_argument("--title",    required=True)
    q.add_argument("--time",     type=float, default=None, dest="time_sec",
                   help="start of the query window in seconds")
    q.add_argument("--k",        type=int, default=12)
    q.add_argument("--no-flip",  action="store_true")

    args = p.parse_args()

    if args.cmd == "build":
        build_index(
            chart_dir=args.chart_dir,
            checkpoint=args.checkpoint,
            output=args.output,
            win_sec=args.win_sec,
            stride_sec=args.stride_sec,
            device_str=args.device,
        )
    elif args.cmd == "query":
        query_index(
            index_path=args.index,
            title=args.title,
            time_sec=args.time_sec,
            k=args.k,
            show_flip=not args.no_flip,
        )


if __name__ == "__main__":
    main()
