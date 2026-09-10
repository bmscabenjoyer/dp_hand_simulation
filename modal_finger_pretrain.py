"""
Modal training for IIDX finger token transformer.

Setup (one-time):
    modal volume create iidx-data       # already exists if iidx_dp_learning is set up
    modal volume create iidx-ckpts      # already exists

Run:
    modal run modal_finger_pretrain.py

Download checkpoint:
    modal volume get iidx-ckpts finger_pretrain/encoder_best.pt checkpoints/finger_pretrain/encoder_best.pt
"""

import modal

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "tqdm")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .add_local_file("chart_io.py",           "/app/chart_io.py")
    .add_local_file("finger_model.py",       "/app/finger_model.py")
    .add_local_file("dp_fingering.py",       "/app/dp_fingering.py")
    .add_local_file("simulate.py",           "/app/simulate.py")
    .add_local_file("finger_tokenize.py",    "/app/finger_tokenize.py")
    .add_local_file("finger_dataset.py",     "/app/finger_dataset.py")
    .add_local_file("finger_transformer.py", "/app/finger_transformer.py")
)

# ── Volumes ───────────────────────────────────────────────────────────────────

data_vol = modal.Volume.from_name("iidx-data")
ckpt_vol = modal.Volume.from_name("iidx-ckpts", create_if_missing=True)

DATA_DIR  = "/iidx-data"
CKPT_DIR  = "/iidx-ckpts"
CHART_DIR = f"{DATA_DIR}/dp12_active/charts"

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("iidx-finger-pretrain")


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
    timeout=86400,
)
def train(
    chart_dir:  str   = CHART_DIR,
    d_model:    int   = 128,
    n_layers:   int   = 4,
    n_heads:    int   = 4,
    d_ff:       int   = 512,
    window:     int   = 64,
    stride:     int   = 32,
    mask_rate:  float = 0.15,
    batch_size: int   = 128,
    lr:         float = 1e-3,
    epochs:     int   = 200,
    ckpt_name:  str   = "finger_pretrain",
):
    import sys
    sys.path.insert(0, "/app")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from pathlib import Path

    from finger_dataset import FingerTokenDataset
    from finger_transformer import FingerTransformer, model_size
    from finger_tokenize import NUM_FINGERS, NUM_LANES

    device = torch.device("cuda")

    # ── Data ──────────────────────────────────────────────────────────────────

    print(f"Preprocessing charts from {chart_dir} ...")
    dataset = FingerTokenDataset(
        chart_dir, window=window, stride=stride, mask_rate=mask_rate
    )
    print(f"  {len(dataset)} windows across {len(dataset.sequences)} sequences")

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────

    model = FingerTransformer(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff
    ).to(device)
    print(f"  Model: {model_size(model)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    ckpt_dir = Path(CKPT_DIR) / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = finger_correct = lane_correct = total_masked = 0

        for batch in loader:
            input_ids    = batch["input_ids"].to(device)      # (B, W, 3)
            lbl_finger   = batch["label_finger"].to(device)   # (B, W)
            lbl_lane     = batch["label_lane"].to(device)     # (B, W)

            finger_logits, lane_logits = model(input_ids)     # (B, W, C)

            loss = (
                criterion(finger_logits.reshape(-1, NUM_FINGERS), lbl_finger.reshape(-1))
                + criterion(lane_logits.reshape(-1, NUM_LANES),   lbl_lane.reshape(-1))
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

            # accuracy on masked positions
            mask = lbl_finger != -100
            n = mask.sum().item()
            if n > 0:
                finger_correct += (finger_logits.argmax(-1)[mask] == lbl_finger[mask]).sum().item()
                lane_correct   += (lane_logits.argmax(-1)[mask]   == lbl_lane[mask]).sum().item()
                total_masked   += n

        scheduler.step()
        avg_loss = total_loss / len(loader)
        f_acc = finger_correct / total_masked if total_masked else 0
        l_acc = lane_correct   / total_masked if total_masked else 0

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:>4}/{epochs}  "
                f"loss={avg_loss:.4f}  "
                f"finger_acc={f_acc:.3f}  lane_acc={l_acc:.3f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict(), "loss": best_loss},
                ckpt_dir / "encoder_best.pt",
            )
            ckpt_vol.commit()

    torch.save(
        {"epoch": epochs, "state_dict": model.state_dict(), "loss": avg_loss},
        ckpt_dir / "encoder_final.pt",
    )
    ckpt_vol.commit()

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"  modal volume get iidx-ckpts {ckpt_name}/encoder_best.pt "
          f"checkpoints/{ckpt_name}/encoder_best.pt")


@app.local_entrypoint()
def main():
    train.remote()
