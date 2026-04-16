"""
AlphaZero-style training loop.

Usage:
    # Install uv (once): https://docs.astral.sh/uv/getting-started/installation/

    # From engine/train/ — install deps and run:
    uv sync
    # For CUDA PyTorch, add it explicitly (example for CUDA 12.4):
    uv add torch --extra-index-url https://download.pytorch.org/whl/cu124

    uv run python train.py

    # Resume from a checkpoint:
    uv run python train.py --resume checkpoints/iter_010.pt

    # After training, the final model is written to ../model.onnx
    # (i.e. engine/model.onnx) which the Rust engine loads at startup.

Hyper-parameters (edit below or pass as --arg):
    --filters       128     residual tower width
    --blocks        10      residual tower depth
    --simulations   200     MCTS simulations per move
    --games         25      self-play games per iteration
    --iterations    100     total training iterations
    --lr            1e-3    initial Adam learning rate
    --batch         256     training batch size
    --epochs        5       gradient epochs per iteration
    --buffer        500000  max replay buffer size
"""

import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model import ChessNet
from mcts import NeuralMCTS
from self_play import play_one_game


# ── Dataset ──────────────────────────────────────────────────────────────────

class ReplayDataset(Dataset):
    def __init__(self, data: list):
        states   = np.array([d[0] for d in data], dtype=np.float32)
        policies = np.array([d[1] for d in data], dtype=np.float32)
        values   = np.array([d[2] for d in data], dtype=np.float32)

        self.states   = torch.from_numpy(states)
        self.policies = torch.from_numpy(policies)
        self.values   = torch.from_numpy(values).unsqueeze(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


# ── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, device) -> float:
    model.train()
    total_loss = 0.0
    use_cuda = device.type == "cuda"

    for states, policies, values in loader:
        states   = states.to(device, non_blocking=use_cuda)
        policies = policies.to(device, non_blocking=use_cuda)
        values   = values.to(device, non_blocking=use_cuda)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                policy_logits, value_pred = model(states)
                policy_loss = -(policies * torch.log_softmax(policy_logits, dim=1)).sum(1).mean()
                value_loss  = nn.functional.mse_loss(value_pred, values)
                loss        = policy_loss + value_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            policy_logits, value_pred = model(states)
            policy_loss = -(policies * torch.log_softmax(policy_logits, dim=1)).sum(1).mean()
            value_loss  = nn.functional.mse_loss(value_pred, values)
            loss        = policy_loss + value_loss
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def export_onnx(model, path: str, device):
    model.eval()
    dummy = torch.zeros(1, 14, 8, 8, device=device)
    torch.onnx.export(
        model.cpu(), dummy.cpu(), path,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    model.to(device)
    print(f"  → exported ONNX model to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume",      default=None)
    p.add_argument("--device",      choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--filters",     type=int,   default=128)
    p.add_argument("--blocks",      type=int,   default=10)
    p.add_argument("--simulations", type=int,   default=200)
    p.add_argument("--games",       type=int,   default=25)
    p.add_argument("--iterations",  type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--batch",       type=int,   default=256)
    p.add_argument("--epochs",      type=int,   default=5)
    p.add_argument("--buffer",      type=int,   default=500_000)
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested with --device cuda, but torch.cuda.is_available() is false.\n"
                "Make sure you installed the CUDA-enabled PyTorch wheel:\n"
                "  uv add torch --extra-index-url https://download.pytorch.org/whl/cu124\n"
                "and that your NVIDIA driver is visible (nvidia-smi should work)."
            )
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available; training on CPU.")

    model = ChessNet(num_filters=args.filters, num_res_blocks=args.blocks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    start_iter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iteration"] + 1
        print(f"Resumed from iteration {start_iter}")

    mcts          = NeuralMCTS(model, device, num_simulations=args.simulations)
    replay_buffer = []

    os.makedirs("checkpoints", exist_ok=True)

    for iteration in range(start_iter, args.iterations):
        print(f"\n{'='*55}")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*55}")

        # ── Self-play ────────────────────────────────────────────
        model.eval()
        t0 = time.time()
        new_samples = []
        for g in range(args.games):
            game_data = play_one_game(mcts)
            new_samples.extend(game_data)
            print(f"  game {g+1:>2}/{args.games}: {len(game_data):>3} positions")
        print(f"  self-play: {time.time()-t0:.1f}s  |  new samples: {len(new_samples)}")

        replay_buffer.extend(new_samples)
        if len(replay_buffer) > args.buffer:
            replay_buffer = replay_buffer[-args.buffer:]
        print(f"  replay buffer: {len(replay_buffer)} samples")

        # ── Training ─────────────────────────────────────────────
        dataset = ReplayDataset(replay_buffer)
        num_workers = 4 if device.type == "cuda" else 0
        loader  = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
        )

        t0 = time.time()
        for ep in range(args.epochs):
            loss = train_one_epoch(model, loader, optimizer, scaler, device)
            print(f"  epoch {ep+1}/{args.epochs}: loss={loss:.4f}")
        print(f"  training: {time.time()-t0:.1f}s")

        scheduler.step()

        # ── Checkpoint ───────────────────────────────────────────
        torch.save({
            "iteration": iteration,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, f"checkpoints/iter_{iteration+1:04d}.pt")

        # ── Periodic ONNX export ─────────────────────────────────
        if (iteration + 1) % 10 == 0:
            export_onnx(model, f"checkpoints/model_iter_{iteration+1:04d}.onnx", device)

    # Final export consumed by the Rust engine
    export_onnx(model, "../model.onnx", device)
    print("\nTraining complete. Model exported to engine/model.onnx")


if __name__ == "__main__":
    main()
