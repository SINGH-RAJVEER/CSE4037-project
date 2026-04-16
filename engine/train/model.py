"""
AlphaZero-style ResNet for chess.

Input:  (batch, 14, 8, 8) float32
Output: policy logits (batch, 4672), value (batch, 1) ∈ [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessNet(nn.Module):
    """
    Residual tower with a policy head and a value head.

    num_filters:    width of the residual tower (128 is a good default)
    num_res_blocks: depth of the tower (10 is a good default; use 20 for stronger play)
    """

    def __init__(self, num_filters: int = 128, num_res_blocks: int = 10):
        super().__init__()

        # Input stem
        self.stem = nn.Sequential(
            nn.Conv2d(14, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
        )

        # Residual tower
        self.tower = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])

        # Policy head: 2-filter conv → flatten → FC to 4672
        self.policy_conv = nn.Conv2d(num_filters, 32, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, 4672)

        # Value head: 3-filter conv → flatten → FC 64 → FC 1 → tanh
        self.value_conv  = nn.Conv2d(num_filters, 3, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(3)
        self.value_fc1   = nn.Linear(3 * 8 * 8, 64)
        self.value_fc2   = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.tower(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.flatten(1))          # (batch, 4672) logits

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.flatten(1)))
        v = torch.tanh(self.value_fc2(v))          # (batch, 1) ∈ [-1, 1]

        return p, v

    def predict(self, board_tensor: torch.Tensor):
        """Convenience wrapper: runs forward in eval mode, returns numpy arrays."""
        self.eval()
        with torch.no_grad():
            p, v = self(board_tensor)
        return p, v
