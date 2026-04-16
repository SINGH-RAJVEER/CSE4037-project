"""
Neural MCTS (AlphaZero-style).

Key differences from vanilla MCTS:
- No random rollouts; the value head replaces simulation.
- Policy head provides prior probabilities; PUCT replaces UCB1.
- All children are expanded at once on the first visit to a node.
"""

import math
import chess
import numpy as np
import torch
from encode import board_to_tensor, move_to_index, legal_move_indices

C_PUCT = 1.5          # exploration constant
DIRICHLET_ALPHA = 0.3 # noise added at root to encourage exploration
DIRICHLET_FRAC  = 0.25


class Node:
    __slots__ = ("board", "parent", "move", "prior",
                 "visit_count", "value_sum", "children", "is_expanded")

    def __init__(self, board: chess.Board, parent=None,
                 move: chess.Move | None = None, prior: float = 0.0):
        self.board       = board
        self.parent      = parent
        self.move        = move       # move that led to this node
        self.prior       = prior
        self.visit_count = 0
        self.value_sum   = 0.0
        self.children: dict[chess.Move, "Node"] = {}
        self.is_expanded = False

    # Mean action value (from the perspective of the player who JUST MOVED)
    @property
    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def puct(self, parent_visits: int) -> float:
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q + u


class NeuralMCTS:
    def __init__(self, model: torch.nn.Module, device: torch.device,
                 num_simulations: int = 200):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations

    # ── Neural network call ──────────────────────────────────────────────────

    @torch.no_grad()
    def _nn_eval(self, board: chess.Board) -> tuple[dict[chess.Move, float], float]:
        """Return ({move: prior}, value) for the current player."""
        tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(tensor)

        moves, indices = legal_move_indices(board)
        if not moves:
            return {}, float(value.item())

        logits = policy_logits[0, indices].cpu().numpy().astype(np.float64)
        logits -= logits.max()                  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()

        policy = {m: float(probs[i]) for i, m in enumerate(moves)}
        return policy, float(value.item())

    # ── Tree operations ──────────────────────────────────────────────────────

    def _expand(self, node: Node) -> float:
        """Evaluate node with the NN, create children, return value estimate."""
        policy, value = self._nn_eval(node.board)
        node.is_expanded = True
        for move, prior in policy.items():
            child_board = node.board.copy(stack=False)
            child_board.push(move)
            node.children[move] = Node(child_board, parent=node, move=move, prior=prior)
        return value     # value is from the current player's perspective

    def _select(self, node: Node) -> Node:
        """Walk down the tree following PUCT until we reach an unexpanded node."""
        while node.is_expanded and not node.board.is_game_over():
            parent_visits = node.visit_count
            node = max(node.children.values(), key=lambda c: c.puct(parent_visits))
        return node

    def _backpropagate(self, node: Node, value: float):
        """Walk back to root, flipping value sign at each edge."""
        while node is not None:
            node.visit_count += 1
            node.value_sum   += value
            value = -value
            node  = node.parent

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self, board: chess.Board, add_noise: bool = True
            ) -> dict[chess.Move, int]:
        """
        Run `num_simulations` iterations from `board`.
        Returns {move: visit_count} for the root's children.
        """
        root = Node(board.copy(stack=False))
        self._expand(root)

        # Dirichlet noise at root for exploration during training
        if add_noise and root.children:
            noise = np.random.dirichlet([DIRICHLET_ALPHA] * len(root.children))
            for node, n in zip(root.children.values(), noise):
                node.prior = (1 - DIRICHLET_FRAC) * node.prior + DIRICHLET_FRAC * n

        for _ in range(self.num_simulations):
            leaf = self._select(root)

            if leaf.board.is_game_over():
                result = leaf.board.result()
                value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)
                # Flip to perspective of the player who just moved into this node
                if leaf.board.turn == chess.WHITE:
                    value = -value
            else:
                value = self._expand(leaf)

            self._backpropagate(leaf, -value)   # negate: value is from child's POV

        return {m: c.visit_count for m, c in root.children.items()}

    def best_move(self, board: chess.Board) -> chess.Move:
        """Return the move with the highest visit count (temperature=0)."""
        visits = self.run(board, add_noise=False)
        return max(visits, key=visits.get)

    def policy_target(self, board: chess.Board,
                       temperature: float = 1.0) -> tuple[dict, np.ndarray]:
        """
        Return (visit_counts, policy_target) for supervised training.
        policy_target is a 4672-element float32 array (visit-count distribution).
        """
        visits = self.run(board, add_noise=True)
        moves  = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float64)

        if temperature != 1.0:
            counts = counts ** (1.0 / temperature)
        counts /= counts.sum()

        target = np.zeros(4672, dtype=np.float32)
        for move, prob in zip(moves, counts):
            target[move_to_index(move)] = float(prob)

        return visits, target
