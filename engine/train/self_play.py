"""
Self-play game generation.

Each game produces a list of (state_tensor, policy_target, value) tuples
where value is from the perspective of the player to move in that state.
"""

import chess
import numpy as np
from encode import board_to_tensor, move_to_index
from mcts import NeuralMCTS

TEMP_THRESHOLD = 30   # use temperature=1 for the first N half-moves, then 0.1


def play_one_game(mcts: NeuralMCTS) -> list[tuple]:
    """
    Play one self-play game.
    Returns [(state_tensor, policy_target, value), ...] with values filled
    retrospectively once the game outcome is known.
    """
    board    = chess.Board()
    states   = []
    policies = []

    half_move = 0
    while not board.is_game_over(claim_draw=True):
        temp = 1.0 if half_move < TEMP_THRESHOLD else 0.1
        _, policy_target = mcts.policy_target(board, temperature=temp)

        states.append(board_to_tensor(board))
        policies.append(policy_target)

        # Sample move proportional to visit counts
        legal      = list(board.legal_moves)
        probs      = np.array([policy_target[move_to_index(m)] for m in legal],
                               dtype=np.float64)
        if probs.sum() < 1e-9:
            probs = np.ones(len(legal), dtype=np.float64)
        probs /= probs.sum()

        chosen = np.random.choice(len(legal), p=probs)
        board.push(legal[chosen])
        half_move += 1

    # Game outcome from White's perspective
    result = board.result(claim_draw=True)
    white_value = {"1-0": 1.0, "0-1": -1.0}.get(result, 0.0)

    # Assign values from each position's player-to-move perspective
    # Even half-moves = White to move, odd = Black to move
    values = []
    for i in range(len(states)):
        if i % 2 == 0:          # White to move
            values.append(white_value)
        else:                   # Black to move
            values.append(-white_value)

    return list(zip(states, policies, values))
