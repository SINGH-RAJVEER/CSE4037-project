"""
Board state and move encoding matching the Rust inference side.

Input tensor: 14 planes × 8 × 8 float32
  Planes  0-5:  white pawns, knights, bishops, rooks, queens, kings
  Planes  6-11: black pawns, knights, bishops, rooks, queens, kings
  Plane  12:    side-to-move (all 1s if white, all 0s if black)
  Plane  13:    en-passant target square (single 1 if any)

Policy tensor: 4672 floats (73 planes × 64 squares), indexed as
  plane * 64 + from_rank * 8 + from_file
  Planes  0-55:  queen moves — dir * 7 + (dist-1), 8 dirs × 7 distances
  Planes 56-63:  knight moves — 8 possible offsets
  Planes 64-72:  underpromotions — piece * 3 + (df+1), pieces={N,B,R}, df∈{-1,0,1}
"""

import numpy as np
import chess

# ── Piece-plane mapping ──────────────────────────────────────────────────────

_PIECE_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4,
    (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.BLACK): 11,
}

# ── Move-plane mappings ──────────────────────────────────────────────────────

# 8 compass directions (dr, df): N NE E SE S SW W NW
_QUEEN_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

# 8 knight offsets
_KNIGHT_OFFSETS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_KNIGHT_OFFSET_INDEX = {off: i for i, off in enumerate(_KNIGHT_OFFSETS)}

# Underpromotion pieces (queen promotion is encoded as a queen move)
_UNDER_PROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Return a (14, 8, 8) float32 numpy array encoding the board state."""
    planes = np.zeros((14, 8, 8), dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            planes[_PIECE_PLANE[(piece.piece_type, piece.color)], rank, file] = 1.0

    if board.turn == chess.WHITE:
        planes[12] = 1.0

    if board.ep_square is not None:
        planes[13, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """Encode a chess.Move as a flat index into the 4672-element policy vector."""
    fr = move.from_square
    from_rank = chess.square_rank(fr)
    from_file = chess.square_file(fr)
    dr = chess.square_rank(move.to_square) - from_rank
    df = chess.square_file(move.to_square) - from_file

    # Underpromotion (queen promotions fall through to queen-move encoding)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        promo_idx = _UNDER_PROMO[move.promotion]
        dir_offset = df + 1           # df ∈ {-1, 0, 1} → {0, 1, 2}
        plane = 64 + promo_idx * 3 + dir_offset
        return plane * 64 + from_rank * 8 + from_file

    # Knight move
    knight_i = _KNIGHT_OFFSET_INDEX.get((dr, df))
    if knight_i is not None:
        plane = 56 + knight_i
        return plane * 64 + from_rank * 8 + from_file

    # Queen move (includes queen promotions)
    for dir_i, (dir_r, dir_f) in enumerate(_QUEEN_DIRS):
        if dir_r == 0 and dir_f == 0:
            continue
        if dir_r != 0:
            if dr % dir_r != 0:
                continue
            dist = dr // dir_r
        else:
            if dr != 0:
                continue
            dist = df // dir_f if dir_f != 0 else 0

        if 1 <= dist <= 7 and dr == dir_r * dist and df == dir_f * dist:
            plane = dir_i * 7 + (dist - 1)
            return plane * 64 + from_rank * 8 + from_file

    raise ValueError(f"Cannot encode move {move} (dr={dr}, df={df})")


def legal_move_indices(board: chess.Board) -> tuple[list, list]:
    """Return (moves, indices) for all legal moves in the position."""
    moves = list(board.legal_moves)
    indices = [move_to_index(m) for m in moves]
    return moves, indices
