"""
Chess rules glue for AlphaZero: board encoding, the 8x8x73 move encoding, and game-over detection.

The rules themselves come from python-chess (CPU-only). Unlike the Connect-4 `Connect4Env` this is not
vectorised -- one MCTS expansion (board copy + push, legal-move generation, encoding) costs ~100us of
Python, and throughput comes from running many self-play processes instead (see `selfplay.py`).

Canonical frame: like `canonicalise_obs` in the Connect-4 code, the network always sees the board from
the side to move. For black to move we flip the board vertically (rank 1 <-> rank 8) and swap colours,
so "my pawns move up the board" always holds. Moves are encoded in the same flipped frame, so the
policy for "push the king's pawn two squares" is the same action index for both colours.

Input planes (19, 8, 8), indexed [plane, rank, file] in the canonical frame (rank 0 = my back rank):
    0-5    my   P N B R Q K
    6-11   their P N B R Q K
    12-15  castling rights: my king-side, my queen-side, their king-side, their queen-side (all-ones planes)
    16     en-passant target square (only if an en-passant capture is actually legal)
    17     position has occurred before (repetition flag, all-ones plane)
    18     half-move clock (raw count stored as uint8; the network input scales it by 1/100)

Action index = plane * 64 + from_square, in the canonical frame (this is exactly the flattened
(73, 8, 8) output of a conv policy head). The 73 planes are:
    0-55   queen-style moves: 8 directions x distance 1..7 (also covers king moves, castling as a
           2-square king move, pawn pushes/captures, en passant, and promotion to a queen)
    56-63  knight moves
    64-72  under-promotions: {knight, bishop, rook} x {capture-left, straight, capture-right}
"""
from __future__ import annotations

import chess
import numpy as np

N_IN_PLANES = 19
N_MOVE_PLANES = 73
N_ACTIONS = 64 * N_MOVE_PLANES        # 4672
MAX_LEGAL = 256                       # the most legal moves in any chess position is 218

QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]   # (dfile, drank)
KNIGHT_DIRS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
UNDERPROMOS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}

_FULL = 0xFFFF_FFFF_FFFF_FFFF


def _build_tables():
    """Lookup tables between (from, to, promotion) in the canonical frame and action indices."""
    move_idx = [[-1] * 64 for _ in range(64)]          # [from][to] -> action (non-underpromotion moves)
    under_idx = {}                                     # (from, to, piece) -> action
    a_from = np.arange(N_ACTIONS) % 64
    a_to = np.full(N_ACTIONS, -1)                      # -1 = this (square, plane) leaves the board
    a_promo = np.zeros(N_ACTIONS, dtype=np.int64)      # under-promotion piece, 0 otherwise
    for sq in range(64):
        f, r = sq % 8, sq // 8
        for d, (df, dr) in enumerate(QUEEN_DIRS):
            for dist in range(1, 8):
                ff, rr = f + df * dist, r + dr * dist
                if 0 <= ff < 8 and 0 <= rr < 8:
                    a = (d * 7 + dist - 1) * 64 + sq
                    move_idx[sq][rr * 8 + ff] = a
                    a_to[a] = rr * 8 + ff
        for k, (df, dr) in enumerate(KNIGHT_DIRS):
            ff, rr = f + df, r + dr
            if 0 <= ff < 8 and 0 <= rr < 8:
                a = (56 + k) * 64 + sq
                move_idx[sq][rr * 8 + ff] = a
                a_to[a] = rr * 8 + ff
        if r == 6:                                     # under-promotions only happen from my 7th rank
            for p_i, piece in enumerate(UNDERPROMOS):
                for dx in (-1, 0, 1):
                    if 0 <= f + dx < 8:
                        a = (64 + 3 * p_i + dx + 1) * 64 + sq
                        under_idx[(sq, 56 + f + dx, piece)] = a
                        a_to[a] = 56 + f + dx
                        a_promo[a] = piece
    return move_idx, under_idx, a_from, a_to, a_promo


_MOVE_IDX, _UNDER_IDX, ACTION_FROM, ACTION_TO, ACTION_PROMO = _build_tables()


# --------------------------------------------------------------------------- moves <-> actions
def move_to_action(move: chess.Move, turn: chess.Color) -> int:
    """Action index (0..4671) of `move`, played by `turn`, in that player's canonical frame."""
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f ^= 56                                        # flip rank: square_mirror
        t ^= 56
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return _UNDER_IDX[(f, t, move.promotion)]
    return _MOVE_IDX[f][t]


def legal_actions(board: chess.Board, moves: list[chess.Move]) -> list[int]:
    """Action indices for a list of legal moves (same order). Inlined `move_to_action` for speed."""
    flip = 56 if board.turn == chess.BLACK else 0
    mi, ui = _MOVE_IDX, _UNDER_IDX
    out = []
    for m in moves:
        f, t = m.from_square ^ flip, m.to_square ^ flip
        p = m.promotion
        out.append(ui[(f, t, p)] if (p is not None and p != chess.QUEEN) else mi[f][t])
    return out


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """Inverse of `move_to_action` for the side to move on `board` (queen promotion is implied)."""
    f, t, promo = int(ACTION_FROM[action]), int(ACTION_TO[action]), int(ACTION_PROMO[action])
    if t < 0:
        raise ValueError(f"action {action} leaves the board")
    if board.turn == chess.BLACK:
        f ^= 56
        t ^= 56
    if promo == 0 and board.piece_type_at(f) == chess.PAWN and chess.square_rank(t) in (0, 7):
        promo = chess.QUEEN
    return chess.Move(f, t, promotion=promo or None)


# --------------------------------------------------------------------------- board -> planes
def encode_board(board: chess.Board) -> np.ndarray:
    """(19, 8, 8) uint8 planes of `board` from the side to move's perspective (see module docstring)."""
    us = board.turn
    them = not us
    occ_us, occ_them = board.occupied_co[us], board.occupied_co[them]
    piece_bbs = (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    bbs = [bb & occ_us for bb in piece_bbs] + [bb & occ_them for bb in piece_bbs]
    bbs.append(_FULL if board.has_kingside_castling_rights(us) else 0)
    bbs.append(_FULL if board.has_queenside_castling_rights(us) else 0)
    bbs.append(_FULL if board.has_kingside_castling_rights(them) else 0)
    bbs.append(_FULL if board.has_queenside_castling_rights(them) else 0)
    bbs.append(chess.BB_SQUARES[board.ep_square] if board.ep_square is not None and board.has_legal_en_passant() else 0)
    bbs.append(_FULL if board.is_repetition(2) else 0)
    if us == chess.BLACK:
        bbs = [chess.flip_vertical(bb) for bb in bbs]
    bits = np.unpackbits(np.array(bbs, dtype=np.uint64).view(np.uint8), bitorder="little")
    out = np.empty((N_IN_PLANES, 8, 8), dtype=np.uint8)
    out[:18] = bits.reshape(18, 8, 8)                  # square s = rank*8 + file -> [rank, file]
    out[18] = min(board.halfmove_clock, 255)
    return out


# --------------------------------------------------------------------------- game over / material
def terminal_value(board: chess.Board, n_legal: int) -> float | None:
    """Game-theoretic value for the side to move if the game is over, else None.

    -1 when checkmated; 0 for stalemate, insufficient material, the 50-move rule, or threefold
    repetition (we treat the claimable draws as automatic, which keeps self-play games short).
    """
    if n_legal == 0:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return None


def termination_reason(board: chess.Board, n_legal: int) -> str:
    if n_legal == 0:
        return "checkmate" if board.is_check() else "stalemate"
    if board.is_insufficient_material():
        return "insufficient"
    if board.halfmove_clock >= 100:
        return "fifty-move"
    if board.is_repetition(3):
        return "repetition"
    return "ongoing"


def material(board: chess.Board, color: chess.Color) -> int:
    """Classical material count (P=1, N=B=3, R=5, Q=9) for `color`."""
    occ = board.occupied_co[color]
    return ((board.pawns & occ).bit_count() + 3 * (board.knights & occ).bit_count()
            + 3 * (board.bishops & occ).bit_count() + 5 * (board.rooks & occ).bit_count()
            + 9 * (board.queens & occ).bit_count())


def adjudicate(board: chess.Board, threshold: int) -> float:
    """Result from WHITE's perspective for a game stopped at the ply cap: a win for the side up at
    least `threshold` points of material, else a draw. (threshold <= 0 disables: always a draw.)"""
    if threshold <= 0:
        return 0.0
    diff = material(board, chess.WHITE) - material(board, chess.BLACK)
    return 1.0 if diff >= threshold else -1.0 if diff <= -threshold else 0.0
