"""
Vectorised chess rules in PyTorch: a batch of B games advanced in lock-step on the GPU, the chess
counterpart of the Connect-4 `Connect4Env`. Ported from pgx's JAX implementation
(pgx/_src/games/chess.py, Apache-2.0), but using this package's conventions so it is a drop-in for the
python-chess glue in `chess_env.py`:

  * squares are python-chess style, s = rank * 8 + file, and the board is always stored in the side to
    move's *canonical frame* (flipped vertically and colour-negated after every move), so "my" pieces
    are positive and my pawns move up the board -- exactly the frame `chess_env.encode_board` uses
  * actions are `chess_env`'s 8x8x73 encoding (plane * 64 + from-square, in the canonical frame)
  * observations are `chess_env.encode_board`'s 19 planes, bit-for-bit (tested against python-chess)
  * draws follow `chess_env.terminal_value`: stalemate, insufficient material (python-chess's rule),
    the 50-move rule and threefold repetition are automatic

Legal move generation (as in pgx): for every own piece, look up its geometric destinations, keep the
ones with a clear path / correct pawn behaviour, then apply each candidate to a copy of the board and
discard it if the mover's king is attacked afterwards. Castling (king and transit squares not attacked)
and under-promotions are added on top. Everything is gathers against precomputed tables; there are no
per-game Python branches.

Repetitions use an additive Zobrist hash of (canonical board, colour, castling rights, legal en-passant
square) and a history of the last `HIST` positions; a position repeats when an equal hash appears an
even number of plies back within the current half-move-clock window (python-chess's definition).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import chess
import numpy as np
import torch

import chapter2_rl.exercises.part5_alphazero_chess.chess_env as E

EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(7)
HIST = 102                  # half-move clock <= 100 within a game, so 101 positions back is enough
A_OO = E.move_to_action(chess.Move(chess.E1, chess.G1), chess.WHITE)
A_OOO = E.move_to_action(chess.Move(chess.E1, chess.C1), chess.WHITE)
REASONS = ["ongoing", "checkmate", "stalemate", "insufficient", "fifty-move", "repetition"]


# --------------------------------------------------------------------------- precomputed tables
def _build_tables():
    dest = -np.ones((7, 64, 27), np.int64)          # geometric destinations per (piece, from)
    can = np.zeros((7, 64, 64), bool)               # piece on `from` could move to `to` (pawns: upward)
    for s in range(64):
        r0, f0 = divmod(s, 8)
        d = {p: [] for p in range(7)}
        for t in range(64):
            if t == s:
                continue
            r1, f1 = divmod(t, 8)
            dr, df = r1 - r0, f1 - f0
            if (dr == 1 and abs(df) <= 1) or (r0 == 1 and dr == 2 and df == 0):
                d[PAWN].append(t)
            if (abs(dr), abs(df)) in ((1, 2), (2, 1)):
                d[KNIGHT].append(t)
            if abs(dr) == abs(df):
                d[BISHOP].append(t)
            if dr == 0 or df == 0:
                d[ROOK].append(t)
            if abs(dr) == abs(df) or dr == 0 or df == 0:
                d[QUEEN].append(t)
            if max(abs(dr), abs(df)) == 1:
                d[KING].append(t)
        for p in range(1, 7):
            dest[p, s, :len(d[p])] = d[p]
            can[p, s, d[p]] = True
    near = -np.ones((64, 16), np.int64)             # king- or knight-distance squares
    far = -np.ones((64, 19), np.int64)              # queen-line squares at distance >= 2
    between = -np.ones((64, 64, 6), np.int64)       # squares strictly between two squares on a line
    for s in range(64):
        nd = sorted(set(dest[KING, s][dest[KING, s] >= 0]) | set(dest[KNIGHT, s][dest[KNIGHT, s] >= 0]))
        near[s, :len(nd)] = nd
        fd = sorted(set(dest[QUEEN, s][dest[QUEEN, s] >= 0]) - set(dest[KING, s][dest[KING, s] >= 0]))
        far[s, :len(fd)] = fd
        r0, f0 = divmod(s, 8)
        for t in range(64):
            r1, f1 = divmod(t, 8)
            if not (r0 == r1 or f0 == f1 or abs(r1 - r0) == abs(f1 - f0)) or s == t:
                continue
            dr, df = np.sign(r1 - r0), np.sign(f1 - f0)
            for i in range(6):
                r, f = r0 + dr * (i + 1), f0 + df * (i + 1)
                if (r, f) == (r1, f1):
                    break
                between[s, t, i] = r * 8 + f
    move_action = np.array(E._MOVE_IDX, dtype=np.int64)            # [from][to] -> action (non-underpromotion)
    under_q, under_a, under_from = [], [], []                      # queen-promotion action -> under-promotions
    for f in range(48, 56):
        for dx in (-1, 0, 1):
            if 0 <= f % 8 + dx < 8:
                to = f + 8 + dx
                under_q.append(move_action[f, to])
                under_a.append([E._UNDER_IDX[(f, to, p)] for p in E.UNDERPROMOS])
                under_from.append(f)
    dark = np.array([(s // 8 + s % 8) % 2 == 0 for s in range(64)])  # a1 is dark
    rays = -np.ones((64, 8, 7), np.int64)            # squares along each queen direction, nearest first
    for s in range(64):
        r0, f0 = divmod(s, 8)
        for d, (df, dr) in enumerate(E.QUEEN_DIRS):
            for i in range(7):
                r, f = r0 + dr * (i + 1), f0 + df * (i + 1)
                if not (0 <= r < 8 and 0 <= f < 8):
                    break
                rays[s, d, i] = r * 8 + f
    orth = np.array([df == 0 or dr == 0 for df, dr in E.QUEEN_DIRS])
    knight_sq = -np.ones((64, 8), np.int64)
    knight_sq[:, :] = np.array([list(dest[KNIGHT, s, :8]) for s in range(64)])
    pawn_checker = -np.ones((64, 2), np.int64)       # where an enemy pawn must stand to attack square s
    enemy_pawn_att = -np.ones((64, 2), np.int64)     # squares an enemy pawn on s attacks (it moves down)
    for s in range(64):
        r, f = divmod(s, 8)
        if r < 7 and f > 0: pawn_checker[s, 0] = s + 7
        if r < 7 and f < 7: pawn_checker[s, 1] = s + 9
        if r > 0 and f < 7: enemy_pawn_att[s, 0] = s - 7
        if r > 0 and f > 0: enemy_pawn_att[s, 1] = s - 9
    return dict(dest=dest, can=can, near=near, far=far, between=between, move_action=move_action,
                under_q=np.array(under_q), under_a=np.array(under_a), under_from=np.array(under_from), dark=dark, rays=rays, orth=orth, knight_sq=knight_sq,
                pawn_checker=pawn_checker, enemy_pawn_att=enemy_pawn_att,
                a_from=E.ACTION_FROM.astype(np.int64), a_to=E.ACTION_TO.astype(np.int64),
                a_promo=E.ACTION_PROMO.astype(np.int64))


_NP = _build_tables()
_DEV_TABLES: dict = {}


def tables(device) -> dict:
    """The lookup tables (and Zobrist keys) as tensors on `device` (cached)."""
    key = str(device)
    if key not in _DEV_TABLES:
        T = {k: torch.as_tensor(v, device=device) for k, v in _NP.items()}
        g = torch.Generator().manual_seed(20231117)
        rnd = lambda *s: torch.randint(-(2 ** 62), 2 ** 62, s, generator=g, dtype=torch.int64).to(device)
        T["z_piece"], T["z_white"], T["z_castle"], T["z_ep"] = rnd(64, 13), rnd(1), rnd(4), rnd(64)
        T["sq"] = torch.arange(64, device=device)
        _DEV_TABLES[key] = T
    return _DEV_TABLES[key]


# --------------------------------------------------------------------------- state
@dataclass
class ChessState:
    """A batch of positions. All fields have leading dim B. Derived fields (from `legal` down) are
    filled in by `finalize`."""
    board: torch.Tensor        # (B, 64) int8, canonical frame: >0 = side to move's pieces
    white: torch.Tensor        # (B,) bool, white to move
    castling: torch.Tensor     # (B, 4) bool: my king-side, my queen-side, their king-side, their queen-side
    ep: torch.Tensor           # (B,) int64 canonical en-passant target square after a double push, else -1
    halfmove: torch.Tensor     # (B,) int64
    ply: torch.Tensor          # (B,) int64 plies since the game started
    hist: torch.Tensor         # (B, HIST) int64 position hashes; hist[:, k] = k plies ago (0 = none)
    legal: torch.Tensor = None       # (B, 4672) bool legal-action mask
    in_check: torch.Tensor = None    # (B,) bool
    ep_legal: torch.Tensor = None    # (B,) bool: an en-passant capture is actually legal
    reps: torch.Tensor = None        # (B,) int64 occurrences of this position (1 = first time)
    terminal: torch.Tensor = None    # (B,) bool
    value: torch.Tensor = None       # (B,) float32 result for the side to move if terminal (-1 mated, 0 draw)
    reason: torch.Tensor = None      # (B,) int64 index into REASONS

    def __len__(self):
        return self.board.shape[0]

    def index(self, idx) -> "ChessState":
        return ChessState(**{f.name: (getattr(self, f.name)[idx] if getattr(self, f.name) is not None else None)
                             for f in fields(self)})

    def assign(self, mask: torch.Tensor, other: "ChessState") -> None:
        """In place: rows where `mask` take `other`'s values (other has the same batch size)."""
        for f in fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            m = mask.view(-1, *([1] * (a.dim() - 1)))
            setattr(self, f.name, torch.where(m, b, a))

    @staticmethod
    def cat(states: list["ChessState"]) -> "ChessState":
        return ChessState(**{f.name: torch.cat([getattr(s, f.name) for s in states]) for f in fields(ChessState)})


def initial_state(B: int, device) -> ChessState:
    b = chess.Board()
    return from_boards([b], device).index(torch.zeros(B, dtype=torch.long, device=device))


def from_boards(boards: list[chess.Board], device) -> ChessState:
    """States for python-chess boards. The move stack (if any) is replayed so repetition history is exact."""
    out = []
    for b in boards:
        root = b.root() if b.move_stack else b
        s = _from_fen_board(root, device)
        for mv in b.move_stack:
            s = step(s, torch.tensor([E.move_to_action(mv, s_turn(s))], device=device))
        out.append(s)
    return ChessState.cat(out)


def s_turn(s: ChessState) -> chess.Color:
    return chess.WHITE if bool(s.white[0]) else chess.BLACK


def _from_fen_board(b: chess.Board, device) -> ChessState:
    us = b.turn
    board = np.zeros(64, np.int8)
    for sq, pc in b.piece_map().items():
        board[sq if us == chess.WHITE else sq ^ 56] = pc.piece_type if pc.color == us else -pc.piece_type
    castling = [b.has_kingside_castling_rights(us), b.has_queenside_castling_rights(us),
                b.has_kingside_castling_rights(not us), b.has_queenside_castling_rights(not us)]
    ep = -1 if b.ep_square is None else (b.ep_square if us == chess.WHITE else b.ep_square ^ 56)
    t = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    s = ChessState(board=t(board, torch.int8)[None], white=t([us == chess.WHITE], torch.bool),
                   castling=t([castling], torch.bool), ep=t([ep], torch.int64),
                   halfmove=t([b.halfmove_clock], torch.int64), ply=t([0], torch.int64),
                   hist=torch.zeros((1, HIST), dtype=torch.int64, device=device))
    return finalize(s, new_position=True)


def to_boards(s: ChessState) -> list[chess.Board]:
    """python-chess boards (no move history) for the positions in `s`."""
    out = []
    board, white, cast, ep = s.board.cpu().numpy(), s.white.cpu().numpy(), s.castling.cpu().numpy(), s.ep.cpu().numpy()
    hm = s.halfmove.cpu().numpy()
    for i in range(len(s)):
        us = chess.WHITE if white[i] else chess.BLACK
        b = chess.Board(None)
        for cs in range(64):
            v = int(board[i, cs])
            if v:
                sq = cs if us == chess.WHITE else cs ^ 56
                b.set_piece_at(sq, chess.Piece(abs(v), us if v > 0 else not us))
        b.turn = us
        rights = 0
        for (k, q), color in (((cast[i, 0], cast[i, 1]), us), ((cast[i, 2], cast[i, 3]), not us)):
            rank = 0 if color == chess.WHITE else 7
            rights |= (chess.BB_SQUARES[chess.square(7, rank)] if k else 0) | (chess.BB_SQUARES[chess.square(0, rank)] if q else 0)
        b.castling_rights = rights
        b.ep_square = None if ep[i] < 0 else (int(ep[i]) if us == chess.WHITE else int(ep[i]) ^ 56)
        b.halfmove_clock = int(hm[i])
        out.append(b)
    return out


# --------------------------------------------------------------------------- rules
def is_attacked(board: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """board (N, 64) int (canonical); pos (N, K) squares -> (N, K) bool: attacked by the opponent (<0)."""
    T = tables(board.device)
    N, K = pos.shape
    b = board.long()
    near = T["near"][pos]                                              # (N, K, 16)
    pn = b.gather(1, near.clamp(min=0).view(N, -1)).view(N, K, -1)
    pt = (-pn).clamp(min=0)                                            # enemy piece type (0 if none)
    p_exp = pos.unsqueeze(-1).expand_as(near)
    can = T["can"].view(-1)[(pt * 4096 + p_exp * 64 + near.clamp(min=0)).view(-1)].view(N, K, -1)
    same_file = (near % 8) == (p_exp % 8)
    hit_near = (near >= 0) & (pt > 0) & can & ~((pt == PAWN) & same_file)   # pawns capture diagonally only
    far = T["far"][pos]                                                # (N, K, 19)
    pf = b.gather(1, far.clamp(min=0).view(N, -1)).view(N, K, -1)
    ptf = (-pf).clamp(min=0)
    slider = (ptf == QUEEN) | (ptf == ROOK) | (ptf == BISHOP)
    pfe = pos.unsqueeze(-1).expand_as(far)
    canf = T["can"].view(-1)[(ptf * 4096 + pfe * 64 + far.clamp(min=0)).view(-1)].view(N, K, -1)
    btw = T["between"].view(-1, 6)[(pfe * 64 + far.clamp(min=0)).view(-1)].view(N, K, -1, 6)
    occ = b.gather(1, btw.clamp(min=0).view(N, -1)).view(N, K, far.shape[-1], 6)
    clear = ((btw < 0) | (occ == 0)).all(-1)
    hit_far = (far >= 0) & slider & canf & clear
    return hit_near.any(-1) | hit_far.any(-1)


def _compact_squares(mask: torch.Tensor, n: int = 16) -> torch.Tensor:
    """(B, 64) bool -> (B, n) the True squares in increasing order, -1 padded (a side has <= 16 pieces)."""
    B = mask.shape[0]
    pos = mask.long().cumsum(1) - 1
    out = torch.full((B, n + 1), -1, dtype=torch.long, device=mask.device)
    out.scatter_(1, torch.where(mask & (pos < n), pos, n), torch.arange(64, device=mask.device).expand(B, -1))
    return out[:, :n]


def _gather3(b: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """b (B, 64), idx (B, ...) squares (may be -1) -> values (B, ...) (garbage where idx < 0)."""
    return b.gather(1, idx.clamp(min=0).reshape(idx.shape[0], -1)).view(idx.shape)


def enemy_attack_map(b: torch.Tensor) -> torch.Tensor:
    """(B, 64) long canonical board -> (B, 64) bool squares attacked by the opponent's (negative) pieces."""
    T = tables(b.device)
    B = b.shape[0]
    sq = _compact_squares(b < 0)                                       # (B, 16)
    s = sq.clamp(min=0)
    ept = (-b.gather(1, s)).clamp(min=0) * (sq >= 0)                   # enemy piece types
    nonpawn = torch.where(ept == PAWN, 0, ept)
    dests = T["dest"].view(-1, 27)[(nonpawn * 64 + s).view(-1)].view(B, 16, 27)
    btw = T["between"].view(-1, 6)[(s.unsqueeze(-1) * 64 + dests.clamp(min=0)).view(-1)].view(B, 16, 27, 6)
    clear = ((btw < 0) | (_gather3(b, btw) == 0)).all(-1)
    att = torch.zeros((B, 65), dtype=torch.bool, device=b.device)
    att.scatter_(1, torch.where((dests >= 0) & clear, dests, 64).view(B, -1), True)
    pa = T["enemy_pawn_att"][s]                                        # (B, 16, 2)
    att.scatter_(1, torch.where((ept == PAWN).unsqueeze(-1) & (pa >= 0), pa, 64).view(B, -1), True)
    return att[:, :64]


def legal_moves(board: torch.Tensor, castling: torch.Tensor, ep: torch.Tensor, chunk: int = 4096):
    """-> legal (B, 4672) bool, in_check (B,), ep_legal (B,).

    Pseudo-legal moves of my <= 16 pieces, filtered with an enemy attack map (king moves, castling),
    check info (non-king moves must capture / block a single checker) and pins (a pinned piece stays on
    its pin ray) -- found by walking the 8 rays from my king. En passant, whose legality has awkward
    corner cases, is tested directly by playing it on a board copy."""
    if board.shape[0] > chunk:
        parts = [legal_moves(board[i:i + chunk], castling[i:i + chunk], ep[i:i + chunk], chunk)
                 for i in range(0, board.shape[0], chunk)]
        return tuple(torch.cat(x) for x in zip(*parts))
    T = tables(board.device)
    B, dev = board.shape[0], board.device
    ar = torch.arange(B, device=dev)
    b = board.long()
    king = (b == KING).long().argmax(1)
    # ---- squares the opponent attacks, seen through my king (for king moves / castling)
    b_noking = b.clone()
    b_noking[ar, king] = 0
    att = enemy_attack_map(b_noking)
    # ---- checks and pins along the 8 rays from my king
    rays = T["rays"][king]                                             # (B, 8, 7)
    rv = _gather3(b, rays) * (rays >= 0)
    occ = rv != 0
    cum = occ.long().cumsum(-1)
    first, second = occ & (cum == 1), occ & (cum == 2)
    fv, sv = (rv * first).sum(-1), (rv * second).sum(-1)               # (B, 8) first / second piece on each ray
    orth = T["orth"]

    def enemy_slider(v):
        t = (-v).clamp(min=0)
        return (t == QUEEN) | ((t == ROOK) & orth) | ((t == BISHOP) & ~orth)
    check_dir = enemy_slider(fv)
    pin_dir = (fv > 0) & enemy_slider(sv)
    kn = T["knight_sq"][king]
    knight_check = (kn >= 0) & (_gather3(b, kn) == -KNIGHT)
    pc = T["pawn_checker"][king]
    pawn_check = (pc >= 0) & (_gather3(b, pc) == -PAWN)
    n_check = check_dir.sum(1) + knight_check.sum(1) + pawn_check.sum(1)
    in_check = n_check > 0
    blk = torch.zeros((B, 65), dtype=torch.bool, device=dev)          # squares that capture / block the checker
    blk.scatter_(1, torch.where(check_dir.unsqueeze(-1) & ((cum == 0) | first) & (rays >= 0), rays, 64).view(B, -1), True)
    blk.scatter_(1, torch.where(knight_check, kn, 64), True)
    blk.scatter_(1, torch.where(pawn_check, pc, 64), True)
    blk = torch.where((n_check == 0).unsqueeze(1), True, blk[:, :64] & (n_check == 1).unsqueeze(1))
    pin_sq = torch.where(pin_dir, (rays.clamp(min=0) * first).sum(-1), 64)     # (B, 8) square of each pinned piece
    pinned_dir = torch.full((B, 65), -1, dtype=torch.long, device=dev)
    pinned_dir.scatter_(1, pin_sq, torch.arange(8, device=dev).expand(B, -1))
    pinned_dir = pinned_dir[:, :64]
    pin_ray = torch.zeros((B, 8, 65), dtype=torch.bool, device=dev)    # squares a piece pinned in dir d may go to
    pin_ray.scatter_(2, torch.where(pin_dir.unsqueeze(-1) & (cum <= 1) & (rays >= 0) | (second & pin_dir.unsqueeze(-1)), rays.clamp(min=0), 64), True)
    pin_ray = pin_ray[:, :, :64].contiguous()
    # ---- pseudo-legal moves of my pieces
    msq = _compact_squares(b > 0)                                      # (B, 16)
    f = msq.clamp(min=0)
    pt = b.gather(1, f) * (msq >= 0)
    dests = T["dest"].view(-1, 27)[(pt * 64 + f).view(-1)].view(B, 16, 27)
    t = dests.clamp(min=0)
    tgt = _gather3(b, dests)
    ok = (dests >= 0) & (tgt <= 0)
    btw = T["between"].view(-1, 6)[(f.unsqueeze(-1) * 64 + t).view(-1)].view(B, 16, 27, 6)
    ok &= ((btw < 0) | (_gather3(b, btw) == 0)).all(-1)
    is_pawn = (pt == PAWN).unsqueeze(-1)
    same_file = (t % 8) == (f % 8).unsqueeze(-1)
    is_ep = is_pawn & ~same_file & (t == ep.view(B, 1, 1)) & (ep.view(B, 1, 1) >= 0)
    ok &= ~is_pawn | (same_file & (tgt == 0)) | (~same_file & (tgt < 0))
    # ---- legality filters
    king_ok = ~_gather3(att, t)
    pdir = pinned_dir.gather(1, f)                                     # (B, 16)
    pin_ok = (pdir < 0).unsqueeze(-1) | pin_ray.view(B, -1).gather(1, (pdir.clamp(min=0).unsqueeze(-1) * 64 + t).view(B, -1)).view(B, 16, 27)
    other_ok = _gather3(blk, t) & pin_ok
    legal_c = ok & torch.where((pt == KING).unsqueeze(-1), king_ok, other_ok)
    act = T["move_action"].view(-1)[(f.unsqueeze(-1) * 64 + t).view(-1)].view(B, 16, 27)
    mask = torch.zeros((B, E.N_ACTIONS + 1), dtype=torch.bool, device=dev)
    mask.scatter_(1, torch.where(legal_c, act, E.N_ACTIONS).view(B, -1), True)
    # ---- en passant: play the (<= 2) candidates on board copies and test the king directly
    e = ep.clamp(min=0)
    ep_from = torch.stack([e - 9, e - 7], 1)                           # capturing pawns (file - 1, file + 1)
    file_ok = torch.stack([e % 8 > 0, e % 8 < 7], 1)
    cand = (ep >= 0).unsqueeze(1) & file_ok & (_gather3(b, ep_from) == PAWN)
    nb = b.unsqueeze(1).expand(B, 2, 64).clone()
    nb.scatter_(2, ep_from.clamp(min=0).unsqueeze(-1), 0)
    nb.scatter_(2, (e - 8).clamp(min=0).view(B, 1, 1).expand(B, 2, 1), 0)
    nb.scatter_(2, e.view(B, 1, 1).expand(B, 2, 1), PAWN)
    ep_ok = cand & ~is_attacked(nb.view(B * 2, 64), king.repeat_interleave(2).view(-1, 1)).view(B, 2)
    ep_act = T["move_action"].view(-1)[(ep_from.clamp(min=0) * 64 + e.unsqueeze(1)).view(-1)].view(B, 2)
    mask.scatter_(1, torch.where(ep_ok, ep_act, E.N_ACTIONS), True)
    ep_legal = ep_ok.any(1)
    # ---- castling: rights, empty squares, king / transit / destination not attacked
    k_home = b[:, 4] == KING
    oo = castling[:, 0] & k_home & (b[:, 5] == 0) & (b[:, 6] == 0) & (b[:, 7] == ROOK) & ~att[:, 4:7].any(1)
    ooo = castling[:, 1] & k_home & (b[:, 3] == 0) & (b[:, 2] == 0) & (b[:, 1] == 0) & (b[:, 0] == ROOK) & ~att[:, 2:5].any(1)
    mask[:, A_OO] |= oo
    mask[:, A_OOO] |= ooo
    # ---- under-promotions mirror the (queen-)promotion moves of pawns on the 7th rank
    pawn7 = b[:, T["under_from"]] == PAWN
    promo_ok = mask[:, T["under_q"]] & pawn7
    mask[:, T["under_a"].view(-1)] = promo_ok.unsqueeze(-1).expand(-1, -1, 3).reshape(B, -1)
    return mask[:, :E.N_ACTIONS], in_check, ep_legal


def insufficient_material(board: torch.Tensor) -> torch.Tensor:
    """python-chess's `is_insufficient_material` (both sides unable to mate), vectorised."""
    T = tables(board.device)
    a = board.long().abs()
    mine, theirs = board > 0, board < 0
    pawns, knights, bishops = (a == PAWN), (a == KNIGHT), (a == BISHOP)
    bishops_same_colour = ~(bishops & T["dark"]).any(1) | ~(bishops & ~T["dark"]).any(1)

    def side(us, them):
        heavy = (us & (pawns | (a == ROOK) | (a == QUEEN))).any(1)
        has_n, has_b = (us & knights).any(1), (us & bishops).any(1)
        them_minor = (them & (pawns | knights | bishops | (a == ROOK))).any(1)    # ~kings & ~queens
        knight_case = (us.sum(1) <= 2) & ~them_minor
        bishop_case = bishops_same_colour & ~pawns.any(1) & ~knights.any(1)
        return ~heavy & torch.where(has_n, knight_case, torch.where(has_b, bishop_case, torch.ones_like(has_n)))
    return side(mine, theirs) & side(theirs, mine)


def position_hash(s: ChessState) -> torch.Tensor:
    T = tables(s.board.device)
    h = T["z_piece"][T["sq"], s.board.long() + 6].sum(1)
    h = h + s.white.long() * T["z_white"][0] + (s.castling.long() * T["z_castle"]).sum(1)
    h = h + torch.where(s.ep_legal, T["z_ep"][s.ep.clamp(min=0)], 0)
    return torch.where(h == 0, 1, h)                                   # 0 marks "no position" in hist


def finalize(s: ChessState, new_position: bool = True) -> ChessState:
    """Fill in the derived fields (legal moves, check, repetition count, game over) of a new position."""
    s.legal, s.in_check, s.ep_legal = legal_moves(s.board, s.castling, s.ep)
    h = position_hash(s)
    if new_position:
        s.hist = torch.cat([h.unsqueeze(1), s.hist[:, :-1]], dim=1)
    k = torch.arange(HIST, device=h.device)
    window = (k % 2 == 0) & (k.unsqueeze(0) <= s.halfmove.unsqueeze(1))   # same side to move, reversible span
    s.reps = ((s.hist == h.unsqueeze(1)) & window).sum(1)
    no_moves = ~s.legal.any(1)
    insuff = insufficient_material(s.board)
    fifty = s.halfmove >= 100
    rep3 = s.reps >= 3
    s.terminal = no_moves | insuff | fifty | rep3
    s.value = torch.where(no_moves & s.in_check, -1.0, 0.0)
    s.reason = torch.where(no_moves, torch.where(s.in_check, 1, 2),
                           torch.where(insuff, 3, torch.where(fifty, 4, torch.where(rep3, 5, 0))))
    return s


def step(s: ChessState, action: torch.Tensor) -> ChessState:
    """Play `action` (B,) (legal, in the canonical frame) in every game; returns the new (finalized) states."""
    T = tables(s.board.device)
    B, dev = len(s), s.board.device
    ar = torch.arange(B, device=dev)
    f, t, promo = T["a_from"][action], T["a_to"][action], T["a_promo"][action]
    b = s.board.long().clone()
    p = b[ar, f]
    captured = b[ar, t] < 0
    epc = (p == PAWN) & (t == s.ep) & (s.ep >= 0)
    b[ar, torch.where(epc, t - 8, f)] = torch.where(epc, 0, b[ar, torch.where(epc, t - 8, f)])
    oo = (p == KING) & (f == 4) & (t == 6)
    ooo = (p == KING) & (f == 4) & (t == 2)
    castle = oo | ooo
    rf, rt = torch.where(oo, 7, 0), torch.where(oo, 5, 3)
    b[ar, rf] = torch.where(castle, 0, b[ar, rf])
    b[ar, rt] = torch.where(castle, ROOK, b[ar, rt])
    newp = torch.where((p == PAWN) & (t >= 56), torch.where(promo > 0, promo, QUEEN), p)
    b[ar, f] = 0
    b[ar, t] = newp
    cr = s.castling.clone()
    cr[:, 0] &= (f != 4) & (f != 7)
    cr[:, 1] &= (f != 4) & (f != 0)
    cr[:, 2] &= t != 63
    cr[:, 3] &= t != 56
    halfmove = torch.where(captured | (p == PAWN), 0, s.halfmove + 1)
    ep = torch.where((p == PAWN) & (t - f == 16), f + 8, -1)
    # hand the move to the opponent: flip ranks, negate colours, swap castling sides
    nb = (-b).view(B, 8, 8).flip(1).reshape(B, 64).to(torch.int8)
    ns = ChessState(board=nb, white=~s.white, castling=cr[:, [2, 3, 0, 1]], ep=torch.where(ep >= 0, ep ^ 56, -1),
                    halfmove=halfmove, ply=s.ply + 1, hist=s.hist)
    return finalize(ns, new_position=True)


def encode(s: ChessState) -> torch.Tensor:
    """(B, 19, 8, 8) uint8 network input, identical to `chess_env.encode_board`."""
    B, dev = len(s), s.board.device
    b = s.board.long()
    k = torch.arange(1, 7, device=dev)
    pieces = torch.cat([(b.unsqueeze(1) == k.view(1, 6, 1)), (b.unsqueeze(1) == -k.view(1, 6, 1))], 1)   # (B, 12, 64)
    cast = s.castling.unsqueeze(-1).expand(B, 4, 64)
    ep = torch.zeros((B, 1, 64), dtype=torch.bool, device=dev)
    ep[torch.arange(B, device=dev), 0, s.ep.clamp(min=0)] = s.ep_legal
    rep = (s.reps >= 2).view(B, 1, 1).expand(B, 1, 64)
    out = torch.cat([pieces, cast, ep, rep], 1).to(torch.uint8)
    hm = s.halfmove.clamp(max=255).to(torch.uint8).view(B, 1, 1).expand(B, 1, 64)
    return torch.cat([out, hm], 1).view(B, 19, 8, 8)


def compact_legal(legal: torch.Tensor, width: int = E.MAX_LEGAL) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, 4672) mask -> (lidx (B, width) long action indices in increasing order, -1 padded; counts (B,))."""
    B = legal.shape[0]
    pos = legal.long().cumsum(1) - 1
    slot = torch.where(legal, pos, width)
    out = torch.full((B, width + 1), -1, dtype=torch.long, device=legal.device)
    out.scatter_(1, slot, torch.arange(E.N_ACTIONS, device=legal.device).expand(B, -1))
    return out[:, :width], legal.sum(1)


def material_balance(s: ChessState) -> torch.Tensor:
    """White's material minus black's (P=1, N=B=3, R=5, Q=9)."""
    vals = torch.tensor([0, 1, 3, 3, 5, 9, 0], device=s.board.device)
    b = s.board.long()
    mine = vals[b.clamp(min=0)].sum(1) - vals[(-b).clamp(min=0)].sum(1)
    return torch.where(s.white, mine, -mine)
