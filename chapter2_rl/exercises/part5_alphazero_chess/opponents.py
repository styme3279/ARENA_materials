"""
Opponents for evaluating the agent, from weakest to strongest:

    random       uniformly random legal move
    greedy       1-ply search on material + piece-square tables (grabs hanging pieces, sees no replies)
    minimax2     2-ply alpha-beta (sees the opponent's best reply)
    minimax3     3-ply alpha-beta
    sf_skill0    Stockfish at "Skill Level" 0 (deliberately weakened; roughly club-beginner strength)
    sf_elo1350   Stockfish with UCI_LimitStrength at its minimum Elo setting (1350)
    sf_d8 / sf_full   Stockfish searching to depth 8 / 0.1s per move: the strong end ("oracle")

plus `play_match`, which plays the MCTS agent against any of them over a set of openings with all
the agent's searches batched into one network call per simulation.
"""
from __future__ import annotations

import random
import shutil

import chess
import chess.engine
import numpy as np

import mcts as M
from chess_env import adjudicate, termination_reason

STOCKFISH = shutil.which("stockfish") or "/usr/games/stockfish"

# --------------------------------------------------------------------------- openings
# A fixed book of common openings (4-6 plies). Deterministic opponents need varied starting positions;
# each opening is played once with the agent as white and once as black (cf. Connect-4 `two_ply_positions`).
OPENINGS = {
    "start": "",
    "ruy_lopez": "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "italian": "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "petrov": "e2e4 e7e5 g1f3 g8f6",
    "vienna": "e2e4 e7e5 b1c3 g8f6",
    "kings_gambit": "e2e4 e7e5 f2f4 e5f4",
    "sicilian_open": "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "sicilian_nc6": "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4",
    "closed_sicilian": "e2e4 c7c5 b1c3 b8c6",
    "french": "e2e4 e7e6 d2d4 d7d5",
    "caro_kann": "e2e4 c7c6 d2d4 d7d5",
    "scandinavian": "e2e4 d7d5 e4d5 d8d5",
    "alekhine": "e2e4 g8f6 e4e5 f6d5",
    "pirc": "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "qgd": "d2d4 d7d5 c2c4 e7e6",
    "qga": "d2d4 d7d5 c2c4 d5c4",
    "slav": "d2d4 d7d5 c2c4 c7c6",
    "london": "d2d4 d7d5 c1f4 g8f6",
    "kings_indian": "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "nimzo_indian": "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "queens_indian": "d2d4 g8f6 c2c4 e7e6 g1f3 b7b6",
    "benko": "d2d4 g8f6 c2c4 c7c5 d4d5 b7b5",
    "dutch": "d2d4 f7f5 g2g3 g8f6",
    "english": "c2c4 e7e5 b1c3 g8f6",
    "reti": "g1f3 d7d5 g2g3 g8f6",
}


def opening_board(name_or_moves: str) -> chess.Board:
    moves = OPENINGS.get(name_or_moves, name_or_moves)
    b = chess.Board()
    for u in moves.split():
        b.push_uci(u)
    return b


# --------------------------------------------------------------------------- players
class Player:
    name = "player"

    def move(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError

    def close(self):
        pass


class RandomPlayer(Player):
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))


# Piece-square tables (Tomasz Michniewski's "Simplified Evaluation Function"), written rank 8 -> rank 1
# from white's point of view, in centipawns.
_PST_TEXT = {
    chess.PAWN: """ 0  0  0  0  0  0  0  0  50 50 50 50 50 50 50 50  10 10 20 30 30 20 10 10   5  5 10 25 25 10  5  5
                    0  0  0 20 20  0  0  0   5 -5-10  0  0-10 -5  5   5 10 10-20-20 10 10  5   0  0  0  0  0  0  0  0""",
    chess.KNIGHT: """-50-40-30-30-30-30-40-50 -40-20  0  0  0  0-20-40 -30  0 10 15 15 10  0-30 -30  5 15 20 20 15  5-30
                    -30  0 15 20 20 15  0-30 -30  5 10 15 15 10  5-30 -40-20  0  5  5  0-20-40 -50-40-30-30-30-30-40-50""",
    chess.BISHOP: """-20-10-10-10-10-10-10-20 -10  0  0  0  0  0  0-10 -10  0  5 10 10  5  0-10 -10  5  5 10 10  5  5-10
                    -10  0 10 10 10 10  0-10 -10 10 10 10 10 10 10-10 -10  5  0  0  0  0  5-10 -20-10-10-10-10-10-10-20""",
    chess.ROOK: """   0  0  0  0  0  0  0  0   5 10 10 10 10 10 10  5  -5  0  0  0  0  0  0 -5  -5  0  0  0  0  0  0 -5
                    -5  0  0  0  0  0  0 -5  -5  0  0  0  0  0  0 -5  -5  0  0  0  0  0  0 -5   0  0  0  5  5  0  0  0""",
    chess.QUEEN: """-20-10-10 -5 -5-10-10-20 -10  0  0  0  0  0  0-10 -10  0  5  5  5  5  0-10  -5  0  5  5  5  5  0 -5
                     0  0  5  5  5  5  0 -5 -10  5  5  5  5  5  0-10 -10  0  5  0  0  0  0-10 -20-10-10 -5 -5-10-10-20""",
    chess.KING: """ -30-40-40-50-50-40-40-30 -30-40-40-50-50-40-40-30 -30-40-40-50-50-40-40-30 -30-40-40-50-50-40-40-30
                    -20-30-30-40-40-30-30-20 -10-20-20-20-20-20-20-10  20 20  0  0  0  0 20 20  20 30 10  0  0 10 30 20""",
}
_CP = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def _parse_pst(text: str) -> list[int]:
    # numbers are packed like "5-10" -> split on whitespace, then on '-' boundaries
    import re
    vals = [int(x) for x in re.findall(r"-?\d+", text)]
    assert len(vals) == 64, len(vals)
    return vals


def _build_pst():
    """pst[color][piece][square] in centipawns, material included (a1 = 0)."""
    pst = {chess.WHITE: {}, chess.BLACK: {}}
    for piece, text in _PST_TEXT.items():
        t = _parse_pst(text)                     # t[(7 - rank) * 8 + file] for white
        pst[chess.WHITE][piece] = [_CP[piece] + t[(7 - (s // 8)) * 8 + s % 8] for s in range(64)]
        pst[chess.BLACK][piece] = [_CP[piece] + t[(s // 8) * 8 + s % 8] for s in range(64)]
    return pst


_PST = _build_pst()
MATE = 100_000


def static_eval(board: chess.Board) -> int:
    """Material + piece-square evaluation in centipawns, from the side to move's perspective."""
    score = 0
    for sq, piece in board.piece_map().items():
        v = _PST[piece.color][piece.piece_type][sq]
        score += v if piece.color == chess.WHITE else -v
    return score if board.turn == chess.WHITE else -score


def _order(board: chess.Board, moves):
    """MVV-LVA-ish move ordering: captures of valuable pieces by cheap pieces, and promotions, first."""
    def key(m):
        s = 0
        if board.is_capture(m):
            victim = board.piece_type_at(m.to_square) or chess.PAWN          # en passant -> pawn
            s += 10 * _CP.get(victim, 0) - _CP.get(board.piece_type_at(m.from_square), 0) + 10_000
        if m.promotion:
            s += 9_000
        return -s
    return sorted(moves, key=key)


class MinimaxPlayer(Player):
    """Fixed-depth negamax with alpha-beta on `static_eval` (no quiescence search, so it is a genuine
    "n-ply solver": at depth 2 it sees its move and the best reply, and nothing beyond).
    Exactly-tied best moves are broken at random so games from the same opening can differ."""

    def __init__(self, depth: int = 2, seed=None):
        self.depth = depth
        self.name = f"minimax{depth}" if depth > 1 else "greedy"
        self.rng = random.Random(seed)
        self.nodes = 0

    def _negamax(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if depth == 0:                       # leaf: only pay for move generation when in check (mate?)
            if board.is_check() and not any(board.generate_legal_moves()):
                return -(MATE - ply)
            return static_eval(board)
        moves = list(board.legal_moves)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0
        if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
            return 0
        best = -MATE - 1
        for m in _order(board, moves):
            board.push(m)
            s = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if s > best:
                best = s
            if s > alpha:
                alpha = s
            if alpha >= beta:
                break
        return best

    def scored_moves(self, board: chess.Board) -> list[tuple[int, chess.Move]]:
        """Exact negamax score of every root move (full window, so scores are comparable)."""
        b = board.copy()
        out = []
        for m in _order(b, list(b.legal_moves)):
            b.push(m)
            out.append((-self._negamax(b, self.depth - 1, -MATE - 1, MATE + 1, 1), m))
            b.pop()
        return out

    def move(self, board):
        b = board.copy()
        moves = _order(b, list(b.legal_moves))
        alpha, best, best_moves = -MATE - 1, -MATE - 1, []
        for m in moves:
            b.push(m)
            s = -self._negamax(b, self.depth - 1, -MATE - 1, -alpha + 1, 1)   # +1: keep exact ties exact
            b.pop()
            if s > best:
                best, best_moves = s, [m]
            elif s == best:
                best_moves.append(m)
            alpha = max(alpha, s)
        return self.rng.choice(best_moves)


class StockfishPlayer(Player):
    """Stockfish via UCI. `options` are UCI options (e.g. {"Skill Level": 0}); `limit` bounds each move."""

    def __init__(self, name: str, options: dict | None = None, limit: chess.engine.Limit | None = None):
        self.name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
        self.engine.configure({"Threads": 1, "Hash": 16, **(options or {})})
        self.limit = limit or chess.engine.Limit(time=0.05)

    def move(self, board):
        return self.engine.play(board, self.limit).move

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass


OPPONENTS = ["random", "greedy", "minimax2", "minimax3", "sf_skill0", "sf_elo1350"]


def make_opponent(name: str, seed=None) -> Player:
    if name == "random":
        return RandomPlayer(seed)
    if name == "greedy":
        return MinimaxPlayer(1, seed)
    if name.startswith("minimax"):
        return MinimaxPlayer(int(name[len("minimax"):]), seed)
    if name == "sf_skill0":
        return StockfishPlayer(name, {"Skill Level": 0}, chess.engine.Limit(time=0.05))
    if name.startswith("sf_elo"):
        return StockfishPlayer(name, {"UCI_LimitStrength": True, "UCI_Elo": int(name[len("sf_elo"):])},
                               chess.engine.Limit(time=0.05))
    if name.startswith("sf_d"):
        return StockfishPlayer(name, {}, chess.engine.Limit(depth=int(name[len("sf_d"):])))
    if name == "sf_full":
        return StockfishPlayer(name, {}, chess.engine.Limit(time=0.1))
    raise ValueError(f"unknown opponent {name!r}")


class MCTSPlayer(Player):
    """The agent as a plain `Player` (fresh search each move; for interactive use and quick tests)."""

    def __init__(self, evaluator, sims: int = 200, c_puct: float = 2.0, name: str = "agent"):
        self.evaluator, self.sims, self.c_puct, self.name = evaluator, sims, c_puct, name
        self.last = None

    def move(self, board):
        self.last = M.analyse(board, self.evaluator, self.sims, self.c_puct)
        return self.last["best"]


# --------------------------------------------------------------------------- matches
def play_match(evaluator, opponent: Player, openings: list[str], sims: int = 128, c_puct: float = 2.0,
               max_plies: int = 300, adjudicate_threshold: int = 3, stop=None, record_values: bool = True,
               fpu_reduction: float | None = 0.0) -> list[dict]:
    """Play the agent against `opponent` from each opening twice (agent as white, then black).

    All games run in lock-step: every round, the games where the agent is to move are searched
    together (one batched network call per simulation, trees reused across moves); the opponent's
    moves are then played one game at a time. Games still running at `max_plies` are adjudicated on
    material. Returns one dict per game: opening, agent_white, result (agent's score: 1 / 0.5 / 0),
    termination, plies, moves (uci list), values ([ply, agent's root value from white's view] before
    each of its moves).
    """
    games = []
    for op in openings:
        for agent_white in (True, False):
            b = opening_board(op)
            games.append({"opening": op, "agent_white": agent_white, "board": b, "root": M.new_root(b),
                          "moves": [], "values": [], "done": False, "start_fen": b.fen()})
    while True:
        live = [g for g in games if not g["done"]]
        if not live or (stop is not None and stop()):
            break
        for g in live:                                       # end-of-game bookkeeping
            root, b = g["root"], g["board"]
            if root.terminal or len(b.move_stack) >= max_plies:
                g["done"] = True
                if root.terminal:
                    reason = termination_reason(b, len(root.moves))
                    white_score = (0.0 if b.turn == chess.WHITE else 1.0) if reason == "checkmate" else 0.5
                else:
                    reason = "adjudicated"
                    white_score = (adjudicate(b, adjudicate_threshold) + 1) / 2
                g["termination"] = reason
                g["result"] = white_score if g["agent_white"] else 1.0 - white_score
                g["plies"] = len(b.move_stack)
        live = [g for g in games if not g["done"]]
        agent_turn = [g for g in live if (g["board"].turn == chess.WHITE) == g["agent_white"]]
        agent_ids = {id(g) for g in agent_turn}
        if agent_turn:
            roots = [g["root"] for g in agent_turn]
            M.prepare_roots(roots, evaluator)
            M.search(roots, evaluator, sims, c_puct, fpu_reduction)
            for g in agent_turn:
                i = M.choose_move(g["root"], 0.0)
                mv = g["root"].moves[i]
                if record_values:                    # [ply since start, agent's value from white's view]
                    v = g["root"].value
                    g["values"].append([len(g["moves"]), v if g["board"].turn == chess.WHITE else -v])
                g["board"].push(mv); g["moves"].append(mv.uci())
                g["root"] = M.advance(g["root"], mv)
        for g in live:
            if id(g) in agent_ids or g["root"].terminal:
                continue
            mv = opponent.move(g["board"])
            g["board"].push(mv); g["moves"].append(mv.uci())
            g["root"] = M.advance(g["root"], mv)
    for g in games:
        g.pop("root", None)
        g["final_fen"] = g.pop("board").fen()
    return [g for g in games if "result" in g]


def score_summary(results: list[dict]) -> dict:
    """Wins / draws / losses and the agent's mean score over a list of `play_match` results."""
    if not results:
        return {"n": 0}
    r = np.array([g["result"] for g in results])
    return {"n": len(r), "score": float(r.mean()), "wins": int((r == 1).sum()),
            "draws": int((r == 0.5).sum()), "losses": int((r == 0).sum())}
