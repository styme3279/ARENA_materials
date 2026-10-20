"""Correctness tests for the chess AlphaZero pieces. Run: `python tests.py` (or `pytest tests.py`)."""
import random

import chess
import numpy as np
import torch

import chess_env as E
import mcts as M
from opponents import MinimaxPlayer, OPENINGS, opening_board

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"          # Rd8#
HANGING_QUEEN = "4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"             # Rxd5
STALEMATE = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
PROMO_FEN = "1n2k3/2P5/8/8/8/8/5p2/4K1N1 w - - 0 1"             # promotions (incl. capture) for both sides


def _random_positions(n=300, seed=0, max_plies=120):
    rng = random.Random(seed)
    out = [chess.Board(PROMO_FEN), chess.Board(PROMO_FEN.replace(" w ", " b "))]
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randrange(max_plies)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            out.append(b)
    return out


def test_move_encoding_roundtrip():
    for b in _random_positions():
        moves = list(b.legal_moves)
        acts = E.legal_actions(b, moves)
        assert len(set(acts)) == len(acts), "two legal moves share an action index"
        assert all(0 <= a < E.N_ACTIONS for a in acts)
        for m, a in zip(moves, acts):
            assert a == E.move_to_action(m, b.turn)
            assert E.action_to_move(a, b) == m, (b.fen(), m, E.action_to_move(a, b))
    # castling = king moves two squares east/west (a queen-style plane), never a special plane
    b = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    a = E.move_to_action(chess.Move.from_uci("e1g1"), chess.WHITE)
    assert a // 64 == 2 * 7 + 1 and a % 64 == chess.E1, "O-O should be 'east, distance 2' from e1"
    print("test_move_encoding_roundtrip passed")


def test_mirror_symmetry():
    """The canonical frame makes a position and its colour-mirror look identical to the network."""
    for b in _random_positions(150, seed=1):
        m = b.mirror()
        assert np.array_equal(E.encode_board(b), E.encode_board(m)), b.fen()
        for mv in b.legal_moves:
            mm = chess.Move(chess.square_mirror(mv.from_square), chess.square_mirror(mv.to_square), mv.promotion)
            assert E.move_to_action(mv, b.turn) == E.move_to_action(mm, m.turn)
    print("test_mirror_symmetry passed")


def test_encode_startpos():
    x = E.encode_board(chess.Board())
    assert x.shape == (19, 8, 8) and x.dtype == np.uint8
    assert x[0, 1].all() and x[0].sum() == 8, "my pawns on my 2nd rank"
    assert x[6, 6].all() and x[6].sum() == 8, "their pawns on their 2nd rank (row 6 in my frame)"
    assert x[5, 0, 4] == 1 and x[11, 7, 4] == 1, "kings on e1 / e8"
    assert x[12:16].all(), "all castling rights"
    assert x[16].sum() == 0 and x[17].sum() == 0 and x[18].max() == 0
    b = chess.Board(); b.push_uci("e2e4")                 # black to move: board flipped, colours swapped
    y = E.encode_board(b)
    assert y[0, 1].all() and y[5, 0, 4] == 1, "black's pawns/king appear on 'my' side"
    assert y[6, 6].sum() == 7 and y[6, 4, 4] == 1, "white's e-pawn on e4 appears on row 4 (rank 5 from black)"
    print("test_encode_startpos passed")


def test_terminal_value():
    b = chess.Board()
    for u in "f2f3 e7e5 g2g4 d8h4".split():
        b.push_uci(u)
    assert E.terminal_value(b, b.legal_moves.count()) == -1.0, "fool's mate: side to move is mated"
    s = chess.Board(STALEMATE)
    assert E.terminal_value(s, s.legal_moves.count()) == 0.0
    assert E.terminal_value(chess.Board(), 20) is None
    k = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    assert E.terminal_value(k, k.legal_moves.count()) == 0.0, "K vs K is a draw"
    r = chess.Board()
    for u in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split():
        r.push_uci(u)
    assert E.terminal_value(r, r.legal_moves.count()) == 0.0, "threefold repetition"
    root = M.new_root(chess.Board())                      # ...and the tree sees it too (stack=8 copies)
    for u in "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8".split():
        root = M.advance(root, chess.Move.from_uci(u))
    assert root.terminal and root.tval == 0.0, "repetition must be detected through tree copies"
    print("test_terminal_value passed")


def test_mcts_finds_mate():
    ev = M.UniformEvaluator()
    # a zero-value net can't see material, so only test that the search itself finds a forced mate
    for flip in (False, True):
        b = chess.Board(MATE_IN_ONE)
        want = chess.Move.from_uci("d1d8")
        if flip:                                     # the same mate for black
            b = b.mirror()
            want = chess.Move(chess.square_mirror(want.from_square), chess.square_mirror(want.to_square))
        res = M.analyse(b, ev, sims=300)
        assert res["best"] == want, f"flip={flip}: MCTS chose {res['best']}, want {want}"
        assert res["moves"][0]["q"] > 0.9, "the mating edge should back up a win"
    # visits sum to sims, and illegal moves don't exist in the tree
    res = M.analyse(chess.Board(), ev, sims=64)
    assert sum(m["visits"] for m in res["moves"]) == 64
    print("test_mcts_finds_mate passed")


def test_batched_search_matches_single():
    """Searching many trees together must give exactly the per-tree results (deterministic evaluator)."""
    torch.manual_seed(0)
    from model import ChessNet
    net = ChessNet(channels=16, blocks=1).eval()
    ev = M.LocalEvaluator(net, torch.device("cpu"))
    boards = [opening_board(o) for o in list(OPENINGS)[:5]]
    roots = [M.new_root(b) for b in boards]
    M.prepare_roots(roots, ev); M.search(roots, ev, 40)
    for b, r in zip(boards, roots):
        single = M.analyse(b, ev, sims=40)["root"]
        assert np.array_equal(single.N, r.N), "batched and single-tree searches disagree"
    print("test_batched_search_matches_single passed")


def test_pipelined_search_matches_plain():
    """selfplay's two-group pipelined search (submit/collect) == mcts.search on the same trees."""
    from model import ChessNet
    from selfplay import _pipelined_search
    torch.manual_seed(0)
    ev = M.LocalEvaluator(ChessNet(channels=16, blocks=1).eval(), torch.device("cpu"))

    class AsyncEv:                                   # the submit/collect interface of RemoteEvaluator
        def submit(self, obs, lidx): self.out = ev(obs, lidx)
        def collect(self): return self.out
    names = list(OPENINGS)[:6]
    plain = [M.new_root(opening_board(o)) for o in names]
    piped = [M.new_root(opening_board(o)) for o in names]
    M.prepare_roots(plain, ev); M.prepare_roots(piped, ev)
    M.search(plain, ev, 30)
    _pipelined_search([piped[:3], piped[3:]], [AsyncEv(), AsyncEv()], 30, 2.0, 0.0)
    for a, b in zip(plain, piped):
        assert np.array_equal(a.N, b.N) and np.allclose(a.W, b.W), "pipelined search disagrees with mcts.search"
    print("test_pipelined_search_matches_plain passed")


def test_minimax():
    assert MinimaxPlayer(2).move(chess.Board(MATE_IN_ONE)).uci() == "d1d8"
    assert MinimaxPlayer(1).move(chess.Board(HANGING_QUEEN)).uci() == "d2d5"
    assert MinimaxPlayer(3).move(chess.Board(HANGING_QUEEN)).uci() == "d2d5"
    # a poisoned pawn: Qxb7?? Rxb7 drops the queen. 1-ply greed grabs it; 2-ply sees the recapture
    b = chess.Board("1r2k3/1p6/8/8/8/8/8/1Q2K3 w - - 0 1")       # Qxb7 Rxb7 drops the queen for a pawn
    assert MinimaxPlayer(1).move(b).uci() == "b1b7", "1-ply greed grabs the pawn"
    assert MinimaxPlayer(2).move(b).uci() != "b1b7", "2-ply sees the recapture"
    print("test_minimax passed")


def test_openings_legal():
    for name in OPENINGS:
        opening_board(name)
    print("test_openings_legal passed")


if __name__ == "__main__":
    test_move_encoding_roundtrip()
    test_mirror_symmetry()
    test_encode_startpos()
    test_terminal_value()
    test_mcts_finds_mate()
    test_batched_search_matches_single()
    test_pipelined_search_matches_plain()
    test_minimax()
    test_openings_legal()
    print("all tests passed")
