"""Tests for the vectorised PyTorch chess rules against python-chess. Run: `python test_torch_chess.py`."""
import random
import sys
import time

import chess
import numpy as np
import torch

import chapter2_rl.exercises.part5_alphazero_chess.chess_env as E
import chapter2_rl.exercises.part5_alphazero_chess.torch_chess as TC

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (name, fen, [perft(1), perft(2), ...]) from https://www.chessprogramming.org/Perft_Results
PERFT = [
    ("start", chess.STARTING_FEN, [20, 400, 8902, 197281]),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", [48, 2039, 97862]),
    ("pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238]),
    ("pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("pos4_mirrored", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", [6, 264, 9467]),
    ("pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379]),
    ("pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079, 89890]),
]


def perft(s: TC.ChessState, depth: int, chunk: int = 4096) -> int:
    if depth == 0:
        return len(s)
    rows, acts = s.legal.nonzero(as_tuple=True)
    if depth == 1:
        return int(rows.numel())
    total = 0
    for i in range(0, rows.numel(), chunk):
        total += perft(TC.step(s.index(rows[i:i + chunk]), acts[i:i + chunk]), depth - 1, chunk)
    return total


def test_perft():
    for name, fen, counts in PERFT:
        s = TC.from_boards([chess.Board(fen)], DEV)
        for d, want in enumerate(counts, start=1):
            got = perft(s, d)
            assert got == want, f"perft {name} depth {d}: got {got}, want {want}"
        print(f"  perft {name}: {counts} ok")
    print("test_perft passed")


def _check_against(pc: chess.Board, s: TC.ChessState, i: int, where: str):
    moves = list(pc.legal_moves)
    want = set(E.legal_actions(pc, moves))
    got = set(s.legal[i].nonzero().flatten().tolist())
    if got != want:
        extra = [E.action_to_move(a, pc).uci() for a in got - want]
        missing = [m.uci() for m, a in zip(moves, E.legal_actions(pc, moves)) if a not in got]
        raise AssertionError(f"{where}: legal moves differ on {pc.fen()}: extra {extra}, missing {missing}")
    obs_t = TC.encode(s.index(torch.tensor([i], device=DEV)))[0].cpu().numpy()
    obs_p = E.encode_board(pc)
    if not np.array_equal(obs_t, obs_p):
        planes = [k for k in range(19) if not np.array_equal(obs_t[k], obs_p[k])]
        raise AssertionError(f"{where}: observation planes {planes} differ on {pc.fen()}")
    tv = E.terminal_value(pc, len(moves))
    term = bool(s.terminal[i])
    assert term == (tv is not None), f"{where}: terminal {term} vs python-chess {tv} on {pc.fen()}"
    if term:
        assert float(s.value[i]) == tv, f"{where}: value {float(s.value[i])} vs {tv}"
        reason = E.termination_reason(pc, len(moves))
        assert TC.REASONS[int(s.reason[i])] == reason, f"{where}: reason {TC.REASONS[int(s.reason[i])]} vs {reason}"
    assert bool(s.white[i]) == (pc.turn == chess.WHITE)


def test_random_games(n_games: int = 256, plies: int = 400, seed: int = 0, shuffle: bool = False):
    """Play random games in lock-step with python-chess, comparing everything at every ply. Half the games
    prefer quiet moves (no captures / pawn moves) so repetitions and 50-move draws actually happen."""
    rng = random.Random(seed)
    pcs = [chess.Board() for _ in range(n_games)]
    quiet = [g % 2 == 1 for g in range(n_games)]
    s = TC.initial_state(n_games, DEV)
    seen = {r: 0 for r in TC.REASONS}
    t0 = time.time()
    for ply in range(plies):
        actions = []
        for i, pc in enumerate(pcs):
            _check_against(pc, s, i, f"game {i} ply {len(pc.move_stack)}")
            if bool(s.terminal[i]):
                seen[TC.REASONS[int(s.reason[i])]] += 1
                actions.append(None)
                continue
            moves = list(pc.legal_moves)
            if shuffle and len(pc.move_stack) >= 2 and rng.random() < 0.7:   # undo my previous move -> repetitions
                prev = pc.move_stack[-2]
                back = chess.Move(prev.to_square, prev.from_square)
                if back in moves and not pc.is_capture(back):
                    moves = [back]
            if quiet[i] and rng.random() < 0.9:
                q = [m for m in moves if not pc.is_capture(m) and pc.piece_type_at(m.from_square) != chess.PAWN]
                moves = q or moves
            m = rng.choice(moves)
            actions.append(E.move_to_action(m, pc.turn))
            pc.push(m)
        done = s.terminal.clone()
        a = torch.tensor([x if x is not None else 0 for x in actions], device=DEV)
        safe = torch.where(done, torch.as_tensor(TC.compact_legal(s.legal)[0][:, 0].clamp(min=0)), a)
        nxt = TC.step(s, safe)
        if bool(done.any()):                                           # restart finished games
            fresh = TC.initial_state(n_games, DEV)
            nxt.assign(done, fresh)
            for i in done.nonzero().flatten().tolist():
                pcs[i] = chess.Board()
        s = nxt
    print(f"test_random_games passed: {n_games} games x {plies} plies in {time.time() - t0:.0f}s; terminations seen: {seen}")


def test_roundtrip_boards():
    pcs = [chess.Board(f) for _, f, _ in PERFT]
    s = TC.from_boards(pcs, DEV)
    for a, b in zip(pcs, TC.to_boards(s)):
        assert a.board_fen() == b.board_fen() and a.turn == b.turn and a.clean_castling_rights() == b.clean_castling_rights()
    print("test_roundtrip_boards passed")


if __name__ == "__main__":
    which = sys.argv[1:] or ["roundtrip", "perft", "random"]
    if "roundtrip" in which:
        test_roundtrip_boards()
    if "perft" in which:
        test_perft()
    if "random" in which:
        test_random_games()
    if "shuffle" in which:
        test_random_games(n_games=128, plies=200, seed=1, shuffle=True)
