"""
The Stockfish "oracle": a frozen set of positions where every legal move has been scored by Stockfish,
the chess analogue of the Connect-4 Pons-solver eval set.

Build it once (a few minutes on many cores):
    python oracle.py build --n 3000 --depth 12

Positions come from games between Stockfish at random skill levels with some random moves mixed in,
sampled at random plies, so the set covers openings, middlegames and endgames of "plausible" quality.
Every legal move gets a centipawn score (Stockfish MultiPV over all moves); we also store the moves
the hand-written baselines (greedy / minimax2 / minimax3) would play, as reference points.

`oracle_metrics(evaluator, S)` then scores the raw network (one batched forward, no search). Move quality is
measured the way Lichess does it, in win-probability terms (a centipawn loss means little in a position
that is already +12), with wp = 50 + 50 * (2 / (1 + exp(-0.00368 * cp)) - 1) the "win %" of an eval:
    top1        argmax-prior move is (one of) Stockfish's best moves (mate distances count)
    good        argmax move loses < 5 win-% points (i.e. not even a Lichess "inaccuracy")
    blunder     argmax move loses >= 15 win-% points (a Lichess "blunder")
    wp_loss     mean win-% points lost by the argmax move
    accuracy    mean Lichess move accuracy, 103.17 * exp(-0.04354 * wp_loss) - 3.17, clipped to [0, 100]
    exp_wp_loss expected win-% loss when *sampling* from the prior
    val_mse     value head vs Stockfish's eval mapped to [-1, 1]
    val_sign    value head has the right sign on clearly decided positions (|eval| >= 150cp)
plus the same per phase (opening: ply < 20, middlegame: 20-59, endgame: >= 60), on "balanced" positions
(|eval| <= 300cp, where the game is still open) and on "tactical" ones (the best move is >= 15 win-%
points better than the second best: an only-move).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

from chapter2_rl.exercises.part5_alphazero_chess.chess_env import MAX_LEGAL, N_IN_PLANES, encode_board, legal_actions

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "oracle_positions.json.gz"
CP_CLIP = 10_000                      # mate scores are +-(10000 - plies to mate)


def cp_to_value(cp):
    """Centipawns (side to move) -> expected score in [-1, 1] (the Lichess win% curve, rescaled)."""
    return 2.0 / (1.0 + np.exp(-0.00368208 * np.clip(np.asarray(cp, dtype=np.float64), -2000, 2000))) - 1.0


def move_accuracy(wp_loss):
    """Lichess's per-move accuracy (0..100) from the win-% points a move gave away."""
    return np.clip(103.1668 * np.exp(-0.04354 * np.asarray(wp_loss)) - 3.1669, 0.0, 100.0)


# --------------------------------------------------------------------------- building
def _gen_positions(args):
    """Worker: play noisy Stockfish games, sample one position per game, label every move."""
    n, depth, seed, with_baselines = args
    from chapter2_rl.exercises.part5_alphazero_chess.opponents import STOCKFISH, MinimaxPlayer
    rng = random.Random(seed)
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    eng.configure({"Threads": 1, "Hash": 32})
    greedy, mm2, mm3 = MinimaxPlayer(1, seed), MinimaxPlayer(2, seed), MinimaxPlayer(3, seed)
    out = []
    try:
        while len(out) < n:
            target = int(rng.triangular(4, 140, 30))            # mostly middlegames, some long endgames
            eng.configure({"Skill Level": rng.randrange(0, 21)})
            eps = rng.choice([0.05, 0.1, 0.2])
            b = chess.Board()
            ok = True
            while len(b.move_stack) < target:
                if b.is_game_over(claim_draw=True):
                    ok = False
                    break
                if rng.random() < eps:
                    b.push(rng.choice(list(b.legal_moves)))
                else:
                    b.push(eng.play(b, chess.engine.Limit(time=0.005)).move)
            if not ok or b.is_game_over(claim_draw=True):
                continue
            moves = list(b.legal_moves)
            eng.configure({"Skill Level": 20})
            infos = eng.analyse(b, chess.engine.Limit(depth=depth), multipv=len(moves))
            scores = {}
            for info in infos:
                if "pv" in info and info["pv"]:
                    scores[info["pv"][0].uci()] = info["score"].relative.score(mate_score=10_000)
            if len(scores) < len(moves):
                continue
            rec = {"fen": b.fen(), "ply": len(b.move_stack), "moves": [m.uci() for m in moves],
                   "cp": [int(scores[m.uci()]) for m in moves]}
            if with_baselines:
                rec["greedy"] = greedy.move(b).uci()
                rec["minimax2"] = mm2.move(b).uci()
                rec["minimax3"] = mm3.move(b).uci()
            out.append(rec)
    finally:
        eng.quit()
    return out


def build(n: int, workers: int, depth: int, path: Path = DEFAULT_PATH, seed: int = 0):
    from multiprocessing import get_context
    per = math.ceil(n / workers)
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        chunks = pool.map(_gen_positions, [(per, depth, seed * 1000 + i, True) for i in range(workers)])
    positions = [p for c in chunks for p in c][:n]
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"n": len(positions), "depth": depth, "engine": "stockfish", "built": time.ctime(),
            "seconds": round(time.time() - t0)}
    with gzip.open(path, "wt") as f:
        json.dump({"meta": meta, "positions": positions}, f)
    print(f"wrote {len(positions)} positions to {path} in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- loading
def load_oracle(path: str | Path = DEFAULT_PATH) -> dict:
    """Load the set and precompute everything the metrics need, aligned with each position's legal moves."""
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    pos = data["positions"]
    N = len(pos)
    obs = np.empty((N, N_IN_PLANES, 8, 8), np.uint8)
    lidx = np.full((N, MAX_LEGAL), -1, np.int16)
    cp = np.full((N, MAX_LEGAL), -np.inf, np.float32)
    boards, move_lists = [], []
    baseline_choice = {k: np.zeros(N, np.int64) for k in ("greedy", "minimax2", "minimax3")}
    for i, p in enumerate(pos):
        b = chess.Board(p["fen"])
        moves = list(b.legal_moves)
        score = dict(zip(p["moves"], p["cp"]))
        obs[i] = encode_board(b)
        lidx[i, :len(moves)] = legal_actions(b, moves)
        cp[i, :len(moves)] = np.clip([score[m.uci()] for m in moves], -CP_CLIP, CP_CLIP)
        ucis = [m.uci() for m in moves]
        for k in baseline_choice:
            baseline_choice[k][i] = ucis.index(p[k]) if k in p else 0
        boards.append(b); move_lists.append(moves)
    L = int((lidx >= 0).sum(1).max())
    lidx, cp = lidx[:, :L], cp[:, :L]
    legal = lidx >= 0
    best = cp.max(1)
    wp = 50.0 * (cp_to_value(np.where(legal, cp, 0.0)) + 1.0)          # win % of each move, 0..100
    wp_loss = np.where(legal, wp.max(1, keepdims=True) - wp, 0.0)
    second = np.where(legal.sum(1) > 1, np.sort(np.where(legal, wp_loss, np.inf), 1)[:, 1], 0.0)
    ply = np.array([p["ply"] for p in pos])
    S = {"obs": obs, "lidx": lidx, "cp": cp, "legal": legal, "best": best, "ply": ply, "N": N,
         "boards": boards, "moves": move_lists, "meta": data["meta"],
         "is_best": legal & (cp == best[:, None]), "wp_loss_all": wp_loss,
         "subsets": {"opening": ply < 20, "middlegame": (ply >= 20) & (ply < 60), "endgame": ply >= 60,
                     "balanced": np.abs(best) <= 300, "tactical": second >= 15}}
    S["baselines"] = baseline_metrics(S, baseline_choice)
    return S


# --------------------------------------------------------------------------- metrics
def _choice_metrics(S, choice, mask=None) -> dict:
    m = np.ones(S["N"], bool) if mask is None else mask
    ar = np.arange(S["N"])
    loss = S["wp_loss_all"][ar, choice][m]
    return {"top1": float(S["is_best"][ar, choice][m].mean()), "good": float((loss < 5).mean()),
            "blunder": float((loss >= 15).mean()), "wp_loss": float(loss.mean()),
            "accuracy": float(move_accuracy(loss).mean())}


def baseline_metrics(S, choices: dict) -> dict:
    """Reference points: a uniformly random mover (in expectation) and the stored minimax baselines' moves."""
    legal, wl = S["legal"], S["wp_loss_all"]
    nleg = legal.sum(1)
    avg = lambda x: float((np.where(legal, x, 0).sum(1) / nleg).mean())
    out = {"random": {"top1": avg(S["is_best"]), "good": avg(wl < 5), "blunder": avg(wl >= 15),
                      "wp_loss": avg(wl), "accuracy": avg(move_accuracy(wl))}}
    for k, c in choices.items():
        out[k] = _choice_metrics(S, c)
    return out


def oracle_metrics(evaluator, S, prefix: str = "oracle") -> dict:
    """Score a network (through any mcts-style evaluator) on the oracle set. One batched forward pass."""
    priors, values = evaluator(S["obs"], S["lidx"])
    priors = np.where(S["legal"], priors, 0.0)
    choice = priors.argmax(1)
    sf_v = cp_to_value(S["best"])
    exp_loss = (priors * S["wp_loss_all"]).sum(1)
    out = {}
    for name, mask in [(None, None)] + list(S["subsets"].items()):
        m = np.ones(S["N"], bool) if mask is None else mask
        if m.sum() == 0:
            continue
        key = prefix if name is None else f"{prefix}_{name}"
        for k, v in _choice_metrics(S, choice, m).items():
            out[f"{key}/{k}"] = v
        out[f"{key}/exp_wp_loss"] = float(exp_loss[m].mean())
        out[f"{key}/val_mse"] = float(((values[m] - sf_v[m]) ** 2).mean())
        decided = m & (np.abs(S["best"]) >= 150)
        if decided.any():
            out[f"{key}/val_sign"] = float((np.sign(values[decided]) == np.sign(S["best"][decided])).mean())
    return out


def oracle_metrics_with_search(evaluator, S, sims: int, c_puct: float = 2.0, chunk: int = 512, limit: int | None = None) -> dict:
    """The agent's *played* move (argmax visits after `sims` simulations) scored on the oracle set."""
    import chapter2_rl.exercises.part5_alphazero_chess.mcts as M
    n = S["N"] if limit is None else min(limit, S["N"])
    choice = np.zeros(n, np.int64)
    for s in range(0, n, chunk):
        roots = [M.new_root(b) for b in S["boards"][s:s + chunk]]
        M.prepare_roots(roots, evaluator)
        M.search(roots, evaluator, sims, c_puct)
        for j, r in enumerate(roots):
            choice[s + j] = M.choose_move(r, 0.0)               # r.moves has the same order as S["moves"]
    sub = {k: (v[:n] if isinstance(v, np.ndarray) and v.shape[:1] == (S["N"],) else v) for k, v in S.items()}
    sub["N"] = n
    return _choice_metrics(sub, choice)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=3000)
    b.add_argument("--workers", type=int, default=0, help="default: usable CPUs (cgroup quota)")
    b.add_argument("--depth", type=int, default=12)
    b.add_argument("--out", type=Path, default=DEFAULT_PATH)
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("baselines")
    s.add_argument("--path", type=Path, default=DEFAULT_PATH)
    a = ap.parse_args()
    if a.cmd == "build":
        from chapter2_rl.exercises.part5_alphazero_chess.selfplay import available_cpus
        build(a.n, a.workers or max(1, int(available_cpus())), a.depth, a.out, a.seed)
    else:
        S = load_oracle(a.path)
        for k, v in S["baselines"].items():
            print(f"{k:>9}: " + "  ".join(f"{m}={x:.3f}" for m, x in v.items()))
