"""
Ladder evaluation running beside GPU training (`train_gpu.py` starts a few of these automatically).

Loops forever: reload `runs/<run>/latest.pt` when it changes, play a match (agent MCTS, no noise, vs the
next opponent on the ladder over random openings, both colours), and append the result to
`runs/<run>/eval.jsonl` / `eval_games.pgn` -- the same files and format the CPU trainer wrote, so
`viz.py` / `explore.py` work unchanged. The opponents (minimax, Stockfish) are CPU-bound, which is the
point: GPU self-play leaves the CPUs free for this.

    python eval_daemon.py --run r2 --index 0 --of 3
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import torch

import mcts as M
from model import load_model
from opponents import OPENINGS, OPPONENTS, make_opponent, play_match, score_summary
from train import game_pgn

HERE = Path(__file__).resolve().parent


def append_locked(path: Path, text: str) -> None:
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(text)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--index", type=int, default=0, help="this daemon's index (offsets the opponent rotation)")
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--openings", type=int, default=6, help="openings per match (x2 colours)")
    ap.add_argument("--opponents", default=",".join(OPPONENTS))
    ap.add_argument("--max-plies", type=int, default=300)
    a = ap.parse_args()
    torch.set_num_threads(1)
    run = HERE / "runs" / a.run
    opponents = a.opponents.split(",")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(1000 + a.index)
    k, mtime, model, ckpt = a.index, None, None, {}
    while True:
        p = run / "latest.pt"
        if not p.exists():
            time.sleep(10)
            continue
        if p.stat().st_mtime != mtime:
            try:
                mtime = p.stat().st_mtime
                model, ckpt = load_model(p, dev)
            except Exception:                     # caught mid-write: retry shortly
                time.sleep(5)
                continue
        opp_name = opponents[k % len(opponents)]
        k += a.of
        opp = make_opponent(opp_name, seed=int(rng.integers(1 << 30)))
        t0 = time.time()
        gen = ckpt.get("gen", 0)
        try:
            games = play_match(M.LocalEvaluator(model), opp, list(rng.choice(list(OPENINGS), a.openings, replace=False)),
                               sims=a.sims, max_plies=a.max_plies)
        finally:
            opp.close()
        s = score_summary(games)
        rec = {"gen": gen, "opponent": opp_name, "time_h": ckpt.get("hours", 0.0), "seconds": time.time() - t0, **s,
               "terminations": [g["termination"] for g in games]}
        append_locked(run / "eval.jsonl", json.dumps(rec) + "\n")
        pgns = []
        for g in games:
            agent = f"agent-gen{gen}"
            white, black = (agent, opp_name) if g["agent_white"] else (opp_name, agent)
            res = {1.0: "1-0", 0.0: "0-1"}.get(g["result"] if g["agent_white"] else 1 - g["result"], "1/2-1/2")
            pgns.append(game_pgn(g["moves"], {"Event": f"eval vs {opp_name}", "White": white, "Black": black,
                                              "Result": res, "Opening": g["opening"], "Termination": g["termination"]},
                                 g["start_fen"]) + "\n\n")
        append_locked(run / "eval_games.pgn", "".join(pgns))


if __name__ == "__main__":
    main()
