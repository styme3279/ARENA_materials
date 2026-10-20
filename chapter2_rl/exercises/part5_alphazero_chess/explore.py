# %%
"""
Poke at a trained chess AlphaZero (run cells in VSCode / Jupyter with `# %%`).

Train first, e.g.  `python train.py --run r1 --hours 6`  (see README.md), then point RUN at the run.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))

import chess
import matplotlib.pyplot as plt
import numpy as np
import torch

import mcts as M
import viz
from model import load_model
from opponents import OPENINGS, make_opponent, play_match, score_summary
from oracle import load_oracle, oracle_metrics, oracle_metrics_with_search

RUN = HERE / "runs" / "r1"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt = load_model(RUN / "latest.pt", dev)
ev = M.LocalEvaluator(model)
print(f"gen {ckpt['gen']}, {ckpt['games']} self-play games, {ckpt['positions'] / 1e6:.1f}M positions, "
      f"{ckpt.get('hours', 0):.1f}h")

# %% Training curves (oracle metrics drawn against the random / greedy / minimax reference lines)
oracle = load_oracle()
for k, v in oracle["baselines"].items():
    print(f"{k:>9}: " + "  ".join(f"{m}={x:.3f}" for m, x in v.items()))
viz.plot_training(RUN, oracle["baselines"])
plt.show()

# %% One position: what does the search think? (cf. the Connect-4 `plot_board_and_policy`)
board = chess.Board()
analysis = M.analyse(board, ev, sims=400)
viz.plot_board_and_policy(board, analysis, title="opening position")
viz.plot_mcts_tree(analysis["root"], max_depth=2, top_k=3, title="search tree (400 sims, top-3 per node)")
plt.show()

# %% Tactics sanity checks: raw policy (no search) vs MCTS
PUZZLES = [
    ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", {"d1d8"}, "mate in one (back rank)"),
    ("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1", {"d2d5"}, "take the hanging queen"),
    ("r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1", {"d5c7"}, "knight fork of king and rook"),
    ("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3", {"g7g6", "d8e7", "d8f6"},
     "defend f7 against Scholar's mate (...Nf6?? Qxf7#)"),
]
for fen, good, desc in PUZZLES:
    b = chess.Board(fen)
    raw = M.analyse(b, ev, sims=1)                      # 1 sim ~ just the prior
    raw_best = max(raw["moves"], key=lambda m: m["prior"])
    a = M.analyse(b, ev, sims=400)
    ok = lambda u: "✓" if u in good else "✗"
    print(f"{desc:<48} prior: {raw_best['san']:<6}{ok(raw_best['uci'])}   search: {a['moves'][0]['san']:<6}"
          f"{ok(a['moves'][0]['uci'])}   value {a['value']:+.2f}")
    viz.plot_board_and_policy(b, a, top_k=8, title=desc)
plt.show()

# %% Oracle set: the raw network, and the agent's played move at several search budgets
print({k: round(v, 3) for k, v in oracle_metrics(ev, oracle).items() if k.startswith("oracle/")})
for sims in [1, 16, 64, 256]:
    r = oracle_metrics_with_search(ev, oracle, sims, limit=1000)
    print(f"sims={sims:4d}: " + "  ".join(f"{k}={v:.3f}" for k, v in r.items()))

# %% A match against one opponent (all agent searches batched), then a replay viewer for it
opponent = make_opponent("minimax2", seed=0)
games = play_match(ev, opponent, list(OPENINGS)[:6], sims=200)
opponent.close()
print(score_summary(games))
viewer = [{"moves": g["moves"], "start_fen": g["start_fen"], "flipped": not g["agent_white"],
           "headers": {"White": "agent" if g["agent_white"] else opponent.name,
                       "Black": opponent.name if g["agent_white"] else "agent",
                       "Opening": g["opening"], "Termination": g["termination"]},
           "values": g["values"]}
          for g in games]
viz.game_viewer_html(viewer, HERE / "match_viewer.html", title="agent vs minimax2")
print("open", HERE / "match_viewer.html")

# %% Replay the ladder games played during training (written by the eval workers)
games = viz.pgn_to_games(RUN / "eval_games.pgn")
viz.game_viewer_html(games[-36:], RUN / "eval_viewer.html", title=f"{RUN.name}: latest ladder games")
print(f"{len(games)} eval games; open", RUN / "eval_viewer.html")

# %% How the network improved: oracle metrics for every saved generation
rows = []
for p in sorted(RUN.glob("gen_*.pt")):
    m, c = load_model(p, dev)
    r = oracle_metrics(M.LocalEvaluator(m), oracle)
    rows.append((c["gen"], r["oracle/top1"], r["oracle/accuracy"], r["oracle/blunder"], r["oracle/val_mse"]))
    print(f"gen {c['gen']:4d}: top1 {rows[-1][1]:.3f}  accuracy {rows[-1][2]:4.1f}  blunder {rows[-1][3]:.3f}  val_mse {rows[-1][4]:.3f}")

# %% Play it yourself in the browser (run in a terminal):
#     python play.py --ckpt runs/r1/latest.pt --sims 400        -> http://localhost:8080
